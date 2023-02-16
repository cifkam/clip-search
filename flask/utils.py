from contextlib import contextmanager, ExitStack
from typing import Callable, Any, Sized, Generator
import threading

class ReadWriteLock:
    """ A lock object that allows many simultaneous "read locks", but
    only one "write lock." """

    def __init__(self) -> None:
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self, blocking : bool = True, timeout : float = -1) -> bool:
        """ Acquire a lock, blocking or non-blocking. """
        acquired = self._read_ready.acquire(blocking, timeout)
        if acquired:
            try:
                self._readers += 1
            finally:
                self._read_ready.release()
            return True
        else:
            return False

    def release_read(self) -> None:
        """ Release a read lock. """
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if not self._readers:
                self._read_ready.notifyAll()
        finally:
            self._read_ready.release()

    def acquire_write(self, blocking : bool = True, timeout : float = -1) -> bool:
        """ Acquire a lock, blocking or non-blocking. If blocking, wait until there are no
        other acquired read or write locks. """
        acquired = self._read_ready.acquire(blocking, timeout)
        if acquired:
            while self._readers > 0:

                if not blocking:
                    self._read_ready.release()
                    return False

                self._read_ready.wait()
            return True
        else:
            return False

    def release_write(self) -> None:
        """ Release a write lock. """
        self._read_ready.release()

@contextmanager
def acquire_read(rwlock : ReadWriteLock, blocking : bool = True, timeout : float = -1) -> Generator[bool, None, None]:
    # Despite using ExitStack, we must use this flag (the code after yielding can raise exceptions)
    success = False
    def cleanup():
        if success : rwlock.release_read()
    with ExitStack() as stack:
        stack.callback(cleanup)
        success = rwlock.acquire_read(blocking, timeout)
        yield success

@contextmanager
def acquire_write(rwlock : ReadWriteLock, blocking : bool = True, timeout : float = -1) -> Generator[bool, None, None]:
    # Despite using ExitStack, we must use this flag (the code after yielding can raise exceptions)
    success = False
    def cleanup():
        if success : rwlock.release_write()
    with ExitStack() as stack:
        stack.callback(cleanup)
        success = rwlock.acquire_write(blocking, timeout)
        yield success

class LockingProgressBarThread(threading.Thread):
    def __init__(self, fn : Callable[..., Any], collection : Sized, rwlock : ReadWriteLock, *fn_args, **fn_kwargs):
        self.progress = -1.0
        self.iterable = collection
        self.funct = fn
        self.rwlock = rwlock
        self.length = len(collection)

        self.fn_args = fn_args
        self.fn_kwargs = fn_kwargs
        super().__init__()

    def run(self) -> None:
        self.rwlock.acquire_write()
        try:
            self.progress = 0.0
            for i, x in enumerate(self.iterable):
                self.funct(x, *self.fn_args, **self.fn_kwargs)
                # Save the progress (between 0 and 1) but with small value subtracted so the exact value 1.0 is present *after* the lock release
                self.progress = (i+1)/self.length - 1e-4
        finally:
            self.rwlock.release_write()
            self.progress = 1.0
            