from contextlib import contextmanager, ExitStack
from typing import Callable, Any, Sized, Generator, Tuple, Iterable, Optional
import threading
from math import ceil


class ReadWriteLock:
    """A lock object that allows many simultaneous "read locks", but
    only one "write lock." """

    def __init__(self) -> None:
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self, blocking: bool = True, timeout: float = -1) -> bool:
        """Acquire a lock, blocking or non-blocking."""
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
        """Release a read lock."""
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if not self._readers:
                self._read_ready.notifyAll()
        finally:
            self._read_ready.release()

    def acquire_write(self, blocking: bool = True, timeout: float = -1) -> bool:
        """Acquire a lock, blocking or non-blocking. If blocking, wait until there are no
        other acquired read or write locks."""
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
        """Release a write lock."""
        self._read_ready.release()


@contextmanager
def acquire_read(
    rwlock: ReadWriteLock, blocking: bool = True, timeout: float = -1
) -> Generator[bool, None, None]:
    # Despite using ExitStack, we must use this flag (the code after yielding can raise exceptions)
    success = False

    def cleanup():
        if success:
            rwlock.release_read()

    with ExitStack() as stack:
        stack.callback(cleanup)
        success = rwlock.acquire_read(blocking, timeout)
        yield success


@contextmanager
def acquire_write(
    rwlock: ReadWriteLock, blocking: bool = True, timeout: float = -1
) -> Generator[bool, None, None]:
    # Despite using ExitStack, we must use this flag (the code after yielding can raise exceptions)
    success = False

    def cleanup():
        if success:
            rwlock.release_write()

    with ExitStack() as stack:
        stack.callback(cleanup)
        success = rwlock.acquire_write(blocking, timeout)
        yield success


class LockingProgressBarThread(threading.Thread):
    def __init__(self, rwlock: ReadWriteLock, fn: Callable[..., Any]):
        self.progress = -1.0
        self.fn = fn
        self.rwlock = rwlock
        self.title = "Something is coming"
        self.description = "Please wait..."

        super().__init__()

    @staticmethod
    def from_function(rwlock: ReadWriteLock, fn: Callable[..., Any]):
        def funct_wrapper(self):
            fn(thr=self)

        return LockingProgressBarThread(rwlock, funct_wrapper)

    @staticmethod
    def from_composite(
        rwlock: ReadWriteLock,
        fns_collections: Iterable[Tuple[Callable[..., bool], Optional[Sized]]]
    ):

        def funct_wrapper(self):
            for fn, collection in fns_collections:

                if collection is None:
                    self.progress = -1
                    if not fn(thr=self):
                        print(f"Composite exiting ({self.title=}")
                        return
                    print(f"Composite: {self.title=}")
                    continue

                self.progress = 0.0
                length = len(collection)
                for i, x in enumerate(collection):
                    if not fn(x, thr=self):
                        print(f"Composite exiting ({self.title=}")
                        return
                    print(f"Composite: {self.title=}")
                    # Save the progress (between 0 and 1) but with small value subtracted
                    # so the exact value 1.0 is present *after* the lock release
                    self.progress = (i + 1) / length - 1e-4

        return LockingProgressBarThread(rwlock, funct_wrapper)

    @staticmethod
    def from_function_collection(
        rwlock: ReadWriteLock,
        fn: Callable[..., Any],
        collection: Sized,
    ):
        def funct_wrapper(self):
            self.progress = 0.0
            length = len(collection)
            for i, x in enumerate(collection):
                fn(x, thr=self)
                # Save the progress (between 0 and 1) but with small value subtracted
                # so the exact value 1.0 is present *after* the lock release
                self.progress = (i + 1) / length - 1e-4

        return LockingProgressBarThread(rwlock, funct_wrapper)

    def run(self) -> None:
        self.rwlock.acquire_write()
        try:
            self.fn(self)
        finally:
            self.rwlock.release_write()
            self.progress = 1.0



def batched(iterable, k=16):
    n = len(iterable)
    for i in range(ceil(n/k)):
        yield iterable[i*k : (i+1)*k]