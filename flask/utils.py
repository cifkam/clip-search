import threading

class LockingProgressBarThread(threading.Thread):
    def __init__(self, fn, collection, rwlock, *fn_args, **fn_kwargs):
        self.progress = -1.0
        self.iterable = collection
        self.funct = fn
        self.rwlock = rwlock
        self.length = len(collection)

        self.fn_args = fn_args
        self.fn_kwargs = fn_kwargs
        super().__init__()

    def run(self):
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


class ReadWriteLock:
    """ A lock object that allows many simultaneous "read locks", but
    only one "write lock." """

    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self, blocking=True, timeout=-1):
        """ Acquire a read lock. Blocks only if a thread has
        acquired the write lock. """
        acquired = self._read_ready.acquire(blocking, timeout)
        if acquired:
            try:
                self._readers += 1
            finally:
                self._read_ready.release()
            return True
        else:
            return False

    def release_read(self):
        """ Release a read lock. """
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if not self._readers:
                self._read_ready.notifyAll()
        finally:
            self._read_ready.release()

    def acquire_write(self, blocking=True, timeout=-1):
        """ Acquire a write lock. Blocks until there are no
        acquired read or write locks. """
        acquired = self._read_ready.acquire(blocking, timeout)
        if acquired:
            while self._readers > 0:
                self._read_ready.wait()
            return True
        else:
            return False

    def release_write(self):
        """ Release a write lock. """
        self._read_ready.release()