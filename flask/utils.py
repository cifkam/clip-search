import threading

class LockingProgressBarProcess(threading.Thread):
    def __init__(self, funct, iterable, rwlock, length=None):
        self.progress = -1.0
        self.collection = iterable
        self.funct = funct
        self.rwlock = rwlock
        self.length = len(iterable) if length is None else length
        super().__init__()

    def run(self):
        self.rwlock.acquire_write()
        try:
            self.progress = 0.0
            for i, x in enumerate(self.collection):
                self.funct(x)
                self.progress = (i+1) / self.length
            self.progress = 1.0
        finally:
            self.rwlock.release_write()


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