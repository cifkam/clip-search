from flask import render_template, request, send_from_directory, redirect, abort
from PIL import Image
from ImageManager import ImageManager
from utils import LockingProgressBarThread, ReadWriteLock, acquire_read, acquire_write
from settings import settings

class Views:
    thr = None
    
    def __init__(self) -> None:
        self.imanager = ImageManager(
            model_name=settings.MODEL_NAME,
            prefer_cuda=settings.PREFER_CUDA
        )
        self.rwlock = ReadWriteLock()

    def progressbar_lock(*, write=False, blocking=True, timeout=0.5):
        """Returns decorator that acquires `self.rwlock` before running the decorated function and returning its result.\n
        If `write==True` the decorator tries to acquire the lock for writing (i.e. with exclusive access).\n
        If `blocking==False` or the waiting for the lock is longer than `timeout`, then returns progressbar (`progress.html`).
        A timeout argument of -1 specifies an unbounded wait. It is forbidden to specify a timeout when blocking is False."""

        def decorator(function):
            def wrapper(self, *args,**kwargs):

                acquire_lock = acquire_write if write else acquire_read
                with acquire_lock(self.rwlock, blocking, timeout) as success:
                    if success:
                        # If acquire_lock was successful, run the function
                        return function(self, *args, **kwargs)
                    else:
                        # Else just show page with progressbar
                        return render_template('progress.html')

            return wrapper
        return decorator

    @staticmethod
    def process_query_result(result):
        return [('/'+x.path, f"/search/id/{x.id}") for x in result]


    @progressbar_lock()
    def index(self):
        return render_template('index.html')

    @progressbar_lock()
    def search(self):
        if request.method == 'POST':
            # Search by image
            try:
                img = Image.open(request.files['upload'])
            except Exception as e:
                abort(415, e)

            result = self.process_query_result(self.imanager.query_image(img, k=settings.QUERY_K))
            return render_template("results.html", result=result)
        
        elif 'q' in request.args and request.args['q'] != "":
            # Search by text
            text = request.args['q']
            print(f"Query: {text}")
            result = self.process_query_result(self.imanager.query_text(text, k=settings.QUERY_K))

            return render_template("results.html", result=result)
        
        return redirect("/")

    @progressbar_lock()
    def search_by_id(self, id):
        try:
            f = float(id)
            id = int(f)
            if id < 1 or id != f:
                raise ValueError()
        except ValueError:
            abort(400)
        result = self.process_query_result(self.imanager.query_id(id, k=settings.QUERY_K))
        return render_template("results.html", result=result)

    @progressbar_lock()
    def classification(self):
        html = 'classification.html'

        if request.method == 'POST':
            labels = request.form['labels']

            try:
                img = Image.open(request.files['upload'])
            except Exception as e:
                abort(415, e)

            labels = list(filter(None, request.form['labels'].splitlines()))
            result = self.imanager.clip.classify(img, labels)
            return render_template(html, result=list(result.items()))
        else:
            return render_template(html)

    def settings(self):
        if request.method == 'POST':
            return self.settings_post()
        else:
            return self.settings_get()

    @progressbar_lock()
    def settings_get(self):
        return render_template('settings.html')

    # No @progress_rwlock() here, we need to handle it manually
    def settings_post(self):
        action = request.form['action']
        if action == 'refresh':
            pass
        elif action == 'save':
            pass

        import time
        # Try to acquire_write (unique access)
        with acquire_write(self.rwlock, True, 1.0) as success:
            # If acquired and EITHER there was no previous self.thr OR there was and it has already finished:
            if success and (self.thr is None or self.thr.progress == 1.0):
                # Start a new thread with our function (and with unique lock)
                self.thr = LockingProgressBarThread(lambda x: time.sleep(1), list(range(5)), self.rwlock)
                self.thr.start()

        return redirect("/settings")

    def progress_status(self):
        if self.thr is None:
            return "1.0"
        else:
            return str(self.thr.progress)

    def error(self, description, title='Error'):
        return render_template('error.html', title=title, description=description)

    def get_db_image(self, filename, as_attachment=False):
        return send_from_directory(settings.DB_IMAGES_ROOT, filename, as_attachment=False)
