from flask import render_template, request, send_from_directory, redirect, abort
from PIL import Image
from ImageManager import ImageManager
from utils import LockingProgressBarProcess, ReadWriteLock
from settings import QUERY_K, MODEL_NAME, PREFER_CUDA, DB_IMAGES_ROOT

class Views:
    thr = None
    
    def __init__(self) -> None:
        self.imanager = ImageManager(
            model_name=MODEL_NAME,
            prefer_cuda=PREFER_CUDA
        )
        self.rwlock = ReadWriteLock()

    def progressbar_lock(blocking=True, timeout=0.5):
        def decorator(function):
            def wrapper(*args,**kwargs):
                # Try to run the function with unique lock
                acquired = args[0].rwlock.acquire_read(blocking, timeout)
                if acquired:
                    try:
                        ret = function(*args, **kwargs)
                    finally:
                        args[0].rwlock.release_read()
                    return ret
                else:
                    # If cannot acquire lock, show the progressbar page
                    return render_template('progress.html')
            return wrapper
        return decorator


    @progressbar_lock()
    def index(self):
        return render_template('index.html')

    @progressbar_lock()
    def search(self):
        if 'q' in request.args and request.args['q'] != "":
            text = request.args['q']
            print(f"Query: {text}")
            result = self.imanager.query_text(text, k=QUERY_K)
            paths = ['/'+x.path for x in result]
            return render_template("results.html", result=paths)
        return render_template('search.html')

    @progressbar_lock()
    def image_search(self):
        if request.method == 'POST':
            try:
                img = Image.open(request.files['upload'])
            except Exception as e:
                abort(415, e)

            result = self.imanager.query_image(img, k=QUERY_K)
            paths = ['/'+x.path for x in result]
            return render_template("results.html", result=paths)
        else:
            return render_template('image_search.html')

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

    def settings_post(self):
        import time

        # Try to acquire_write (unique access)
        acquired = self.rwlock.acquire_write(True, 1.0)
        
        # If acquired and EITHER there was no previous self.thr OR the previous self.thr has already finished:
        if acquired and (self.thr is None  or  self.thr.progress == 1.0):
            try:
                # Start a new thread with our function (and with unique lock)
                self.thr = LockingProgressBarProcess(lambda x: time.sleep(1), list(range(10)), self.rwlock)
                self.thr.start()
            finally:
                self.rwlock.release_write()

        return redirect("/settings")


    def progress_status(self):
        if self.thr is None:
            return "-1"
        else:
            return str(self.thr.progress)

    def error(self, description, title='Error'):
        return render_template('error.html', title=title, description=description)

    def get_db_image(self, filename, as_attachment=False):
        return send_from_directory(DB_IMAGES_ROOT, filename, as_attachment=False)
