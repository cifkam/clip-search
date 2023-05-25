from flask import render_template, request, send_from_directory, redirect, abort
from PIL import Image
from ImageManager import ImageManager
from utils import LockingProgressBarThread, ReadWriteLock, acquire_read, acquire_write
from settings import settings
from itertools import islice
from EmbeddingTagCache import EmbeddingTagCache
import secrets
import clip
from time import sleep
import functools

class Views:
    thr = None

    def __init__(self) -> None:
        self.load_image_manager()
        self.progressbar_rwlock = ReadWriteLock()
        self.progressbar_title = ""
        self.progressbar_description = ""
        self.embedding_tag_cache = EmbeddingTagCache()
        self.session_ids = set()

    def progressbar_lock(title="Something is comming...", description="Oh no! You have to wait for a while...",
            *, write=False, blocking=True, timeout=0.5, progress_unknown=False
    ):
        """Returns decorator that acquires `self.progressbar_rwlock` before running the decorated function
        and returning its result.\n
        If `write==True` the decorator tries to acquire lock for writing (with exclusive access).
        If `blocking==False` or the waiting is longer than `timeout`, returns page `progress.html`.
        A timeout argument of -1 specifies an unbounded wait.
        It is forbidden to specify a timeout when blocking is False."""

        def decorator(function):
            def wrapper(self, *args, **kwargs):

                acquire_lock = acquire_write if write else acquire_read
                with acquire_lock(self.progressbar_rwlock, blocking, timeout) as success:
                    if success:
                        self.progress_title = title
                        self.progress_description = description
                        # If acquire_lock was successful, run the function
                        return function(self, *args, **kwargs)
                    else:
                        # Else just show page with progressbar
                        return render_template(
                            "progress.html", title=self.progressbar_title, description=self.progressbar_description
                        )

            return wrapper

        return decorator

    def load_image_manager(self):
        self.imanager = ImageManager(
            model_name=settings.MODEL_NAME, prefer_cuda=settings.PREFER_CUDA
        )

    @staticmethod
    def process_query_result(result, page=1):
        # Get the correct "page" of results
        result = islice(result, settings.QUERY_K * (page - 1), settings.QUERY_K * page)

        # Map it to pairs ('/url_for_file', '/url_for_search_by_id')
        return [("/" + x.path, f"/search/id/{x.id}") for x in result]

    @staticmethod
    def parse_int(string):
        try:
            f = float(string)
            i = int(f)
        except ValueError:
            return None
        if i == f:
            return i
        else:
            return None

    @staticmethod
    def render_settings(error_msg=None):
        return render_template(
            "settings.html",
            models=clip.available_models(),
            model_selected=settings.MODEL_NAME,
            results_per_page=[15, 20, 25, 30, 40, 50],
            results_per_page_selected=settings.QUERY_K,
            error_msg=error_msg,
        )

    @staticmethod
    def render_search_results(result, page, args=None):
        if args is None:
            args = dict()

        if page > 1:
            prev_args = dict(args)
            prev_args["page"] = str(page - 1)
            prev_href = "?" + "&".join(
                [f"{key}={value}" for key, value in prev_args.items()]
            )
        else:
            prev_href = None

        if len(result) >= settings.QUERY_K:
            next_args = dict(args)
            next_args["page"] = str(page + 1)
            next_href = "?" + "&".join(
                [f"{key}={value}" for key, value in next_args.items()]
            )
        else:
            next_href = None

        return render_template(
            "results.html", result=result, prev_href=prev_href, next_href=next_href
        )

    def query_image(self, img, page=1):
        result = self.imanager.query_image(img, k=settings.QUERY_K * page)
        return self.process_query_result(result, page)

    def query_text(self, text, page=1):
        result = self.imanager.query_text(text, k=settings.QUERY_K * page)
        return self.process_query_result(result, page)

    def query_id(self, id, page=1):
        result = self.imanager.query_id(id, k=settings.QUERY_K * page)
        return self.process_query_result(result, page)

    def query_embedding(self, embedding, page=1):
        result = self.imanager.query(embedding, k=settings.QUERY_K * page)
        return self.process_query_result(result, page)

    @progressbar_lock()
    def index(self):
        return render_template("index.html")

    @progressbar_lock()
    def search_cached(self, tag):
        if "page" in request.args and request.args["page"] != "":
            page = self.parse_int(request.args["page"])
            if page is None:
                page = 1
        else:
            page = 1

        try:
            embedding = self.embedding_tag_cache.get(tag, request.cookies["session_id"])
        except KeyError:
            return self.error(
                "Your query session has been already released,\
                               please upload your image again."
            )

        result = self.query_embedding(embedding, page)
        return self.render_search_results(result, page, request.args)

    @progressbar_lock()
    def search(self):
        if "page" in request.args and request.args["page"] != "":
            page = self.parse_int(request.args["page"])
            if page is None:
                page = 1
        else:
            page = 1

        if request.method == "POST":
            # Search by image
            try:
                img = Image.open(request.files["upload"])
            except Exception as e:
                abort(415, e)

            cookies = request.cookies
            if (
                "session_id" not in cookies
                or cookies["session_id"] not in self.session_ids
            ):
                result = self.query_image(img, page)
                return self.render_search_results(result, page, request.args)
            else:
                embedding = self.imanager.embed_image(img)
                tag = self.embedding_tag_cache.add(embedding, cookies["session_id"])
                return redirect(f"/search/img/{tag}")

        elif "q" in request.args and request.args["q"] != "":
            # Search by text
            text = request.args["q"]
            print(f"Query: {text}")
            result = self.query_text(text, page)
            return self.render_search_results(result, page, request.args)

        return redirect("/")

    @progressbar_lock()
    def search_by_id(self, id):
        id = self.parse_int(id)
        if id is None or id < 1:
            abort(400)

        if "page" in request.args and request.args["page"] != "":
            page = self.parse_int(request.args["page"])
            if page is None:
                page = 1
        else:
            page = 1

        result = self.query_id(id, page)
        return self.render_search_results(result, page, request.args)

    @progressbar_lock()
    def classification(self):
        html = "classification.html"

        if request.method == "POST":
            labels = request.form["labels"]

            try:
                img = Image.open(request.files["upload"])
            except Exception as e:
                abort(415, e)

            labels = list(filter(None, request.form["labels"].splitlines()))
            result = self.imanager.clip.classify(img, labels)
            return render_template(html, result=list(result.items()))
        else:
            return render_template(html)

    def settings(self):
        if request.method == "POST":
            return self.settings_post()
        else:
            return self.settings_get()

    @progressbar_lock()
    def settings_get(self):
        return self.render_settings()

    # No @progressbar_lock() here, we need to handle it manually
    def settings_post(self):
        action = request.form["action"]
        model_change = None
        backup = None

        def set_and_save_settings():
            nonlocal model_change
            nonlocal backup
            backup = settings.get_values()
            K = int(request.form["results_per_page"])
            model_name = request.form["model"]

            model_change = request.form["model"] != settings.MODEL_NAME
            settings.MODEL_NAME = model_name
            settings.QUERY_K = K

            if not settings.save():
                settings.set_values(backup)
                return False
            return True

        def try_lock_and_run(composite):
            if model_change or self.imanager.kdtree is None:
                with acquire_write(self.progressbar_rwlock, True, 1.0) as success:
                    # If acquired and EITHER there was no previous self.thr
                    # OR there was and it has already finished:
                    if success and (self.thr is None or self.thr.progress == 1.0):
                        # Start a new thread with our function (and with unique lock)

                        self.thr = LockingProgressBarThread.from_composite(
                            self.progressbar_rwlock, composite
                        )
                        self.thr.start()

        def func_save(thr=None):
            thr.title = "Trying to load new model..."
            # 3. clear self.embedding_tag_cache
            self.embedding_tag_cache = EmbeddingTagCache()
            self.load_image_manager()
            if self.imanager.kdtree is None:
                settings.set_values(backup)
                settings.save()
                self.load_image_manager()
            return True

        def func_refresh(thr=None):
            thr.title = "Trying to load new model..."
            # 3. clear self.embedding_tag_cache
            self.embedding_tag_cache = EmbeddingTagCache()
            self.load_image_manager()

            thr.title = "Refreshing the database..."
            #TODO: RELOAD APP AND REFRESH! 
            return True

        if action == "save":
            print("action: save")
            # TODO:
            # 1. Set settings and save it to json
            # 2. try to load self.immanger with new model name, but do not change neither
            # the kdtree nor the database - if the kdtree file cannot be found, show
            # the settings with some error notification
            if not set_and_save_settings():
                return self.render_settings(error_msg="Error: Couldn't save settings!")
            
            try_lock_and_run([(func_save, None)])
            return redirect("/settings")

        elif action == "refresh":
            print("action: refresh")
            # TODO:
            # 1. Set settings and save it to json
            # 2. clear self.embedding_tag_cache
            # 3. try to load self.immanager with new model name, and refresh the database
            # and create new kdtree (and save it to file)
            if not set_and_save_settings():
                return self.render_settings(error_msg="Error: Couldn't save settings!")
            
            self.progressbar_title = "Refreshing the database..."
            self.progressbar_description = "This may take a while..."
            try_lock_and_run([(func_refresh, None)])

        elif action == "reset":
            print("action: reset")
            # TODO: !!!
            # 1. Set settings and save it to json
            if not set_and_save_settings():
                return self.render_settings(error_msg="Error: Couldn't save settings!")
            # 2. clear self.embedding_tag_cache
            self.embedding_tag_cache = EmbeddingTagCache()
            # 3. similar to 'refresh' but clear the databse and kdtree
            # and create it from scratch, instead of refreshing it

        else:
            raise Exception(f'Settings: Received unknown action "{action}"!')

        import time

        # TODO: This is just for testing, remove it later and replace with a proper implementation
        # Try to acquire_write (unique access)
        with acquire_write(self.progressbar_rwlock, True, 1.0) as success:
            # If acquired and EITHER there was no previous self.thr OR it has already finished:
            if success and (self.thr is None or self.thr.progress == 1.0):
                # Start a new thread with our function (and with unique lock)

                def fn_a(x, thr=None):
                    thr.title = "A"
                    time.sleep(1)

                def fn_b(thr=None):
                    thr.title = "B"
                    time.sleep(5)

                self.progressbar_title = "Testing"
                self.progressbar_description = "Testing description"
                self.thr = LockingProgressBarThread.from_composite(
                    self.progressbar_rwlock, [(fn_a, list(range(5))), (fn_b, None), (fn_a, list(range(5)))]
                )
                #
                # self.thr = LockingProgressBarThread.from_function_collection(
                #    self.progressbar_rwlock, lambda x: time.sleep(1), list(range(5))
                # )
                # self.thr = LockingProgressBarThread.from_function(
                #    self.progressbar_rwlock,
                #    lambda: time.sleep(5))
                self.thr.start()

        return redirect("/settings")

    def progress_status(self):
        if self.thr is None:
            return "1.0"
        else:
            return str(self.thr.progress)

    def session_id(self):
        if "session_id" in request.cookies:
            id = request.cookies["session_id"]
            if id in self.session_ids:
                return id

        id = secrets.token_hex(16)
        while id in self.session_ids:
            id = secrets.token_hex(16)
        self.session_ids.add(id)
        return id

    def get_db_image(self, filename, as_attachment=False):
        return send_from_directory(settings.DB_IMAGES_ROOT, filename, as_attachment=as_attachment)
    
    def error(self, description, title="Error"):
        return render_template("error.html", title=title, description=description)
    

    def shutdown(self, conn):
        with acquire_write(self.progressbar_rwlock, True, 1.0) as success:
            # If acquired and EITHER there was no previous self.thr
            # OR there was and it has already finished:
            if success and (self.thr is None or self.thr.progress == 1.0):
                # Start a new thread with our function (and with unique lock)
                
                self.progressbar_title = "Shutting down"
                self.progressbar_description = "The application is shutting down..."
                
                def dummy_func(thr, secs=10):
                    sleep(10)

                self.thr = LockingProgressBarThread.from_function(
                    self.progressbar_rwlock, dummy_func)
                self.thr.start()

            else:
                return render_template("error.html",
                    title="Couldn't shutdown the application",
                    description="Application shutting down failed, please try again in a moment.")
            
        if conn is not None:
            conn.send("shutdown")
            conn.close()

        return redirect("/")

    def restart(self, conn=None):
        with acquire_write(self.progressbar_rwlock, True, 1.0) as success:
            # If acquired and EITHER there was no previous self.thr
            # OR there was and it has already finished:
            if success and (self.thr is None or self.thr.progress == 1.0):
                # Start a new thread with our function (and with unique lock)
                
                self.progressbar_title = "Restarting"
                self.progressbar_description = "The application is restarting. Please reload the page in a few seconds..."
                
                def dummy_func(thr, secs=10):
                    sleep(10)

                self.thr = LockingProgressBarThread.from_function(
                    self.progressbar_rwlock, dummy_func)
                self.thr.start()

            else:
                return render_template("error.html",
                    title="Couldn't restart the application",
                    description="Application restarting failed, please try again in a moment.")
            
        if conn is not None:
            conn.send("restart")
            conn.close()

        return redirect("/")
    

