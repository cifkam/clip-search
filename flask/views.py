from flask import render_template, request, send_from_directory, redirect, abort
from PIL import Image
from ImageManager import ImageManager
from utils import LockingProgressBarThread, ReadWriteLock, acquire_read, acquire_write
from settings import settings
from itertools import islice
from EmbeddingTagCache import EmbeddingTagCache
import secrets
import clip
import json
from time import sleep
import urllib
import functools

HTTP_BAD_REQUEST = 400
HTTP_UNSUPPORTED_MEDIA_TYPE = 415

class Views:
    thr = None

    def __init__(self, app, runner_conn=None) -> None:
        self.app = app
        self.runner_conn = runner_conn
        self.progressbar_rwlock = ReadWriteLock()
        self.progressbar_description = ""
        self.embedding_tag_cache = EmbeddingTagCache()
        self.session_ids = set()
        
        self.load_image_manager()

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
                            "progress.html"
                            #,title=self.thr.title, description=self.thr.description
                        )

            return wrapper

        return decorator

    def  load_image_manager(self, create_new_kdtree=True):
        self.imanager = ImageManager(
            model_name=settings.MODEL_NAME, prefer_cuda=settings.PREFER_CUDA
        )

        if create_new_kdtree and self.imanager.kdtree is None:
            print("Kdtree not found, building new...")
            self.imanager.full_refresh()

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
                [f"{key}={urllib.parse.quote(value)}" for key, value in prev_args.items()]
            )
        else:
            prev_href = None

        if len(result) >= settings.QUERY_K:
            next_args = dict(args)
            next_args["page"] = str(page + 1)
            next_href = "?" + "&".join(
                [f"{key}={urllib.parse.quote(value)}" for key, value in next_args.items()]
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
                abort(HTTP_UNSUPPORTED_MEDIA_TYPE, e)

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
            abort(HTTP_BAD_REQUEST)

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
                abort(HTTP_UNSUPPORTED_MEDIA_TYPE, e)

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

        def func_restart(thr=None):
            with self.app.app_context():
                thr.title = "Restarting"
                thr.description = "Application is restarting. Please reload the page in a few seconds."

                if self.runner_conn is not None:
                    self.runner_conn.send("restart")
                    self.runner_conn.close()
                    
                sleep(10)
                return True

        if action == "save":
            print("action: save")
            if not set_and_save_settings():
                return self.render_settings(error_msg="Error: Couldn't save settings!")
            
            if model_change:
                try_lock_and_run([(func_restart, None)])
                return redirect("/settings")
        else:
            raise Exception(f'Settings: Received unknown action "{action}"!')
        
        return redirect("/settings/")

    def progress_status(self):
        if self.thr is None:
            return json.dumps({"progress": 1.0, "title":"Something is coming", "description":"Please wait..."})
        else:
            return json.dumps({"progress": self.thr.progress, "title": self.thr.title, "description": self.thr.description})
        
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
    

    def shutdown(self):
        with acquire_write(self.progressbar_rwlock, True, 1.0) as success:
            # If acquired and EITHER there was no previous self.thr
            # OR there was and it has already finished:
            if success and (self.thr is None or self.thr.progress == 1.0):
                # Start a new thread with our function (and with unique lock)
                
                def dummy_func(thr, secs=10):
                    thr.title = "Shutting down"
                    thr.description = "The application is shutting down... Refresh the page to check if it's still running."
                    sleep(10)

                self.thr = LockingProgressBarThread.from_function(
                    self.progressbar_rwlock, dummy_func)
                self.thr.start()

            else:
                return render_template("error.html",
                    title="Couldn't shutdown the application",
                    description="Application shutting down failed, please try again in a moment.")
            
        if self.runner_conn is not None:
            self.runner_conn.send("shutdown")
            self.runner_conn.close()

        return redirect("/")

    def restart(self):
        with acquire_write(self.progressbar_rwlock, True, 1.0) as success:
            # If acquired and EITHER there was no previous self.thr
            # OR there was and it has already finished:
            if success and (self.thr is None or self.thr.progress == 1.0):
                # Start a new thread with our function (and with unique lock)
                
                def dummy_func(thr, secs=10):
                    thr.title = "Restarting"
                    thr.description = "The application is restarting. Please reload the page in a few seconds..."
                    sleep(10)

                self.thr = LockingProgressBarThread.from_function(
                    self.progressbar_rwlock, dummy_func)
                self.thr.start()

            else:
                return render_template("error.html",
                    title="Couldn't restart the application",
                    description="Application restarting failed, please reload the page and try again in a moment.")
            
        if self.runner_conn is not None:
            self.runner_conn.send("restart")
            self.runner_conn.close()

        return redirect("/")
    
    def db_reset(self):
        with acquire_write(self.progressbar_rwlock, True, 1.0) as success:
            # If acquired and EITHER there was no previous self.thr
            # OR there was and it has already finished:
            if success and (self.thr is None or self.thr.progress == 1.0):
                # Start a new thread with our function (and with unique lock)
                
                def refresh_function(thr):
                    thr.title = "Resetting library"
                    thr.description = "The database is being refreshed. Please wait... The page will reload automatically."

                    self.embedding_tag_cache = EmbeddingTagCache()
                    with self.app.app_context():
                        for gen, n, description in self.imanager.get_full_refresh_generators():
                            thr.description = description
                            for i, _ in enumerate(gen):
                                thr.progress = i/n

                self.thr = LockingProgressBarThread.from_function(
                    self.progressbar_rwlock, refresh_function)
                self.thr.start()

            else:
                return render_template("error.html",
                    title="Couldn't reset the library",
                    description="Library resetting failed, please reload the page and try again in a moment.")
        return redirect("/settings/")

    def db_refresh(self):
        with acquire_write(self.progressbar_rwlock, True, 1.0) as success:
            # If acquired and EITHER there was no previous self.thr
            # OR there was and it has already finished:
            if success and (self.thr is None or self.thr.progress == 1.0):
                # Start a new thread with our function (and with unique lock)
                
                def refresh_function(thr):
                    thr.title = "Refreshing library"
                    thr.description = "The database is being refreshed. Please wait... The page will reload automatically."

                    with self.app.app_context():
                        for gen, n, description in self.imanager.get_refresh_generators():
                            thr.description = description
                            for i, _ in enumerate(gen):
                                thr.progress = i/n

                self.thr = LockingProgressBarThread.from_function(
                    self.progressbar_rwlock, refresh_function)
                self.thr.start()

            else:
                return render_template("error.html",
                    title="Couldn't refresh the library",
                    description="Library refreshing failed, please reload the page and try again in a moment.")
        return redirect("/settings/")
