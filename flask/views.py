from flask import render_template, request, send_from_directory, redirect, abort
from PIL import Image
from ImageManager import ImageManager
from utils import LockingProgressBarThread, ReadWriteLock, acquire_read, acquire_write
from settings import settings
from itertools import islice
from EmbeddingTagCache import EmbeddingTagCache
import secrets

class Views:
    thr = None
    
    def __init__(self) -> None:
        self.imanager = ImageManager(
            model_name=settings.MODEL_NAME,
            prefer_cuda=settings.PREFER_CUDA
        )
        self.rwlock = ReadWriteLock()
        self.embedding_tag_cache = EmbeddingTagCache()
        self.session_ids = set()

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
    def process_query_result(result, page=1):
        # Get the correct "page" of results
        result = islice(result, settings.QUERY_K*(page-1), settings.QUERY_K*page)
        
        # Map it to pairs ('/url_for_file', '/url_for_search_by_id')
        return [('/'+x.path, f'/search/id/{x.id}') for x in result]
    
    @staticmethod
    def parse_int(string):
        try:
            f = float(string)
            i = int(f)
        except:
            return None
        if i == f:
            return i
        else:
            return None
        
    @staticmethod
    def render_search_results(result, page, args=None):
        if args == None:
            args = dict()

        if page > 1:
            prev_args = dict(args)
            prev_args['page'] = str(page-1)
            prev_href = '?' + '&'.join([f'{key}={value}' for key,value in prev_args.items()])
        else:
            prev_href = None 
        
        if len(result) >= settings.QUERY_K: 
            next_args = dict(args)
            next_args['page'] = str(page+1) 
            next_href = '?' + '&'.join([f'{key}={value}' for key,value in next_args.items()])
        else:
            next_href = None

        return render_template('results.html', result=result, prev_href=prev_href, next_href=next_href)


    def query_image(self, img, page=1):
        result = self.imanager.query_image(img, k=settings.QUERY_K*page)
        return self.process_query_result(result, page)

    def query_text(self, text, page=1):
        result = self.imanager.query_text(text, k=settings.QUERY_K*page)
        return self.process_query_result(result, page)

    def query_id(self, id, page=1):
        result = self.imanager.query_id(id, k=settings.QUERY_K*page)
        return self.process_query_result(result, page)

    def query_embedding(self, embedding, page=1):
        result = self.imanager.query(embedding, k=settings.QUERY_K*page)
        return self.process_query_result(result, page)

    @progressbar_lock()
    def index(self):
        return render_template('index.html')


    @progressbar_lock()
    def search_cached(self, tag):
        if 'page' in request.args and request.args['page'] != '':
            page = self.parse_int(request.args['page'])
            if page is None:
                page = 1
        else:
            page = 1
        
        try:
            embedding = self.embedding_tag_cache.get(tag, request.cookies['session_id'])
        except KeyError:
            return self.error('Your query session has been already released, please upload your image again.')

        result = self.query_embedding(embedding, page)
        return self.render_search_results(result, page, request.args)


    @progressbar_lock()
    def search(self):
        if 'page' in request.args and request.args['page'] != '':
            page = self.parse_int(request.args['page'])
            if page is None:
                page = 1
        else:
            page = 1

        if request.method == 'POST':
            # Search by image
            try:
                img = Image.open(request.files['upload'])
            except Exception as e:
                abort(415, e)

            if 'session_id' not in request.cookies or request.cookies['session_id'] not in self.session_ids:
                result = self.query_image(img, page)
                return self.render_search_results(result, page, request.args)
            else:
                embedding = self.imanager.embed_image(img)
                tag = self.embedding_tag_cache.add(embedding, request.cookies['session_id'])
                return redirect(f'/search/img/{tag}')

        
        elif 'q' in request.args and request.args['q'] != '':
            # Search by text
            text = request.args['q']
            print(f'Query: {text}')
            result = self.query_text(text, page)
            return self.render_search_results(result, page, request.args)
        
        return redirect('/')

    @progressbar_lock()
    def search_by_id(self, id):
        id = self.parse_int(id)
        if id is None or id < 1:
            abort(400)

        if 'page' in request.args and request.args['page'] != '':
            page = self.parse_int(request.args['page'])
            if page is None:
                page = 1
        else:
            page = 1
        
        result = self.query_id(id, page)
        return self.render_search_results(result, page, request.args)

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

        return redirect('/settings')

    def progress_status(self):
        if self.thr is None:
            return '1.0'
        else:
            return str(self.thr.progress)
        
    def session_id(self):
        if 'session_id' in request.cookies:
            id = request.cookies['session_id']
            if id in self.session_ids:
                return id

        id = secrets.token_hex(16)
        while id in self.session_ids:
            id = secrets.token_hex(16)
        self.session_ids.add(id)
        return id

    def error(self, description, title='Error'):
        return render_template('error.html', title=title, description=description)

    def get_db_image(self, filename, as_attachment=False):
        return send_from_directory(settings.DB_IMAGES_ROOT, filename, as_attachment=False)
