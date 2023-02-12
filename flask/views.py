from flask import render_template, request, send_from_directory, redirect, abort
from PIL import Image
from settings import *
from ImageManager import ImageManager
import numpy as np

class Views:
    def __init__(self) -> None:
        self.imanager = ImageManager(
            vector_size=EMBEDDING_SIZE,
            model_name=MODEL_NAME,
            prefer_cuda=PREFER_CUDA
        )

    def index(self):
        return render_template('index.html')

    def search(self):
        if 'q' in request.args:
            #print(f"request: {request.args['q']}")
            pass
        return render_template('search.html')

    def image_search(self):
        if request.method == 'POST':
            try:
                img = Image.open(request.files['upload'])
            except Exception as e:
                abort(415, e)

            result = self.imanager.query_image(img, k=6)
            paths = ['/'+x.path for x in result]
            return render_template("results.html", result=paths)
        else:
            return render_template('image_search.html')

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
        return render_template('settings.html')

    def error(self, description, title='Error'):
        return render_template('error.html', title=title, description=description)

    def get_db_image(self, filename, as_attachment=False):
        return send_from_directory(DB_IMAGES_ROOT, filename, as_attachment=False)
