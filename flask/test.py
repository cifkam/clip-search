#!/usr/bin/env python3

from app import app, db
import models
import numpy as np
from datetime import datetime
from ImageManager import ImageManager
from settings import MODEL_NAME, PREFER_CUDA
app.app_context().push()

im = ImageManager(model_name=MODEL_NAME, prefer_cuda=PREFER_CUDA)


if True:
    dir = 'db_images'
    im.clear_all()
    im.init_from_dir(dir)
else:
    assert im.load_kdtree()

pass