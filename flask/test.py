#!/usr/bin/env python3

from app import app, db
import models
import numpy as np
from settings import EMBEDDING_SIZE

app.app_context().push()

from datetime import datetime
from ImageManager import ImageManager

im = ImageManager()


if True:
    im.clear_all()
    dir = 'db_images'
    im.init_from_dir(dir)
    im.refresh(dir)
else:
    assert im.load_kdtree()

pass