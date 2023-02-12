#!/usr/bin/env python3

from app import app, db
import models
import torch
app.app_context().push()

from datetime import datetime
from ImageManager import ImageManager

im = ImageManager(block_size=10)
