import os

DEBUG = True
USE_RELOADER = True
SQLALCHEMY_TRACK_MODIFICATIONS = True

PREFER_CUDA = False

MODEL_NAME = "ViT-B/32"
EMBEDDING_SIZE = 512


# WARNING: this allows to access whole filesystem/current drive,
# change it to some other path where your images are stored
#FILES_ROOT = os.path.abspath('.').split(os.path.sep)[0]+os.path.sep

DB_IMAGES_ROOT = 'db_images'
