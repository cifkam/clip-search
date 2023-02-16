DEBUG = True
USE_RELOADER = True
SQLALCHEMY_TRACK_MODIFICATIONS = True

PREFER_CUDA = True
QUERY_K = 15

#clip.available_models(): ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
MODEL_NAME = "RN50"

DB_IMAGES_ROOT = 'db_images'



import json
class Settings:
    settings_path = 'settings.json'

    def __init__(self):
        pass

    def load_defaults():
        pass

    def load():
        pass

    def save():
        pass