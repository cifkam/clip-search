import json

class Settings:
    settings_json_path = 'settings.json'

    def __check_exception(function):
        def wrapper(*args,**kwargs):
            try:
                function(*args,**kwargs)
                return True
            except:
                return False
        return wrapper

    def load_defaults(self):
        self.PREFER_CUDA = True
        self.QUERY_K = 15
        self.DB_IMAGES_ROOT = 'db_images'
        #clip.available_models(): ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        self.MODEL_NAME = 'RN50'
        self.SQLITE_DB_NAME = 'CLIPSearch.db'
        self.TAG_EMBED_CACHE_TTL = 15*60 # 15 minutes before expiration
        self.TAG_EMBED_CACHE_SIZE = 32

        self.DEBUG=True
        self.USE_RELOADER = True
        self.SQLALCHEMY_TRACK_MODIFICATIONS = True

    @__check_exception
    def load(self):
        with open(Settings.settings_json_path, 'r') as f:
            d = json.load(f)
    
        for key, value in d.items():
            setattr(self, key, value)
    
    @__check_exception
    def save(self):
            with open(Settings.settings_json_path, 'w') as f:
                json.dump(self.__dict__, f, indent='\t')


settings = Settings()
if not settings.load():
    settings.load_defaults()
    settings.save()