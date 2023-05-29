import json


class Settings:
    settings_json_path = "settings.json"

    def __check_exception(function):
        def wrapper(*args, **kwargs):
            try:
                function(*args, **kwargs)
                return True
            except Exception:
                return False

        return wrapper

    def load_defaults(self):
        self.PREFER_CUDA = True
        self.QUERY_K = 15
        self.DB_IMAGES_ROOT = "db_images"
        # clip.available_models(): ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
        #                           'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        self.MODEL_NAME = "RN50"
        #self.SQLITE_DB_NAME = "CLIPSearch.db"

        self.TAG_EMBED_CACHE_TTL = 15 * 60  # 15 minutes before expiration
        self.TAG_EMBED_CACHE_SIZE = 32

        self.DEBUG = False
        self.USE_RELOADER = False
        self.SQLALCHEMY_TRACK_MODIFICATIONS = False

        self.LOCALHOST_PORT = 16060

    def get_values(self):
        return dict(self.__dict__)

    def set_values(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

    @__check_exception
    def load(self):
        with open(Settings.settings_json_path, "r") as f:
            d = json.load(f)
            print("Loading settins.json...")

        self.set_values(d)

    @__check_exception
    def save(self):
        with open(Settings.settings_json_path, "w") as f:
            json.dump(self.__dict__, f, indent="\t")
            print("Saving settins.json...")


settings = Settings()
if not settings.load():
    settings.load_defaults()
    settings.save()
