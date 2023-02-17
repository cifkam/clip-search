from cachetools import TTLCache
from settings import settings
import sys
import secrets

class EmbeddingTagCache:
    def __init__(self):
        self.cache = TTLCache(settings.TAG_EMBED_CACHE_SIZE, settings.TAG_EMBED_CACHE_TTL)
    
    @staticmethod
    def get_full_tag(tag, session_id):
        return session_id + "_" + tag

    def get(self, tag, session_id):
        return self.cache[self.get_full_tag(tag, session_id)]

    def add(self, embedding, session_id):
        h = hash(embedding.tobytes())
        if h < 0:
            h += sys.maxsize+1
        tag = hex(h)[2:]
        full_tag = self.get_full_tag(tag, session_id)
        self.cache[full_tag] = embedding
        return tag
