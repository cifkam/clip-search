import models
import numpy as np
import os, os.path
import sys
import pickle
import PIL
import re
import torch
from itertools import count
from pathlib import Path
from datetime import datetime
from scipy.spatial import KDTree
from glob import glob
from models import db
from CLIPWrapper import CLIPWrapper


class ImageManager:
    image_formats = ['jpg','jpeg','png','gif','bmp','ico','tiff','tga','webp']

    def __init__(self, *, clip_wrapper = None, vector_size=512, model_name = "ViT-B/32", prefer_cuda=False) -> None:
        self.dir = os.path.dirname(os.path.abspath(sys.argv[0]))

        if clip_wrapper is not None:
            self.clip = clip_wrapper
        self.clip = CLIPWrapper.Create(model_name=model_name, prefer_cuda=prefer_cuda)

        if (self.load_kdtree()):
            print("Loaded existing kd-tree")
        else:
            print("Warning! File 'kd-tree.pkl' not found!")

    def images(self): return db.session.query(models.Image)

    def query_text(self, text, k=1):
        return self.query(self.clip.text2vec(text), k=k)
    
    def query_image(self, image, k=1):
        return self.query(self.clip.img2vec(image), k=k)
    
    def query(self, embedding, k=1):
        indices = self.kdtree.query(embedding, k=k)[1].reshape(-1)
        indices = 1 + indices[indices < self.kdtree.data.shape[0]]
        db_query = list( models.Image.query.filter(models.Image.id.in_(indices.tolist())).order_by(models.Image.id) )

        # wee need to order the 
        order = dict(zip(indices, count()))
        db_query.sort(key=lambda x: order[x.id])
        return db_query
        

    def create_kdtree(self, data):
        kdtree = KDTree(data)
        with open("kdtree.pkl", "wb") as f:
            pickle.dump(kdtree, f)
        self.kdtree = kdtree

    def load_kdtree(self):
        if Path("kdtree.pkl").is_file():
            with open("kdtree.pkl", "rb") as f:
                self.kdtree = pickle.load(f)
            return True
        else:
            return False

    def get_embedding(self, path):
        with PIL.Image.open(path) as img:
            with torch.no_grad():
                return self.clip.img2vec(img)

    def full_refresh(self, dir):
        self.clear_all()
        self.init_from_dir(dir)

    def refresh(self, dir):
        print()
        # Find missing files   and   files that are already present
        db_paths = set(map(lambda x: x.path, self.images()))
        dir_paths = set(self.find_images(dir))
        
        missing = dir_paths-db_paths
        intersection = db_paths.intersection(dir_paths)
        intersection = list(models.Image.query.filter(models.Image.path.in_(intersection)))
        ids = list(map(lambda x: x.id-1, intersection))

        # Clear the old databse
        old_data = self.kdtree.data[ids]
        self.clear_all()
        
        # Add old images to the database and update the embeddings if necessary
        update = False
        for i, old_img in enumerate(intersection):
            new_img = self.insert_image(old_img.path)
            if old_img.timestamp !=  new_img.timestamp:
                update = True
                old_data[i] = self.get_embedding(new_img.path)

        # Add new images to the databse
        new_data = []
        for file in missing:
            self.insert_image(file)
            embedding = self.get_embedding(file)
            new_data.append(embedding)

        if len(new_data) == 0:
            data = old_data
        else:
            new_data = torch.cat(new_data).cpu().numpy()    
            data = np.concatenate([old_data, new_data], axis=0)

        try:
            db.session.commit()
            self.create_kdtree(data)
        except Exception as e:
            db.session.rollback()
            raise e


    def init_from_dir(self, dir):
        vectors = []
        for file in self.find_images(dir):
            self.insert_image(file)
            embedding = self.get_embedding(file)
            vectors.append(embedding)
        
        data = torch.cat(vectors).cpu().numpy()

        try:
            db.session.commit()
            self.create_kdtree(data)
        except Exception as e:
            db.session.rollback()
            raise e

    def update_by_path(self, path):
        img = models.Image.query.filter_by(path=path).first()
        return self.update(img)

    def clear_all(self):
        self.images().delete()
        try:
            db.session.commit()
            self.kdtree = None
        except Exception as e:
            db.session.rollback()
            raise e

    def find_images(self, path, abs_path=False, return_string=True):
        return self.find_files(path, abs_path, formats=self.image_formats, return_string=return_string)

    def find_files(self, path, abs_path=False, formats=None, return_string=True):
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError("No such directory: {path}")

        if abs_path:
            path = path.resolve()
        
        if formats is None:
            for file in glob(str(path/"**"), recursive=True):
                yield (str(file) if return_string else file)
        else:
            r = re.compile('.*\.(' + '|'.join(formats) + ')$', re.IGNORECASE)
            for file in glob(str(path/"**"), recursive=True):
                if r.search(str(file)):
                    yield (str(file) if return_string else file)

    def insert_image(self, path):
        timestamp = datetime.fromtimestamp(os.path.getmtime(path))
        img = models.Image(path=path, timestamp=timestamp)
        db.session.add(img)
        return img
    
