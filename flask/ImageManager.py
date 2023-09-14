import models
import numpy as np
import os
import os.path
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
from tqdm import tqdm

from settings import settings


class ImageManager:
    image_formats = ["jpg", "jpeg", "png", "gif", "bmp", "ico", "tiff", "tga", "webp"]

    def __init__(self, *, clip_wrapper=None, model_name="ViT-B/32", prefer_cuda=False):
        self.dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.model_name = model_name

        if clip_wrapper is not None:
            self.clip = clip_wrapper
        self.clip = CLIPWrapper.Create(model_name=model_name, prefer_cuda=prefer_cuda)
        self.try_load_kdtree()

    # Returns all images in database
    def images(self):
        return db.session.query(models.Image)

    # Returns result of the k-d tree query for the k nearest neighbours of the text embedding.
    def query_text(self, text, k=1):
        return self.query(self.clip.text2vec(text).cpu().numpy(), k=k)

    # Embeds the image and return result of the k-d tree query for the k nearest neighbours of the embedding.
    def query_image(self, image, k=1):
        return self.query(self.clip.img2vec(image).cpu().numpy(), k=k)

    # Embeds the image and returns the embedding
    def embed_image(self, image):
        return self.clip.img2vec(image).cpu().numpy()

    # Returns result of the k-d tree query for the k nearest neighbours of the image embedding given by it's databse id
    def query_id(self, id, k=1):
        return self.query(self.kdtree.data[id - 1], k=k)

    # Returns result of the k-d tree query for the k nearest neighbours of the given embedding.
    def query(self, embedding, k=1):
        indices = self.kdtree.query(embedding, k=k)[1].reshape(-1)
        indices = 1 + indices[indices < self.kdtree.data.shape[0]]
        db_query = models.Image.query.filter(
            models.Image.id.in_(indices.tolist())
        ).order_by(models.Image.id)
        db_query = list(db_query)

        # wee need to order the results
        order = dict(zip(indices, count()))
        db_query.sort(key=lambda x: order[x.id])
        return db_query


    """
    Creates a k-d tree from the given data, saves it in self.kdtree, and dumps
    it to the disk.
    """
    def create_kdtree(self, data):
        filename = f"kdtrees/kdtree_{self.model_name.replace('/','-')}.pkl"
        kdtree = KDTree(data)
        with open(filename, "wb") as f:
            pickle.dump(kdtree, f)
        self.kdtree = kdtree

    """
    Tries to load the k-d tree from the disk. If the file is not found,
    the k-d tree is set to None and False is returned.
    """
    def try_load_kdtree(self):
        filename = f"kdtrees/kdtree_{self.model_name.replace('/','-')}.pkl"
        if Path(filename).is_file():
            with open(filename, "rb") as f:
                self.kdtree = pickle.load(f)
            print(f"Successfully loaded {filename}.")
            return True
        else:
            print(f"Warning: '{filename}' not found!")
            self.kdtree = None
            return False

    # Returns the embedding of the image given by path.
    def get_embedding(self, path):
        with PIL.Image.open(path) as img:
            with torch.no_grad():
                return self.clip.img2vec(img)

    # Clears the databse and kd-tree, and returns the action (generator) that
    # rebuilds the database and the k-d tree from scratch.
    def get_full_refresh_generators(self):
        self.clear_all()
        yield from self.get_init_generators()

    # Clears the database and the k-d tree, and rebuilds them from scratch by calling self.init().
    def full_refresh(self):
        self.clear_all()
        self.init()


    """
    Refreshes the database and the k-d tree by executing the generators returned
    by get_refresh_generators(). Non-existing images are removed from the databse,
    modified images are updated, and new images are added to the database. The k-d
    tree is rebuilt from scratch.
    """
    def refresh(self):
        for gen, n, description in self.get_refresh_generators():
            for _ in gen: # Execute the generator, ignore the outputs (None)
                pass

    """
    Returns a generator with a sequence of actions. Each action is a tuple of
    (generator, n, description) where:
     - generator is a generator/iterator that performs the action, but return always None
        (i.e. the steps of the action are executed, but we don't care about the output)
     - n is the number of steps of the action (-1 if unknown)
     - description is a string describing the action
    This way the progressbar can easily get the progress of each action without
    having to know the details of each action, or the action having the know the
    implementation of the progressbar.
    """
    def get_refresh_generators(self):
        ########################
        def update_images(imgs):
            print("Adding and updating old images:")
            for i, old_img in imgs:
                yield
                new_img = self.insert_image(old_img.path)
                if old_img.timestamp != new_img.timestamp:
                    old_data[i] = self.get_embedding(new_img.path)
        ########################
        new_data = []
        def add_images(missing):
            print("Adding new images:")
            for file in missing:
                yield
                self.insert_image(file)
                embedding = self.get_embedding(file)
                new_data.append(embedding)
        ########################
        def finish():
            nonlocal new_data, old_data
            yield
            if len(new_data) == 0:
                data = old_data
            else:
                new_data = torch.cat(new_data).cpu().numpy()
                if old_data is not None:
                    data = np.concatenate([old_data, new_data], axis=0)
                else:
                    data = new_data
            try:
                db.session.commit()
                print("Building k-d tree")
                self.create_kdtree(data)
            except Exception as e:
                db.session.rollback()
                raise 
        ########################
        
        dir = settings.DB_IMAGES_ROOT

        # Find missing files and files that are already present
        db_paths = set(map(lambda x: x.path, self.images()))
        dir_paths = set(self.find_images(dir))

        missing = dir_paths - db_paths
        intersection = db_paths.intersection(dir_paths)
        intersection = list(
            models.Image.query.filter(models.Image.path.in_(intersection))
        )
        ids = list(map(lambda x: x.id - 1, intersection))

        # Clear the old databse
        if self.kdtree is not None:
            old_data = self.kdtree.data[ids]
        else:
            old_data = None
        self.clear_all()

        # Add old images to the database and update the embeddings if necessary
        imgs = tqdm(list(enumerate(intersection)), ncols=100)
        yield update_images(imgs), len(imgs), "Updating old images..."

        # Add new images to the databse
        missing = tqdm(missing, ncols=100)
        yield add_images(missing), len(missing), "Adding new images..."

        # Commit the changes to the database and rebuild the k-d tree
        yield finish(), -1, "Finishing up"


    """
    Returns a generator with a sequence of actions. Each action is a tuple of
    (generator, n, description) where:
     - generator is a generator/iterator that performs the action, but return always None
        (i.e. the steps of the action are executed, but we don't care about the output)
     - n is the number of steps of the action (-1 if unknown)
     - description is a string describing the action
    This way the progressbar can easily get the progress of each action without
    having to know the details of each action, or the action having the know the
    implementation of the progressbar.
    """
    def get_init_generators(self):
        ########################
        data = None
        def add_images(paths):
            nonlocal data
            print("Adding new images:")
            vectors = []
            for file in paths:
                yield
                self.insert_image(file)
                embedding = self.get_embedding(file)
                vectors.append(embedding)
    
            data = torch.cat(vectors).cpu().numpy()
        ########################
        def finish():
            nonlocal data
            yield
            try:
                db.session.commit()
                print("Building k-d tree")
                self.create_kdtree(data)
            except Exception as e:
                db.session.rollback()
                raise e
        ########################
        
        # Find images
        paths = tqdm(list(self.find_images(settings.DB_IMAGES_ROOT)), ncols=100)
        # Add images to the databse
        yield add_images(paths), len(paths), "Adding new images..."
        # Commit and build the k-d tree
        yield finish(), -1, "Finishing up"


    """
    Inits the database and the k-d tree from scratch by executing
    the generators returned by get_init_generators().
    """
    def init(self):
        for gen, n, description in self.get_init_generators():
            for _ in gen: # Execute the generator, ignore the outputs (None)
                pass

    # Updates the embedding of the image given by path.
    def update_by_path(self, path):
        img = models.Image.query.filter_by(path=path).first()
        return self.update(img)

    """
    Clears the databse and the k-d tree.
    """
    def clear_all(self):
        self.images().delete()
        try:
            db.session.commit()
            self.kdtree = None
        except Exception as e:
            db.session.rollback()
            raise e

    # Finds and returns the images within the given directory (recursively).
    # The supported image formats are specified in the class variable image_formats.
    def find_images(self, path, abs_path=False, return_str=True):
        return self.find_files(
            path, abs_path, formats=self.image_formats, return_str=return_str
        )

    # Finds and returns the files within the given directory (recursively).
    def find_files(self, path, abs_path=False, formats=None, return_str=True):
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError("No such directory: {path}")

        if abs_path:
            path = path.resolve()

        if formats is None:
            for file in glob(str(path / "**"), recursive=True):
                yield (str(file) if return_str else file)
        else:
            r = re.compile(".*\.(" + "|".join(formats) + ")$", re.IGNORECASE)
            for file in glob(str(path / "**"), recursive=True):
                if r.search(str(file)):
                    yield (str(file) if return_str else file)

    # Inserts an image given by the path into the database with the modified time as the timestamp.
    # Does not commit the changes, nor modify the k-d tree.
    def insert_image(self, path):
        timestamp = datetime.fromtimestamp(os.path.getmtime(path))
        img = models.Image(path=path, timestamp=timestamp)
        db.session.add(img)
        return img
