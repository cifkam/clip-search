# CLIP-Search

CLIP-Search is an application for image search of user's image library using [OpenAI's CLIP](https://github.com/openai/CLIP) model, which can be prompted either via text or image. The application allows users to choose between various CLIP models, such as various versions of ResNet and ViT models. Any image from the search results can be used again as a query to search for similar images. Additionally, the app offers a "Zero-Shot Classification" tab, where users can experiment with CLIP's capabilities by submitting an image with multiple text labels to perform a classification task.

## Features

- Image search using OpenAI's CLIP model
- Choose between various CLIP models (ResNet/ViT variants)
- "Browsing" by clicking on the image from search results to search again for similar images
- Zero-shot classification for uploaded image and user-defined labels
- Web-based interface for easy use (mainly intended to be used locally but can be theoretically modified to be used as a public web-server as well)

## Running the Application

To set up the required Python environment, you have two options: using `virtualenv` and our installation script or any other Python virtual environment manager like Anaconda or `venv` and installing the packages using the `requirements`` files.

### Installing virtual environment

#### Using virtualenv

1. Install `virtualenv` if you don't have it:
   ```
   pip install virtualenv
   ```
2. For GPU support, run:
   ```
   ./setup-cuda-virtualenv.sh
   ```
   For CPU-only support, use the following command instead:
   ```
   ./setup-cpu-virtualenv.sh
   ```

#### Using other virtual environment managers
1. Create a Python virtual environment using your preferred manager (Anaconda/`venv`...).
2. Activate the environment.
3. Install the required packages for GPU or CPU support, depending on your preference:
   ```
   pip install -r requirements-cuda.txt
   ```
   or 
   ```
   pip install -r requirements-cpu.txt

   ```

### Setting up the image library
By default the application searches for images in `flask/db_images` directory. You can either create the directory and copy your image files here (including any subdirectories), or you can create symlink that points to your desired destination. Alternatively, you can change the image library path by modifying the value of `DB_IMAGES_ROOT` in the [flask/settings.json](flask/settings.json) file. Note that specifying e.g. root of your filesystem is not considered safe as this will give access to ANY file of your filesystem through the application endpoints. Also, changing the path after the application already created its database will cause error in displaying the image results, in that case please [reset the databse](#reseting-and-refreshing-the-image-databse).

### Running the application
After you created the the Python environment and the image library has been prepared, navigate to the `flask` directory and start the application:
```
cd flask
python run.py
```
After loading the CLIP models, the application needs to create its internal database of image embeddings. Note that this step may take a while and depends on number of your images. After this step has been finished, the application's web interface will be available at [127.0.0.1:5000](http://127.0.0.1:5000/).

### Settings
Application settings can be changed on the [Settings page](http://127.0.0.1:5000/settings/). It allows user to set the some basic settings, e.g. CLIP model and number of results per page. By default, the `RN50` model is used. Some other settings are accessible only in in the [flask/settings.json](flask/settings.json) file, please be careful when manually editing the file. When in trouble, it is possible to delete the file and it will be automatically recreated with default values when starting the application.

#### Reseting and refreshing the image databse
Whenever user adds, removes or changes an image in the image library, the application won't reflect the changes automatically. When this happens, user needs to reset or refresh the library. This can be done in settings of the application under "Library Control". "Refreshing" the library will automatically add new files, remove non-existing files and update embeddings of all the changed files. "Fully reseting" the library will remove all images from the database and re-embed all images in the directory from scratch.

Note: Image embeddings are specific for each CLIP model and image library controls apply only for the currently selected CLIP model. Whenever you change the image files and refresh the library, the database files for other models will still be outdated. 

### Example images and models
We provide a subset of images from the [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html) and already pre-computed embeddings for this subset. If you wish to try our application without waiting for the embeddings to create for you own images, you can follow the next steps:
1. With your environment activated, go to `example_ds` directory and run the dataset downloader without any arguments:
   ```
   cd example_ds
   python download_dataset.py
   ```
2. Download the internal databse files and unzip them into the `flask` directory:
   ```
   cd ../flask
   gdown https://drive.google.com/uc?id=14oLiRJjVj_eB3uNAxsQHQYGn4yXmvfjZ
   unzip clip-search-open_image_db.zip
   ```

   After running the application, you can start searching the images directly. 

## Documentation
Please refer to [docs/doc.md](docs/doc.md).
