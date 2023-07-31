# CLIP-Search

CLIP-Search is an application for image search of user's image database using [OpenAI's CLIP](https://github.com/openai/CLIP) model, which can be prompted either via text or image. The application allows users to choose between various CLIP models, such as various versions of ResNet and ViT models. Any image from the search results can be used again as a query to search for similar images. Additionally, the app offers a "Zero-Shot Classification" tab, where users can experiment with CLIP's capabilities by submitting an image with multiple text labels to perform a classification task.

## Features

- Image search using OpenAI's CLIP model
- Choose between various CLIP models (ResNet, ViT, etc.)
- Zero-Shot Classification for image and text tasks
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

### Setting up the image databse
Automatically the application searches for images in `flask/db_images` directory. You can either create the directory and copy your images files here (including any subdirectories), or you can create symlink that points to your desired destination. Alternatively, you can change the image database path by modifying the value of `DB_IMAGES_ROOT` in the [flask/settings.json](flask/settings.json) file.

### Running the application
After you created the the Python environment and the image database has been prepared, navigate to the `flask` directory and start the application:
```
cd flask
python run.py
```
After loading the models, the application needs to create its internal database of image embeddings. Note that this step may take a while and depends on number of your images. After this step has finished, he application's web interface will be available at [127.0.0.1:5000](http://127.0.0.1:5000/`).

### Example images and models
We provide a subset of images from the [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html) and already pre-computed embeddings for this subset. If you wish to try our application without waiting for the embeddings to create, follow the next steps:
1. A
2. B
3. C

## Documentation
Please refer to [docs/doc.md](docs/doc.md).
