# CLIP-Search Documentation
TODO: Add ToC


## Introduction
TODO.... For basic information, please consult the [README.md](../README.md).

## Application architecture
The Python application can be divided into two basic parts: the <em>Flask</em> frontend running a server accessible at port `5000` by default, and a backend consisting mainly of the `CLIP` model and related classes for querying and storing the necessary information such as embeddings and metadata into the database etc. Application itself runs as a separate child process started in the [run.py](../flask/run.py) file - this allows us to stop the application in need of restart, as the Flask does not offer any reasonable way to terminate the application. The main process thus takes care only of the starting, stopping and restarting the subprocess with Flask app, while using simple `multiprocessing.connection`'s `Client` and `Listener` for inter-process comunication.

### Flask frontend
The Flask app offers very simple GUI to the user through a locally hosted web-server. It allows searching in the pre-defined database of images, while utilize CLIP model's capabilities of querying either by image or by text label. That is, user can select an image file to search for similar images, or use a text input field to describe the desired image. When the result images are shown, it is possible to perform browsing in the database, i.e. searching for images similar to one of the results by simply clicking on the image. In addition, there is a separate (just-for-fun) module for zero-shot classification, allowing to perform an image classification into user-defined classes. This can be used to explore capabilities of the CLIP models. Finally, there is a settings that allows user to edit number of results per page, specific CLIP model, control the image library and shutdown or restart the application.

The app is being started in the [app.py](../flask/app.py) file, where also the REST API endpoints are defined. The endpoints can be divided into 3 main parts. First, there are search related endpoints (GET endpoints, unless specified otherwise):
- `/search/`: Redirects to the `/` main page with the search fields. 
- `/search/`: POST/GET endpoint. For GET requests it returns results page of the text-based search specified by the `q` query attribute. When a POST request is received, it performs search-by-image, but the actual results are displayed after redirection to the endpoint described in the following bullet point.
- `/search/img/<tag>`: Returns a results page for search-by-image, where `<tag>` is a special image-specific identifier. See the [Caching and tags](#caching-and-tags) section for more details.
- `/search/id/<id>`: Returns a results page when browsing in the image database, i.e. after clicking on one of the results images. Even though technically this is performing again search-by-image, images in the database have their own embedding precomputed, thus we use different endpoint.

All the endpoints returning results also support the pagination with the `page` query attribute.

Seconds, we have settings related endpoints:
- `/settings/`: POST/GET endpoint. For GET requests it returns a settings page, while the POST requests sets new settings values and then redirect back the settings page with GET request.
- `/settings/restart/`: Restarts the whole application.
- `/settings/shutdown/`: Shuts down the whole application.
- `/settings/db_refresh/`: Refreshes the database. See the [Image library](#image-library) section for more details.
- `/settings/db_reset/`: Fully resets the database. See the [Database](#image-library) section for more details.

Third, we have additional endpoints:
- `/`: The main page with the search fields. Can be also accessed on `/search/` endpoint which redirects here.
- `/classification/`: POST/GET endpoint. For GET requests it returns the page with the form for zero-shot image classification. For POST requests it performs the classification and returns page with the results below the form.
- `/db_images/<path:filename>`: Returns an image defined by path relative to the `DB_IMAGES_ROOT` settings variable.
- `/progress_status/`: Simple endpoint returning JSON message with progress status of the current action. The JSON has 3 key-value pairs, with keys `progress` (a floating-point value between 0 and 1, where 1 means "finished"), and `title` and `description` describing the current action.
- `/session_id/`: Checks and generates the random session id. If there is an `session_id` in cookies and it is valid (i.e. it is saved in the list of known identifiers on the server and it is not expired), then returns the same session id as the response. Otherwise, it returns new random 16-bytes long hexadecimal identifier.

When an operation is being performed (e.g. refreshing the library), the endpoints returning normal pages are locked. In that case, the endpoints will return a page with a progress bar instead.

### Backend
The backend part of our application handles the inference of the CLIP model and takes care of the databse etc. These actions are available via the `ImageManager` class and its functions. When initialized, it creates an instance of the `CLIPWrapper` class, which simplifies the calls to the CLIP model (preprocess the inputs before inference, and prepares the outputs for the user).

TODO: Querying

#### Image library
We consider the image libary to be quite static, i.e. we do not expect often updates. As going through all the images in `DB_IMAGES_ROOT` directory might be time-wise expensive operation, the `ImageManager` does not try to update the image library when it already exists during application startup. However, if the library files cannot be found at all, we initialize them automatically as running the application with empty library is pointless.

Initializing the library requires to find all image files in the `DB_IMAGES_ROOT` directory, read the file metadata and the image itself and perform the embedding. When all the images have been processed and metadata added to database, a <em>k-d tree</em> with all the image embeddings is created. This allows to perform a fast search for similar embeddings. The database stores only an ID for each image, its path and datatime of last modification.

Updating the libary might be of two different kinds. First, we might want to fully reset the library, i.e. clearing the databse, deleting the whole k-d tree and initializing everything from scratch. This shouldn't be needed at all, but we keep this option as a safety net. Second, the library can be refreshed. Refreshing also requires going through all the files in the `DB_IMAGES_ROOT` directory, however we skip all the files that have already been in the databse and its modified time has not changed -- this can save a lot of time as we do not need to run the CLIP model for them to get the embeddings. However we must compute new embeddings for any files with different modified time, and of course compute embeddings for completely new files. As there might be files that have been deleted since the last library update, we need to identify those and remove them. The last step is creating the k-d tree. This operation is quite fast and implementing a k-d tree supporting update might be tricky. Thus we use implementation from `scipy.spatial.KDTree` and recreate the tree everytime we refresh the library. The database is cleared and recreated as well to have the IDs in database matching indices in the k-d tree.

#### Embeddings
As image embeddings we use directly the CLIP embeddings. CLIP offers several models (based on ResNet - e.g. `RN50`, `RN101`; or based on Vision Transformer - `ViT-B/32`, `ViT-L/16` etc.). The embedding dimension of different models varies from 512 to 1024. As the embeddings from different models have different meaning (even though they might have the same dimension), the k-d tree and the databse needs to be created separately for each model type.

##### Caching and tags
When a user searches for a similar image in our application, they need to upload the image and the application computes the embedding and queries the k-d tree. As for the results we use pagination, when user switches between the result pages we need to run the query again. To simply store the information about the uploaded image, we introduce tags and caching. <em>Tag</em> is simply a hash of the embedding converted to hexadecimal string. When user uploads an image and we compute the image embedding, the embedding is stored in the TTL cache with its tag as a key. The user is then redirected to results page which has the tag in the URL, thus we can use the saved embedding from cache. Also, as application runs in browser, going back in history would unnecessarily send the POST request again and therefore upload the image and compute the embedding again. The cache solves this problem as well.

In addition, the tag is internally prefixed with <em>session ID</em> in the cache, which each user receives and is stored in their browser cookies, i.e. one user cannot access cached embeddings of another user, unless one reveals their session ID to the other.

### Settings
The modifiable application settings are store in the [settings.json](/flask/settings.json) file. If the file doesn't exist, it is automatically created with default values. The settings include number of results per page, CLIP model, the image library directory path, tag cache settings etc. The first two can be also changed via GUI. In the settings file, it is also possible to change to port on which the application is running, and the port for inter-process communication, and wheter the CLIP model should run on GPU (if available). Lastly, there are few settings useful for application debugging. On application startup, the JSON file is parsed and stored in a `Settings` class instance. Please note that any changes in the JSON file won't have any effect until application restart. Also, when changing settings in GUI, the changes in JSON file will be overwritten.

Be careful when changing the `DB_IMAGES_ROOT` or moving large subdirectories - if there is database created, the image paths in databse may become invalid and the images won't be shown. In that case, resetting/refreshing will be needed.