#!/usr/bin/env python3


class FlaskExitException(Exception):
    pass

def run_app(conn=None, log_file=None):
    import werkzeug.exceptions
    from flask import Flask, request, render_template
    from models import db
    from views import Views
    from settings import settings
    import sys

    if log_file is not None:
        sys.stderr = log_file
        sys.stdout = log_file


    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + settings.MODEL_NAME.replace("/", "_") + '.db'
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = settings.SQLALCHEMY_TRACK_MODIFICATIONS
    db.init_app(app)

    # Prevent from loading the CLIP model twice on startup and when reloading
    # if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
    views = Views()
    progress_thr = None

    with app.app_context():
        db.create_all()


    # === endpoints === #
    @app.route("/")
    def index():
        return views.index()


    #  [GET] /search/             (default search page)
    #  [GET] /search/?q=something (text search)
    # [POST] /search/             (image search)
    @app.route("/search/", methods=["GET", "POST"])
    def search():
        return views.search()


    @app.route("/search/img/<tag>")
    def search_cached(tag):
        return views.search_cached(tag)


    # search by image from databse
    @app.route("/search/id/<id>")
    def search_by_id(id):
        return views.search_by_id(id)


    @app.route("/classification/", methods=["GET", "POST"])
    def classification():
        return views.classification()


    @app.route("/settings/", methods=["GET", "POST"])
    def settings_page():
        return views.settings()


    @app.route("/progress_status")
    def progress_status():
        return views.progress_status()


    @app.route("/db_images/<path:filename>")
    def get_file(filename):
        return views.get_db_image(filename)


    @app.route("/session_id/")
    def session_id():
        return views.session_id()


    @app.route("/shutdown/")
    def shutdown():
        return views.shutdown(conn)
    
    @app.route("/restart/")
    def restart():
        return views.restart(conn)
        


    ###############################
    @app.errorhandler(werkzeug.exceptions.HTTPException)
    def generic_error_handler(e):
        description = ""
        title = ""
        try:
            description = str(e.description)
        except Exception:
            if isinstance(e, FlaskExitException):
                raise e
        try:
            title = f"{e.code} {e.name}"
        except Exception:
            title = "Unkown Error"

        return views.error(description=description, title=title)


    ###############################

    app.run(debug=settings.DEBUG, use_reloader=settings.USE_RELOADER)
    
if __name__ == "__main__":    
    run_app()
    
