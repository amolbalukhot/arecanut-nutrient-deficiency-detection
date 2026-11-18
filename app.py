"""Flask application factory. Starts even when datasets or models are missing."""
import logging

from flask import Flask, render_template

import config
from services import database_service


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(SECRET_KEY=config.SECRET_KEY, MAX_CONTENT_LENGTH=config.MAX_UPLOAD_BYTES + 1024 * 1024,
                      DATABASE_PATH=config.DATABASE_PATH, UPLOAD_DIR=config.UPLOAD_DIR,
                      HEATMAP_DIR=config.HEATMAP_DIR, REPORT_DIR=config.REPORT_DIR)
    if test_config:
        app.config.update(test_config)
    logging.basicConfig(level=logging.INFO)

    try:
        from flask_wtf.csrf import CSRFProtect
        csrf = CSRFProtect()
        csrf.init_app(app)  # honours WTF_CSRF_ENABLED; always exposes csrf_token() to templates
    except ImportError:  # CSRF library optional at import time; documented in README
        app.logger.warning("Flask-WTF not installed: CSRF protection disabled.")
        app.jinja_env.globals["csrf_token"] = lambda: ""

    database_service.init_db(app.config["DATABASE_PATH"])

    from routes.main_routes import bp as main_bp
    from routes.prediction_routes import bp as pred_bp
    from routes.dashboard_routes import bp as dash_bp
    for bp in (main_bp, pred_bp, dash_bp):
        app.register_blueprint(bp)

    @app.errorhandler(413)
    def too_large(_):
        return render_template("index.html", error="File is too large (16 MB maximum)."), 413

    @app.errorhandler(404)
    def not_found(_):
        return render_template("index.html", error="Page not found."), 404

    @app.errorhandler(400)
    def bad_request(_):
        return render_template("index.html", error="Invalid request. Please reload the page and try again."), 400

    @app.errorhandler(500)
    def server_error(_):
        return render_template("index.html", error="Something went wrong. Please try again."), 500

    if not app.config.get("TESTING") and not app.config.get("SKIP_PRELOAD"):
        import threading
        from services import prediction_service
        threading.Thread(target=prediction_service.preload, daemon=True).start()  # fast first upload

    return app


if __name__ == "__main__":
    create_app().run(debug=False, host="127.0.0.1", port=5000)