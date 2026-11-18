from flask import Blueprint, render_template

import config

bp = Blueprint("main", __name__)


@bp.get("/")
def index():
    models_ready = {"verification": config.VERIFICATION_MODEL_PATH.is_file(),
                    "mobilenetv2": config.MOBILENETV2_MODEL_PATH.is_file()}
    return render_template("index.html", models_ready=models_ready, max_mb=config.MAX_UPLOAD_BYTES // (1024 * 1024))
