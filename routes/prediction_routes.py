import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Blueprint, current_app, render_template, request, send_file, abort, jsonify, url_for
from PIL import Image

from services import database_service, pdf_service, prediction_service
from utils import image_utils

bp = Blueprint("prediction", __name__)
_heatmap_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gradcam")
_heatmap_lock = threading.Lock()
_heatmap_futures = {}
_futures_lock = threading.Lock()


def _generate_heatmap(pid, image_name, class_name, upload_dir, heatmap_dir, database_path):
    """Generate the optional visualization off-request, then persist its filename."""
    started = time.perf_counter()
    try:
        from config import CLASS_NAMES
        model = prediction_service.load_nutrient_model()
        with Image.open(Path(upload_dir) / Path(image_name).name) as source:
            img = source.convert("RGB")
        batch = image_utils.to_nutrient_input(img)
        
        with _heatmap_lock:
            name = prediction_service.gradcam_service.generate_gradcam(
                model, img, batch, CLASS_NAMES.index(class_name), heatmap_dir)
        database_service.update_heatmap(pid, name, database_path)
        logging.getLogger(__name__).info("prediction_timing id=%s gradcam_background=%.4fs", pid,
                                         time.perf_counter() - started)
    except Exception:  # noqa: BLE001 - heatmap is optional and must not invalidate the prediction
        logging.getLogger(__name__).exception("Background Grad-CAM failed for prediction %s", pid)
        database_service.update_heatmap(pid, None, database_path)
        return None
    return name


def _enqueue_heatmap(rec, app_config):
    pid = rec["id"]
    with _futures_lock:
        existing = _heatmap_futures.get(pid)
        if existing is not None and not existing.done():
            return existing
        future = _heatmap_executor.submit(
            _generate_heatmap, pid, rec["image_path"], rec["nutrient_class"],
            app_config["UPLOAD_DIR"], app_config["HEATMAP_DIR"], app_config["DATABASE_PATH"])
        _heatmap_futures[pid] = future
        return future


def _index_error(msg, status=400):
    return render_template("index.html", error=msg, models_ready=None, max_mb=16), status


@bp.post("/predict")
def predict():
    request_started = time.perf_counter()
    f = request.files.get("image")
    if f is None or not f.filename:
        return _index_error("Please choose an image to upload.")
    if not image_utils.allowed_extension(f.filename):
        return _index_error("Unsupported file type. Use JPG, PNG or WEBP.")
    try:
        stage_started = time.perf_counter()
        data = f.read()
        img = image_utils.validate_image_bytes(data)
        logging.getLogger(__name__).info("prediction_timing decode_validate=%.4fs", time.perf_counter()-stage_started)
    except image_utils.ImageValidationError as e:
        return _index_error(str(e))
    try:
        
        stage_started = time.perf_counter()
        vm = prediction_service.load_verification_model()
        nm = prediction_service.load_nutrient_model()
        logging.getLogger(__name__).info("prediction_timing model_load_or_cache=%.4fs",
                                         time.perf_counter()-stage_started)
    except prediction_service.ModelUnavailableError as e:
        return _index_error(str(e), 503)
    stage_started = time.perf_counter()
    name = image_utils.save_upload(img, f.filename, current_app.config["UPLOAD_DIR"])
    logging.getLogger(__name__).info("prediction_timing upload_save=%.4fs", time.perf_counter()-stage_started)
    try:
        stage_started = time.perf_counter()
        rec = prediction_service.run_pipeline(img, name, vm, nm, with_gradcam=False)
        logging.getLogger(__name__).info("prediction_timing pipeline=%.4fs", time.perf_counter()-stage_started)
        stage_started = time.perf_counter()
        pid = database_service.insert_prediction(rec, current_app.config["DATABASE_PATH"])
        logging.getLogger(__name__).info("prediction_timing database_insert=%.4fs", time.perf_counter()-stage_started)
    except Exception:  # noqa: BLE001
        current_app.logger.exception("Prediction failed")
        return _index_error("The image could not be analysed. Please try again.", 500)
    stored = dict(rec, id=pid)
    from services.recommendation_service import get_recommendation
    stage_started = time.perf_counter()
    detail = get_recommendation(stored["nutrient_class"]) if stored["nutrient_class"] else None
    logging.getLogger(__name__).info("prediction_timing recommendation=%.4fs total_response=%.4fs",
                                     time.perf_counter()-stage_started, time.perf_counter()-request_started)
    if stored["verification_accepted"] and stored["nutrient_class"]:
        _enqueue_heatmap(stored, current_app.config)
    return render_template("result.html", p=stored, detail=detail)


@bp.get("/heatmap/<int:pid>")
def heatmap_status(pid: int):
    rec = database_service.get_prediction(pid, current_app.config["DATABASE_PATH"])
    if not rec or not rec["verification_accepted"]:
        abort(404)
    if rec.get("heatmap_path"):
        return jsonify(status="ready", url=url_for("static", filename="heatmaps/" + Path(rec["heatmap_path"]).name))
    with _futures_lock:
        future = _heatmap_futures.get(pid)
    if future is not None and future.done():
        name = future.result()
        with _futures_lock:
            _heatmap_futures.pop(pid, None)
        if name:
            return jsonify(status="ready", url=url_for("static", filename="heatmaps/" + Path(name).name))
        return jsonify(status="unavailable"), 200
    if future is None and rec.get("nutrient_class"):
        _enqueue_heatmap(rec, current_app.config)
    return jsonify(status="pending"), 202


@bp.get("/download/<int:pid>")
def download(pid: int):
    rec = database_service.get_prediction(pid, current_app.config["DATABASE_PATH"])
    if not rec:
        abort(404)
    with _futures_lock:
        future = _heatmap_futures.get(pid)
    if future is None and rec.get("verification_accepted") and rec.get("nutrient_class") and not rec.get("heatmap_path"):
        future = _enqueue_heatmap(rec, current_app.config)
    if future is not None:
        future.result()
        with _futures_lock:
            _heatmap_futures.pop(pid, None)
        rec = database_service.get_prediction(pid, current_app.config["DATABASE_PATH"])
    report_started = time.perf_counter()
    path = pdf_service.generate_report(rec, current_app.config["REPORT_DIR"], current_app.config["UPLOAD_DIR"],
                                       current_app.config["HEATMAP_DIR"])
    logging.getLogger(__name__).info("prediction_timing pdf_generation=%.4fs", time.perf_counter()-report_started)
    return send_file(path, as_attachment=True, download_name=f"arecanut_report_{pid}.pdf", mimetype="application/pdf")
