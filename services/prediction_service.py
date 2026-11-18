"""Two-stage pipeline: verification CNN -> (only if accepted) MobileNetV2."""
import logging
import threading
import time
from pathlib import Path

import numpy as np

import config
from services import gradcam_service, recommendation_service
from utils import image_utils

log = logging.getLogger("prediction")
_lock = threading.Lock()
_cache: dict = {}


class ModelUnavailableError(RuntimeError):
    pass


def _load(name: str, path: Path, train_cmd: str):
    with _lock:
        if name in _cache:
            return _cache[name]
        if not Path(path).is_file():
            raise ModelUnavailableError(f"The {name} model is missing ({Path(path).name}). Train it with: {train_cmd}")
        try:
            import tensorflow as tf
            _cache[name] = tf.keras.models.load_model(path, compile=False)
        except Exception as e:  # noqa: BLE001
            raise ModelUnavailableError(
                f"The {name} model file could not be loaded (corrupted or incompatible). Retrain with: {train_cmd}") from e
        return _cache[name]


def load_verification_model():
    return _load("verification", config.VERIFICATION_MODEL_PATH, "python train_verification.py")


def load_nutrient_model():
    return _load("MobileNetV2", config.MOBILENETV2_MODEL_PATH, "python train.py")


def _infer(model, x: np.ndarray) -> np.ndarray:
    """Direct call is much faster than model.predict() for a single image (no dataset/iterator setup)."""
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            return np.asarray(model(x, training=False))
    except ImportError:
        pass
    return np.asarray(model.predict(x, verbose=0))


def preload():
    """Load both models and run one dummy inference each, so the first upload is not slow."""
    t0 = time.perf_counter()
    for loader, shape in ((load_verification_model, (*config.VERIFICATION_IMAGE_SIZE, 3)),
                          (load_nutrient_model, (*config.NUTRIENT_IMAGE_SIZE, 3))):
        try:
            _infer(loader(), np.zeros((1, *shape), dtype="float32"))
            if loader is load_nutrient_model:
                gradcam_service.warm_up(load_nutrient_model())
        except ModelUnavailableError as e:
            log.warning("Preload skipped: %s", e)
    log.info("Models preloaded in %.1fs", time.perf_counter() - t0)


def clear_model_cache():
    _cache.clear()


def verify(img, model=None):
    """Sigmoid = P(Non_Arecanut). Arecanut confidence = 1 - p. Accepted iff it meets the threshold."""
    model = model or load_verification_model()
    p_non = float(_infer(model, image_utils.to_verification_input(img)).ravel()[0])
    p_are = 1.0 - p_non
    accepted = p_are >= config.VERIFICATION_THRESHOLD
    label = "Arecanut" if p_are >= p_non else "Non_Arecanut"
    return {"class": label, "confidence": max(p_are, p_non), "arecanut_probability": p_are, "accepted": accepted}


def classify(img, model=None, with_gradcam=True):
    model = model or load_nutrient_model()
    batch = image_utils.to_nutrient_input(img)
    t = time.perf_counter()
    probs = _infer(model, batch)[0]
    idx = int(np.argmax(probs))
    t_pred = time.perf_counter() - t
    t = time.perf_counter()
    heatmap = gradcam_service.generate_gradcam(model, img, batch, idx) if with_gradcam else None
    log.info("stage2 predict %.2fs, gradcam %.2fs (%s)", t_pred, time.perf_counter() - t,
             "on" if with_gradcam else "off")
    return {"class": config.CLASS_NAMES[idx], "confidence": float(probs[idx]), "index": idx,
            "probabilities": {c: float(p) for c, p in zip(config.CLASS_NAMES, probs)}, "heatmap": heatmap}


def run_pipeline(img, image_name: str, verification_model=None, nutrient_model=None,
                 with_gradcam: bool = True) -> dict:
    """Returns a record ready for database insertion. Stage 2 never runs after a rejection."""
    # Both models must be present before any prediction is produced.
    vmodel = verification_model or load_verification_model()
    nmodel = nutrient_model or load_nutrient_model()
    t = time.perf_counter()
    v = verify(img, vmodel)
    log.info("stage1 verify %.2fs", time.perf_counter() - t)
    rec = {"image_path": image_name, "verification_class": v["class"], "verification_confidence": v["confidence"],
           "verification_accepted": v["accepted"], "nutrient_class": None, "nutrient_confidence": None,
           "heatmap_path": None, "recommendation": None, "status": "ok" if v["accepted"] else "rejected"}
    if not v["accepted"]:
        return rec
    n = classify(img, nmodel, with_gradcam=with_gradcam)
    rec.update(nutrient_class=n["class"], nutrient_confidence=n["confidence"], heatmap_path=n["heatmap"],
               recommendation=recommendation_service.recommendation_text(n["class"]))
    return rec