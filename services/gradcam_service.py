"""Grad-CAM for the MobileNetV2 classifier (nested base model supported).
Grad-CAM shows regions that influenced the model; it is not proof of a nutrient deficiency."""
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

import config


def find_last_conv_layer(model):
    """Return (container_model_or_None, layer_name) for the last 4D conv-like output in the loaded model."""
    import tensorflow as tf
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for inner in reversed(layer.layers):
                if len(inner.output.shape) == 4 and ("conv" in inner.name.lower() or "relu" in inner.name.lower()):
                    return layer, inner.name
        elif len(layer.output.shape) == 4 and "conv" in layer.name.lower():
            return None, layer.name
    raise ValueError("No convolutional layer found for Grad-CAM.")


MAX_OVERLAY_SIDE = 512
_fn_cache: dict = {}


def _get_cam_fn(model):
    """Build the Grad-CAM graph ONCE per loaded model and compile it with tf.function.
    (Rebuilding sub-models and running eagerly on every request was the slow part.)"""
    import tensorflow as tf
    key = id(model)
    if key in _fn_cache and _fn_cache[key][0] is model:
        return _fn_cache[key][1]
    base, layer_name = find_last_conv_layer(model)
    if base is None:
        grad_model = tf.keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])

        @tf.function(reduce_retracing=True)
        def cam_fn(x, class_index):
            with tf.GradientTape() as tape:
                conv, preds = grad_model(x, training=False)
                score = tf.gather(preds, class_index, axis=1)
            return conv, tape.gradient(score, conv)
    else:
        feat_model = tf.keras.Model(base.inputs, [base.get_layer(layer_name).output, base.output])
        tail = model.layers[model.layers.index(base) + 1:]

        @tf.function(reduce_retracing=True)
        def cam_fn(x, class_index):
            with tf.GradientTape() as tape:
                conv, out = feat_model(x, training=False)
                tape.watch(conv)
                h = out
                for layer in tail:
                    h = layer(h, training=False)
                score = tf.gather(h, class_index, axis=1)
            return conv, tape.gradient(score, conv)
    _fn_cache[key] = (model, cam_fn)
    return cam_fn


def compute_heatmap(model, batch: np.ndarray, class_index: int) -> np.ndarray:
    import tensorflow as tf
    conv, grads = _get_cam_fn(model)(tf.convert_to_tensor(batch), tf.constant(int(class_index), dtype=tf.int32))
    if grads is None:
        raise ValueError("Gradients could not be computed.")
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * weights[:, None, None, :], axis=-1))[0].numpy()
    cam = cam - cam.min()
    return cam / cam.max() if cam.max() > 0 else cam


def warm_up(model):
    """Trace the Grad-CAM graph once at model-load time so the first user request is fast."""
    try:
        compute_heatmap(model, np.zeros((1, *config.NUTRIENT_IMAGE_SIZE, 3), dtype="float32"), 0)
    except Exception:  # noqa: BLE001 - warm-up is best effort
        pass


def overlay_heatmap(original: Image.Image, cam: np.ndarray, alpha=0.4) -> Image.Image:
    import matplotlib
    cmap = matplotlib.colormaps["jet"]
    heat = Image.fromarray(np.uint8(255 * cam)).resize(original.size, Image.BILINEAR)
    colored = (cmap(np.asarray(heat) / 255.0)[..., :3] * 255).astype("uint8")
    return Image.blend(original.convert("RGB"), Image.fromarray(colored), alpha)


def generate_gradcam(model, original: Image.Image, batch: np.ndarray, class_index: int,
                     out_dir: Path | None = None) -> str | None:
    """Return saved file name, or None on failure (never raises into the request)."""
    try:
        cam = compute_heatmap(model, batch, class_index)
        out_dir = Path(out_dir or config.HEATMAP_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Phone photos are ~12 MP: blending/saving at full size was the slow part. Overlay at <=512 px.
        small = original.convert("RGB")
        small.thumbnail((MAX_OVERLAY_SIDE, MAX_OVERLAY_SIDE))
        name = f"{uuid.uuid4().hex}.jpg"
        overlay_heatmap(small, cam).save(out_dir / name, quality=88)
        return name
    except Exception:  # noqa: BLE001 - heatmap is optional
        return None