"""Manifest-driven tf.data pipelines, reports and plots shared by both training scripts."""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

import config
from utils import data_split


def _load(path: Path, size, mode):
    img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype="float32")
    return arr / 127.5 - 1.0 if mode == "mobilenet" else arr  


def augment(arr: np.ndarray, rng: random.Random, mode: str) -> np.ndarray:
    """Training-only augmentation on raw 0-255 or [-1,1] arrays (flip, brightness, small shift)."""
    if rng.random() < 0.5:
        arr = arr[:, ::-1]
    if rng.random() < 0.5:
        arr = arr[::-1, :]
    lo, hi = (-1.0, 1.0) if mode == "mobilenet" else (0.0, 255.0)
    arr = np.clip(arr + (hi - lo) * rng.uniform(-0.1, 0.1), lo, hi)
    dx, dy = rng.randint(-8, 8), rng.randint(-8, 8)
    return np.roll(arr, (dy, dx), axis=(0, 1))


def make_dataset(rows, root, classes, size, mode, train, batch=32, seed=config.RANDOM_SEED):
    import tensorflow as tf
    idx = {c: i for i, c in enumerate(classes)}  
    rows = list(rows)

    def gen():
        rng = random.Random(seed)
        order = list(rows)
        if train:
            rng.shuffle(order)
        for r in order:
            a = _load(Path(root) / r["path"], size, mode)
            if train:
                a = augment(a, rng, mode)
            yield np.ascontiguousarray(a), idx[r["label"]]

    sig = (tf.TensorSpec((*size, 3), tf.float32), tf.TensorSpec((), tf.int32))
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig).batch(batch)
    return ds.prefetch(2)


def split_rows(rows):
    return {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}


def class_weights(rows, classes):
    counts = np.array([sum(1 for r in rows if r["label"] == c) for c in classes], dtype=float)
    w = counts.sum() / (len(classes) * np.maximum(counts, 1))
    return {i: float(x) for i, x in enumerate(w)}


def save_plots_and_reports(out_dir: Path, history: dict, y_true, y_pred, classes, metrics_extra: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    for key in ("accuracy", "loss"):
        plt.figure()
        plt.plot(history.get(key, []), label=f"train {key}")
        plt.plot(history.get(f"val_{key}", []), label=f"validation {key}")
        plt.xlabel("epoch"); plt.legend(); plt.title(key); plt.savefig(out_dir / f"{key}.png", dpi=120); plt.close()
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5)); plt.imshow(cm, cmap="Greens"); plt.colorbar()
    plt.xticks(labels, classes, rotation=45, ha="right"); plt.yticks(labels, classes)
    for i in labels:
        for j in labels:
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title("Test confusion matrix"); plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=120); plt.close()
    report = classification_report(y_true, y_pred, labels=labels, target_names=classes,
                                   output_dict=True, zero_division=0)
    (out_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
    metrics = {"evaluation_set": "held-out TEST split", "test_accuracy": float(accuracy_score(y_true, y_pred)),
               "confusion_matrix": cm.tolist(), "class_mapping": {c: i for i, c in enumerate(classes)},
               **metrics_extra}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(classification_report(y_true, y_pred, labels=labels, target_names=classes, zero_division=0))
    return metrics
