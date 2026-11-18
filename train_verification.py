"""Train the Stage 1 binary verification CNN.  python train_verification.py [--epochs N]"""
import argparse, json, sys
import numpy as np

import config
from utils import data_split, training_utils as tu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--groups", type=str, default=None)
    ap.add_argument("--regenerate-split", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    import tensorflow as tf
    from models.verification_model import build_verification_model
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    classes = config.VERIFICATION_CLASS_NAMES
    try:
        res = data_split.prepare_split("verification", config.VERIFICATION_DATASET_DIR, classes, config.RANDOM_SEED,
                                       a.groups, a.regenerate_split, a.yes)
    except data_split.SplitError as e:
        print(f"ERROR: {e}"); return 1
    parts = tu.split_rows(res["rows"])
    root, size = config.VERIFICATION_DATASET_DIR, config.VERIFICATION_IMAGE_SIZE
    mk = lambda s, tr: tu.make_dataset(parts[s], root, classes, size, "raw", tr, a.batch)
    train, val, test = mk("train", True), mk("val", False), mk("test", False)
    model = build_verification_model()
    config.SAVED_MODELS_DIR.mkdir(exist_ok=True)
    cb = [tf.keras.callbacks.ModelCheckpoint(config.VERIFICATION_MODEL_PATH, monitor="val_loss",
                                             save_best_only=True),
          tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
          tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)]
    hist = model.fit(train, validation_data=val, epochs=a.epochs, callbacks=cb,
                     class_weight=tu.class_weights(parts["train"], classes))
    best = tf.keras.models.load_model(config.VERIFICATION_MODEL_PATH)
    y_true = np.array([classes.index(r["label"]) for r in parts["test"]])
    p = best.predict(test, verbose=0).ravel()
    y_pred = (p >= 0.5).astype(int)
    from sklearn.metrics import precision_recall_fscore_support
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    extra = {"precision_non_arecanut": float(pr), "recall_non_arecanut": float(rc), "f1_non_arecanut": float(f1),
             "seed": config.RANDOM_SEED, "split_method": res["rows"][0]["method"],
             "epochs_run": len(hist.history["loss"]), "input_shape": [*size, 3],
             "output": "sigmoid = P(Non_Arecanut); accept if 1-p >= threshold"}
    out = config.TRAINING_RESULTS_DIR / "verification"
    tu.save_plots_and_reports(out, hist.history, y_true, y_pred, classes, extra)
    print(f"Saved model: {config.VERIFICATION_MODEL_PATH}\nResults: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
