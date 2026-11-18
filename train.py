"""Train Stage 2 MobileNetV2 only, reading a ready-made folder split (no manifest needed).

Expected layout (class folder names may be full names or h/n/k/m, case-insensitive):
    dataset/train/<Healthy|Nitrogen_Deficiency|Potassium_Deficiency|Magnesium_Deficiency>/
    dataset/val/<same 4 folders>/        (also accepted: valid, validation)
    dataset/test/<same 4 folders>/

Run:  python train.py [--epochs N] [--finetune-epochs N] [--batch N] [--no-pretrained]
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

import config
from utils import training_utils as tu

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SPLIT_ALIASES = {"train": ["train", "training"], "val": ["val", "valid", "validation"], "test": ["test", "testing"]}
CLASS_ALIASES = {  
    "Healthy": ["healthy", "h"],
    "Nitrogen_Deficiency": ["nitrogen_deficiency", "nitrogen", "n"],
    "Potassium_Deficiency": ["potassium_deficiency", "potassium", "k", "p"],
    "Magnesium_Deficiency": ["magnesium_deficiency", "magnesium", "m"],
}


def _find_child(parent: Path, aliases: list[str], what: str) -> Path:
    kids = [d for d in parent.iterdir() if d.is_dir() and d.name.lower() in aliases]
    if len(kids) != 1:
        found = sorted(d.name for d in parent.iterdir() if d.is_dir())
        raise SystemExit(f"ERROR: expected exactly one {what} folder in {parent} named one of {aliases}; "
                         f"found {len(kids)}. Folders present: {found}")
    return kids[0]


def collect_rows(root: Path):
    """Return rows [{'path','label','split','sha256'}] using the explicit class mapping."""
    if not root.is_dir():
        raise SystemExit(f"ERROR: dataset folder not found: {root}")
    rows = []
    for split, aliases in SPLIT_ALIASES.items():
        sdir = _find_child(root, aliases, f"'{split}'")
        for cls in config.CLASS_NAMES:
            cdir = _find_child(sdir, CLASS_ALIASES[cls], f"class '{cls}'")
            files = sorted(p for p in cdir.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
            for p in files:
                rows.append({"path": p.relative_to(root).as_posix(), "label": cls, "split": split,
                             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    return rows


def validate_rows(rows):
    from PIL import Image
    good, bad = [], []
    for r in rows:
        try:
            with Image.open(config.NUTRIENT_DATASET_DIR / r["path"]) as im:
                im.verify()
            good.append(r)
        except Exception:
            bad.append(r["path"])
    return good, bad


def check_leakage(rows):
    """Identical image (same SHA-256) in different splits, or under different labels, is data leakage/mislabel."""
    seen, leaks, conflicts = {}, [], []
    for r in rows:
        h = r["sha256"]
        if h in seen:
            if seen[h]["split"] != r["split"]:
                leaks.append((seen[h]["path"], r["path"]))
            if seen[h]["label"] != r["label"]:
                conflicts.append((seen[h]["path"], r["path"]))
        else:
            seen[h] = r
    return leaks, conflicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--finetune-epochs", type=int, default=0, help="0 disables fine-tuning")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="Offline: random init. Needs substantially more data to work well.")
    ap.add_argument("--allow-leakage", action="store_true",
                    help="Continue even if identical images appear in different splits (NOT recommended).")
    a = ap.parse_args()

    root = config.NUTRIENT_DATASET_DIR
    classes = config.CLASS_NAMES
    rows, bad = validate_rows(collect_rows(root))
    if bad:
        print(f"WARNING: {len(bad)} unreadable images skipped, e.g. {bad[:3]}")
    parts = tu.split_rows(rows)

    print("Class mapping:", {c: i for i, c in enumerate(classes)})
    print(f"{'split':<6}" + "".join(f"{c[:12]:>14}" for c in classes) + f"{'total':>8}")
    for s in ("train", "val", "test"):
        counts = [sum(1 for r in parts[s] if r["label"] == c) for c in classes]
        print(f"{s:<6}" + "".join(f"{n:>14}" for n in counts) + f"{sum(counts):>8}")
        if 0 in counts:
            raise SystemExit(f"ERROR: a class has no images in '{s}'.")

    leaks, conflicts = check_leakage(rows)
    if conflicts:
        print(f"WARNING: {len(conflicts)} identical images carry DIFFERENT class labels, e.g. {conflicts[:3]}")
    if leaks:
        print(f"{'WARNING' if a.allow_leakage else 'ERROR'}: {len(leaks)} identical images appear in different "
              f"splits (data leakage inflates test scores), e.g. {leaks[:3]}")
        if not a.allow_leakage:
            print("Move or delete the duplicate copies, then rerun (or pass --allow-leakage).")
            return 1

    import tensorflow as tf
    from models.mobilenetv2_model import build_mobilenetv2, unfreeze_for_finetuning
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    size = config.NUTRIENT_IMAGE_SIZE
    mk = lambda s, tr: tu.make_dataset(parts[s], root, classes, size, "mobilenet", tr, a.batch)
    train, val, test = mk("train", True), mk("val", False), mk("test", False)

    model = build_mobilenetv2(pretrained=not a.no_pretrained)
    config.SAVED_MODELS_DIR.mkdir(exist_ok=True)
    cb = lambda: [tf.keras.callbacks.ModelCheckpoint(config.MOBILENETV2_MODEL_PATH, monitor="val_loss",
                                                     save_best_only=True),
                  tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                  tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)]
    cw = tu.class_weights(parts["train"], classes)
    h = model.fit(train, validation_data=val, epochs=a.epochs, callbacks=cb(), class_weight=cw)
    history = {k: list(v) for k, v in h.history.items()}
    if a.finetune_epochs > 0:
        unfreeze_for_finetuning(model)
        h2 = model.fit(train, validation_data=val, epochs=a.finetune_epochs, callbacks=cb(), class_weight=cw)
        for k, v in h2.history.items():
            history.setdefault(k, []).extend(v)

    best = tf.keras.models.load_model(config.MOBILENETV2_MODEL_PATH)  
    y_true = np.array([classes.index(r["label"]) for r in parts["test"]])
    y_pred = np.argmax(best.predict(test, verbose=0), axis=1)
    extra = {"seed": config.RANDOM_SEED, "pretrained_imagenet": not a.no_pretrained,
             "finetune_epochs": a.finetune_epochs, "split_method": "manual folder split (train/val/test folders)",
             "split_counts": {s: len(r) for s, r in parts.items()}, "input_shape": [*size, 3],
             "note": "Test metrics come from a single final evaluation; they do not prove generalisation "
                     "to new farms, cameras or lighting."}
    out = config.TRAINING_RESULTS_DIR / "mobilenetv2"
    tu.save_plots_and_reports(out, history, y_true, y_pred, classes, extra)
    print(f"Saved model: {config.MOBILENETV2_MODEL_PATH}\nResults: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())