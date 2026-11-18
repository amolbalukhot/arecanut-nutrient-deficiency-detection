"""Dataset inspection and stable, leakage-aware splitting.

Usage:
    python utils/data_split.py --dataset verification|nutrient|all [--regenerate --yes] [--groups groups.csv]

Group metadata: optional CSV with columns `path,group` (path relative to the dataset root).
If absent, no group-aware splitting is claimed; exact + near-duplicate groups are still kept together.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, UnidentifiedImageError

import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MANIFEST_FIELDS = ["path", "label", "split", "sha256", "duplicate_group", "source_group",
                   "dataset_fingerprint", "seed", "method"]
MIN_PER_CLASS = 3
NEAR_DUP_DISTANCE = 10  


class SplitError(Exception):
    pass


class StaleManifestError(SplitError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path, size: int = 16) -> int:
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype="int16")
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def inspect_dataset(root: Path, classes: list[str]) -> dict:
    """Inspect a dataset; never modifies or deletes files."""
    root = Path(root)
    report = {"root": str(root), "required_classes": classes, "discovered_class_folders": [],
              "missing_classes": [], "total_files": 0, "valid_images": 0, "corrupted": [],
              "unsupported": [], "per_class": {}, "records": []}
    if not root.is_dir():
        report["missing_classes"] = list(classes)
        return report
    report["discovered_class_folders"] = sorted(p.name for p in root.iterdir() if p.is_dir())
    for cls in classes:
        d = root / cls
        if not d.is_dir():
            report["missing_classes"].append(cls)
            continue
        count = 0
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.name.startswith("."):
                continue
            report["total_files"] += 1
            rel = p.relative_to(root).as_posix()
            if p.suffix.lower() not in IMAGE_EXTS:
                report["unsupported"].append(rel)
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    im.load()
                h, dh = sha256_file(p), dhash(p)
            except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
                report["corrupted"].append(rel)
                continue
            report["valid_images"] += 1
            count += 1
            report["records"].append({"path": rel, "label": cls, "sha256": h, "dhash": dh})
        report["per_class"][cls] = count
    counts = [c for c in report["per_class"].values() if c]
    report["imbalance_ratio"] = round(max(counts) / min(counts), 2) if counts and min(counts) else None
    return report


def assign_duplicate_groups(records: list[dict], merge_near: bool = False) -> dict:
    """Union-find over EXACT (sha256) duplicates. Near-duplicates (dHash) are only reported for manual review,
    because chaining visually similar leaf photos merges unrelated images into one giant group and wrecks the split.
    Pass merge_near=True to also merge them (not recommended for leaf datasets)."""
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    by_hash = defaultdict(list)
    for i, r in enumerate(records):
        by_hash[r["sha256"]].append(i)
    exact = [v for v in by_hash.values() if len(v) > 1]
    for v in exact:
        for i in v[1:]:
            union(v[0], i)
    near_pairs = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if records[i]["sha256"] != records[j]["sha256"] and \
                    hamming(records[i]["dhash"], records[j]["dhash"]) <= NEAR_DUP_DISTANCE:
                if len(near_pairs) < 500:
                    near_pairs.append((records[i]["path"], records[j]["path"]))
                if merge_near:
                    union(i, j)
    for i, r in enumerate(records):
        r["duplicate_group"] = f"dup_{find(i)}"
    conflicts = []
    groups = defaultdict(set)
    for r in records:
        groups[r["duplicate_group"]].add(r["label"])
    for g, labels in groups.items():
        if len(labels) > 1:
            conflicts.append({"group": g, "labels": sorted(labels),
                              "paths": [r["path"] for r in records if r["duplicate_group"] == g]})
    return {"exact_duplicate_groups": [[records[i]["path"] for i in v] for v in exact],
            "near_duplicate_pairs_for_review": near_pairs, "label_conflicts": conflicts}


def load_groups(groups_csv: Path | None) -> dict:
    if not groups_csv:
        return {}
    with open(groups_csv, newline="") as f:
        return {row["path"]: row["group"] for row in csv.DictReader(f)}


def fingerprint(records: list[dict]) -> str:
    h = hashlib.sha256()
    for r in sorted(records, key=lambda x: x["path"]):
        h.update(f"{r['path']}|{r['label']}|{r['sha256']}\n".encode())
    return h.hexdigest()


def _unit_key(r: dict) -> str:
    """Atomic unit kept in a single split: merge duplicate group and known source group."""
    return r["_unit"]


def _merge_units(records: list[dict]):
    """Merge duplicate groups and source groups transitively into units."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in records:
        a = find(r["duplicate_group"])
        if r.get("source_group"):
            b = find("src_" + r["source_group"])
            parent[a] = b
    for r in records:
        r["_unit"] = find(r["duplicate_group"])


def build_split(records: list[dict], classes: list[str], seed: int, ratios=None) -> dict[str, str]:
    """Stratified (by unit's majority class) split of units into train/val/test."""
    ratios = ratios or (config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO)
    _merge_units(records)
    units = defaultdict(list)
    for r in records:
        units[r["_unit"]].append(r)
    per_class_units = defaultdict(list)
    for u, rs in units.items():
        labels = [r["label"] for r in rs]
        per_class_units[max(set(labels), key=labels.count)].append(u)
    rng = random.Random(seed)
    assignment = {}
    for cls in classes:
        us = sorted(per_class_units.get(cls, []))
        rng.shuffle(us)
        total = sum(len(units[u]) for u in us)
        targets = {"train": total * ratios[0], "val": total * ratios[1], "test": total * ratios[2]}
        filled = {"train": 0, "val": 0, "test": 0}
        
        order = ["val", "test", "train"] if len(us) >= 3 else ["train"]
        for k, u in enumerate(us):
            if k < len(order) and len(us) >= 3:
                s = order[k]
            else:
                s = max(targets, key=lambda x: targets[x] - filled[x])
            filled[s] += len(units[u])
            assignment[u] = s
    return {r["path"]: assignment[r["_unit"]] for r in records}


def verify_no_leakage(records: list[dict], assignment: dict) -> list[str]:
    problems = []
    by_unit = defaultdict(set)
    for r in records:
        by_unit[r["_unit"]].add(assignment[r["path"]])
    for u, splits in by_unit.items():
        if len(splits) > 1:
            problems.append(f"unit {u} spans splits {sorted(splits)}")
    by_hash = defaultdict(set)
    for r in records:
        by_hash[r["sha256"]].add(assignment[r["path"]])
    problems += [f"exact duplicate {h[:10]} spans splits" for h, s in by_hash.items() if len(s) > 1]
    return problems


def manifest_path(name: str) -> Path:
    return config.SPLIT_DIR / f"{name}_split_manifest.csv"


def write_manifest(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def check_manifest_fresh(manifest_rows: list[dict], root: Path, classes: list[str]):
    """Raise StaleManifestError if files were added, removed or modified."""
    current = inspect_dataset(root, classes)
    cur = {r["path"]: r["sha256"] for r in current["records"]}
    old = {r["path"]: r["sha256"] for r in manifest_rows}
    added = sorted(set(cur) - set(old))
    removed = sorted(set(old) - set(cur))
    modified = sorted(p for p in set(cur) & set(old) if cur[p] != old[p])
    if added or removed or modified:
        raise StaleManifestError(
            f"Manifest is stale: {len(added)} added, {len(removed)} removed, {len(modified)} modified files. "
            f"Re-run with --regenerate --yes to create a new split (this changes train/val/test membership).")


def prepare_split(name: str, root: Path, classes: list[str], seed: int = config.RANDOM_SEED,
                  groups_csv: Path | None = None, regenerate: bool = False, confirmed: bool = False) -> dict:
    """Return the manifest (creating it only if missing, or regenerating with explicit confirmation)."""
    mpath = manifest_path(name)
    if mpath.exists() and not regenerate:
        rows = read_manifest(mpath)
        check_manifest_fresh(rows, root, classes)
        return {"manifest": mpath, "rows": rows, "reused": True}
    if mpath.exists() and regenerate and not confirmed:
        raise SplitError("Refusing to replace an existing manifest without --yes confirmation.")

    rep = inspect_dataset(root, classes)
    if rep["missing_classes"]:
        raise SplitError(f"Missing required class folders in {root}: {rep['missing_classes']}")
    for cls, n in rep["per_class"].items():
        if n < MIN_PER_CLASS:
            raise SplitError(f"Class '{cls}' has only {n} valid images; need at least {MIN_PER_CLASS} "
                             f"to place it in train, validation and test.")
    records = rep["records"]
    dup = assign_duplicate_groups(records)
    groups = load_groups(groups_csv)
    for r in records:
        r["source_group"] = groups.get(r["path"], "")
    group_aware = bool(groups) and all(r["source_group"] for r in records)
    assignment = build_split(records, classes, seed)
    problems = verify_no_leakage(records, assignment)
    if problems:
        raise SplitError("Leakage check failed: " + "; ".join(problems[:5]))
    fp = fingerprint(records)
    method = ("stratified+duplicate-aware+group-aware" if group_aware
              else "stratified+duplicate-aware (no group metadata supplied)")
    rows = [{"path": r["path"], "label": r["label"], "split": assignment[r["path"]], "sha256": r["sha256"],
             "duplicate_group": r["duplicate_group"], "source_group": r["source_group"],
             "dataset_fingerprint": fp, "seed": seed, "method": method} for r in records]
    write_manifest(mpath, rows)

    total = len(rows)
    split_counts = {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "test")}
    summary = {"dataset": name, "method": method, "seed": seed, "total": total,
               "split_counts": split_counts,
               "split_percentages": {s: round(100 * c / total, 2) for s, c in split_counts.items()},
               "target_percentages": {"train": 80, "val": 10, "test": 10},
               "per_class_per_split": {c: {s: sum(1 for r in rows if r["label"] == c and r["split"] == s)
                                           for s in split_counts} for c in classes},
               "group_aware": group_aware,
               "corrupted": rep["corrupted"], "unsupported": rep["unsupported"],
               "imbalance_ratio": rep["imbalance_ratio"], "duplicates": dup,
               "leakage_problems": problems}
    config.SPLIT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.SPLIT_REPORT_DIR / f"{name}_split_report.json").write_text(json.dumps(summary, indent=2))
    return {"manifest": mpath, "rows": rows, "reused": False, "summary": summary}


DATASETS = {
    "verification": (lambda: config.VERIFICATION_DATASET_DIR, config.VERIFICATION_CLASS_NAMES),
    "nutrient": (lambda: config.NUTRIENT_DATASET_DIR, config.CLASS_NAMES),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Inspect datasets and create stable split manifests.")
    ap.add_argument("--dataset", choices=["verification", "nutrient", "all"], required=True)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--groups", type=Path, help="CSV with columns path,group (path relative to dataset root)")
    ap.add_argument("--regenerate", action="store_true", help="Replace an existing manifest")
    ap.add_argument("--yes", action="store_true", help="Confirm manifest replacement")
    args = ap.parse_args(argv)
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    status = 0
    for n in names:
        root, classes = DATASETS[n][0](), DATASETS[n][1]
        print(f"\n=== {n} dataset: {root}")
        try:
            res = prepare_split(n, root, classes, args.seed, args.groups, args.regenerate, args.yes)
        except SplitError as e:
            print(f"ERROR: {e}")
            status = 1
            continue
        if res["reused"]:
            print(f"Existing manifest is valid and unchanged: {res['manifest']}")
        else:
            s = res["summary"]
            print(f"Method: {s['method']}\nCounts: {s['split_counts']}  Percent: {s['split_percentages']} (target 80/10/10)")
            print(f"Exact duplicate groups: {len(s['duplicates']['exact_duplicate_groups'])}; "
                  f"near-duplicate pairs to review: {len(s['duplicates']['near_duplicate_pairs_for_review'])}; "
                  f"label conflicts: {len(s['duplicates']['label_conflicts'])}")
            print(f"Manifest: {res['manifest']}\nReport: {config.SPLIT_REPORT_DIR / (n + '_split_report.json')}")
    return status


if __name__ == "__main__":
    sys.exit(main())