"""Manual stratified 80/10/10 split for the VERIFICATION dataset only.
Run from project root:  python split_verification.py
Writes data_splits/verification_split_manifest.csv (format train_verification.py reads).
Exact duplicates (same SHA-256) always stay in one split. Images are not moved or changed."""
import csv, hashlib, random, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("verification_dataset")
CLASSES = ["Arecanut", "Non_Arecanut"]
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
OUT = Path("data_splits/verification_split_manifest.csv")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    from PIL import Image
    rng = random.Random(SEED)
    rows, bad = [], []
    for cls in CLASSES:
        d = ROOT / cls
        if not d.is_dir():
            sys.exit(f"Missing folder: {d}")
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in EXTS:
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                bad.append(p.as_posix()); continue
            rows.append({"path": p.relative_to(ROOT).as_posix(), "label": cls, "sha256": sha256(p)})

    
    groups = defaultdict(list)
    for r in rows:
        groups[r["sha256"]].append(r)
    cross = [g for g in groups.values() if len({x["label"] for x in g}) > 1]

    assign = {}
    for cls in CLASSES:
        units = [g for g in groups.values() if g[0]["label"] == cls]
        rng.shuffle(units)
        units.sort(key=len, reverse=True)  
        total = sum(len(u) for u in units)
        target = {s: total * r for s, r in RATIOS.items()}
        got = {s: 0 for s in RATIOS}
        for u in units:
            s = max(RATIOS, key=lambda k: target[k] - got[k])
            got[s] += len(u)
            for x in u:
                assign[x["path"]] = s

    fp = hashlib.sha256("".join(sorted(f"{r['path']}|{r['label']}|{r['sha256']}" for r in rows)).encode()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "label", "split", "sha256", "duplicate_group", "source_group",
              "dataset_fingerprint", "seed", "method"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({"path": r["path"], "label": r["label"], "split": assign[r["path"]], "sha256": r["sha256"],
                        "duplicate_group": "dup_" + r["sha256"][:12], "source_group": "",
                        "dataset_fingerprint": fp, "seed": SEED,
                        "method": "stratified+duplicate-aware (no group metadata supplied)"})

    n = len(rows)
    print(f"Images: {n}  unreadable skipped: {len(bad)}  exact-duplicate groups: {sum(len(g) > 1 for g in groups.values())}")
    for cls in CLASSES:
        c = {s: sum(1 for r in rows if r["label"] == cls and assign[r["path"]] == s) for s in RATIOS}
        print(f"  {cls}: {c}")
    tot = {s: sum(1 for r in rows if assign[r["path"]] == s) for s in RATIOS}
    print("Total:", tot, {s: f"{100 * v / n:.1f}%" for s, v in tot.items()})
    if cross:
        print(f"WARNING: {len(cross)} identical images appear in BOTH classes. Fix labels:")
        for g in cross[:10]:
            print("  ", [x["path"] for x in g])
    print("Wrote", OUT)


if __name__ == "__main__":
    main()