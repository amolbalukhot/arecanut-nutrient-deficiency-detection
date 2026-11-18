import pytest
import config
from utils import data_split as ds
from tests.conftest import make_png


def test_missing_dataset_and_classes(tmp_path, make_dataset):
    rep = ds.inspect_dataset(tmp_path / "nope", config.CLASS_NAMES)
    assert rep["missing_classes"] == config.CLASS_NAMES
    root = make_dataset(config.CLASS_NAMES[:2])
    with pytest.raises(ds.SplitError, match="Missing required class"):
        ds.prepare_split("nutrient", root, config.CLASS_NAMES)


def test_invalid_images_reported_not_deleted(tmp_path, make_dataset):
    root = make_dataset(["A", "B"])
    bad = root / "A" / "bad.png"; bad.write_bytes(b"not an image")
    (root / "A" / "notes.txt").write_text("x")
    rep = ds.inspect_dataset(root, ["A", "B"])
    assert "A/bad.png" in rep["corrupted"] and "A/notes.txt" in rep["unsupported"] and bad.exists()


def test_exact_duplicates_detected_across_extensions_and_names(tmp_path, make_dataset):
    root = make_dataset(["A", "B"])
    data = (root / "A" / "0.png").read_bytes()
    (root / "A" / "copy_renamed.webp").write_bytes(data)  # same bytes, different name/extension
    rep = ds.inspect_dataset(root, ["A", "B"])
    d = ds.assign_duplicate_groups(rep["records"])
    assert any(len(g) == 2 for g in d["exact_duplicate_groups"])


def test_split_reproducible_and_no_leakage(tmp_path, make_dataset):
    root = make_dataset(["A", "B"], n=20)
    data = (root / "A" / "0.png").read_bytes()
    for k in range(3):
        (root / "B" / f"dup{k}.png").write_bytes(data)  # label conflict + duplicates
    r1 = ds.prepare_split("t", root, ["A", "B"], seed=1)
    ds.manifest_path("t").unlink()
    r2 = ds.prepare_split("t", root, ["A", "B"], seed=1)
    assert [x["split"] for x in r1["rows"]] == [x["split"] for x in r2["rows"]]
    by_hash = {}
    for r in r2["rows"]:
        by_hash.setdefault(r["sha256"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_hash.values())
    assert r2["summary"]["duplicates"]["label_conflicts"]
    assert {"train", "val", "test"} == {r["split"] for r in r2["rows"]}


def test_manifest_reuse_and_stale_detection(tmp_path, make_dataset):
    root = make_dataset(["A", "B"])
    assert not ds.prepare_split("t", root, ["A", "B"])["reused"]
    assert ds.prepare_split("t", root, ["A", "B"])["reused"]
    (root / "A" / "new.png").write_bytes(make_png(seed=999))
    with pytest.raises(ds.StaleManifestError):
        ds.prepare_split("t", root, ["A", "B"])
    with pytest.raises(ds.SplitError, match="--yes"):
        ds.prepare_split("t", root, ["A", "B"], regenerate=True)
    assert not ds.prepare_split("t", root, ["A", "B"], regenerate=True, confirmed=True)["reused"]


def test_group_aware_keeps_groups_together(tmp_path, make_dataset):
    root = make_dataset(["A", "B"], n=12)
    gcsv = tmp_path / "g.csv"
    lines = ["path,group"] + [f"{c}/{i}.png,{c}_plant{i // 3}" for c in "AB" for i in range(12)]
    gcsv.write_text("\n".join(lines))
    res = ds.prepare_split("t", root, ["A", "B"], groups_csv=gcsv)
    assert res["summary"]["group_aware"]
    groups = {}
    for r in res["rows"]:
        groups.setdefault(r["source_group"], set()).add(r["split"])
    assert all(len(s) == 1 for s in groups.values())


def test_no_groups_is_not_claimed(tmp_path, make_dataset):
    res = ds.prepare_split("t", make_dataset(["A", "B"]), ["A", "B"])
    assert "no group metadata" in res["summary"]["method"] and not res["summary"]["group_aware"]


def test_too_few_images(tmp_path, make_dataset):
    with pytest.raises(ds.SplitError, match="at least"):
        ds.prepare_split("t", make_dataset(["A", "B"], n=2), ["A", "B"])
