import io, sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from app import create_app


def make_png(color=(30, 120, 40), size=(64, 64), seed=None):
    arr = np.full((*size[::-1], 3), color, dtype="uint8")
    if seed is not None:
        arr = np.random.RandomState(seed).randint(0, 255, arr.shape, dtype="uint8")
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, "PNG"); return buf.getvalue()


@pytest.fixture
def make_dataset(tmp_path):
    def _make(classes, n=10, root=None):
        root = Path(root or tmp_path / "ds")
        for c in classes:
            (root / c).mkdir(parents=True, exist_ok=True)
            for i in range(n):
                (root / c / f"{i}.png").write_bytes(make_png(seed=hash((c, i)) % 10000))
        return root
    return _make


@pytest.fixture
def app(tmp_path):
    return create_app({"TESTING": True, "WTF_CSRF_ENABLED": False, "DATABASE_PATH": tmp_path / "t.db",
                       "UPLOAD_DIR": tmp_path / "up", "HEATMAP_DIR": tmp_path / "hm", "REPORT_DIR": tmp_path / "rep"})


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture(autouse=True)
def _split_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SPLIT_DIR", tmp_path / "splits")
    monkeypatch.setattr(config, "SPLIT_REPORT_DIR", tmp_path / "splits" / "reports")
