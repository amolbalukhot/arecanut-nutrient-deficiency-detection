import io
import numpy as np
import pytest
import config
from services import prediction_service as ps
from tests.conftest import make_png


class FakeV:
    def __init__(self, p): self.p = p
    def predict(self, x, verbose=0): return np.array([[self.p]])


class FakeN:
    def predict(self, x, verbose=0): return np.array([[0.7, 0.1, 0.1, 0.1]])


def post(client, data, name="leaf.png"):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def test_pages_load_with_empty_db(client):
    for url in ("/", "/dashboard", "/history"):
        assert client.get(url).status_code == 200
    assert b"No nutrient predictions yet" in client.get("/dashboard").data


def test_app_starts_and_predict_reports_missing_model(client, monkeypatch, tmp_path):
    ps.clear_model_cache()
    monkeypatch.setattr(config, "VERIFICATION_MODEL_PATH", tmp_path / "none.keras")
    r = post(client, make_png())
    assert r.status_code == 503 and b"python train_verification.py" in r.data


def test_upload_validation(client):
    assert client.post("/predict", data={}).status_code == 400
    assert post(client, b"junk").status_code == 400
    assert post(client, make_png(), "x.exe").status_code == 400
    assert post(client, b"", "x.png").status_code == 400


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(ps, "load_nutrient_model", lambda: FakeN())
    monkeypatch.setattr(ps.gradcam_service, "generate_gradcam", lambda *a, **k: None)


def test_accepted_flow_history_and_pdf(client, monkeypatch, fake_models):
    monkeypatch.setattr(ps, "load_verification_model", lambda: FakeV(0.05))
    r = post(client, make_png())
    assert r.status_code == 200 and b"Healthy" in r.data
    assert b"Healthy" in client.get("/history").data
    pdf = client.get("/download/1")
    assert pdf.status_code == 200 and pdf.data.startswith(b"%PDF")
    assert client.get("/download/999").status_code == 404
    assert b"Total predictions" in client.get("/dashboard").data


def test_rejected_flow_shows_no_diagnosis(client, monkeypatch, fake_models):
    monkeypatch.setattr(ps, "load_verification_model", lambda: FakeV(0.95))
    r = post(client, make_png())
    assert r.status_code == 200 and b"nutrient classification was not performed" in r.data
    assert b"Grad-CAM" not in r.data and b"Stage 2" not in r.data


def test_csrf_enforced_when_enabled(tmp_path):
    from app import create_app
    a = create_app({"TESTING": True, "DATABASE_PATH": tmp_path / "c.db", "UPLOAD_DIR": tmp_path / "u"})
    assert a.test_client().post("/predict", data={}).status_code == 400
