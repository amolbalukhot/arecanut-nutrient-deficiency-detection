import numpy as np
import pytest
import config
from services import prediction_service as ps
from utils import image_utils as iu
from tests.conftest import make_png


class FakeV:
    def __init__(self, p_non): self.p = p_non
    def predict(self, x, verbose=0): return np.array([[self.p]])


class FakeN:
    calls = 0
    def predict(self, x, verbose=0):
        FakeN.calls += 1
        return np.array([[0.1, 0.1, 0.7, 0.1]])


@pytest.fixture
def img(): return iu.validate_image_bytes(make_png())


def test_stage1_rejection_never_calls_stage2(img, monkeypatch):
    FakeN.calls = 0
    monkeypatch.setattr(ps.gradcam_service, "generate_gradcam", lambda *a, **k: None)
    rec = ps.run_pipeline(img, "x.png", FakeV(0.9), FakeN())
    assert rec["verification_accepted"] is False and rec["nutrient_class"] is None and FakeN.calls == 0
    assert rec["status"] == "rejected"


def test_stage2_runs_after_accept_with_exact_class_order(img, monkeypatch):
    FakeN.calls = 0
    monkeypatch.setattr(ps.gradcam_service, "generate_gradcam", lambda *a, **k: "h.png")
    rec = ps.run_pipeline(img, "x.png", FakeV(0.1), FakeN())
    assert rec["verification_accepted"] and FakeN.calls == 1
    assert rec["nutrient_class"] == config.CLASS_NAMES[2] == "Potassium_Deficiency"
    assert rec["heatmap_path"] == "h.png" and "potassium" in rec["recommendation"].lower()


def test_threshold_boundary(img):
    assert ps.verify(img, FakeV(0.4))["accepted"]        # 0.60 arecanut -> accepted
    assert not ps.verify(img, FakeV(0.41))["accepted"]


def test_missing_models_give_actionable_errors(monkeypatch, tmp_path):
    ps.clear_model_cache()
    monkeypatch.setattr(config, "VERIFICATION_MODEL_PATH", tmp_path / "v.keras")
    monkeypatch.setattr(config, "MOBILENETV2_MODEL_PATH", tmp_path / "m.keras")
    with pytest.raises(ps.ModelUnavailableError, match="python train_verification.py"):
        ps.load_verification_model()
    with pytest.raises(ps.ModelUnavailableError, match="python train.py"):
        ps.load_nutrient_model()


def test_corrupted_model_file(monkeypatch, tmp_path):
    ps.clear_model_cache()
    bad = tmp_path / "m.keras"; bad.write_bytes(b"junk")
    monkeypatch.setattr(config, "MOBILENETV2_MODEL_PATH", bad)
    with pytest.raises(ps.ModelUnavailableError, match="could not be loaded"):
        ps.load_nutrient_model()
    ps.clear_model_cache()
