import numpy as np
import pytest
tf = pytest.importorskip("tensorflow")
from PIL import Image
from models.mobilenetv2_model import build_mobilenetv2
from services import gradcam_service as g


@pytest.fixture(scope="module")
def model():
    return build_mobilenetv2(pretrained=False)  # no ImageNet download


def test_finds_nested_conv_layer(model):
    base, name = g.find_last_conv_layer(model)
    assert base is not None and name in [l.name for l in base.layers]


def test_heatmap_and_overlay(model, tmp_path):
    x = np.random.RandomState(0).uniform(-1, 1, (1, 224, 224, 3)).astype("float32")
    cam = g.compute_heatmap(model, x, 1)
    assert cam.ndim == 2 and 0 <= cam.min() and cam.max() <= 1
    name = g.generate_gradcam(model, Image.new("RGB", (100, 80)), x, 1, tmp_path)
    assert name and (tmp_path / name).is_file()


def test_graceful_failure(model, tmp_path):
    assert g.generate_gradcam(model, Image.new("RGB", (10, 10)), np.zeros((1, 5, 5, 3), "float32"), 0, tmp_path) is None


def test_model_has_four_class_softmax(model):
    assert model.output_shape == (None, 4)
