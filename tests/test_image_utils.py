import pytest
from utils import image_utils as iu
from tests.conftest import make_png


def test_valid_and_invalid_images():
    assert iu.validate_image_bytes(make_png()).mode == "RGB"
    for bad in (b"", b"garbage"):
        with pytest.raises(iu.ImageValidationError):
            iu.validate_image_bytes(bad)
    with pytest.raises(iu.ImageValidationError):
        iu.validate_image_bytes(make_png(), max_bytes=10)


def test_content_checked_not_extension():
    assert iu.allowed_extension("x.png")
    with pytest.raises(iu.ImageValidationError):
        iu.validate_image_bytes(b"<html>fake.png</html>")


def test_safe_filename_and_traversal(tmp_path):
    n = iu.safe_filename("../../etc/passwd.png")
    assert "/" not in n and ".." not in n and n.endswith(".png")
    assert iu.safe_filename("evil.exe").endswith(".jpg")
    img = iu.validate_image_bytes(make_png())
    saved = iu.save_upload(img, "../x.png", tmp_path)
    assert (tmp_path / saved).is_file()


def test_preprocessing_ranges_and_shapes():
    img = iu.validate_image_bytes(make_png((255, 255, 255)))
    v, n = iu.to_verification_input(img), iu.to_nutrient_input(img)
    assert v.shape == (1, 128, 128, 3) and v.max() == 255.0
    assert n.shape == (1, 224, 224, 3) and abs(n.max() - 1.0) < 1e-6
