"""Image validation, safe filenames and consistent preprocessing."""
import io
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

import config

Image.MAX_IMAGE_PIXELS = 50_000_000  



ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP"}


class ImageValidationError(ValueError):
    pass


def allowed_extension(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


def validate_image_bytes(data: bytes, max_bytes: int | None = None) -> Image.Image:
    """Decode with Pillow (content check, not extension). Returns an RGB image."""
    limit = max_bytes or config.MAX_UPLOAD_BYTES
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    if len(data) > limit:
        raise ImageValidationError(f"File exceeds the {limit // (1024 * 1024)} MB limit.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        img = Image.open(io.BytesIO(data))
        if (img.format or "").upper() not in ALLOWED_FORMATS:
            raise ImageValidationError(f"Unsupported image format ({img.format}). Use JPG, PNG or WEBP.")
        img.load()
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError):
        raise ImageValidationError("The file is not a valid or readable image.")
    return img.convert("RGB")


def safe_filename(original_name: str) -> str:
    """Never trust user filenames: generate a random one, keep only a whitelisted extension."""
    ext = original_name.rsplit(".", 1)[1].lower() if "." in original_name else "jpg"
    if ext not in config.ALLOWED_EXTENSIONS:
        ext = "jpg"
    ext = "jpg" if ext == "jpeg" else ext
    return f"{uuid.uuid4().hex}.{ext}"


def save_upload(img: Image.Image, original_name: str, upload_dir: Path | None = None) -> str:
    """Save validated RGB image; returns the safe file name (no directories)."""
    upload_dir = Path(upload_dir or config.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    name = safe_filename(original_name)
    target = (upload_dir / name).resolve()
    if upload_dir.resolve() not in target.parents:
        raise ImageValidationError("Invalid file path.")
    img.save(target)
    return name


def resize(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return img.convert("RGB").resize(size, Image.BILINEAR)


def to_verification_input(img: Image.Image) -> np.ndarray:
    """Raw 0-255 float32 batch (1,H,W,3). The verification model rescales by 1/255 internally."""
    arr = np.asarray(resize(img, config.VERIFICATION_IMAGE_SIZE), dtype="float32")
    return arr[None, ...]


def to_nutrient_input(img: Image.Image) -> np.ndarray:
    """MobileNetV2 preprocessing applied exactly once: raw 0-255 -> [-1, 1]."""
    arr = np.asarray(resize(img, config.NUTRIENT_IMAGE_SIZE), dtype="float32")
    arr = arr / 127.5 - 1.0  
    return arr[None, ...]