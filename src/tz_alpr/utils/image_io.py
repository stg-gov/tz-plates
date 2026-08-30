"""Image decoding / IO. Centralized so EXIF orientation and colour order are
handled once (phone-camera uploads are frequently rotated — spec §1)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode an uploaded image (JPEG/PNG/WebP) to a BGR uint8 array, EXIF-corrected."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return _apply_exif_orientation(data, img)


def imread(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def imwrite(path: str | Path, img: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise OSError(f"Could not write image: {path}")


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


_EXIF_ORIENTATION_TAG = 0x0112


def _apply_exif_orientation(data: bytes, img: np.ndarray) -> np.ndarray:
    try:
        import io

        from PIL import ExifTags, Image  # noqa: F401

        with Image.open(io.BytesIO(data)) as pil:
            exif = pil.getexif()
        orientation = exif.get(_EXIF_ORIENTATION_TAG, 1)
    except Exception:
        orientation = 1

    ops = {
        3: lambda x: cv2.rotate(x, cv2.ROTATE_180),
        6: lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
        8: lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    return ops.get(orientation, lambda x: x)(img)
