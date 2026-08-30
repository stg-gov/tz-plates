"""Inference-time OCR preprocessing (numpy only).

Kept dependency-light on purpose: the ONNX runtime path must work without torch
or albumentations, and the torch path just wraps the array. Must stay bit-for-bit
consistent with the eval transform in ``transforms.py`` (Resize -> [ToGray] ->
Normalize(0.5, 0.5)).
"""

from __future__ import annotations

import cv2
import numpy as np

NORM_MEAN = 0.5
NORM_STD = 0.5


def preprocess_plate(
    plate_bgr: np.ndarray, input_height: int, input_width: int, gray: bool
) -> np.ndarray:
    """Return a (1, C, H, W) float32 array ready for the OCR model."""
    img = cv2.resize(plate_bgr, (input_width, input_height), interpolation=cv2.INTER_CUBIC)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
    elif img.ndim == 2:
        img = img[..., None]
    img = img.astype(np.float32) / 255.0
    img = (img - NORM_MEAN) / NORM_STD
    return img.transpose(2, 0, 1)[None, ...]
