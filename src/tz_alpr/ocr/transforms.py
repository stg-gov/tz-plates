"""Augmentation pipelines for OCR training (spec §8).

Each difficult real-world condition named in the spec maps to one or more
Albumentations ops. Probabilities are read from configs/ocr.yaml so the mix can
be tuned without code changes.
"""

from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from tz_alpr.ocr.preprocess import NORM_MEAN as _MEAN
from tz_alpr.ocr.preprocess import NORM_STD as _STD


def _to_gray_1ch() -> A.BasicTransform:
    """Grayscale to a genuine single channel, so training input matches the
    1-channel inference preprocessing in ``preprocess.py``."""
    try:
        return A.ToGray(num_output_channels=1, method="weighted_average", p=1.0)
    except TypeError:  # older Albumentations without num_output_channels
        return A.Compose([A.ToGray(p=1.0), A.ToFloat(max_value=1.0)])


def _night(p: float) -> A.BasicTransform:
    return A.OneOf(
        [
            A.RandomBrightnessContrast(brightness_limit=(-0.5, -0.2), contrast_limit=(-0.1, 0.2)),
            A.RandomGamma(gamma_limit=(140, 220)),
        ],
        p=p,
    )


def build_train_transform(cfg: dict) -> A.Compose:
    model = cfg["model"]
    aug = cfg["augmentation"]
    h, w = int(model["input_height"]), int(model["input_width"])
    gray = int(model.get("input_channels", 1)) == 1
    q_lo, q_hi = aug.get("jpeg_quality", [30, 90])

    ops: list[A.BasicTransform] = [
        A.Resize(h, w, interpolation=cv2.INTER_CUBIC),
        A.Affine(
            rotate=(-aug["rotate_limit_deg"], aug["rotate_limit_deg"]),
            shear={"x": (-4, 4), "y": (-2, 2)},
            scale=(0.94, 1.06),
            translate_percent=(0.0, 0.03),
            fit_output=False,
            p=0.6,
        ),
        A.Perspective(scale=(0.02, aug["perspective_scale"]), p=aug["perspective_p"]),
        A.RandomBrightnessContrast(p=aug["brightness_contrast_p"]),
        _night(aug["night_p"]),
        A.OneOf(
            [
                A.MotionBlur(blur_limit=(3, 9)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.Defocus(radius=(1, 3)),
            ],
            p=max(aug["motion_blur_p"], aug["gaussian_blur_p"]),
        ),
        A.ISONoise(intensity=(0.1, 0.6), p=aug["iso_noise_p"]),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2), p=aug["shadow_p"]),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_radius=80, p=aug["glare_p"]),
        A.RandomRain(blur_value=2, drop_length=8, p=aug["rain_p"]),
        A.ImageCompression(quality_range=(int(q_lo), int(q_hi)), p=aug["jpeg_p"]),
        A.GridDropout(ratio=0.2, holes_number_xy=(6, 2), p=aug["grid_dropout_p"]),
    ]
    if gray:
        ops.append(_to_gray_1ch())
    ops += [A.Normalize(mean=_MEAN, std=_STD, max_pixel_value=255.0), ToTensorV2()]
    return A.Compose(ops)


def build_eval_transform(cfg: dict) -> A.Compose:
    model = cfg["model"]
    h, w = int(model["input_height"]), int(model["input_width"])
    gray = int(model.get("input_channels", 1)) == 1
    ops: list[A.BasicTransform] = [A.Resize(h, w, interpolation=cv2.INTER_CUBIC)]
    if gray:
        ops.append(_to_gray_1ch())
    ops += [A.Normalize(mean=_MEAN, std=_STD, max_pixel_value=255.0), ToTensorV2()]
    return A.Compose(ops)
