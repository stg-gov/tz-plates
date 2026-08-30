"""Image enhancement (first box of the target workflow, spec §1).

Deliberately conservative: enhancement that helps a human read the plate also
helps the detector and OCR, but aggressive filtering destroys fine stroke
detail. We do adaptive contrast + optional light denoise, nothing more.
"""

from __future__ import annotations

import cv2
import numpy as np


class ImageEnhancer:
    def __init__(
        self,
        clahe_clip: float = 2.0,
        clahe_grid: int = 8,
        denoise: bool = False,
        gamma_target_mean: float = 0.5,
    ) -> None:
        self._clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
        self._denoise = denoise
        self._gamma_target_mean = gamma_target_mean

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.enhance(img)

    def enhance(self, img: np.ndarray) -> np.ndarray:
        out = img
        if out.ndim == 3:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        else:
            out = self._clahe.apply(out)

        out = self._auto_gamma(out)

        if self._denoise:
            out = (
                cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)
                if out.ndim == 3
                else cv2.fastNlMeansDenoising(out, None, 3, 7, 21)
            )
        return out

    def _auto_gamma(self, img: np.ndarray) -> np.ndarray:
        mean = float(np.mean(img)) / 255.0
        if mean <= 1e-3 or abs(mean - self._gamma_target_mean) < 0.06:
            return img
        gamma = np.log(self._gamma_target_mean) / np.log(mean)
        gamma = float(np.clip(gamma, 0.4, 2.5))
        lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, lut)


_DEFAULT = ImageEnhancer()


def auto_enhance(img: np.ndarray) -> np.ndarray:
    return _DEFAULT.enhance(img)
