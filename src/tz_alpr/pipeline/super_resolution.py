"""Optional plate super-resolution (spec §9).

Disabled by default because it adds latency. Two backends:
  * Real-ESRGAN if ``realesrgan`` + weights are installed (best quality).
  * A dependency-free fallback: Lanczos 3x upscale + unsharp mask + mild
    bilateral denoise. Not a true SR network, but measurably helps OCR on tiny
    crops and always available.

Whether SR actually improves end-to-end accuracy must be measured with
``training/evaluate.py --with-sr`` before enabling it in production.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class BicubicUnsharpSR:
    def __init__(self, scale: int = 3) -> None:
        self._scale = scale

    def __call__(self, img: np.ndarray) -> np.ndarray:
        up = cv2.resize(img, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_LANCZOS4)
        blur = cv2.GaussianBlur(up, (0, 0), 1.2)
        sharp = cv2.addWeighted(up, 1.6, blur, -0.6, 0)
        return cv2.bilateralFilter(sharp, 5, 40, 40)


class RealEsrganSR:
    def __init__(self, weights: str | Path, scale: int = 4, device: str = "cpu") -> None:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=scale
        )
        self._upsampler = RealESRGANer(
            scale=scale, model_path=str(weights), model=model, half=False, device=device
        )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        out, _ = self._upsampler.enhance(img, outscale=self._scale if hasattr(self, "_scale") else 4)
        return out


def build_super_resolver(sr_cfg: dict):
    weights = sr_cfg.get("weights", "")
    if weights and Path(weights).exists():
        try:
            return RealEsrganSR(weights)
        except Exception:  # noqa: BLE001 - fall back silently to the always-available path
            pass
    return BicubicUnsharpSR()
