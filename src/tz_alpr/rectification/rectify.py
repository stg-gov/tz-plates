"""Plate rectification (spec §5).

Pipeline: detected plate -> corner detection -> perspective correction ->
normalized plate image.

Approach chosen: homography from a 4-point quad.
  * If the detector supplied a quad (oriented-box YOLO or the classical
    contour approximation), use it directly.
  * Otherwise recover corners inside the padded bbox crop with a contour /
    ``minAreaRect`` search on a high-contrast mask.
  * If neither yields a plausible quad, fall back to an aspect-preserving
    resize so the pipeline still produces an OCR-ready image.

A Spatial Transformer Network is a natural upgrade (learned, differentiable
rectification jointly trained with OCR); the interface here — ``rectify`` returns
a fixed-size image — does not change when that lands.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tz_alpr.plate_detection.base import PlateDetection
from tz_alpr.utils.geometry import order_quad

# Tanzanian plates: single-line ~4.5:1, two-line (stacked) ~1.2-1.7:1. A wide
# gap, so 2.6 separates them cleanly. Most TZ car plates in the wild are two-line.
_TWO_LINE_ASPECT_MAX = 2.6


@dataclass
class RectifiedPlate:
    image: np.ndarray                 # BGR, (out_h, out_w, 3)
    method: str                       # "quad" | "contour" | "resize"
    quad_image_coords: np.ndarray | None
    is_two_line: bool


class PlateRectifier:
    def __init__(
        self,
        output_width: int = 192,
        output_height: int = 48,
        two_line_output_height: int = 96,
        pad_ratio: float = 0.12,
        contour_search: bool = False,
    ) -> None:
        self._out_w = output_width
        self._out_h = output_height
        self._two_line_h = two_line_output_height
        self._pad = pad_ratio
        # Speculative Canny/contour quad search inside a box-only detection often
        # warps a good crop into garbage on noisy phone photos. Off by default;
        # a detector that supplies its own oriented quad still gets a warp.
        self._contour_search = contour_search

    def rectify(self, image: np.ndarray, det: PlateDetection) -> RectifiedPlate:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = det.bbox_xyxy
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        is_two_line = (bw / bh) < _TWO_LINE_ASPECT_MAX

        quad = det.quad if det.quad is not None else None
        method = "quad"
        if quad is None and self._contour_search:
            quad = self._find_quad_in_crop(image, det)
            method = "contour" if quad is not None else "resize"
        elif quad is None:
            method = "resize"

        if quad is None:
            px1, py1, px2, py2 = self._pad_box(x1, y1, x2, y2, w, h)
            warped = image[py1:py2, px1:px2]
            quad_img = None
        else:
            quad = order_quad(quad)
            warped = self._warp(image, quad, is_two_line)
            quad_img = quad

        if is_two_line:
            warped = self._destack(warped)

        warped = cv2.resize(warped, (self._out_w, self._out_h), interpolation=cv2.INTER_CUBIC)
        return RectifiedPlate(
            image=warped, method=method, quad_image_coords=quad_img, is_two_line=is_two_line
        )

    # ------------------------------------------------------------------ helpers
    def _pad_box(self, x1, y1, x2, y2, w, h):
        dx, dy = int((x2 - x1) * self._pad), int((y2 - y1) * self._pad)
        return (max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy))

    def _warp(self, image: np.ndarray, quad: np.ndarray, is_two_line: bool) -> np.ndarray:
        out_h = self._two_line_h if is_two_line else self._out_h
        dst = np.array(
            [[0, 0], [self._out_w - 1, 0], [self._out_w - 1, out_h - 1], [0, out_h - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        return cv2.warpPerspective(image, M, (self._out_w, out_h), flags=cv2.INTER_CUBIC)

    def _find_quad_in_crop(self, image: np.ndarray, det: PlateDetection) -> np.ndarray | None:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = self._pad_box(*det.bbox_xyxy, w, h)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 60, 60)
        edges = cv2.Canny(gray, 40, 140)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        crop_area = crop.shape[0] * crop.shape[1]
        best, best_area = None, 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.15 * crop_area or area <= best_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            pts = (
                approx.reshape(-1, 2)
                if len(approx) == 4
                else cv2.boxPoints(cv2.minAreaRect(cnt))
            )
            if len(pts) != 4:
                continue
            best, best_area = pts.astype(np.float32), area

        if best is None:
            return None
        best[:, 0] += x1
        best[:, 1] += y1
        return best

    def _destack(self, plate_img: np.ndarray) -> np.ndarray:
        """Turn a 2-line plate into one line: bottom row placed after the top row.

        Uses overlapping bands (not an exact 50/50 cut) so neither line is clipped
        when the split is slightly off, and trims the outer padding rows.
        """
        h, w = plate_img.shape[:2]
        top = plate_img[int(0.04 * h) : int(0.58 * h)]
        bottom = plate_img[int(0.42 * h) : int(0.96 * h)]
        band_h = max(top.shape[0], bottom.shape[0], 8)
        out_w = max(self._out_w, w)
        top = cv2.resize(top, (out_w, band_h), interpolation=cv2.INTER_CUBIC)
        bottom = cv2.resize(bottom, (out_w, band_h), interpolation=cv2.INTER_CUBIC)
        return np.hstack([top, bottom])
