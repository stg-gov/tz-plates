"""Classical CV plate detector.

Purpose: make the end-to-end pipeline runnable from a clean checkout, before any
YOLO plate weights exist, and provide a cheap CPU fallback. It looks for the
Tanzanian yellow (private/commercial) or white (government) plate rectangle.

This is intentionally a heuristic. It is replaced by the trained detector
(``yolo_plate.YoloPlateDetector``) as soon as weights are available — see
``factory.build_plate_detector``.
"""

from __future__ import annotations

import cv2
import numpy as np

from tz_alpr.plate_detection.base import PlateDetection, PlateDetector

_IDEAL_ASPECT = 3.4                       # single-line TZ plate w/h
_IDEAL_ASPECT_TWO_LINE = 1.5              # motorcycle stacked plate


class ClassicalPlateDetector(PlateDetector):
    name = "classical"

    def __init__(self, cfg: dict) -> None:
        self._yellow_lo = np.array(cfg.get("yellow_hsv_low", [15, 60, 60]), np.uint8)
        self._yellow_hi = np.array(cfg.get("yellow_hsv_high", [45, 255, 255]), np.uint8)
        self._white_lo = np.array(cfg.get("white_hsv_low", [0, 0, 150]), np.uint8)
        self._white_hi = np.array(cfg.get("white_hsv_high", [180, 60, 255]), np.uint8)
        self._min_area_ratio = float(cfg.get("min_area_ratio", 0.0008))
        self._max_area_ratio = float(cfg.get("max_area_ratio", 0.35))
        self._min_aspect = float(cfg.get("min_aspect", 1.6))
        self._max_aspect = float(cfg.get("max_aspect", 6.0))
        self._eps_ratio = float(cfg.get("approx_epsilon_ratio", 0.02))

    def detect(self, image: np.ndarray, max_plates: int = 8) -> list[PlateDetection]:
        h, w = image.shape[:2]
        img_area = float(h * w)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.bitwise_or(
            cv2.inRange(hsv, self._yellow_lo, self._yellow_hi),
            cv2.inRange(hsv, self._white_lo, self._white_hi),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[PlateDetection] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area / img_area < self._min_area_ratio or area / img_area > self._max_area_ratio:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            if rw < 1 or rh < 1:
                continue
            long_side, short_side = max(rw, rh), min(rw, rh)
            aspect = long_side / short_side
            if not (self._min_aspect <= aspect <= self._max_aspect):
                continue

            box = cv2.boxPoints(rect).astype(np.float32)
            fill = area / (rw * rh + 1e-6)
            ideal = _IDEAL_ASPECT if aspect > 2.2 else _IDEAL_ASPECT_TWO_LINE
            aspect_score = max(0.0, 1.0 - abs(aspect - ideal) / ideal)
            conf = float(np.clip(0.35 * fill + 0.45 * aspect_score + 0.20, 0.05, 0.9))

            quad = _quad_from_contour(cnt, self._eps_ratio)
            if quad is None:
                quad = _order(box)
            x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            candidates.append(
                PlateDetection(
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=conf,
                    source=self.name,
                    quad=quad,
                    extra={"aspect": aspect, "fill": fill},
                )
            )

        candidates.sort(key=lambda d: d.confidence, reverse=True)
        return _nms(candidates, iou_thresh=0.3)[:max_plates]


def _order(box: np.ndarray) -> np.ndarray:
    from tz_alpr.utils.geometry import order_quad

    return order_quad(box)


def _quad_from_contour(cnt: np.ndarray, eps_ratio: float) -> np.ndarray | None:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps_ratio * peri, True)
    if len(approx) == 4:
        return _order(approx.reshape(4, 2).astype(np.float32))
    return None


def _nms(dets: list[PlateDetection], iou_thresh: float) -> list[PlateDetection]:
    from tz_alpr.utils.geometry import iou

    kept: list[PlateDetection] = []
    for det in dets:
        if all(iou(det.bbox_xyxy, k.bbox_xyxy) < iou_thresh for k in kept):
            kept.append(det)
    return kept
