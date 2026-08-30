"""Trained YOLO plate detector (Ultralytics). Used whenever weights are present."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tz_alpr.plate_detection.base import PlateDetection, PlateDetector


class YoloPlateDetector(PlateDetector):
    name = "yolo"

    def __init__(
        self,
        weights: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.50,
        imgsz: int = 960,
        device: str = "cpu",
        max_det: int = 20,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ultralytics is required for YoloPlateDetector. "
                'Install with: pip install -e ".[detect]"'
            ) from exc

        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"Plate detector weights not found: {weights}")

        self._model = YOLO(str(weights))
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._imgsz = imgsz
        self._device = device
        self._max_det = max_det

    def detect(self, image: np.ndarray, max_plates: int = 8) -> list[PlateDetection]:
        results = self._model.predict(
            image[:, :, ::-1],  # ultralytics expects RGB
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=self._device,
            max_det=self._max_det,
            verbose=False,
        )
        dets: list[PlateDetection] = []
        if not results:
            return dets
        res = results[0]
        boxes = getattr(res, "obb", None) or res.boxes
        if res.boxes is None:
            return dets

        for i in range(len(res.boxes)):
            xyxy = res.boxes.xyxy[i].tolist()
            conf = float(res.boxes.conf[i])
            quad = None
            if getattr(res, "obb", None) is not None and res.obb is not None:
                quad = np.asarray(res.obb.xyxyxyxy[i].tolist(), dtype=np.float32).reshape(4, 2)
            dets.append(
                PlateDetection(
                    bbox_xyxy=tuple(int(round(v)) for v in xyxy),
                    confidence=conf,
                    source=self.name,
                    quad=quad,
                )
            )

        dets.sort(key=lambda d: d.confidence, reverse=True)
        return dets[:max_plates]
