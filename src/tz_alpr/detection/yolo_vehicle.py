"""Ultralytics YOLO vehicle detector with COCO->platform class remapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tz_alpr.detection.base import VehicleDetection, VehicleDetector

_COCO_TO_PLATFORM = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class YoloVehicleDetector(VehicleDetector):
    name = "yolo"

    def __init__(
        self,
        weights: str | Path,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.50,
        imgsz: int = 640,
        device: str = "cpu",
        max_det: int = 50,
        class_map: dict[int, str] | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ImportError('Install ultralytics: pip install -e ".[detect]"') from exc

        self._model = YOLO(str(weights))
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._imgsz = imgsz
        self._device = device
        self._max_det = max_det
        self._class_map = class_map or _COCO_TO_PLATFORM

    def detect(self, image: np.ndarray) -> list[VehicleDetection]:
        results = self._model.predict(
            image[:, :, ::-1],
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=self._device,
            max_det=self._max_det,
            verbose=False,
        )
        out: list[VehicleDetection] = []
        if not results or results[0].boxes is None:
            return out
        res = results[0]
        for i in range(len(res.boxes)):
            cls_id = int(res.boxes.cls[i])
            name = self._class_map.get(cls_id, res.names.get(cls_id, "other"))
            if name not in ("car", "motorcycle", "bus", "truck", "minibus", "tuktuk"):
                continue
            out.append(
                VehicleDetection(
                    bbox_xyxy=tuple(int(round(v)) for v in res.boxes.xyxy[i].tolist()),
                    vehicle_class=name,
                    confidence=float(res.boxes.conf[i]),
                )
            )
        return out
