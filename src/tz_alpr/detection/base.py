"""Vehicle detector contract: ``VehicleDetector.detect(image)`` (spec §3)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck", "minibus", "tuktuk", "other")


@dataclass
class VehicleDetection:
    bbox_xyxy: tuple[int, int, int, int]
    vehicle_class: str
    confidence: float
    track_id: int | None = None


class VehicleDetector(ABC):
    name: str = "base"

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[VehicleDetection]:
        ...
