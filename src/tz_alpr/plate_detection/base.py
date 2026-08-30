"""License-plate detector contract (spec §4).

The plate detector is independent of the vehicle detector. In Phase 1 it runs on
the full frame; in Phase 2 it runs on vehicle crops. It always reports the plate
box in *image* coordinates; the pipeline adds the vehicle-relative box.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PlateDetection:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    source: str = "unknown"
    quad: np.ndarray | None = None            # (4, 2) float32, image coords, TL-TR-BR-BL
    extra: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> int:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]


class PlateDetector(ABC):
    name: str = "base"

    @abstractmethod
    def detect(self, image: np.ndarray, max_plates: int = 8) -> list[PlateDetection]:
        """Return plate detections in `image` coordinates, best first."""
