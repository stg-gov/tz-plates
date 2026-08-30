"""Stage-1 vehicle detection (spec §3). Active from Phase 2."""

from tz_alpr.detection.base import VEHICLE_CLASSES, VehicleDetection, VehicleDetector
from tz_alpr.detection.factory import build_vehicle_detector

__all__ = [
    "VEHICLE_CLASSES",
    "VehicleDetection",
    "VehicleDetector",
    "build_vehicle_detector",
]
