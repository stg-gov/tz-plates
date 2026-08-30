from tz_alpr.plate_detection.base import PlateDetection, PlateDetector
from tz_alpr.plate_detection.classical import ClassicalPlateDetector
from tz_alpr.plate_detection.factory import build_plate_detector

__all__ = [
    "ClassicalPlateDetector",
    "PlateDetection",
    "PlateDetector",
    "build_plate_detector",
]
