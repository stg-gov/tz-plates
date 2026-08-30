"""Pick the best available plate detector: trained YOLO if weights exist,
otherwise the classical fallback (if enabled in config)."""

from __future__ import annotations

from tz_alpr.config import get_settings, load_yaml, resolve_path
from tz_alpr.logging_conf import get_logger
from tz_alpr.plate_detection.base import PlateDetector
from tz_alpr.plate_detection.classical import ClassicalPlateDetector

log = get_logger(__name__)


def build_plate_detector(config_path: str = "configs/plate_detector.yaml") -> PlateDetector:
    cfg = load_yaml(config_path)
    settings = get_settings()

    weights = resolve_path(settings.plate_detector_weights)
    if weights and weights.exists():
        try:
            from tz_alpr.plate_detection.yolo_plate import YoloPlateDetector

            model_cfg = cfg.get("model", {})
            log.info("plate_detector.load", backend="yolo", weights=str(weights))
            return YoloPlateDetector(
                weights=weights,
                conf_threshold=model_cfg.get("conf_threshold", 0.25),
                iou_threshold=model_cfg.get("iou_threshold", 0.50),
                imgsz=model_cfg.get("imgsz", 960),
                device=settings.device,
                max_det=model_cfg.get("max_det", 20),
            )
        except (ImportError, FileNotFoundError) as exc:
            log.warning("plate_detector.yolo_unavailable", error=str(exc))

    fb = cfg.get("fallback", {})
    if not fb.get("enabled", True):
        raise RuntimeError(
            "No plate-detector weights found and classical fallback is disabled. "
            "Set TZ_ALPR_PLATE_DETECTOR_WEIGHTS or enable fallback in plate_detector.yaml."
        )
    log.warning("plate_detector.load", backend="classical", reason="no trained weights")
    return ClassicalPlateDetector(fb)
