from __future__ import annotations

from tz_alpr.config import get_settings, load_yaml, resolve_path
from tz_alpr.detection.base import VehicleDetector
from tz_alpr.logging_conf import get_logger

log = get_logger(__name__)


def build_vehicle_detector(config_path: str = "configs/detector.yaml") -> VehicleDetector | None:
    """Best available stage-1 vehicle detector, or ``None``.

    Priority:
      1. Custom fine-tuned weights (``TZ_ALPR_VEHICLE_DETECTOR_WEIGHTS``) — adds
         minibus / tuk-tuk on top of the COCO classes.
      2. COCO-pretrained ``yolov8n.pt`` (open weights, auto-downloaded) — covers
         car / motorcycle / bus / truck out of the box.
      3. ``None`` — ultralytics missing or offline; the pipeline then runs
         full-frame plate detection (Phase 1 behaviour).
    """
    settings = get_settings()
    model_cfg = load_yaml(config_path).get("model", {})
    common = dict(
        conf_threshold=model_cfg.get("conf_threshold", 0.30),
        iou_threshold=model_cfg.get("iou_threshold", 0.50),
        imgsz=model_cfg.get("imgsz", 640),
        device=settings.device,
    )

    custom = resolve_path(settings.vehicle_detector_weights)
    if custom and custom.exists():
        try:
            from tz_alpr.detection.yolo_vehicle import YoloVehicleDetector

            log.info("vehicle_detector.load", backend="yolo-custom", weights=str(custom))
            return YoloVehicleDetector(weights=custom, **common)
        except Exception as exc:  # noqa: BLE001
            log.warning("vehicle_detector.custom_failed", error=str(exc))

    try:
        from tz_alpr.detection.yolo_vehicle import YoloVehicleDetector

        coco_weights = model_cfg.get("weights", "yolov8n.pt")
        detector = YoloVehicleDetector(weights=coco_weights, **common)
        log.info("vehicle_detector.load", backend="yolo-coco", weights=coco_weights)
        return detector
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "vehicle_detector.unavailable",
            error=str(exc),
            hint='pip install -e ".[detect]" for Phase 2 vehicle detection',
        )
        return None
