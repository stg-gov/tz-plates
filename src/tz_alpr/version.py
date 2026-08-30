"""Single source of truth for package and model-registry versions.

`__version__` tracks the codebase. `MODEL_VERSION` tracks the deployed model bundle
(detector + OCR + rules) and is echoed in every API response so a recognition can
always be traced back to the exact artifacts that produced it (spec §30).
"""

import os

# 1.0.0 = Phase 1 (image -> plate -> OCR).  1.1.0 = Phase 2 (vehicle detection).
# 1.2.0 = Phase 3 (video: tracking + temporal OCR aggregation + event dedup).
__version__ = "1.2.0"

# Overridable per-deployment without a code change; the registry layout under
# models/<component>/<vN>/ is the durable record.
MODEL_VERSION = os.environ.get("TZ_ALPR_MODEL_VERSION", "tz-alpr-1.2.0")

COMPONENT_VERSIONS = {
    "vehicle_detector": os.environ.get("TZ_ALPR_VEHICLE_DETECTOR_VERSION", "coco-yolov8n"),
    "plate_detector": os.environ.get("TZ_ALPR_PLATE_DETECTOR_VERSION", "v1"),
    "ocr": os.environ.get("TZ_ALPR_OCR_VERSION", "v1"),
    "country_rules": os.environ.get("TZ_ALPR_RULES_VERSION", "tz-2024.1"),
}
