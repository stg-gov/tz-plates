#!/usr/bin/env python3
"""Run the ALPR pipeline on a single image and print the JSON result.

    python tools/predict_image.py path/to/car.jpg
    python tools/predict_image.py labeled_images/T336CAG-0007245e981f11ee9347df110422d5da.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tz_alpr.pipeline import get_pipeline
from tz_alpr.utils.image_io import imread


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path)
    ap.add_argument("--config", default="configs/inference.yaml")
    args = ap.parse_args()

    pipeline = get_pipeline(args.config)
    response = pipeline.read_image(imread(args.image))
    print(json.dumps(response.model_dump(), indent=2))


if __name__ == "__main__":
    main()
