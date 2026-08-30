#!/usr/bin/env python3
"""Fetch open-source pretrained weights used to bootstrap the pipeline.

Currently: an open-source YOLOv11-nano license-plate detector, used to auto-crop
plates from the labelled dataset for OCR training and as the runtime plate
detector until a project-specific detector is trained.

    python tools/fetch_pretrained.py                 # plate detector -> models/plate_detector/v1/plate_yolo.pt

Source: https://huggingface.co/morsetechlab/yolov11-license-plate-detection
(YOLOv11 weights; released under AGPL-3.0 / commercial per Ultralytics' terms —
swap for an Apache-2.0 detector if that matters for your deployment).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PLATE_DETECTOR = {
    "repo_id": "morsetechlab/yolov11-license-plate-detection",
    "filename": "license-plate-finetune-v1n.pt",
    "dest": REPO_ROOT / "models/plate_detector/v1/plate_yolo.pt",
    "sha_hint": None,
}


def _download(repo_id: str, filename: str, dest: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit('huggingface_hub is required. Install:  pip install -e ".[detect]"')

    dest.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id=repo_id, filename=filename)
    shutil.copy(cached, dest)
    print(f"{repo_id}/{filename}  ->  {dest}  ({dest.stat().st_size // 1024} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = ap.parse_args()

    d = PLATE_DETECTOR
    if d["dest"].exists() and not args.force:
        print(f"already present: {d['dest']}  (use --force to re-download)")
    else:
        _download(d["repo_id"], d["filename"], d["dest"])

    print(
        "\nSet this so the pipeline and tools/prepare_dataset.py use it:\n"
        f"  export TZ_ALPR_PLATE_DETECTOR_WEIGHTS={d['dest'].relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
