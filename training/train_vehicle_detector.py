#!/usr/bin/env python3
"""Fine-tune the stage-1 vehicle detector (Ultralytics YOLO), spec §3, §14.

Only needed to add the local classes COCO lacks — **minibus** (daladala) and
**tuktuk** (bajaji) — and to sharpen car/motorcycle/bus/truck on Tanzanian street
scenes. Until this runs, the pipeline uses COCO-pretrained ``yolov8n.pt`` and a
daladala reads as "bus", a bajaji as "motorcycle" (does not block plate OCR).

    python training/train_vehicle_detector.py --data datasets/annotations/vehicles/vehicles.yaml

`vehicles.yaml` (YOLO format) with classes:
    0: car  1: motorcycle  2: bus  3: truck  4: minibus  5: tuktuk
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs/detector.yaml")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg = yaml.safe_load(args.config.read_text())
    tcfg = cfg["train"]
    model = YOLO(args.weights or cfg["model"]["weights"])

    model.train(
        data=str(args.data or tcfg["data_yaml"]),
        epochs=args.epochs or tcfg["epochs"],
        imgsz=tcfg["imgsz"],
        batch=tcfg["batch"],
        patience=tcfg["patience"],
        device=args.device or tcfg["device"],
        project=tcfg["project"],
        name=tcfg["name"],
    )
    best = Path(tcfg["project"]) / tcfg["name"] / "weights" / "best.pt"
    dst = REPO_ROOT / "models/detector/v1/vehicle_yolo.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        dst.write_bytes(best.read_bytes())
        print(f"Copied {best} -> {dst}")
        print("Set TZ_ALPR_VEHICLE_DETECTOR_WEIGHTS=models/detector/v1/vehicle_yolo.pt")


if __name__ == "__main__":
    main()
