#!/usr/bin/env python3
"""Train the dedicated license-plate detector (Ultralytics YOLO), spec §4, §14.

    python training/train_plate_detector.py --data datasets/annotations/plates/plates.yaml

Requires hand-verified boxes. tools/prepare_dataset.py writes weak boxes from the
classical detector to bootstrap; correct a few hundred before a production run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs/plate_detector.yaml")
    ap.add_argument("--data", type=Path, default=None, help="override data yaml")
    ap.add_argument("--weights", type=str, default=None, help="override start weights")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg = yaml.safe_load(args.config.read_text())
    tcfg = cfg["train"]
    model = YOLO(args.weights or cfg["model"]["weights"])

    results = model.train(
        data=str(args.data or tcfg["data_yaml"]),
        epochs=args.epochs or tcfg["epochs"],
        imgsz=tcfg["imgsz"],
        batch=tcfg["batch"],
        patience=tcfg["patience"],
        device=args.device or tcfg["device"],
        project=tcfg["project"],
        name=tcfg["name"],
        augment=tcfg.get("augment", True),
        single_cls=True,
    )
    print(results)
    best = Path(tcfg["project"]) / tcfg["name"] / "weights" / "best.pt"
    dst = REPO_ROOT / "models/plate_detector/v1/plate_yolo.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        dst.write_bytes(best.read_bytes())
        print(f"Copied {best} -> {dst}")
        print("Set TZ_ALPR_PLATE_DETECTOR_WEIGHTS=models/plate_detector/v1/plate_yolo.pt")


if __name__ == "__main__":
    main()
