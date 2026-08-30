#!/usr/bin/env python3
"""Dataset preparation for the Tanzanian ALPR dataset (spec §10, §11, §13).

Input  : labeled_images/*.jpg  +  labels.jsonl  ({"image","plate_text","confidence"})
Output :
  datasets/processed/plate_crops/*.jpg        rectified plate crops for OCR
  datasets/ocr/ocr_annotations.csv            OCR labels + leakage-safe split column
  datasets/annotations/plates/{images,labels} YOLO-format weak plate boxes + plates.yaml
  datasets/annotations/metadata.jsonl         unified per-image metadata
  datasets/splits/{train,val,test,hard_test}.txt

Splitting is grouped by plate identity (a plate that appears in several photos
never straddles splits). Crops that are small / low-confidence / motorcycle /
unrecognised-pattern are routed to ``hard_test`` (spec §13).

The plate boxes written here are WEAK labels produced by the current detector
(classical fallback unless YOLO weights are set). Hand-correct a few hundred in
any YOLO annotation tool before training the production plate detector; the OCR
crops are usable as-is because the text label is human-verified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from tz_alpr.country_rules import get_country_rules
from tz_alpr.plate_detection.factory import build_plate_detector
from tz_alpr.rectification import PlateRectifier


@dataclass
class PreparedItem:
    src_image: str
    crop_path: str
    plate_text: str
    plate_type: str
    plate_bbox_xyxy: list[int]
    det_conf: float
    crop_w: int
    lighting: str
    blur_score: float
    ok: bool


def _worker(args: tuple[str, str, str]) -> PreparedItem | None:
    image_rel, plate_text, images_root = args
    src = Path(images_root) / image_rel
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        return None

    detector = _get_detector()
    rectifier = _get_rectifier()
    rules = get_country_rules("TZ")

    dets = detector.detect(img, max_plates=3)
    plate_type = rules.classify(rules.clean(plate_text)).name
    mean_v = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    lighting = "night" if mean_v < 60 else "day" if mean_v > 110 else "dim"

    crop_dir = REPO_ROOT / "datasets/processed/plate_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{plate_text}-{hashlib.md5(image_rel.encode()).hexdigest()[:12]}.jpg"
    crop_path = crop_dir / stem

    if dets:
        det = dets[0]
        rect = rectifier.rectify(img, det)
        crop = rect.image
        bbox = list(det.bbox_xyxy)
        conf = float(det.confidence)
        ok = True
    else:
        h, w = img.shape[:2]
        y1, y2 = int(h * 0.45), int(h * 0.85)
        crop = cv2.resize(img[y1:y2, :], (192, 48))
        bbox = [0, y1, w, y2]
        conf = 0.0
        ok = False

    cv2.imwrite(str(crop_path), crop)
    blur = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

    return PreparedItem(
        src_image=str(src.relative_to(REPO_ROOT)) if src.is_relative_to(REPO_ROOT) else str(src),
        crop_path=str(crop_path.relative_to(REPO_ROOT)),
        plate_text=plate_text,
        plate_type=plate_type,
        plate_bbox_xyxy=[int(v) for v in bbox],
        det_conf=conf,
        crop_w=int(crop.shape[1]),
        lighting=lighting,
        blur_score=blur,
        ok=ok,
    )


_DETECTOR = None
_RECTIFIER = None


def _get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = build_plate_detector()
    return _DETECTOR


def _get_rectifier():
    global _RECTIFIER
    if _RECTIFIER is None:
        _RECTIFIER = PlateRectifier()
    return _RECTIFIER


def _assign_splits(items: list[PreparedItem], cfg: dict) -> dict[str, str]:
    split_cfg = cfg["split"]
    ratios = split_cfg["ratios"]
    hard = split_cfg["hard_test"]
    rng = random.Random(split_cfg.get("seed", 1337))

    groups: dict[str, list[PreparedItem]] = {}
    for it in items:
        groups.setdefault(it.plate_text, []).append(it)

    keys = sorted(groups)
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    group_split = {}
    for i, k in enumerate(keys):
        group_split[k] = "train" if i < n_train else "val" if i < n_train + n_val else "test"

    per_image: dict[str, str] = {}
    for it in items:
        base = group_split[it.plate_text]
        is_hard = (
            it.crop_w <= hard["max_plate_px_width"]
            or it.det_conf < hard["min_detector_conf"]
            or it.plate_type in set(hard.get("include_categories", []))
            or not it.ok
        )
        per_image[it.crop_path] = "hard_test" if (is_hard and base != "train") else base
    return per_image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--labels", type=Path, default=REPO_ROOT / "datasets/annotations/accepted_labels.jsonl",
        help="audited manifest; run tools/audit_labeled_images.py first",
    )
    ap.add_argument("--images", type=Path, default=REPO_ROOT / "labeled_images")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-label-conf", type=float, default=1.0)
    ap.add_argument(
        "--skip-yolo-export",
        action="store_true",
        help="skip writing the YOLO plate-detection dataset (only OCR crops + csv)",
    )
    args = ap.parse_args()

    if not args.labels.exists():
        raise FileNotFoundError(
            f"{args.labels} does not exist. Run: python tools/audit_labeled_images.py"
        )

    from tz_alpr.config import load_yaml

    train_cfg = load_yaml("configs/training.yaml")

    rows: list[tuple[str, str, str]] = []
    with args.labels.open() as fh:
        for line in fh:
            d = json.loads(line)
            if float(d.get("confidence", 1.0)) < args.min_label_conf:
                continue
            text = d["plate_text"].strip().upper()
            if not (4 <= len(text) <= 10):
                continue
            rows.append((d["image"], text, str(args.images)))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Preparing {len(rows)} images with {args.workers} workers ...")

    items: list[PreparedItem] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_worker, r) for r in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res is not None:
                items.append(res)
            if i % 1000 == 0:
                print(f"  {i}/{len(rows)}  (detected={sum(x.ok for x in items)})")

    splits = _assign_splits(items, train_cfg)

    ocr_csv = REPO_ROOT / "datasets/ocr/ocr_annotations.csv"
    ocr_csv.parent.mkdir(parents=True, exist_ok=True)
    with ocr_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["image_path", "plate_text", "country", "plate_type", "split", "group_key",
             "source", "det_conf", "crop_w"]
        )
        for it in items:
            w.writerow(
                [it.crop_path, it.plate_text, "TZ", it.plate_type,
                 splits[it.crop_path], it.plate_text, "real",
                 round(it.det_conf, 4), it.crop_w]
            )

    if not args.skip_yolo_export:
        _write_yolo_dataset(items, splits)
    _write_metadata(items)
    _write_split_lists(items, splits)

    counts: dict[str, int] = {}
    for s in splits.values():
        counts[s] = counts.get(s, 0) + 1
    det_rate = sum(x.ok for x in items) / max(1, len(items))
    print(f"\nDone. crops={len(items)}  detect_rate={det_rate:.1%}")
    print(f"split sizes: {counts}")
    print(f"OCR annotations: {ocr_csv}")


def _write_yolo_dataset(items: list[PreparedItem], splits: dict[str, str]) -> None:
    root = REPO_ROOT / "datasets/annotations/plates"
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    for it in items:
        if not it.ok:
            continue
        split = "val" if splits[it.crop_path] in ("val", "test", "hard_test") else "train"
        src = REPO_ROOT / it.src_image
        img = cv2.imread(str(src))
        if img is None:
            continue
        h, w = img.shape[:2]
        x1, y1, x2, y2 = it.plate_bbox_xyxy
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        name = Path(it.crop_path).stem
        cv2.imwrite(str(root / f"images/{split}/{name}.jpg"), img)
        (root / f"labels/{split}/{name}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    (root / "plates.yaml").write_text(
        "path: " + str(root) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: plate\n"
    )


def _write_metadata(items: list[PreparedItem]) -> None:
    path = REPO_ROOT / "datasets/annotations/metadata.jsonl"
    with path.open("w") as fh:
        for it in items:
            fh.write(
                json.dumps(
                    {
                        "image_id": Path(it.crop_path).stem,
                        "camera_id": None,
                        "timestamp": None,
                        "vehicle_bbox": None,
                        "plate_bbox": it.plate_bbox_xyxy,
                        "plate_text": it.plate_text,
                        "plate_type": it.plate_type,
                        "vehicle_type": None,
                        "weather": None,
                        "lighting": it.lighting,
                        "angle": None,
                        "occlusion": None,
                        "blur_score": round(it.blur_score, 2),
                    }
                )
                + "\n"
            )


def _write_split_lists(items: list[PreparedItem], splits: dict[str, str]) -> None:
    out = REPO_ROOT / "datasets/splits"
    out.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[str]] = {"train": [], "val": [], "test": [], "hard_test": []}
    for it in items:
        buckets[splits[it.crop_path]].append(it.crop_path)
    for name, paths in buckets.items():
        (out / f"{name}.txt").write_text("\n".join(sorted(paths)) + "\n")


if __name__ == "__main__":
    main()
