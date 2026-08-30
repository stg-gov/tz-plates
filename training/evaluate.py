#!/usr/bin/env python3
"""Evaluation + confidence calibration (spec §15, §16).

Runs OCR -> Tanzania-aware decode -> confidence fusion over a split of
``ocr_annotations.csv`` and reports:
  * OCR: character accuracy, CER, sequence accuracy
  * End-to-end: exact normalized-plate match accuracy (the only success metric
    that counts, spec §15)
  * Slices: by plate_type, by lighting (from metadata.jsonl), by predicted
    review band
  * A reliability table (confidence vs. empirical accuracy) and a fitted Platt
    calibration written to models/ocr/v1/confidence_calibration.json

    python training/evaluate.py --split test
    python training/evaluate.py --split hard_test --calibrate
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from tz_alpr.country_rules import get_country_rules
from tz_alpr.ocr.dataset import load_samples_from_csv
from tz_alpr.ocr.engine import build_ocr_engine
from tz_alpr.ocr.metrics import score_batch
from tz_alpr.postprocessing.confidence import StageScores, build_confidence_model
from tz_alpr.postprocessing.tz_aware import TanzaniaAwareDecoder


def _load_lighting_map() -> dict[str, str]:
    path = REPO_ROOT / "datasets/annotations/metadata.jsonl"
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            d = json.loads(line)
            out[d["image_id"]] = d.get("lighting") or "unknown"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=REPO_ROOT / "datasets/ocr/ocr_annotations.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "reports")
    args = ap.parse_args()

    samples = load_samples_from_csv(args.csv, split=args.split)
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit(f"No samples for split={args.split} in {args.csv}")

    ocr = build_ocr_engine()
    rules = get_country_rules("TZ")
    decoder = TanzaniaAwareDecoder(rules)
    conf_model = build_confidence_model()
    lighting_map = _load_lighting_map()

    preds, gts = [], []
    e2e_correct = 0
    slices: dict[str, list[int]] = defaultdict(list)
    calib_pairs: list[tuple[float, bool]] = []
    reliability = defaultdict(lambda: [0, 0])  # bucket -> [correct, total]

    for s in samples:
        img = cv2.imread(str(REPO_ROOT / s.image_path))
        if img is None:
            continue
        pred = ocr.predict(img)
        tz = decoder.decode(pred.raw_text, pred.positions, pred.seq_confidence)
        gt_norm = rules.clean(s.text)

        preds.append(tz.raw_ocr)
        gts.append(gt_norm)
        is_exact = tz.normalized_text == gt_norm
        e2e_correct += int(is_exact)

        val_conf = rules.validation_confidence(tz.normalized_text, tz.category, tz.n_swaps)
        length_mismatch = bool(
            tz.category.slots and len(tz.category.slots) != len(tz.normalized_text)
        )
        fused = conf_model.fuse(
            StageScores(0.0, 0.9, pred.seq_confidence, val_conf),
            n_swaps=tz.n_swaps,
            length_mismatch=length_mismatch,
        )
        calib_pairs.append((fused.final_confidence, is_exact))
        bucket = round(fused.final_confidence * 10) / 10
        reliability[bucket][0] += int(is_exact)
        reliability[bucket][1] += 1

        img_id = Path(s.image_path).stem
        slices[f"type:{s.plate_type}"].append(int(is_exact))
        slices[f"light:{lighting_map.get(img_id, 'unknown')}"].append(int(is_exact))
        slices[f"review:{fused.review_status}"].append(int(is_exact))

    ocr_scores = score_batch(preds, gts)
    n = len(gts)
    report = {
        "split": args.split,
        "n": n,
        "ocr": ocr_scores.as_dict(),
        "end_to_end_exact_match": round(e2e_correct / n, 4),
        "slices": {
            k: {"exact_match": round(sum(v) / len(v), 4), "n": len(v)}
            for k, v in sorted(slices.items())
        },
        "reliability": {
            str(k): {"acc": round(c / t, 3), "n": t} for k, (c, t) in sorted(reliability.items())
        },
    }

    args.report.mkdir(parents=True, exist_ok=True)
    out = args.report / f"eval_{args.split}.json"
    out.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nReport -> {out}")

    if args.calibrate:
        try:
            a, b = conf_model.calibrate(calib_pairs)
            path = REPO_ROOT / "models/ocr/v1/confidence_calibration.json"
            conf_model.save(path)
            print(f"Fitted Platt scaling a={a:.3f} b={b:.3f} -> {path}")
        except ValueError as exc:
            print(f"Calibration skipped: {exc}")


if __name__ == "__main__":
    main()
