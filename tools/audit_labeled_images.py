#!/usr/bin/env python3
"""Validate original labelled images before any GPU training.

It rejects objective failures and writes an explicit report; visual mistakes
remain reviewable rather than being silently relabelled by a script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "PRIVATE": re.compile(r"^T\d{3}[A-Z]{3}$"),
    "MOTORCYCLE": re.compile(r"^MC\d{3}[A-Z]{3}$"),
    "GOVERNMENT": re.compile(r"^[A-Z]{2,3}\d{3,4}$"),
    "DIPLOMATIC": re.compile(r"^\d{1,3}[A-Z]{2}\d{2,4}$"),
}


def classify(text: str) -> str | None:
    return next((name for name, pattern in PATTERNS.items() if pattern.fullmatch(text)), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=REPO_ROOT / "labels.jsonl")
    parser.add_argument("--images", type=Path, default=REPO_ROOT / "labeled_images")
    parser.add_argument("--accepted", type=Path, default=REPO_ROOT / "datasets/annotations/accepted_labels.jsonl")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "reports/data_audit.jsonl")
    parser.add_argument("--min-confidence", type=float, default=1.0)
    args = parser.parse_args()
    args.accepted.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    accepted_tmp = args.accepted.with_suffix(args.accepted.suffix + ".tmp")
    report_tmp = args.report.with_suffix(args.report.suffix + ".tmp")
    seen_hashes: set[str] = set()
    counts: Counter[str] = Counter()
    # Only publish both files after the entire source has been scanned. This is
    # important on short-lived notebook/CI workers where a scan can be stopped.
    with args.labels.open() as source, accepted_tmp.open("w") as accepted, report_tmp.open("w") as report:
        for line_no, line in enumerate(source, 1):
            row = json.loads(line)
            text = re.sub(r"[^A-Z0-9]", "", str(row.get("plate_text", "")).upper())
            filename = str(row.get("image", ""))
            path = args.images / filename
            reasons: list[str] = []
            category = classify(text)
            if category is None:
                reasons.append("unsupported_plate_format")
            if float(row.get("confidence", 1.0)) < args.min_confidence:
                reasons.append("label_confidence_below_threshold")
            if filename.split("-", 1)[0].upper() != text:
                reasons.append("filename_label_mismatch")
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                reasons.append("missing_or_unreadable_image")
            else:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in seen_hashes:
                    reasons.append("duplicate_image")
                seen_hashes.add(digest)
            status = "accepted" if not reasons else "rejected"
            report.write(json.dumps({"line": line_no, "image": filename, "plate_text": text, "category": category, "status": status, "reasons": reasons}) + "\n")
            counts[status] += 1
            if status == "accepted":
                row["plate_text"] = text
                accepted.write(json.dumps(row) + "\n")
    os.replace(accepted_tmp, args.accepted)
    os.replace(report_tmp, args.report)
    print(f"Audit complete: accepted={counts['accepted']} rejected={counts['rejected']}")
    print(f"Accepted manifest: {args.accepted}\nReview report: {args.report}")


if __name__ == "__main__":
    main()
