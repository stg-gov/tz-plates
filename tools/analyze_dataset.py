#!/usr/bin/env python3
"""Summarise the labelled set and OCR crops before / during training.

    python tools/analyze_dataset.py
    # -> reports/dataset_analysis.json
    # -> reports/dataset_analysis.md

Reads labels.jsonl, the audit report, ocr_annotations.csv and metadata.jsonl
when they exist. Safe to run after audit only (partial) or after prepare.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _count_jsonl(path: Path) -> tuple[int, Counter[str], Counter[str]]:
    n = 0
    prefixes: Counter[str] = Counter()
    types: Counter[str] = Counter()
    if not path.exists():
        return 0, prefixes, types
    for line in path.open():
        if not line.strip():
            continue
        n += 1
        rec = json.loads(line)
        text = str(rec.get("plate_text", "")).upper().replace(" ", "")
        if text.startswith("MC"):
            types["MOTORCYCLE"] += 1
        elif len(text) == 7 and text[0] == "T" and text[1:4].isdigit():
            types["PRIVATE"] += 1
        else:
            types["OTHER"] += 1
        prefixes[text[:3] if len(text) >= 3 else text] += 1
    return n, prefixes, types


def _audit_counts(path: Path) -> dict:
    accepted = rejected = 0
    reasons: Counter[str] = Counter()
    if not path.exists():
        return {"accepted": 0, "rejected": 0, "reasons": {}}
    for line in path.open():
        rec = json.loads(line)
        if rec.get("status") == "accepted":
            accepted += 1
        else:
            rejected += 1
            for r in rec.get("reasons") or []:
                reasons[r] += 1
    return {"accepted": accepted, "rejected": rejected, "reasons": dict(reasons)}


def _ocr_stats(path: Path) -> dict:
    splits: Counter[str] = Counter()
    types: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    det_ok = 0
    widths: list[int] = []
    unique_plates: set[str] = set()
    n = 0
    if not path.exists():
        return {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            n += 1
            splits[row.get("split") or "unknown"] += 1
            types[row.get("plate_type") or "UNKNOWN"] += 1
            sources[row.get("source") or "unknown"] += 1
            unique_plates.add(row.get("group_key") or row.get("plate_text") or "")
            try:
                conf = float(row.get("det_conf") or 0)
            except ValueError:
                conf = 0.0
            if conf >= 0.5:
                det_ok += 1
            try:
                widths.append(int(row.get("crop_w") or 0))
            except ValueError:
                pass
    return {
        "rows": n,
        "unique_plates": len(unique_plates),
        "splits": dict(splits),
        "plate_types": dict(types),
        "sources": dict(sources),
        "det_conf_ge_0.5": det_ok,
        "det_rate_ge_0.5": round(det_ok / n, 4) if n else 0.0,
        "crop_w_mean": round(sum(widths) / len(widths), 1) if widths else 0.0,
    }


def _lighting(path: Path) -> dict[str, int]:
    c: Counter[str] = Counter()
    if not path.exists():
        return {}
    for line in path.open():
        rec = json.loads(line)
        c[rec.get("lighting") or "unknown"] += 1
    return dict(c)


def _md(report: dict) -> str:
    labels = report.get("labels") or {}
    audit = report.get("audit") or {}
    ocr = report.get("ocr") or {}
    lines = [
        f"# Dataset analysis — {report.get('generated_at', '')}",
        "",
        f"Source images on disk: **{report.get('images_on_disk', 0):,}**",
        f"`labels.jsonl` rows: **{labels.get('n', 0):,}**",
        "",
        "## Plate categories (from labels.jsonl)",
        "",
        "| Type | Count |",
        "|---|---|",
    ]
    for k, v in sorted((labels.get("types") or {}).items()):
        lines.append(f"| {k} | {v:,} |")
    lines += [
        "",
        "## Audit",
        "",
        f"Accepted **{audit.get('accepted', 0):,}** · rejected **{audit.get('rejected', 0):,}**",
        "",
    ]
    reasons = audit.get("reasons") or {}
    if reasons:
        lines += ["| Reject reason | Count |", "|---|---|"]
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {k} | {v:,} |")
        lines.append("")
    if ocr:
        lines += [
            "## OCR crops (after prepare_dataset.py)",
            "",
            f"Rows **{ocr.get('rows', 0):,}** · unique plates **{ocr.get('unique_plates', 0):,}**",
            f"Detector conf ≥ 0.5: **{ocr.get('det_rate_ge_0.5', 0):.1%}**",
            "",
            "| Split | Count |",
            "|---|---|",
        ]
        for k, v in sorted((ocr.get("splits") or {}).items()):
            lines.append(f"| {k} | {v:,} |")
        lines += ["", "| Plate type | Count |", "|---|---|"]
        for k, v in sorted((ocr.get("plate_types") or {}).items()):
            lines.append(f"| {k} | {v:,} |")
        lighting = report.get("lighting") or {}
        if lighting:
            lines += ["", "| Lighting | Count |", "|---|---|"]
            for k, v in sorted(lighting.items()):
                lines.append(f"| {k} | {v:,} |")
        lines.append("")
    lines += [
        "Add more photos to `labeled_images/` and rows to `labels.jsonl`, then:",
        "",
        "```bash",
        "bash scripts/retrain.sh",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "labels.jsonl")
    ap.add_argument("--images", type=Path, default=REPO_ROOT / "labeled_images")
    ap.add_argument("--audit", type=Path, default=REPO_ROOT / "reports/data_audit.jsonl")
    ap.add_argument("--ocr-csv", type=Path, default=REPO_ROOT / "datasets/ocr/ocr_annotations.csv")
    ap.add_argument("--metadata", type=Path, default=REPO_ROOT / "datasets/annotations/metadata.jsonl")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports")
    args = ap.parse_args()

    n_labels, prefixes, types = _count_jsonl(args.labels)
    n_images = len(list(args.images.glob("*"))) if args.images.exists() else 0
    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "images_on_disk": n_images,
        "labels": {"n": n_labels, "types": dict(types), "top_prefixes": prefixes.most_common(12)},
        "audit": _audit_counts(args.audit),
        "ocr": _ocr_stats(args.ocr_csv),
        "lighting": _lighting(args.metadata),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "dataset_analysis.json"
    md_path = args.out_dir / "dataset_analysis.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_md(report))
    print(md_path.read_text())
    print(f"Wrote {json_path}  and  {md_path}")


if __name__ == "__main__":
    main()
