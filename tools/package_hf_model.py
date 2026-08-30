#!/usr/bin/env python3
"""Assemble a Hugging Face model folder from the trained OCR weights.

    python tools/package_hf_model.py
    # -> hf_export/tz-alpr-ocr/{ocr_crnn.pt,config.json,README.md,...}

Does not upload. Use:  bash scripts/push_hf.sh
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_eval(name: str) -> dict:
    p = REPO_ROOT / "reports" / name
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _model_card(cfg: dict, test: dict, hard: dict) -> str:
    ocr = test.get("ocr") or {}
    hard_ocr = hard.get("ocr") or {}
    return f"""---
language:
  - en
license: apache-2.0
library_name: pytorch
tags:
  - license-plate-recognition
  - ocr
  - crnn
  - ctc
  - tanzania
  - alpr
pipeline_tag: image-to-text
---

# tz-alpr OCR (CRNN + CTC)

**Author:** [Japhari](https://huggingface.co/Japhari)

Tanzanian license-plate OCR used by [stg-gov/tz-plates](https://github.com/stg-gov/tz-plates).
Input is a grayscale plate crop `1×{cfg["input_height"]}×{cfg["input_width"]}`; output is a
CTC sequence over digits + A–Z.

## Metrics (held-out, {datetime.now(timezone.utc).date().isoformat()})

| Split | Exact match | Char acc | CER | n |
|---|---|---|---|---|
| test | {test.get("end_to_end_exact_match", "—")} | {ocr.get("char_acc", "—")} | {ocr.get("cer", "—")} | {test.get("n", "—")} |
| hard_test | {hard.get("end_to_end_exact_match", "—")} | {hard_ocr.get("char_acc", "—")} | {hard_ocr.get("cer", "—")} | {hard.get("n", "—")} |

## Files

- `ocr_crnn.pt` — PyTorch `state_dict` of the CRNN
- `confidence_calibration.json` — Platt scaling fitted on the test split
- `config.json` — architecture + charset
- `ocr.yaml` / `tanzania.yaml` — training and country-rule configs

## Load

```python
import torch
from huggingface_hub import hf_hub_download

# After cloning tz-plates and `pip install -e ".[train]"`:
from tz_alpr.ocr.model import build_crnn
from tz_alpr.config import load_yaml

cfg = load_yaml("configs/ocr.yaml")["model"]
weights = hf_hub_download("<repo_id>", "ocr_crnn.pt")
model = build_crnn(cfg, num_classes=37)  # 36 alnum + CTC blank
model.load_state_dict(torch.load(weights, map_location="cpu"))
model.eval()
```

Or serve the full pipeline:

```bash
export TZ_ALPR_OCR_WEIGHTS=ocr_crnn.pt
export TZ_ALPR_RUNTIME=torch
uvicorn tz_alpr.api.main:app --port 8081
```

## Intended use

Parking / road-reserve enforcement in Tanzania. Two-line motorcycle and bajaji
plates are de-stacked before OCR. Vehicle type is **not** predicted by this
checkpoint (COCO YOLO is a separate stage).

## Limitations

Daytime exact-match is below the 95% engineering target. Hard / night /
motorcycle crops remain weak. Review any reading with confidence &lt; 0.90.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "hf_export/tz-alpr-ocr")
    args = ap.parse_args()

    weights = REPO_ROOT / "models/ocr/v1/ocr_crnn.pt"
    if not weights.exists():
        raise SystemExit(f"Missing {weights} — train first (bash scripts/gpu_train.sh)")

    ocr_yaml = yaml.safe_load((REPO_ROOT / "configs/ocr.yaml").read_text())
    tz_yaml = yaml.safe_load((REPO_ROOT / "configs/tanzania.yaml").read_text())
    model_cfg = ocr_yaml["model"]
    charset = tz_yaml.get("charset", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, args.out / "ocr_crnn.pt")
    cal = REPO_ROOT / "models/ocr/v1/confidence_calibration.json"
    if cal.exists():
        shutil.copy2(cal, args.out / "confidence_calibration.json")
    shutil.copy2(REPO_ROOT / "configs/ocr.yaml", args.out / "ocr.yaml")
    shutil.copy2(REPO_ROOT / "configs/tanzania.yaml", args.out / "tanzania.yaml")

    config = {
        "architectures": ["CRNN"],
        "model_type": "tz-alpr-crnn-ctc",
        "backbone": model_cfg.get("backbone"),
        "input_height": model_cfg.get("input_height"),
        "input_width": model_cfg.get("input_width"),
        "input_channels": model_cfg.get("input_channels"),
        "rnn_hidden": model_cfg.get("rnn_hidden"),
        "rnn_layers": model_cfg.get("rnn_layers"),
        "dropout": model_cfg.get("dropout"),
        "charset": charset,
        "num_classes": len(charset) + 1,
        "blank_index": 0,
        "library": "pytorch",
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    test = _load_eval("eval_test.json")
    hard = _load_eval("eval_hard_test.json")
    (args.out / "README.md").write_text(_model_card(model_cfg, test, hard))

    print(f"Packed {args.out}")
    for p in sorted(args.out.iterdir()):
        print(f"  {p.name:32} {p.stat().st_size // 1024:7} KB")


if __name__ == "__main__":
    main()
