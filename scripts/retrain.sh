#!/usr/bin/env bash
# Incremental OCR training when more labelled photos are added.
#
# Drop new files into labeled_images/ and matching rows into labels.jsonl, then:
#   bash scripts/retrain.sh
#
# Re-audits, re-crops, writes dataset analysis, then fine-tunes from the last
# ocr_crnn.pt (or runs a full train if no weights exist yet).
#
# Tunables:
#   FORCE_RETRAIN=1     retrain even if the label/image fingerprint is unchanged
#   PREP_WORKERS=8
#   PREP_LIMIT=0
#   FINETUNE_EPOCHS=40  fewer epochs than a cold start (warm weights)
#   SKIP_OVERFIT_CHECK=1
set -euo pipefail
cd "$(dirname "$0")/.."

PREP_WORKERS="${PREP_WORKERS:-8}"
PREP_LIMIT="${PREP_LIMIT:-0}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-40}"
SKIP_OVERFIT_CHECK="${SKIP_OVERFIT_CHECK:-1}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
STAMP="datasets/.last_train_manifest"

export TZ_ALPR_PLATE_DETECTOR_WEIGHTS="${TZ_ALPR_PLATE_DETECTOR_WEIGHTS:-models/plate_detector/v1/plate_yolo.pt}"
mkdir -p logs datasets

say() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

fingerprint() {
  python - <<'PY'
from pathlib import Path
import hashlib
root = Path(".")
h = hashlib.sha256()
labels = root / "labels.jsonl"
images = root / "labeled_images"
h.update(str(labels.stat().st_size if labels.exists() else 0).encode())
h.update(b"|")
if labels.exists():
    h.update(labels.read_bytes())
n = len(list(images.glob("*"))) if images.exists() else 0
h.update(f"|images={n}".encode())
print(f"{n} {h.hexdigest()[:16]}")
PY
}

say "0. environment"
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
assert torch.cuda.is_available(), "No CUDA GPU visible"
PY

NEW_FP="$(fingerprint)"
OLD_FP=""
[ -f "$STAMP" ] && OLD_FP="$(cat "$STAMP")"
echo "data fingerprint: $NEW_FP"
[ -n "$OLD_FP" ] && echo "last trained on : $OLD_FP"

if [ "$FORCE_RETRAIN" != "1" ] && [ "$NEW_FP" = "$OLD_FP" ] && [ -f models/ocr/v1/ocr_crnn.pt ]; then
  echo "No new labelled data since the last train. Add images + labels.jsonl rows, or FORCE_RETRAIN=1."
  exit 0
fi

if [ ! -d labeled_images ]; then
  echo "labeled_images/ is missing"
  exit 1
fi
if [ ! -f labels.jsonl ]; then
  echo "labels.jsonl is missing"
  exit 1
fi

say "1. fetch plate detector if needed"
python tools/fetch_pretrained.py

say "2. audit + recrop (new photos included)"
python tools/audit_labeled_images.py 2>&1 | tee logs/data_audit.log
python tools/prepare_dataset.py --workers "$PREP_WORKERS" --limit "$PREP_LIMIT" \
  --skip-yolo-export 2>&1 | tee logs/prepare_dataset.log

say "3. dataset analysis"
python tools/analyze_dataset.py 2>&1 | tee logs/dataset_analysis.log

if [ "$SKIP_OVERFIT_CHECK" != "1" ]; then
  say "4. overfit check"
  python tools/overfit_check.py --samples 512 --steps 1500 --device cuda \
    2>&1 | tee logs/overfit_check.log
fi

if [ -f models/ocr/v1/ocr_crnn.pt ]; then
  say "5. fine-tune from existing OCR weights (${FINETUNE_EPOCHS} epochs)"
  python training/train_ocr.py --stage finetune \
    --init models/ocr/v1/ocr_crnn.pt \
    --max-epochs "$FINETUNE_EPOCHS" \
    2>&1 | tee logs/retrain.log
else
  say "5. no ocr_crnn.pt yet — full two-stage train"
  bash scripts/gpu_train.sh
  echo "$NEW_FP" > "$STAMP"
  exit 0
fi

say "6. evaluate + calibrate"
python training/evaluate.py --split test --calibrate 2>&1 | tee logs/eval_test.log
python training/evaluate.py --split hard_test 2>&1 | tee logs/eval_hard_test.log

say "7. pack Hugging Face folder (does not upload)"
python tools/package_hf_model.py

echo "$NEW_FP" > "$STAMP"

say "DONE — incremental retrain"
echo "OCR weights : models/ocr/v1/ocr_crnn.pt"
echo "Analysis    : reports/dataset_analysis.md"
echo "HF folder   : hf_export/tz-alpr-ocr/"
echo "Upload      : bash scripts/push_hf.sh"
