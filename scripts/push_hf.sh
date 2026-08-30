#!/usr/bin/env bash
# Upload the packed OCR model to the Hugging Face Hub.
#
#   export HF_REPO=your-user/tz-alpr-ocr    # required
#   export HF_PRIVATE=1                     # optional, default public
#   bash scripts/push_hf.sh
#
# Needs `hf auth login` (or HF_TOKEN) once. Packs first if hf_export/ is missing.
set -euo pipefail
cd "$(dirname "$0")/.."

HF_REPO="${HF_REPO:-}"
HF_PRIVATE="${HF_PRIVATE:-0}"
OUT="hf_export/tz-alpr-ocr"

if [ -z "$HF_REPO" ]; then
  echo "Set HF_REPO=namespace/tz-alpr-ocr  (e.g. Japhari/tz-alpr-ocr)"
  exit 1
fi

if [ ! -f models/ocr/v1/ocr_crnn.pt ]; then
  echo "No trained weights at models/ocr/v1/ocr_crnn.pt"
  exit 1
fi

python tools/package_hf_model.py --out "$OUT"

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI not on PATH. Install: pip install huggingface_hub"
  echo "Then: hf auth login"
  exit 1
fi

PRIV=()
[ "$HF_PRIVATE" = "1" ] && PRIV=(--private)

echo "Uploading $OUT  ->  https://huggingface.co/$HF_REPO"
hf upload "$HF_REPO" "$OUT" --repo-type model "${PRIV[@]}" \
  --commit-message "tz-alpr CRNN OCR weights"

echo "Done: https://huggingface.co/$HF_REPO"
