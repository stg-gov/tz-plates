#!/usr/bin/env bash
# End-to-end OCR training pipeline for a single-GPU box.
# See GPU_TRAINING.md for prerequisites and a step-by-step walkthrough.
#
#   bash scripts/gpu_train.sh
#
# Tunables (env vars):
#   PREP_LIMIT=0            0 = all ~26k images; set e.g. 12000 for a quick pass
#   PREP_WORKERS=8
#   SYNTH_COUNT=60000
#   SKIP_OVERFIT_CHECK=0    1 = don't run the pre-flight sanity check
#   SKIP_PRETRAIN=0         1 = train directly on real crops (--stage scratch)
#   FINETUNE_EPOCHS=""      override configs/ocr.yaml train.max_epochs
set -euo pipefail
cd "$(dirname "$0")/.."

PREP_LIMIT="${PREP_LIMIT:-0}"
PREP_WORKERS="${PREP_WORKERS:-8}"
SYNTH_COUNT="${SYNTH_COUNT:-60000}"
SKIP_OVERFIT_CHECK="${SKIP_OVERFIT_CHECK:-0}"
SKIP_PRETRAIN="${SKIP_PRETRAIN:-0}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-}"

export TZ_ALPR_PLATE_DETECTOR_WEIGHTS="${TZ_ALPR_PLATE_DETECTOR_WEIGHTS:-models/plate_detector/v1/plate_yolo.pt}"
mkdir -p logs

say() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

say "0. environment"
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
assert torch.cuda.is_available(), "No CUDA GPU visible — check drivers / --gpus all"
PY

say "1. fetch pretrained plate detector"
python tools/fetch_pretrained.py

say "2. audit original labels/images (reject invalid or corrupted samples)"
python tools/audit_labeled_images.py 2>&1 | tee logs/data_audit.log

say "3. prepare audited original data (auto-crop plates -> OCR csv + leakage-safe splits)"
if [ -f datasets/ocr/ocr_annotations.csv ] && [ "${FORCE_PREP:-0}" != "1" ]; then
  echo "datasets/ocr/ocr_annotations.csv exists — skipping (FORCE_PREP=1 to redo)"
else
  python tools/prepare_dataset.py --workers "$PREP_WORKERS" --limit "$PREP_LIMIT" \
    --skip-yolo-export 2>&1 | tee logs/prepare_dataset.log
fi

say "4. generate layout-correct synthetic OCR crops (OCR pretraining)"
if [ -f datasets/synthetic/generated_plates_v2/ocr_annotations.csv ] && [ "${FORCE_SYNTH:-0}" != "1" ]; then
  echo "synthetic set exists — skipping (FORCE_SYNTH=1 to redo)"
else
  python tools/generate_tanzania_plates.py --count "$SYNTH_COUNT" \
    --out datasets/synthetic/generated_plates_v2 2>&1 | tee logs/synth.log
fi

if [ "$SKIP_OVERFIT_CHECK" != "1" ]; then
  say "5. pre-flight: can the model overfit a small clean set? (~2 min)"
  if ! python tools/overfit_check.py --samples 512 --steps 1500 --device cuda \
        2>&1 | tee logs/overfit_check.log; then
    echo -e "\n\033[1;31mPre-flight FAILED — aborting before the full run.\033[0m"
    echo "Inspect crops in datasets/processed/plate_crops/ and logs/overfit_check.log."
    exit 1
  fi
fi

if [ "$SKIP_PRETRAIN" != "1" ]; then
  say "6. stage 1 — pretrain OCR on synthetic plates"
  python training/train_ocr.py --stage pretrain 2>&1 | tee logs/pretrain.log
  INIT_ARG=(--init models/ocr/v1/ocr_crnn_pretrained.pt)
  STAGE=finetune
else
  INIT_ARG=()
  STAGE=scratch
fi

say "7. stage 2 — train OCR on real crops (${STAGE})"
EPOCH_ARG=()
[ -n "$FINETUNE_EPOCHS" ] && EPOCH_ARG=(--max-epochs "$FINETUNE_EPOCHS")
python training/train_ocr.py --stage "$STAGE" "${INIT_ARG[@]}" "${EPOCH_ARG[@]}" \
  2>&1 | tee logs/train.log

say "8. evaluate + calibrate confidence"
python training/evaluate.py --split test --calibrate 2>&1 | tee logs/eval_test.log
python training/evaluate.py --split hard_test          2>&1 | tee logs/eval_hard_test.log

say "9. export ONNX for the runtime"
CKPT="$(ls -t checkpoints/ocr-${STAGE}-*.ckpt 2>/dev/null | head -1)"
[ -z "$CKPT" ] && CKPT="checkpoints/last.ckpt"
python tools/export_onnx.py ocr --ckpt "$CKPT" 2>&1 | tee logs/export.log

say "DONE"
echo "OCR weights : models/ocr/v1/ocr_crnn.pt  (+ .onnx)"
echo "Reports     : reports/eval_test.json  reports/eval_hard_test.json"
echo "Serve       : export TZ_ALPR_OCR_WEIGHTS=models/ocr/v1/ocr_crnn.pt"
echo "              export TZ_ALPR_OCR_ONNX=models/ocr/v1/ocr_crnn.onnx TZ_ALPR_RUNTIME=onnx"
echo "              uvicorn tz_alpr.api.main:app --port 8080"
