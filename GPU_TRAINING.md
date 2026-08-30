# GPU training runbook — Tanzanian ALPR OCR

Train the plate-OCR model (and, optionally, the plate detector) on a CUDA GPU
server. Everything is already wired; this is the operational checklist.

The image pipeline (detector → rectify → **OCR** → Tanzania rules → confidence)
runs today with an **open-source YOLO plate detector** and produces well-framed
crops. The only piece that needs GPU training is the CRNN+CTC OCR model — CPU/MPS
training stalled in dev (CTC's slow "turn-on" needs more steps than was practical
there). The fixes below are already in the repo.

---

## 0. What to copy to the GPU box

```
tz-alpr/                     # the whole repo
labeled_images/              # 26,137 vehicle photos  (~800 MB)  — NOT in git
labels.jsonl                 # {"image","plate_text","confidence"}  — NOT in git
```

`labeled_images/` + `labels.jsonl` are git-ignored. `rsync` or `scp` them
alongside the repo (or point `tools/prepare_dataset.py --images/--labels` at
wherever they live). The pretrained plate detector is fetched by the script; it
is not committed.

---

## 1. Prerequisites

- NVIDIA GPU, driver ≥ 525 (for CUDA 12.1), ≥ 12 GB VRAM comfortable (the model
  is ~9 M params; batch 256 fits in far less, scale `data.batch_size` down for
  smaller cards).
- Either Docker + `nvidia-container-toolkit`, **or** a Python 3.10–3.11 env.

### Option A — Docker (recommended)

```bash
docker build -f docker/Dockerfile.train -t tz-alpr-train .
docker run --gpus all -it --rm \
  -v "$PWD":/workspace -w /workspace \
  -v /path/to/labeled_images:/workspace/labeled_images:ro \
  -v /path/to/labels.jsonl:/workspace/labels.jsonl:ro \
  tz-alpr-train bash scripts/gpu_train.sh
```

### Option B — bare env

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.3.*" "torchvision==0.18.*"
pip install -e ".[train,detect,onnx,dev]"
```

---

## 2. One command

```bash
bash scripts/gpu_train.sh
```

Runs, in order (each step is skipped if its output already exists):

| # | Step | Output | ~time on 1×A100 / 1×3090 |
|---|---|---|---|
| 1 | fetch pretrained plate detector | `models/plate_detector/v1/plate_yolo.pt` | seconds |
| 2 | `audit_labeled_images.py` — reject invalid/corrupt/duplicate originals and write review report | `datasets/annotations/accepted_labels.jsonl`, `reports/data_audit.jsonl` | 1–3 min |
| 3 | `prepare_dataset.py` — auto-crop audited originals, build `ocr_annotations.csv` + leakage-safe splits | `datasets/ocr/`, `datasets/processed/plate_crops/` | 3–6 min / 8–15 min |
| 4 | `generate_tanzania_plates.py` (`SYNTH_COUNT=60000`) | `datasets/synthetic/generated_plates_v2/` | 3–5 min |
| 4 | **pre-flight** `overfit_check.py` — model must overfit 512 clean crops | pass/fail (aborts on fail) | ~2 min |
| 5 | `train_ocr.py --stage pretrain` (synthetic only) | `models/ocr/v1/ocr_crnn_pretrained.pt` | 15–30 min / 40–70 min |
| 6 | `train_ocr.py --stage finetune` (real + synthetic, warm-started) | `models/ocr/v1/ocr_crnn.pt`, `checkpoints/ocr-finetune-*.ckpt` | 30–60 min / 1.5–3 h |
| 7 | `evaluate.py` on `test` + `hard_test`, fit confidence calibration | `reports/eval_*.json`, `models/ocr/v1/confidence_calibration.json` | 1–3 min |
| 8 | `export_onnx.py ocr` | `models/ocr/v1/ocr_crnn.onnx` | seconds |

Tunables (env vars): `PREP_LIMIT` (0 = all), `SYNTH_COUNT`, `FINETUNE_EPOCHS`,
`SKIP_PRETRAIN=1` (train from scratch on real only), `SKIP_OVERFIT_CHECK=1`,
`FORCE_PREP=1`, `FORCE_SYNTH=1`.

Watch progress from another shell: `tools/progress.sh -w 15`
or TensorBoard: `tensorboard --logdir lightning_logs`.

---

## 3. Manual, step by step

```bash
# detector + data
python tools/fetch_pretrained.py
export TZ_ALPR_PLATE_DETECTOR_WEIGHTS=models/plate_detector/v1/plate_yolo.pt
python tools/audit_labeled_images.py
python tools/prepare_dataset.py --workers 8 --skip-yolo-export
python tools/generate_tanzania_plates.py --count 60000 --out datasets/synthetic/generated_plates_v2

# ALWAYS run this first — if it FAILs, do not start the full run
python tools/overfit_check.py --device cuda        # expect train seq-acc -> ~1.0

# two-stage training
python training/train_ocr.py --stage pretrain
python training/train_ocr.py --stage finetune --init models/ocr/v1/ocr_crnn_pretrained.pt

# eval + calibrate + export
python training/evaluate.py --split test --calibrate
python training/evaluate.py --split hard_test
python tools/export_onnx.py ocr --ckpt "$(ls -t checkpoints/ocr-finetune-*.ckpt | head -1)"
```

---

## 4. Serve the trained model

```bash
export TZ_ALPR_OCR_WEIGHTS=models/ocr/v1/ocr_crnn.pt
export TZ_ALPR_OCR_ONNX=models/ocr/v1/ocr_crnn.onnx
export TZ_ALPR_PLATE_DETECTOR_WEIGHTS=models/plate_detector/v1/plate_yolo.pt
export TZ_ALPR_RUNTIME=onnx            # or torch
uvicorn tz_alpr.api.main:app --port 8080

curl -s -F "upload=@some_car.jpg" http://localhost:8080/v1/plate-reader | jq
python tools/benchmark.py --limit 300
```

Copy `models/ocr/v1/*` and `models/plate_detector/v1/plate_yolo.pt` back to the
inference host (or bake them into the `runtime` Docker image / mount them).

---

## 5. What was already fixed for this (context)

CPU/MPS training in dev stalled at ~25 % char-accuracy. Root causes found and
fixed in-repo — the GPU run inherits all of these:

| Fix | Where |
|---|---|
| Train aug emitted 3-channel tensors; model + inference expect 1 | `ocr/transforms.py` `_to_gray_1ch` |
| Rectifier's speculative perspective-warp mangled good crops | `rectification/rectify.py` `contour_search=False` default |
| Low-confidence YOLO boxes polluting train/val | `data.min_det_conf: 0.5` in `configs/ocr.yaml` |
| **Two-line plates** (`T336` / `CAG`) fed vertically overlaid | de-stack threshold 1.9→2.6, overlapping-band split in `rectify._destack` |
| Default Conv2d init → forward-signal decay through the CNN (dead backbone) | `ocr/model.py` `CRNN._init_weights` (Kaiming fan-out) |
| Early-stop on `val/seq_acc` killed runs before CTC turned on | `min_epochs: 25`, monitor `val/char_acc`, patience 15 |

`configs/ocr.yaml` now has `input_height: 32`, `input_width: 256` and a canonical
VGG-style CRNN backbone that pools height to 1 inside the conv stack.

---

## 6. Troubleshooting

- **`overfit_check.py` FAILs** (train seq-acc stays < 0.6): the model can't fit a
  tiny clean set — a full run is pointless. Check `datasets/processed/plate_crops/*.jpg`
  by eye (are two-line plates de-stacked side-by-side and readable?), then the
  charset / target encoding. Try `--stage scratch` with `augmentation.enabled:
  false` in the config to isolate data vs. regularization.
- **Loss plateaus ~2.3, predictions like `TDAM` for everything**: the classic CTC
  blank-collapse. It usually breaks after 400–1500 steps — make sure `min_epochs`
  is being honored (don't pass a tiny `--max-epochs`). If it never breaks, lower
  `lr` to 5e-4 and set `scheduler: cosine`.
- **OOM**: drop `data.batch_size` (256 → 128 → 64) and raise
  `accumulate_grad_batches` to keep the effective batch.
- **`prepare_dataset.py` slow**: it runs YOLO per image. On GPU it's fast; if it's
  CPU-bound, lower `--workers` (thread contention) or set `--limit`.
- **Detector weights 401/404 on fetch**: HF repo moved. Search
  `huggingface_hub` for "license plate detection yolo", drop any `*.pt` into
  `models/plate_detector/v1/plate_yolo.pt`, and it just works
  (`YoloPlateDetector` only needs a YOLO `.pt`).

---

## 7. Targets (spec §33 — engineering targets, measure don't assume)

- End-to-end exact-match ≥ 95 % under normal daytime conditions
- Separate numbers for night / blur / motorcycle / angled (`reports/eval_hard_test.json`)
- GPU latency < 100 ms/image; CPU deployment usable without a GPU
