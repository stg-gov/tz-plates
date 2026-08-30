# tz-alpr — Tanzanian Automatic License Plate Recognition

Open-source ALPR platform built entirely from open technologies and models. Initial
target country: **Tanzania**. Architecture is country-pluggable (Kenya, Uganda,
Rwanda, Burundi, Zambia, Malawi, Mozambique … add a rule module, nothing else
changes). First production use case: parking / road-reserve parking enforcement.

No proprietary code, weights or APIs are used or reproduced.

---

## Status — Phase 3 (video: tracking + temporal OCR) · `tz-alpr-1.2.0`

Implemented and tested in this repository:

| Area | State |
|---|---|
| `POST /v1/plate-reader` (image → plate JSON) | ✅ complete |
| `POST /v1/video` (video → deduplicated `VehicleEvent[]`) | ✅ complete |
| `GET /health`, `GET /version` | ✅ complete |
| **ByteTrack** multi-object tracker (dependency-free, low+high-score association) | ✅ complete, unit-tested |
| **Temporal OCR aggregation** — OCR-probability-weighted voting over a track | ✅ complete, unit-tested |
| **Event deduplication** — one dwell → one event, cross-track window | ✅ complete, unit-tested |
| Vehicle detector — COCO-pretrained YOLOv8n + fine-tune script | ✅ (auto-downloaded; degrades to full-frame / plate-tracking if absent) |
| Per-vehicle plate detection + full-frame safety net; box in image + vehicle coords | ✅ complete |
| Plate detector — trained **YOLO** + **classical fallback** | ✅ (fallback runs with no weights) |
| Plate rectification (homography / contour / resize) | ✅ complete |
| OCR — **CRNN + CTC** model, Lightning training, ONNX export | ✅ code complete, needs training |
| Tanzania rule engine + Tanzania-aware decoding | ✅ complete, unit-tested |
| Calibrated confidence + review routing | ✅ complete, unit-tested |
| Synthetic plate generator · dataset prep / leakage-safe splitting · eval + calibration | ✅ complete |
| Docker (`alpr-api`) + Compose | ✅ complete |
| RTSP `stream-worker` / webhooks / review-UI | ⏳ Phase 4–5, endpoints stubbed `501` with frozen contracts |

On a clean checkout the API runs immediately: **classical plate detector**, **null
OCR engine**, and — unless `.[detect]` extras are installed — **no vehicle
detector** (full-frame plate detection). Responses are always well-formed and
every degraded stage is listed in `warnings`. Install extras + train the models
(below) to make it production-grade.

---

## 1. Architecture

```
Camera / Image / POS photo
      │
      ▼
Image enhancement            src/tz_alpr/preprocessing/         CLAHE + auto-gamma
      │
      ▼
Vehicle detection            src/tz_alpr/detection/             YOLOv8 (COCO or fine-tuned), COCO→taxonomy remap
      │                                                        (Phase 2 — active; falls back to full-frame if absent)
      ▼
License-plate detection      src/tz_alpr/plate_detection/       per vehicle crop + full-frame safety net
                                                               YOLO (trained) | classical CV (fallback)
      │                                                        box reported in image AND vehicle coords
      │
      ▼
Plate rectification          src/tz_alpr/rectification/         quad → homography; de-stack 2-line
      │
      ▼
(optional) super-resolution  src/tz_alpr/pipeline/super_resolution.py   Real-ESRGAN | bicubic+unsharp
      │
      ▼
OCR  (CRNN + CTC)            src/tz_alpr/ocr/                   per-char posteriors + seq confidence
      │
      ▼
Tanzania-aware post-proc     src/tz_alpr/postprocessing/tz_aware.py     OCR probs × plate schema × confusions
      │
      ▼
Confidence fusion            src/tz_alpr/postprocessing/confidence.py   weighted geo-mean + penalties + Platt
      │
      ▼
Structured result  →  REST API (FastAPI)  →  (Phase 4) webhook

── Video / stream path (Phase 3) ─────────────────────────────────────────────
frame sampling → vehicle detection → ByteTrack        src/tz_alpr/tracking/bytetrack.py
      → per-track plate recognition (recognize() above)
      → temporal OCR aggregation (weighted vote)       src/tz_alpr/tracking/aggregator.py
      → event deduplication (1 dwell = 1 event)        src/tz_alpr/tracking/events.py
      → VehicleEvent[]   (orchestrator: src/tz_alpr/pipeline/video.py)
```

Every stage is an independent, replaceable component with a small interface
(`VehicleDetector.detect`, `PlateDetector.detect`, `OcrEngine.predict`,
`CountryRules.normalize`). `AlprPipeline.recognize()` is the shared
rectify→OCR→decode→fuse step used by both the image and the video pipelines.
Orchestrators: `src/tz_alpr/pipeline/alpr.py` (image), `.../video.py` (video).

## 2. Technology choices

| Concern | Choice | Why |
|---|---|---|
| Vehicle detection | Ultralytics **YOLOv8n**, COCO-pretrained (car/motorcycle/bus/truck) + optional fine-tune for minibus/tuk-tuk | open weights, runs out of the box, one interface (`VehicleDetector.detect`); Phase 1 needed no training data for it |
| Plate detection | Ultralytics **YOLOv8** single-class, + classical HSV/contour fallback; runs **per vehicle crop** with a full-frame safety net | strong open detector, OBB support for tilted plates; per-vehicle cropping raises small-plate recall and ties each plate to a vehicle; fallback keeps the system runnable and gives a CPU baseline |
| Rectification | 4-point **homography** (detector quad → contour search → resize) | no per-corner labels needed; STN is a drop-in upgrade behind the same interface |
| **OCR** | **CRNN (CNN-lite + BiLSTM) + CTC** | closed 36-char alphabet, short 7–8 char sequences, alignment-free (handles the flag/dot spacing on TZ plates), ~3.7 M params, few-ms CPU, clean ONNX. PARSeq/TrOCR need far more data/compute for no real gain here; LPRNet is lighter but less robust to perspective residue; SVTR is provided as an alternate backbone hook |
| Training | **PyTorch + Lightning**, YAML configs, TensorBoard (+ optional W&B) | checkpointing / early-stop / mixed precision out of the box |
| Augmentation | **Albumentations** | one op per "difficult condition" in the spec, probabilities in `configs/ocr.yaml` |
| Serving | **FastAPI + Uvicorn** | typed request/response, OpenAPI schema clients can build against now |
| Runtime | **PyTorch** or **ONNX Runtime** (CPU/CUDA), TensorRT-ready | on-prem, Jetson / CPU-server / GPU-server targets |
| Tracking | **ByteTrack** (own dependency-free implementation) | two-stage high/low-score association recovers occluded tracks; no scipy/lap; EMA constant-velocity motion is enough at parking-camera FPS |
| Temporal fusion | OCR-probability-weighted per-character voting + bounded agreement boost | uses the CTC posteriors already produced per frame; correlated frames get diminishing returns, not a naive noisy-OR |

## 3. Repository structure

```
tz-alpr/
  configs/            detector.yaml plate_detector.yaml ocr.yaml training.yaml tanzania.yaml inference.yaml
  src/tz_alpr/
    preprocessing/    image enhancement
    detection/        vehicle detector — YOLOv8 wrapper + COCO/custom factory (active)
    plate_detection/  YOLO + classical fallback + factory
    rectification/    perspective correction
    ocr/              charset · CRNN model · CTC decode · datamodule · lightning module · engines · metrics
    postprocessing/   tanzania-aware decoder · calibrated confidence
    country_rules/    base registry + tanzania.py
    pipeline/         image orchestrator (alpr.py) · video orchestrator (video.py) · super-resolution
    tracking/         bytetrack.py · aggregator.py (temporal vote) · events.py (dedup)  (active)
    api/              FastAPI app + routes
    streaming/ webhooks/   Phase 4 (documented stubs)
    utils/            image IO · geometry · timing
  configs/            + tracking.yaml (sampling / tracker / event-dedup knobs)
  training/           train_vehicle_detector.py  train_plate_detector.py  train_ocr.py  evaluate.py
  tools/              prepare_dataset.py  generate_tanzania_plates.py  extract_frames.py
                      export_onnx.py  benchmark.py  predict_image.py  process_video.py
  tests/              rules · normalization · OCR post-proc · confidence · CTC · tracking ·
                      temporal aggregation · event dedup · API
  datasets/  models/  checkpoints/  notebooks/
  Dockerfile  docker-compose.yml  Makefile  pyproject.toml  requirements.txt
```

## 4. Dataset specification

```
datasets/
  raw/{images,videos}/
  annotations/{vehicles,plates,ocr}/     YOLO txt + plates.yaml + metadata.jsonl
  processed/{vehicle_crops,plate_crops,rectified_plates}/
  synthetic/generated_plates_v2/         layout-correct synthetic OCR pretraining set
  ocr/ocr_annotations.csv                image_path,plate_text,country,plate_type,split,group_key,source
  splits/{train,val,test,hard_test}.txt
  retraining_queue/                      active-learning drop zone (Phase ≥4)
```

Provided data (`labeled_images/` + `labels.jsonl`, 26 137 phone photos of vehicle
rears, each labelled with a human-verified plate string; ~24.5 k `T### ABC`
private, ~1.6 k `MC### ABC` motorcycle). No boxes — `tools/prepare_dataset.py`
produces plate crops and *weak* YOLO boxes from the detector; hand-correct a few
hundred boxes before training the production detector. OCR crops are usable as-is
because the text is verified.

**Unified metadata** (`datasets/annotations/metadata.jsonl`): `image_id, camera_id,
timestamp, vehicle_bbox, plate_bbox, plate_text, plate_type, vehicle_type,
weather, lighting, angle, occlusion, blur_score` (unknowns are `null`; `lighting`
and `blur_score` are computed).

## 5. Annotation strategy

* **Detection** — YOLO format, single class `plate` (`0 cx cy w h`, normalized).
* **OCR** — `ocr_annotations.csv` (schema above), one row per plate crop.
* **Splitting is grouped, never random per-frame** (spec §13): grouped by plate
  identity here; `tools/extract_frames.py` stamps a `session_id` per video so
  video frames stay within one split. `hard_test` collects night / tiny / low-
  confidence / motorcycle / unrecognised-pattern crops.

## 6. Model architecture (OCR)

`src/tz_alpr/ocr/model.py`. Input `1×48×192` grayscale (rectified plate).
6 conv blocks downsample height→1 and width→time (stride 4), → 2-layer BiLSTM(256)
→ linear → `log_softmax` over 37 classes (36 alnum + CTC blank). Greedy CTC
decoder (`ctc_decode.py`) returns the string **and** per-output-character
posteriors with top-k alternatives — required by the Tanzania-aware step and the
review UI.

## 7. Tanzania plate-rule engine

`src/tz_alpr/country_rules/tanzania.py` + `configs/tanzania.yaml` (all policy is
data): category regex & priority, per-position character classes
(`PRIVATE = A N N N A A A`), OCR-confusion table (`0↔O`, `1↔I`, `2↔Z`, `5↔S`,
`6↔G`, `8↔B`, …), confidence bonuses, review thresholds.

Tanzania-aware decoding (`postprocessing/tz_aware.py`): if a position's emitted
character violates the plate schema, the decoder consults the confusion table and
the OCR posteriors and swaps **only** when a schema-valid alternative is within
`swap_margin` of the argmax and the argmax is below a hard-lock probability.
**Every swap is recorded** and returned:

```json
{ "raw_ocr": "T33IEBG", "normalized_plate": "T331EBG",
  "corrections": ["pos 3: I->1 (p 0.55->0.42, slot=N)"] }
```

## 8. Confidence (spec §16)

Not a plain product. `postprocessing/confidence.py`:
temperature-scale OCR sequence confidence → **weighted geometric mean** of
`{plate_detection, ocr, plate_validation}` (each floored) → subtract an explicit
penalty per rule swap and for length mismatch → optional **Platt scaling** fitted
from operator-verified outcomes (`training/evaluate.py --calibrate`) → route:
`≥0.90 auto_accept`, `0.70–0.89 review`, `<0.70 manual`.

---

## Quick start (inference)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                       # runtime deps only

uvicorn tz_alpr.api.main:app --port 8080
# or: docker compose up --build alpr-api

curl -s http://localhost:8080/health
python tools/predict_image.py labeled_images/T336CAG-0007245e981f11ee9347df110422d5da.jpg
```

## Train the models on your Tanzanian dataset

> **On a GPU box, use [`GPU_TRAINING.md`](GPU_TRAINING.md)** — it's a one-command
> runbook (`bash scripts/gpu_train.sh`: fetch detector → prep data → synthetic →
> pre-flight sanity check → pretrain → fine-tune → evaluate → export ONNX) plus a
> CUDA `docker/Dockerfile.train`. The steps below are the manual equivalent.

Install training extras once:

```bash
pip install -e ".[train,detect,onnx,dev]"
python tools/fetch_pretrained.py     # open-source YOLO plate detector for auto-cropping
```

Always run the pre-flight check before a full run:

```bash
python tools/overfit_check.py        # model must overfit a small clean set (train seq-acc -> ~1.0)
```

### A. Prepare the dataset

```bash
python tools/prepare_dataset.py --workers 8
#   -> datasets/processed/plate_crops/*.jpg
#   -> datasets/ocr/ocr_annotations.csv         (train/val/test/hard_test, grouped by plate identity)
#   -> datasets/annotations/plates/{images,labels}/  + plates.yaml   (WEAK boxes — verify a subset)
#   -> datasets/annotations/metadata.jsonl
```

### B. Generate synthetic plates (OCR pretraining)

```bash
python tools/generate_tanzania_plates.py --count 60000
#   -> datasets/synthetic/generated_plates_v2/{images, ocr_annotations.csv}
```

### C. Train OCR (CRNN + CTC), two-stage

```bash
# stage 1 — pretrain on synthetic only
python training/train_ocr.py --stage pretrain
#   -> models/ocr/v1/ocr_crnn_pretrained.pt

# stage 2 — fine-tune on real crops (synthetic mixed in), warm-started
python training/train_ocr.py --stage finetune --init models/ocr/v1/ocr_crnn_pretrained.pt
#   -> checkpoints/ocr-finetune-*.ckpt   and   models/ocr/v1/ocr_crnn.pt

# sanity check without a GPU
python training/train_ocr.py --stage scratch --fast-dev-run
```

TensorBoard: `tensorboard --logdir lightning_logs`. Enable W&B in `configs/training.yaml`.

### D. Train the plate detector (after verifying boxes)

```bash
python training/train_plate_detector.py --data datasets/annotations/plates/plates.yaml
#   -> models/plate_detector/v1/plate_yolo.pt
```

### E. Evaluate + calibrate confidence

```bash
python training/evaluate.py --split test --calibrate
python training/evaluate.py --split hard_test
#   -> reports/eval_*.json  (OCR char-acc / CER / seq-acc, end-to-end exact-match,
#      slices by plate_type & lighting & review band, reliability table)
#   -> models/ocr/v1/confidence_calibration.json
```

### F. Export for the runtime

```bash
python tools/export_onnx.py ocr   --ckpt checkpoints/ocr-finetune-XX-0.98.ckpt
python tools/export_onnx.py plate --weights runs/plate_detector/v1/weights/best.pt
```

### G. Serve with the trained models

```bash
export TZ_ALPR_OCR_WEIGHTS=models/ocr/v1/ocr_crnn.pt
export TZ_ALPR_OCR_ONNX=models/ocr/v1/ocr_crnn.onnx
export TZ_ALPR_PLATE_DETECTOR_WEIGHTS=models/plate_detector/v1/plate_yolo.pt
export TZ_ALPR_RUNTIME=onnx            # or torch
uvicorn tz_alpr.api.main:app --port 8080
```

### H. Test on an image

```bash
python tools/predict_image.py path/to/car.jpg
curl -s -F "upload=@path/to/car.jpg" http://localhost:8080/v1/plate-reader | jq
```

### Benchmark (spec §29)

```bash
python tools/benchmark.py --images labeled_images --limit 300
# FPS, mean/P50/P95/P99 latency, per-stage ms, CPU/RAM (+ GPU/VRAM if pynvml present)
```

---

## Example API calls

`POST /v1/plate-reader` (multipart, field name `upload`):

```bash
curl -s -F "upload=@car.jpg" http://localhost:8080/v1/plate-reader
```

```json
{
  "processing_time_ms": 83,
  "model_version": "tz-alpr-1.1.0",
  "results": [
    {
      "plate": "T331EBG",
      "raw_text": "T 331 EBG",
      "raw_ocr": "T33IEBG",
      "normalized_text": "T331EBG",
      "confidence": 0.97,
      "confidence_breakdown": {
        "vehicle_confidence": 0.95,
        "plate_detection_confidence": 0.95,
        "ocr_confidence": 0.96,
        "plate_validation_confidence": 0.93,
        "final_confidence": 0.97
      },
      "plate_bbox": { "x": 420, "y": 320, "width": 180, "height": 55 },
      "plate_bbox_in_vehicle": { "x": 92, "y": 140, "width": 180, "height": 55 },
      "plate_quad": [[420,320],[600,318],[602,373],[421,375]],
      "vehicle": {
        "type": "car",
        "confidence": 0.95,
        "bbox": { "x": 328, "y": 180, "width": 520, "height": 430 }
      },
      "vehicle_track_id": null,
      "country": "TZ",
      "plate_type": "PRIVATE",
      "plate_colour": "yellow",
      "corrections": ["pos 3: I->1 (p 0.55->0.42, slot=N)"],
      "ocr_candidates": [{ "text": "T331EBG", "confidence": 0.96 }],
      "review_status": "auto_accept"
    }
  ],
  "warnings": []
}
```

When the vehicle detector is unavailable, `vehicle.type` is `"unknown"`,
`plate_bbox_in_vehicle` is `null`, and a `warnings` entry says so — the rest of
the contract is identical.

```bash
curl -s http://localhost:8080/health
curl -s http://localhost:8080/version
```

### `POST /v1/video` (Phase 3)

Multipart: `upload` (video file), `camera_id`, `sample_fps` (default 5),
`max_seconds` (0 = whole clip). Synchronous — a job queue is Phase 4.

```bash
curl -s -F "upload=@clip.mp4" -F "camera_id=DODOMA_PARKING_01" -F "sample_fps=5" \
     http://localhost:8080/v1/video | jq
# or
python tools/process_video.py clip.mp4 --camera-id DODOMA_PARKING_01 --sample-fps 5
```

```json
{
  "processing_time_ms": 4120,
  "model_version": "tz-alpr-1.2.0",
  "camera_id": "DODOMA_PARKING_01",
  "video": { "duration_s": 12.0, "fps_source": 25.0, "frames_total": 300,
             "frames_sampled": 60, "sample_fps": 5 },
  "tracks_seen": 3,
  "events": [
    {
      "event_id": "9f2c1e7b8a...",
      "event": "vehicle.detected",
      "camera_id": "DODOMA_PARKING_01",
      "track_id": 2,
      "plate": "T331EBG",
      "raw_ocr": "T33IEBG",
      "confidence": 0.97,
      "plate_type": "PRIVATE",
      "vehicle_type": "car",
      "review_status": "auto_accept",
      "corrections": ["pos 3: I->1 (p 0.55->0.42, slot=N)"],
      "first_seen": "2024-06-01T08:14:03.200000+00:00",
      "last_seen": "2024-06-01T08:14:06.800000+00:00",
      "frame_count": 18,
      "model_version": "tz-alpr-1.2.0",
      "per_frame": [[71, "T331E8G", 0.72], [76, "T331EBG", 0.91], [81, "T331EBG", 0.96]]
    }
  ],
  "warnings": []
}
```

One physical vehicle passing the camera produces **one** `VehicleEvent`, not one
per frame (spec §18). Field names match the Phase 4 webhook payload (spec §20).

`POST /v1/streams`, `GET|DELETE /v1/streams/{id}` return `501` until Phase 4;
their schemas are already in `/docs`.

---

## Testing

```bash
pytest -q          # 79 tests: rules, normalization, OCR post-processing, confidence,
                   # CTC decoding, vehicle pipeline, ByteTracker, temporal aggregation,
                   # event dedup, image + video API contracts, later-phase 501s
ruff check src tools training tests
```

## Phase 2 — vehicle detection

Enabled by default (`configs/inference.yaml → pipeline.use_vehicle_detector: true`).

```bash
pip install -e ".[detect]"     # ultralytics; yolov8n.pt auto-downloads on first run
python tools/predict_image.py labeled_images/T336CAG-*.jpg   # now populates `vehicle` + `plate_bbox_in_vehicle`
```

Flow: `vehicle detector → for each vehicle: plate detector on the padded crop →
translate plate box to image coords + record vehicle-relative box → optional
full-frame safety-net pass for plates outside any vehicle box (bajaji, cut-off
vehicles) → IoU de-dup → associate leftovers with the smallest containing
vehicle`. Multiple vehicles and multiple plates per image are handled. If
ultralytics is absent or offline, the pipeline logs a warning and runs the
Phase 1 full-frame path.

Optional — fine-tune to add **minibus** (daladala) and **tuktuk** (bajaji), which
COCO lacks (until then a daladala reads as `bus`, a bajaji as `motorcycle`; plate
OCR is unaffected — spec §32):

```bash
# annotate vehicle boxes -> datasets/annotations/vehicles/vehicles.yaml
#   classes: 0 car  1 motorcycle  2 bus  3 truck  4 minibus  5 tuktuk
python training/train_vehicle_detector.py --data datasets/annotations/vehicles/vehicles.yaml
export TZ_ALPR_VEHICLE_DETECTOR_WEIGHTS=models/detector/v1/vehicle_yolo.pt
```

## Phase 3 — video: tracking + temporal OCR

`POST /v1/video` / `tools/process_video.py` / `src/tz_alpr/pipeline/video.py`.

Flow: `sample frames at sample_fps → vehicle detection → ByteTrack (stable
track ids) → per confirmed track: plate detection + recognize() → push the
reading (with its CTC posteriors) into the temporal aggregator → OCR-probability-
weighted per-character vote → event deduplication`. Knobs live in
`configs/tracking.yaml`.

- **ByteTracker** (`tracking/bytetrack.py`) — high-score detections drive
  association; low-score ones recover tracks through brief occlusion; tracks die
  after `track_max_age` unmatched sampled frames. No vehicle detector? It tracks
  plate boxes directly (thresholds auto-relaxed for the classical detector).
- **Temporal aggregation** (`tracking/aggregator.py`) — per position, sum
  `frame_confidence × OCR_posterior(char)` across frames; the winning string is
  re-validated against the Tanzania rules. Aggregated confidence = best single
  frame + a bounded boost for agreeing frames (correlated, so diminishing
  returns — not a naive noisy-OR). Reproduces the spec §17 example.
- **Event dedup** (`tracking/events.py`) — one track → at most one
  `VehicleEvent`; a plate correction re-issues the *same* `event_id`; the same
  plate from a fragmented track is suppressed for `event_dedup_window_s`; tracks
  that end below the emit bar still flush one event so nothing is lost silently.

Events fire only once the **OCR model is trained** — with the null OCR engine the
video pipeline still tracks and samples but produces no readings (a `warnings`
entry says so).

## Roadmap

* **Phase 1 — ALPR MVP** ✅  `tz-alpr-1.0.0`
* **Phase 2 — vehicle detection stage** ✅  `tz-alpr-1.1.0`
* **Phase 3 — video: tracking + temporal OCR + event dedup** ✅  `tz-alpr-1.2.0` — `POST /v1/video`, ByteTrack, weighted temporal voting, one-dwell-one-event
* **Phase 4** — RTSP multi-camera `stream-worker` (reuses `VideoPipeline`'s per-frame step), webhooks (HMAC, retries, idempotency, event IDs), review UI, Postgres/Redis
* **Phase 5** — parking-system integration (ALPR emits recognition events; billing is a separate service)
* **Phase 6** — vehicle attributes (make / model / colour / direction), advanced analytics

To continue: **"Proceed to Phase 4"** — the same repository is extended, the
architecture above does not change.

## License

Apache-2.0. Uses open models/tools only (Ultralytics YOLO — AGPL/commercial per
their terms; swap for an Apache detector if that matters to you).
