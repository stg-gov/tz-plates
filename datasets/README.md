# datasets/

Layout, contents and how each file is produced.

```
raw/
  images/            original stills (drop yours here; phone/POS/CCTV)
  videos/            source clips for tools/extract_frames.py
annotations/
  vehicles/          YOLO labels + vehicles.yaml       (Phase 2)
  plates/            images/{train,val}  labels/{train,val}  plates.yaml   <- tools/prepare_dataset.py (WEAK boxes)
  ocr/               reserved
  accepted_labels.jsonl  original labels after audit    <- tools/audit_labeled_images.py
  metadata.jsonl     unified per-image metadata        <- tools/prepare_dataset.py
processed/
  vehicle_crops/     Phase 2
  plate_crops/       rectified plate crops for OCR      <- tools/prepare_dataset.py
  rectified_plates/  reserved for cached rectifications
synthetic/
  generated_plates_v2/ images/ + ocr_annotations.csv   <- tools/generate_tanzania_plates.py
ocr/
  ocr_annotations.csv   image_path,plate_text,country,plate_type,split,group_key,source   <- tools/prepare_dataset.py
splits/
  {train,val,test,hard_test}.txt   crop paths per split   <- tools/prepare_dataset.py
retraining_queue/    active-learning drop zone (Phase >=4)
```

## Rules

* **No leakage.** Splitting is grouped by `group_key` (plate identity for stills;
  `session_id` for video frames). Never split frames of one video/session across
  train/val/test.
* **`hard_test`** = night / motion-blur / tiny / angled / motorcycle / partially
  occluded / unrecognised-pattern. Report metrics on `test` **and** `hard_test`
  separately.
* Run `tools/audit_labeled_images.py` before `prepare_dataset.py`. It produces
  an accepted manifest and a review report, rejecting invalid formats, corrupt
  files, filename-label disagreements and exact duplicates. It cannot determine
  whether a readable but incorrectly labelled photo is visually wrong; review
  `reports/data_audit.jsonl` and the rejected examples before training.
* Weak plate boxes from `prepare_dataset.py` bootstrap detector training —
  hand-verify a few hundred in any YOLO annotation tool before a production run.
  OCR crops need no box correction (text labels are human-verified).
* Large binaries here are git-ignored. Keep raw data backed up out-of-band.

## Provided dataset

`../labeled_images/*.jpg` + `../labels.jsonl`
(`{"image","plate_text","confidence"}`), 26 137 vehicle-rear photos with
human-verified plate strings. ~24.5 k private `T### ABC`, ~1.6 k motorcycle
`MC### ABC`, plus a few government/other. Run `tools/prepare_dataset.py` to turn
it into OCR crops + splits.
