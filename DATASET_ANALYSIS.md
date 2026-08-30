# Dataset analysis — 2026-08-30

Snapshot from `python tools/analyze_dataset.py` after the first GPU train.
Regenerate any time (also runs inside `gpu_train.sh` / `retrain.sh`) →
`reports/dataset_analysis.md`.

Source images on disk: **26,137**  
`labels.jsonl` rows: **26,137**

## Plate categories (from labels.jsonl)

| Type | Count |
|---|---|
| PRIVATE | 24,497 |
| MOTORCYCLE (incl. bajaji / tuk-tuk `MC` plates) | 1,621 |
| OTHER | 19 |

## Audit

Accepted **26,108** · rejected **29**

| Reject reason | Count |
|---|---|
| unsupported_plate_format | 18 |
| duplicate_image | 8 |
| missing_or_unreadable_image | 3 |

## OCR crops (after `prepare_dataset.py`)

Rows **26,108** · unique plates **24,748**  
Detector conf ≥ 0.5: **76.7%**

| Split | Count |
|---|---|
| train | 21,408 |
| val | 1,808 |
| test | 1,808 |
| hard_test | 1,084 |

| Plate type (crops) | Count |
|---|---|
| PRIVATE | 24,492 |
| MOTORCYCLE | 1,614 |
| GOVERNMENT | 2 |

| Lighting | Count |
|---|---|
| day | 15,721 |
| dim | 9,266 |
| night | 1,121 |

## When more photos arrive

```bash
# 1. copy new jpgs into labeled_images/
# 2. append {"image":"...jpg","plate_text":"T331EBG","confidence":1.0} to labels.jsonl
bash scripts/retrain.sh
```
