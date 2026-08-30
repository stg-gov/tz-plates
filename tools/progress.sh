#!/usr/bin/env bash
# Live status of the tz-alpr data-prep / training pipeline.
#   tools/progress.sh            one-shot snapshot
#   tools/progress.sh -w [SECS]  watch, refreshing every SECS (default 10)
set -u
cd "$(dirname "$0")/.."

TOTAL_IMAGES=$(wc -l < labels.jsonl 2>/dev/null | tr -d ' ')
BAR_W=32

hr()   { printf '%*s\n' 60 '' | tr ' ' '-'; }
bar()  { # bar <done> <total>
  local d=$1 t=${2:-1} filled
  [ "$t" -eq 0 ] && t=1
  filled=$(( d * BAR_W / t )); [ "$filled" -gt "$BAR_W" ] && filled=$BAR_W
  printf '['
  printf '%*s' "$filled" '' | tr ' ' '#'
  printf '%*s' "$((BAR_W - filled))" '' | tr ' ' '.'
  printf '] %s/%s (%d%%)\n' "$d" "$t" "$(( d * 100 / t ))"
}
proc() { pgrep -f "$1" >/dev/null && echo "RUNNING (pid $(pgrep -f "$1" | tr '\n' ' '))" || echo "not running"; }
newest() { ls -t $1 2>/dev/null | head -1; }

snapshot() {
  date '+%Y-%m-%d %H:%M:%S'
  hr
  echo "processes"
  printf '  prepare_dataset       : %s\n' "$(proc 'tools/prepare_dataset.py')"
  printf '  generate_tz_plates    : %s\n' "$(proc 'tools/generate_tanzania_plates.py')"
  printf '  train_ocr             : %s\n' "$(proc 'training/train_ocr.py')"
  printf '  evaluate              : %s\n' "$(proc 'training/evaluate.py')"
  printf '  load average          : %s\n' "$(uptime | sed 's/.*load averages*: //')"
  hr

  echo "dataset prep"
  local crops
  crops=$(find datasets/processed/plate_crops -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
  printf '  plate crops           : '; bar "${crops:-0}" "${TOTAL_IMAGES:-1}"
  if [ -f datasets/ocr/ocr_annotations.csv ]; then
    printf '  ocr_annotations.csv   : %s rows\n' "$(( $(wc -l < datasets/ocr/ocr_annotations.csv) - 1 ))"
    awk -F, 'NR>1{c[$5]++} END{for(k in c) printf "    %-10s %d\n", k, c[k]}' datasets/ocr/ocr_annotations.csv
  else
    echo "  ocr_annotations.csv   : (not written yet — appears when prep finishes)"
  fi
  hr

  echo "synthetic"
  local syn
  syn=$(find datasets/synthetic/generated_plates/images -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
  printf '  synthetic plates      : %s images\n' "${syn:-0}"
  hr

  echo "training"
  local logdir metrics ckpt
  logdir=$(newest 'lightning_logs/*/version_*')
  if [ -n "$logdir" ]; then
    printf '  run dir               : %s\n' "$logdir"
    metrics="$logdir/metrics.csv"
    if [ -f "$metrics" ]; then
      python3 - "$metrics" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("  metrics.csv           : (empty)"); raise SystemExit
def last(key):
    for r in reversed(rows):
        if r.get(key) not in (None, "", "nan"):
            return r[key]
    return None
ep = last("epoch")
step = last("step")
print(f"  epoch / step          : {ep} / {step}")
for k in ("train/loss_epoch", "val/loss", "val/seq_acc", "val/char_acc", "val/cer",
          "test/seq_acc", "test/char_acc", "test/cer"):
    v = last(k)
    if v is not None:
        try: v = f"{float(v):.4f}"
        except ValueError: pass
        print(f"  {k:22s}: {v}")
PY
    else
      echo "  metrics.csv           : (not written yet)"
    fi
  else
    echo "  no lightning run yet"
  fi
  ckpt=$(newest 'checkpoints/*.ckpt')
  [ -n "$ckpt" ] && printf '  newest checkpoint     : %s (%s)\n' "$ckpt" "$(date -r "$ckpt" '+%H:%M:%S')"
  for f in models/ocr/v1/ocr_crnn.pt models/ocr/v1/ocr_crnn_pretrained.pt models/ocr/v1/ocr_crnn.onnx; do
    [ -f "$f" ] && printf '  exported              : %s (%s KB)\n' "$f" "$(( $(wc -c < "$f") / 1024 ))"
  done
  hr

  echo "reports"
  for f in reports/eval_test.json reports/eval_hard_test.json; do
    if [ -f "$f" ]; then
      printf '  %s\n' "$f"
      python3 -c "import json,sys; d=json.load(open('$f')); print('    n=%s  exact_match=%s  ocr=%s' % (d['n'], d['end_to_end_exact_match'], d['ocr']))" 2>/dev/null
    fi
  done
}

if [ "${1:-}" = "-w" ]; then
  interval="${2:-10}"
  while true; do clear; snapshot; sleep "$interval"; done
else
  snapshot
fi
