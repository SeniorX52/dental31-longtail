#!/usr/bin/env bash

# gpu_busy: true only when a REAL python training/eval process is alive, or the
# GPU has a compute app attached.
#
# Why not plain `pgrep -f train_seg.py`: pgrep -f matches full command lines, so
# ANY shell whose arguments merely mention those script names (a monitoring
# command, a grep, this script's own launcher) counts as "busy". That made the
# wait loop sleep another 300 s every time a diagnostic command happened to be
# running, and it could stall indefinitely. Filtering by the process's comm
# (python*) counts only genuine workloads.
gpu_busy() {
  local p c
  for p in $(pgrep -f "train_seg\.py|main\.py --output_dir|predict_to_coco\.py|export_dino_preds\.py" 2>/dev/null); do
    c=$(ps -o comm= -p "$p" 2>/dev/null)
    case "$c" in python*) return 0 ;; esac
  done
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]' && return 0
  return 1
}

# Project-1 FINAL: pick the best detection arm from the VALID ablation table
# and evaluate it ONCE on the held-out test split. No retraining -- every arm
# is already a full 12-epoch (official 1x) run, identical budget to the
# baseline, so the winner's checkpoint is the final model.
#
# Launch:  setsid nohup bash final_dino_run.sh > logs/final_dino.log 2>&1 </dev/null &
set -o pipefail

ROOT="$HOME/Documents/ML_SOTA"
DINO="$HOME/DINO"
cd "$ROOT"
source "$HOME/miniconda3/bin/activate" dental
export PYTHONPATH="$ROOT:$DINO:$PYTHONPATH"

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
[ -f reports/final_dino_test_bbox.json ] && { echo "final dino already scored"; exit 0; }

echo "=== [$(stamp)] waiting for the DINO ablation / GPU ==="
while gpu_busy; do
  sleep 300
done
sleep 20

read -r WINNER OPTS <<< "$(python3 - <<'PY'
import json, glob, os
ARM_OPTS = {
 "A0": "", "A1": "lt_rfs=True",
 "A2": "lt_rfs=True lt_la_loss=True",
 "A3": "lt_rfs=True lt_la_loss=True lt_la_cost=True",
 "A4": "lt_rfs=True lt_la_loss=True lt_la_cost=True lt_freq_dn=True",
 "A5": "lt_rfs=True lt_la_loss=True lt_la_cost=True lt_freq_dn=True lt_clahe=True",
 "A2p": "lt_rfs=True lt_la_cost=True",
}
best, name = -1, "A0"
for p in glob.glob("reports/dino_ablation_A*_valid_bbox.json"):
    m = json.load(open(p))["coco_stats"]["mAP"]
    a = os.path.basename(p).split("_")[2]
    if m > best: best, name = m, a
print(name, ARM_OPTS.get(name, ""))
PY
)"
echo "winner: $WINNER  (opts: ${OPTS:-none})"

CKPT="$ROOT/runs/dino_abl/$WINNER/checkpoint.pth"
[ "$WINNER" = "A0" ] && CKPT="$ROOT/runs/dino_baseline/checkpoint.pth"
[ -f "$CKPT" ] || { echo "winner checkpoint missing: $CKPT"; exit 1; }
FLAGS=""; case "$OPTS" in *lt_clahe=True*) FLAGS="--clahe" ;; esac

echo "=== [$(stamp)] ONE-TIME test evaluation of $WINNER ==="
python dino_longtail/export_dino_preds.py \
  --dino-root "$DINO" --config "$DINO/config/DINO/DINO_4scale.py" \
  --checkpoint "$CKPT" --coco-path "$ROOT/data_coco" --split test2017 \
  --gt data_clean/annotations/instances_test.json \
  --out preds/final_dino_test.json $FLAGS \
  --options num_classes=32 dn_labelbook_size=32 $OPTS 2>&1 | tail -3
python eval/coco_eval_report.py --gt data_clean/annotations/instances_test.json \
  --dt preds/final_dino_test.json --train-json data_clean/annotations/instances_train.json \
  --iou-type bbox --out reports/final_dino_test_bbox 2>&1 | tail -4
echo "$WINNER" > reports/_final_dino_winner.txt
echo "=== [$(stamp)] PROJECT 1 FINAL DONE ==="
