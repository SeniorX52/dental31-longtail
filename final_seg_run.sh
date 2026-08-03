#!/usr/bin/env bash
# Project-2 FINAL run: retrain the winning ablation configuration at 100 epochs
# (same budget as the 100-epoch baseline) and evaluate ONCE on the held-out
# test split. This is the number that goes in the deliverable.
#
# The winner is picked from the VALID ablation reports at runtime (mask mAP),
# so this script can be queued before S3/S4 finish.
#
# Launch:  setsid nohup bash final_seg_run.sh > logs/final_seg.log 2>&1 </dev/null &
set -o pipefail

ROOT="$HOME/Documents/ML_SOTA"
cd "$ROOT"
mkdir -p logs reports preds runs
source "$HOME/miniconda3/bin/activate" dental
export PYTHONPATH="$ROOT:$PYTHONPATH"

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
step()  { echo; echo "=== [$(stamp)] $* ==="; }

[ -f reports/final_seg_test_segm.json ] && { echo "final already scored"; exit 0; }

step "waiting for the ablation queues / GPU"
while pgrep -f "run_seg_ablation|train_seg.py|predict_to_coco" >/dev/null 2>&1 \
   || nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; do
  sleep 300
done
sleep 20

step "picking the winner from the VALID ablation table"
WINNER=$(python3 - <<'PY'
import json, glob, os
best, bname = -1, None
for p in glob.glob("reports/ablation_S*_valid_segm.json"):
    m = json.load(open(p))["coco_stats"]["mAP"]
    a = os.path.basename(p).split("_")[1]
    if m > best: best, bname = m, a
print(bname or "S1c")
PY
)
echo "winner: $WINNER"
case "$WINNER" in
  S0)  WS=none;    BW=0;   CP=0 ;;
  S1a) WS=0.9;     BW=0;   CP=0 ;;
  S1b) WS=0.99;    BW=0;   CP=0 ;;
  S1c) WS=invsqrt; BW=0;   CP=0 ;;
  S2)  WS=invsqrt; BW=0.5; CP=0 ;;
  S3)  WS=invsqrt; BW=0;   CP=1 ;;
  S4)  WS=invsqrt; BW=0.5; CP=1 ;;
  *)   WS=invsqrt; BW=0;   CP=0 ;;
esac
if [ "$CP" = "1" ]; then
  DATA="$ROOT/data_clean_cp/data.yaml"; TL="--train-labels $ROOT/data_clean_cp/train/labels"
else
  DATA="$ROOT/data_clean/data.yaml"; TL=""
fi
echo "config: weights=$WS boundary=$BW copy-paste=$CP"

NAME="final_${WINNER}_100ep"
best_run() {
  local best="" bestn=-1 d n
  for d in runs/*/${1} runs/*/${1}-*; do
    [ -d "$d" ] || continue
    n=$(( $(wc -l < "$d/results.csv" 2>/dev/null || echo 1) - 1 )); [ "$n" -lt 0 ] && n=0
    if [ "$n" -gt "$bestn" ]; then bestn=$n; best="$d"; fi
  done
  [ -n "$best" ] && printf '%s\t%s\n' "$bestn" "$best"
}
info=$(best_run "$NAME"); n=$(echo "$info" | cut -f1); dir=$(echo "$info" | cut -f2)

if [ -n "$dir" ] && [ "${n:-0}" -ge 100 ]; then
  echo "final already trained ($dir)"
elif [ -n "$dir" ] && [ -f "$dir/weights/last.pt" ] && [ "${n:-0}" -ge 2 ]; then
  step "RESUMING final from epoch $n"
  python yolov8_seg_longtail/train_seg.py --data "$DATA" --model yolov8x-seg.pt $TL \
    --epochs 100 --imgsz 640 --batch 8 --seed 42 \
    --weights "$WS" --boundary-weight "$BW" \
    --resume "$dir/weights/last.pt" --name "$NAME" 2>&1 | tail -15
else
  step "training final config, 100 epochs"
  python yolov8_seg_longtail/train_seg.py --data "$DATA" --model yolov8x-seg.pt $TL \
    --epochs 100 --imgsz 640 --batch 8 --seed 42 \
    --weights "$WS" --boundary-weight "$BW" --name "$NAME" 2>&1 | tail -15
fi

info=$(best_run "$NAME"); dir=$(echo "$info" | cut -f2)
W="$dir/weights/best.pt"
[ -f "$W" ] || { echo "no final weights produced"; exit 1; }

step "ONE-TIME test-split evaluation ($W)"
python yolov8_seg_longtail/predict_to_coco.py \
  --weights "$W" --gt data_clean/annotations/instances_test.json \
  --images data_clean/test/images --out preds/final_seg_test.json \
  --imgsz 640 --conf 0.001 --seed 42 2>&1 | tail -3
for t in segm bbox; do
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_test.json \
    --dt preds/final_seg_test.json --train-json data_clean/annotations/instances_train.json \
    --iou-type $t --out reports/final_seg_test_$t 2>&1 | tail -4
done
echo "$WINNER" > reports/_final_seg_winner.txt
python3 _internal/update_results.py 2>/dev/null || true
step "PROJECT 2 FINAL DONE"
