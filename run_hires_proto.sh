#!/usr/bin/env bash
# Test the intervention the ceiling measurement points at: raise the prototype
# grid from input/4 to input/2.
#
# WHY THIS ARM EXISTS
# tools/mask_resolution_ceiling.py round-trips every ground-truth mask through
# the prototype grid and scores it against itself, which bounds what ANY model
# can achieve regardless of its loss. At the stock input/4 grid:
#
#   mean Dice ceiling                      0.8963
#   instances that cannot reach IoU 0.75    17.7 %
#   root canal treatment IoU ceiling        0.622   (below the 0.75 threshold)
#   caries IoU ceiling                      0.721   (below the 0.75 threshold)
#
# So AP75 is capped by the representation before training starts, and the two
# most clinically important classes are structurally excluded from it. That is
# consistent with what the model actually does: mask AP50 reaches 87 % of box
# AP50, but mask AP75 only 49 % of box AP75.
#
# Doubling the grid moves the impossible share to 5.4 % and the Dice ceiling to
# 0.9492. This arm measures how much of that headroom the model can use.
#
# WHAT MAKES IT A CLEAN TEST
# HR differs from S0 in exactly one respect: the prototype grid. Same data,
# same schedule, same seed, same augmentation, no class weighting, no auxiliary
# loss term. The head reuses the pretrained cv1/upsample/cv2 and adds a
# sub-pixel convolution stage, so the model ends up with 7712 FEWER parameters
# than stock -- any gain cannot be explained by added capacity.
#
# 50 epochs against the existing 50-epoch S0 arm. That budget is well past the
# measured peak (every arm on this dataset tops out around epoch 25-30), so a
# longer schedule would only add overtraining.
#
# Launch: setsid nohup bash run_hires_proto.sh > logs/hires_proto.log 2>&1 </dev/null &
set -o pipefail

ROOT="$HOME/Documents/ML_SOTA"
cd "$ROOT"
mkdir -p logs reports preds runs
source "$HOME/miniconda3/bin/activate" dental
export PYTHONPATH="$ROOT:$PYTHONPATH"

EPOCHS=50
BATCH=8
IMGSZ=640
TAG=HR
CONF=0.15

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
step()  { echo; echo "=== [$(stamp)] $* ==="; }

gpu_busy() {
  local p c
  for p in $(pgrep -f "train_seg\.py|main\.py --output_dir|predict_to_coco\.py|export_dino_preds\.py" 2>/dev/null); do
    c=$(ps -o comm= -p "$p" 2>/dev/null)
    case "$c" in python*) return 0 ;; esac
  done
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]' && return 0
  return 1
}

run_epoch() {
  [ -f "$1/results.csv" ] || { echo 0; return; }
  awk -F, 'NR==1{for(i=1;i<=NF;i++){gsub(/^[ \t]+|[ \t]+$/,"",$i); if($i=="epoch") c=i} next}
           c && $c+0>m {m=$c+0} END{printf "%d", m+0}' "$1/results.csv"
}

best_run() {
  local best="" bestn=-1 d n
  for d in runs/segment/${1} runs/segment/${1}-*; do
    [ -d "$d" ] || continue
    n=$(run_epoch "$d"); [ "${n:-0}" -gt "$bestn" ] && { bestn=$n; best="$d"; }
  done
  [ -n "$best" ] && printf '%s\t%s\n' "$bestn" "$best"
}

step "waiting for the GPU (SB is expected to be running)"
while gpu_busy; do sleep 300; done
sleep 20

NAME="abl_${TAG}"
REPORT="reports/ablation_${TAG}_valid_segm"

if [ ! -f "${REPORT}.json" ]; then
  info=$(best_run "$NAME"); n=$(echo "$info" | cut -f1); dir=$(echo "$info" | cut -f2)
  if [ -n "$dir" ] && [ "${n:-0}" -ge "$EPOCHS" ]; then
    echo "[$(stamp)] $TAG already trained ($dir)"
  elif [ -n "$dir" ] && [ -f "$dir/weights/last.pt" ] && [ "${n:-0}" -ge 2 ]; then
    step "resuming $TAG from epoch $n"
    python yolov8_seg_longtail/train_seg.py --data data_clean/data.yaml \
      --model yolov8x-seg.pt --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" \
      --seed 42 --cache ram --channels-last --weights none --boundary-weight 0 \
      --proto-scale 2 --resume "$dir/weights/last.pt" --name "$NAME" 2>&1 | tail -14
  else
    step "$TAG: prototype grid input/2, no weighting, no auxiliary loss, ${EPOCHS} ep"
    python yolov8_seg_longtail/train_seg.py --data data_clean/data.yaml \
      --model yolov8x-seg.pt --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" \
      --seed 42 --cache ram --channels-last --weights none --boundary-weight 0 \
      --proto-scale 2 --name "$NAME" 2>&1 | tail -14
  fi

  info=$(best_run "$NAME"); n=$(echo "$info" | cut -f1); dir=$(echo "$info" | cut -f2)
  W="$dir/weights/best.pt"
  if [ ! -f "$W" ]; then
    echo "[$(stamp)] $TAG: no weights produced"; exit 1
  fi
  # completion guard -- never score a run that did not finish its schedule
  if [ "${n:-0}" -lt "$EPOCHS" ] && ! python3 tools/run_finished.py "$dir" 2>/dev/null; then
    echo "[$(stamp)] $TAG reached only ${n:-0}/$EPOCHS epochs and is not marked"
    echo "            finished -- refusing to score. Relaunch to resume."
    exit 1
  fi

  step "score $TAG on VALID (epochs ${n}/$EPOCHS)"
  python yolov8_seg_longtail/predict_to_coco.py --weights "$W" \
    --gt data_clean/annotations/instances_valid.json \
    --images data_clean/valid/images --out "preds/ablation_${TAG}_valid.json" \
    --imgsz "$IMGSZ" --conf 0.001 --seed 42 2>&1 | tail -3
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
    --dt "preds/ablation_${TAG}_valid.json" \
    --train-json data_clean/annotations/instances_train.json \
    --iou-type segm --out "$REPORT" 2>&1 | tail -4
fi

step "contour metrics and paired comparison against S0"
[ -f "reports/contour_${TAG}_valid.json" ] || python eval/contour_metrics.py \
  --gt data_clean/annotations/instances_valid.json \
  --dt "preds/ablation_${TAG}_valid.json" \
  --train-json data_clean/annotations/instances_train.json \
  --conf "$CONF" --boot 200 --out "reports/contour_${TAG}_valid" 2>&1 | tail -3
[ -f "reports/paired_contour_S0_${TAG}_valid.json" ] || \
  PYTHONPATH="$ROOT/eval:$PYTHONPATH" python eval/paired_contour.py \
    --gt data_clean/annotations/instances_valid.json \
    --dt-a preds/ablation_S0_valid.json --label-a S0 \
    --dt-b "preds/ablation_${TAG}_valid.json" --label-b "$TAG" \
    --conf "$CONF" --boot 500 --out "reports/paired_contour_S0_${TAG}_valid" 2>&1 | tail -8

step "RESULT: prototype grid input/2 vs stock input/4"
python3 - <<'PY'
import json, os
def g(p):
    d=json.load(open(p)); s,q=d["coco_stats"],d["group_AP"]
    return s["mAP"],s["AP50"],s["AP75"],q.get("head",0),q.get("mid",0),q.get("tail",0)
a="reports/ablation_S0_valid_segm.json"; b="reports/ablation_HR_valid_segm.json"
if os.path.exists(a) and os.path.exists(b):
    s0,hr=g(a),g(b)
    print("  %-26s %8s %8s %8s %8s %8s"%("arm","mAP","AP50","AP75","head","mid"))
    print("  %-26s %8.4f %8.4f %8.4f %8.4f %8.4f"%(("S0  proto input/4",)+s0[:5]))
    print("  %-26s %8.4f %8.4f %8.4f %8.4f %8.4f"%(("HR  proto input/2",)+hr[:5]))
    print("\n  delta (pp):  mAP %+.2f  AP50 %+.2f  AP75 %+.2f  head %+.2f  mid %+.2f"
          % tuple(100*(hr[i]-s0[i]) for i in range(5)))
    print("\n  ceiling context: input/4 allows mean Dice 0.8963 and leaves 17.7%% of")
    print("  instances unable to reach IoU 0.75; input/2 gives 0.9492 and 5.4%%.")
PY
step "HI-RES PROTO ARM DONE"
