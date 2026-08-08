#!/usr/bin/env bash
# P0 resolution probe. 40 percent of every split is native 1615x840 and has
# been training with its long side downsampled 2.5x to 640. The ceiling
# measurement says input/2 raises representable Dice from 0.896 to 0.949, and
# the near-floor classes (caries 0.092, periapical lesion 0.031, bone loss
# 0.020) are small low-contrast findings, exactly what resolution starves.
#
# This is a PROBE, not a clean cell: it fine-tunes the finished S0 weights for
# 25 epochs at 1280, so the delta against S0 bundles the extra budget with the
# resolution. If it moves the pathology classes, the clean 50-epoch from-scratch
# comparison is the follow-up. If it moves nothing, resolution is closed too.
#
# batch 2 (not 8): 4x the pixels per image. No RAM cache: the 640 cache already
# used ~25 GB and 1280 would quadruple it past physical RAM.
#
# Usage:  nohup ./run_hr1280.sh > logs/hr1280.log 2>&1 &

cd "$HOME/Documents/ML_SOTA" || exit 1
source "$HOME/miniconda3/bin/activate" dental
set -u
export PYTHONPATH="$PWD:$PYTHONPATH"
mkdir -p logs reports preds

TAG=abl_HR1280ft
CONF=0.15
stamp() { date '+%F %T'; }
finished() { python tools/run_finished.py "runs/segment/$1" >/dev/null 2>&1; }

gpu_busy() {
  local p c
  for p in $(pgrep -f "train_seg\.py|main\.py --output_dir|predict_to_coco" 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    c=$(ps -o comm= -p "$p" 2>/dev/null)
    case "$c" in python*|pt_data*) return 0 ;; esac
  done
  return 1
}
while gpu_busy; do sleep 60; done

if ! finished "$TAG"; then
  # Resume if an interrupted run left a checkpoint; a fresh start would
  # otherwise reinitialise from S0 and silently discard the finished epochs.
  RESUME=()
  if [ -f "runs/segment/$TAG/weights/last.pt" ]; then
    RESUME=(--resume "runs/segment/$TAG/weights/last.pt")
    echo "[$(stamp)] found last.pt, resuming in place"
  fi
  echo "[$(stamp)] === $TAG: S0 weights, imgsz 1280, 25 epochs ==="
  python yolov8_seg_longtail/train_seg.py \
      --data "$PWD/data_clean/data.yaml" \
      --model runs/segment/abl_S0/weights/best.pt --nc 31 \
      --epochs 25 --imgsz 1280 --batch 2 --seed 42 \
      --channels-last --weights none --boundary-weight 0 \
      --name "$TAG" "${RESUME[@]}" 2>&1 | tail -20
fi

if finished "$TAG"; then
  echo "[$(stamp)] scoring $TAG at its own resolution"
  dt="preds/ablation_${TAG}_valid.json"
  [ -f "$dt" ] || python yolov8_seg_longtail/predict_to_coco.py \
      --weights "runs/segment/$TAG/weights/best.pt" \
      --gt data_clean/annotations/instances_valid.json \
      --images data_clean/valid/images --out "$dt" \
      --imgsz 1280 --conf 0.001 --seed 42 2>&1 | tail -3
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
      --dt "$dt" --train-json data_clean/annotations/instances_train.json \
      --iou-type segm --out "reports/eval_${TAG}_valid" 2>&1 | tail -4
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
      --dt "$dt" --train-json data_clean/annotations/instances_train.json \
      --iou-type bbox --out "reports/bboxchk_${TAG}_valid" 2>&1 | tail -3
  PYTHONPATH="$PWD/eval:$PYTHONPATH" python eval/paired_contour.py \
      --gt data_clean/annotations/instances_valid.json \
      --dt-a preds/ablation_S0_valid.json --label-a S0 \
      --dt-b "$dt" --label-b "$TAG" --conf "$CONF" --boot 500 \
      --out "reports/paired_contour_S0_${TAG}_valid" 2>&1 | tail -8
else
  echo "[$(stamp)] *** $TAG did not finish; not scoring"
fi
echo "[$(stamp)] done"
