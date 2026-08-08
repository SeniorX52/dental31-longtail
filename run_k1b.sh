#!/usr/bin/env bash
# K1b -- coefficient distillation, second attempt, with the target repaired.
#
# WHY THERE IS A SECOND ATTEMPT. The first arm, abl_K1_coeffdistill, trained all
# 50 epochs and produced a number (0.1106 against the 0.1244 reference), but the
# number measures a bug, not the method. Its seg_loss ran 182,188 at epoch 1,
# NaN by epoch 10 and 7.2e7 by epoch 40, against 2.9 falling to 1.6 for the same
# configuration without the term.
#
# The cause was the ridge. A = P B P^T is built from learned features restricted
# to one instance box, and it is severely ill-conditioned: measured on
# COCO-pretrained prototypes over real dental instances its eigenvalues span
# 1e-6 to 1e3, with the smallest slightly negative from rounding. A fixed ridge
# of 1e-3 regularises some instances and leaves others singular, and one
# singular instance out of 95,745 annotations is enough to send c* to infinity.
#
# The weight calibration is what let it through. Mean c*^2 was measured as 2.86
# on the CONVERGED reference model, then applied to a run starting from COCO
# weights, where the same quantity measures 4 to 53 with a much worse tail.
#
# WHAT CHANGED. The ridge is now relative to trace(A)/nm, which bounds the
# condition number whatever the prototype scale or box size. The term is a
# relative error rather than a raw MSE, so the weight means the same thing at
# every step. Targets are clamped and non-finite values dropped. Verified on the
# exact case that broke it: every target finite, max element 3.86, and the
# teacher still reaches Dice 0.8991 from COCO-pretrained prototypes, against
# 0.6969 for what the trained model achieves.
#
# The broken run is kept on disk rather than overwritten. It is the evidence for
# the failure and it belongs in the write-up.
#
# Usage:  nohup ./run_k1b.sh > logs/k1b.log 2>&1 &

cd "$HOME/Documents/ML_SOTA" || exit 1
source "$HOME/miniconda3/bin/activate" dental
set -u
export PYTHONPATH="$PWD:$PYTHONPATH"
mkdir -p logs reports preds

TAG=abl_K1b_coeffdistill
COEFF_W="${COEFF_W:-0.2}"
CONF=0.15

stamp() { date '+%F %T'; }

# run_finished.py takes the run directory POSITIONALLY. Passing flags it does
# not define makes it exit 2, which reads as "not finished" and silently skips
# scoring a completed run. That is what hid the first arm's failure.
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

wait_for_gpu() {
  local w=0
  while gpu_busy; do
    [ $((w % 600)) -eq 0 ] && echo "[$(stamp)] GPU busy, waiting (${w}s)"
    sleep 60; w=$((w + 60))
  done
  sleep 45
  echo "[$(stamp)] GPU free after ${w}s"
}

score() {
  local tag="$1" dt
  finished "$tag" || { echo "[$(stamp)] $tag unfinished, refusing to score"; return 0; }
  dt="preds/ablation_${tag}_valid.json"
  if [ ! -f "$dt" ]; then
    python yolov8_seg_longtail/predict_to_coco.py \
      --weights "runs/segment/$tag/weights/best.pt" \
      --gt data_clean/annotations/instances_valid.json \
      --images data_clean/valid/images --out "$dt" \
      --imgsz 640 --conf 0.001 --seed 42 2>&1 | tail -3 || return 0
  fi
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
    --dt "$dt" --train-json data_clean/annotations/instances_train.json \
    --iou-type segm --out "reports/eval_${tag}_valid" 2>&1 | tail -4 || true
  python eval/contour_metrics.py \
    --gt data_clean/annotations/instances_valid.json --dt "$dt" \
    --train-json data_clean/annotations/instances_train.json \
    --conf "$CONF" --boot 200 --out "reports/contour_${tag}_valid" 2>&1 | tail -3 || true
  if [ -f preds/ablation_S0_valid.json ]; then
    PYTHONPATH="$PWD/eval:$PYTHONPATH" python eval/paired_contour.py \
      --gt data_clean/annotations/instances_valid.json \
      --dt-a preds/ablation_S0_valid.json --label-a S0 \
      --dt-b "$dt" --label-b "$tag" --conf "$CONF" --boot 500 \
      --out "reports/paired_contour_S0_${tag}_valid" 2>&1 | tail -8 || true
  fi
}

echo "[$(stamp)] K1b queued behind whatever is on the GPU (K2 is finishing)"

if finished "$TAG"; then
  echo "[$(stamp)] $TAG already complete"
else
  wait_for_gpu
  echo "[$(stamp)] === $TAG (coeff-weight $COEFF_W, relative ridge) ==="
  python yolov8_seg_longtail/train_seg.py \
      --data "$PWD/data_clean/data.yaml" --model yolov8x-seg.pt --nc 31 \
      --epochs 50 --imgsz 640 --batch 8 --seed 42 --cache ram \
      --channels-last --weights none --boundary-weight 0 \
      --coeff-weight "$COEFF_W" --name "$TAG" 2>&1 | tail -25
  finished "$TAG" && echo "[$(stamp)] $TAG finished" \
                  || echo "[$(stamp)] *** $TAG did NOT finish"
fi

# A blown-up auxiliary term is visible in the training curve long before the
# metric, so check it explicitly rather than trusting the mAP alone.
echo "[$(stamp)] seg_loss sanity (must stay order 1, not 1e5):"
awk -F, 'NR==1{next} $1==1||$1%10==0 {printf "    ep %-3s seg_loss %s\n",$1,$4}' \
    "runs/segment/$TAG/results.csv" 2>/dev/null

echo "[$(stamp)] scoring K1b and K2"
score "$TAG"
score abl_K2_cv4wide
echo "[$(stamp)] done"
