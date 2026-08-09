#!/usr/bin/env bash
# Two arms against the measured mask bottleneck.
#
# WHY THESE TWO. `tools/oracle_coefficients.py` solved, in closed form, for the
# best coefficients obtainable from the trained model's OWN prototypes. It split
# the 0.1994 Dice gap between what the model achieves (0.6969) and what its
# prototype grid allows (0.8963):
#
#     prototype basis cannot represent it   0.0304   15 %
#     coefficient head fails to find it     0.1690   85 %
#
# So the basis is close to sufficient and the head is not finding the right
# point in it. Every earlier intervention -- boundary losses, class weighting,
# higher-resolution prototypes, a P2-fed prototype head -- either supervised in
# pixel space or enlarged the basis. Both address the 15 %. That is why they all
# landed inside the +-0.21 pp noise floor, and why the P2 head came out 0.72 pp
# WORSE: it spent capacity on the part that was not broken.
#
#   K1  supervise the coefficients directly (--coeff-weight)
#       An auxiliary term || c_pred - c* ||^2 where c* is the closed-form
#       optimum on the model's own prototypes, recomputed every step and
#       detached. The teacher is exact and costs one 32x32 solve per image.
#       This is the first term in the project that puts gradient on the
#       coefficients without routing it through the prototype product.
#
#   K2  widen the coefficient branch (--coeff-width)
#       cv4 is 320 -> 80 -> 80 -> 32, and that 80 is `max(ch[0]//4, nm)`, an
#       ultralytics default for COCO that nobody tuned. The branch holds 1.33 M
#       parameters against 2.26 M in the prototype head and 7.41 M in the
#       classifier: the smallest head carries the largest share of the error.
#       This is the capacity control for K1. Widening helps => the problem is
#       capacity. Only K1 helps => the problem is supervision. Both help => the
#       2x2 says so.
#
# Everything else is held at the abl_S0 reference exactly: yolov8x-seg.pt,
# data_clean, 50 epochs, imgsz 640, batch 8, seed 42, cache ram, no class
# weighting, no boundary term. One variable per arm.
#
# ORDER. K1 first. It is the arm that targets the measured 85 %; K2 is its
# control. If only one finishes, the one that finishes should be the one that
# tests the hypothesis.
#
# Usage:  nohup ./run_coeff_arms.sh > logs/coeff_arms.log 2>&1 &
cd "$HOME/Documents/ML_SOTA" || exit 1
# conda's activation hooks reference unset variables, so -u goes on afterwards
source "$HOME/miniconda3/bin/activate" dental
set -u
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs reports

DATA="$PWD/data_clean/data.yaml"
# Weight chosen from a measurement, not a guess. On the trained S0 prototypes
# over 181 validation instances the closed-form target has mean c*^2 of 2.86 per
# coefficient, while the BCE term ultralytics computes is
# `crop_mask(bce).mean(dim=(1,2)) / area`, which normalises to roughly the mean
# BCE inside the box -- order 0.3 to 1.0 per instance. At weight 1.0 the
# distillation term would be several times the mask loss it is supposed to
# assist. 0.2 makes it a substantial minority of the primary term.
#
# The same 181 instances put the box-restricted target at Dice 0.9814 against
# the 0.6969 the model achieves, so the teacher has real headroom to teach.
COEFF_W="${COEFF_W:-0.2}"
WIDTH="${WIDTH:-256}"

# ---------------------------------------------------------------- gpu_busy ---
# A training job is running if any *python* process owns a train script. Match
# on the process NAME, never on `pgrep -f` alone: `pgrep -f train_seg` also
# matches this script, any editor holding the file, and any diagnostic shell
# that happens to mention it. That self-match silently stalled the queue three
# separate times in this project -- the chain waited forever on itself.
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
  local waited=0
  while gpu_busy; do
    if [ $((waited % 600)) -eq 0 ]; then
      echo "[$(date '+%F %T')] GPU busy, waiting (${waited}s)"
    fi
    sleep 60; waited=$((waited + 60))
  done
  # let the previous process release its allocation before claiming memory
  sleep 45
  echo "[$(date '+%F %T')] GPU free after ${waited}s"
}

# ------------------------------------------------------------------- guard ---
# COMPLETION GUARD. An OOM-killed run leaves a best.pt on disk that looks
# exactly like a finished one, and this queue scored such a partial model as a
# final result once already (a 30-epoch checkpoint reported as a 100-epoch P2
# run at mAP 0.1007). Never score a run without asking whether it finished its
# schedule. run_finished.py reads the epoch number recorded in the checkpoint,
# not the row count of results.csv.
# run_finished.py takes the run directory POSITIONALLY and reads the epoch = -1
# completion marker from the checkpoint. Calling it with --run/--epochs flags it
# does not define makes it exit 2 on an argparse error, which reads as "not
# finished" and silently refuses to score a run that completed. That is exactly
# what happened to abl_K1_coeffdistill on the 07 Aug overnight chain: it trained
# all 50 epochs and was never scored.
finished() {
  python tools/run_finished.py "runs/segment/$1" >/dev/null 2>&1
}

run_arm() {
  local tag="$1"; shift
  if [ -d "runs/segment/$tag" ] && finished "$tag"; then
    echo "[$(date '+%F %T')] $tag already complete, skipping"
    return 0
  fi
  wait_for_gpu
  echo "[$(date '+%F %T')] === $tag ==="
  echo "    extra args: $*"
  python yolov8_seg_longtail/train_seg.py \
      --data "$DATA" --model yolov8x-seg.pt --nc 31 \
      --epochs 50 --imgsz 640 --batch 8 --seed 42 --cache ram \
      --channels-last --weights none --boundary-weight 0 \
      --name "$tag" "$@" 2>&1 | tail -25

  if finished "$tag"; then
    echo "[$(date '+%F %T')] $tag finished its schedule"
  else
    echo "[$(date '+%F %T')] *** $tag did NOT finish 50 epochs -- NOT scoring it"
    return 1
  fi
}

echo "[$(date '+%F %T')] coefficient arms queued behind whatever is on the GPU"
echo "    K1 coefficient distillation, weight $COEFF_W"
echo "    K2 cv4 widened to $WIDTH"

run_arm abl_K1_coeffdistill --coeff-weight "$COEFF_W"
run_arm abl_K2_cv4wide      --coeff-width  "$WIDTH"

# ------------------------------------------------------------------ score ---
# Same path every other arm in this project went through, and for the same
# reason. COCO segm mAP alone cannot judge these two: HD95 and ASSD average
# only over cases where BOTH masks are non-empty, so an arm that predicts less
# gets an easier denominator and can look better while being worse. That
# already happened once here -- a boundary arm read 17-25 % better on distance
# until it was recomputed on the intersection of cases, where it was
# significantly worse. So every arm gets the paired comparison against S0 on
# common cases, with coverage printed beside it.
#
# CONF is the operating point chosen on validation using the BASELINE arm and
# then frozen. It is never re-tuned per arm.
CONF=0.15
REF=abl_S0

score() {
  local tag="$1" dt rdt
  finished "$tag" || { echo "[$(date '+%F %T')] $tag unfinished, refusing to score"; return 0; }
  dt="preds/ablation_${tag}_valid.json"
  local w="runs/segment/$tag/weights/best.pt"

  [ -f "$dt" ] || python yolov8_seg_longtail/predict_to_coco.py \
      --weights "$w" --gt data_clean/annotations/instances_valid.json \
      --images data_clean/valid/images --out "$dt" \
      --imgsz 640 --conf 0.001 --seed 42 2>&1 | tail -3 || return 0

  python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
      --dt "$dt" --train-json data_clean/annotations/instances_train.json \
      --iou-type segm --out "reports/eval_${tag}_valid" 2>&1 | tail -4 || true

  python eval/contour_metrics.py \
      --gt data_clean/annotations/instances_valid.json --dt "$dt" \
      --train-json data_clean/annotations/instances_train.json \
      --conf "$CONF" --boot 200 --out "reports/contour_${tag}_valid" 2>&1 | tail -3 || true

  rdt="preds/ablation_${REF#abl_}_valid.json"
  [ -f "$rdt" ] || rdt="preds/ablation_S0_valid.json"
  if [ -f "$rdt" ]; then
    PYTHONPATH="$PWD/eval:${PYTHONPATH:-}" python eval/paired_contour.py \
        --gt data_clean/annotations/instances_valid.json \
        --dt-a "$rdt" --label-a S0 --dt-b "$dt" --label-b "$tag" \
        --conf "$CONF" --boot 500 \
        --out "reports/paired_contour_S0_${tag}_valid" 2>&1 | tail -8 || true
  else
    echo "[$(date '+%F %T')] S0 predictions absent, paired comparison skipped"
  fi
}

echo "[$(date '+%F %T')] both arms attempted; scoring what completed"
score abl_K1_coeffdistill
score abl_K2_cv4wide
echo "[$(date '+%F %T')] done"
