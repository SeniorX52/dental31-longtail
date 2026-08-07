#!/usr/bin/env bash
# Overnight chain: finish C15, then the two coefficient-head arms, then score.
#
# Start it with:
#     cd ~/Documents/ML_SOTA && nohup ./run_overnight.sh > logs/overnight.log 2>&1 &
#
# Then check on it any time with:
#     tail -f ~/Documents/ML_SOTA/logs/overnight.log
#
# Roughly 8 hours end to end. Safe to leave unattended: every stage writes a
# checkpoint each epoch and refuses to score a run that did not finish its
# schedule, so an interruption costs at most the epoch in progress.
#
# The PAUSE file is deliberately left in place. The cron watchdog checks for it
# every 15 minutes and does nothing while it exists, which is what keeps the old
# 38-hour queue from launching on top of this one. DO NOT delete PAUSE to start
# this script -- this script does not need the watchdog, and removing the file
# would let the watchdog start a competing job on the same GPU.

cd "$HOME/Documents/ML_SOTA" || exit 1
source "$HOME/miniconda3/bin/activate" dental
set -u
export PYTHONPATH="$PWD:$PYTHONPATH"
mkdir -p logs reports

stamp() { date '+%F %T'; }

finished() {
  python tools/run_finished.py --run "runs/segment/$1" --epochs "$2" >/dev/null 2>&1
}

echo "[$(stamp)] overnight chain starting"
echo "          PAUSE flag left in place on purpose; watchdog stays idle"

# ---------------------------------------------------------------- stage 1 ---
# C15 was interrupted at epoch 39/50 with optimizer state intact. Resuming
# continues in the same run directory to the same epoch target, and the trainer
# re-attaches the class-weight and boundary settings, so the criterion after
# resume is identical to before it.
if finished abl_C15 50; then
  echo "[$(stamp)] stage 1: abl_C15 already complete, skipping"
else
  # The original C15 flags are repeated in full. --data is a required argument
  # even when resuming, and ultralytics restores the interrupted run's own
  # settings from the checkpoint afterwards, so these are parsed and then
  # superseded rather than able to change the run mid-flight.
  echo "[$(stamp)] stage 1: resuming abl_C15 from last.pt"
  python yolov8_seg_longtail/train_seg.py \
      --data "$PWD/data_clean_15/data.yaml" --model yolov8x-seg.pt --nc 15 \
      --epochs 50 --imgsz 640 --batch 8 --seed 42 --cache ram \
      --channels-last --weights none --boundary-weight 0 --name abl_C15 \
      --resume runs/segment/abl_C15/weights/last.pt 2>&1 | tail -20
  if finished abl_C15 50; then
    echo "[$(stamp)] stage 1: abl_C15 finished 50 epochs"
  else
    echo "[$(stamp)] stage 1: abl_C15 did NOT reach 50 epochs."
    echo "          Continuing to the coefficient arms regardless -- they are the"
    echo "          experiments that matter, and C15 can be resumed again later."
  fi
fi

# ---------------------------------------------------------------- stage 2 ---
# K1 coefficient distillation, then K2 cv4 widening. run_coeff_arms.sh carries
# the reasoning for both, waits for the GPU to be free, guards completion before
# scoring, and runs the paired contour comparison against S0 on common cases.
echo "[$(stamp)] stage 2: coefficient-head arms (K1 then K2)"
./run_coeff_arms.sh

echo "[$(stamp)] overnight chain done"
echo
echo "Results to look at in the morning:"
echo "  reports/paired_contour_S0_abl_K1_coeffdistill_valid.md   <- the one that matters"
echo "  reports/paired_contour_S0_abl_K2_cv4wide_valid.md"
echo "  reports/eval_abl_K1_coeffdistill_valid.md"
echo "  reports/eval_abl_K2_cv4wide_valid.md"
