#!/usr/bin/env bash
# Relaunch the ablation queue if it dies. Both queues are resumable -- an arm
# whose report already exists is skipped -- so restarting never repeats work.
#
# Deliberately does NOT relaunch run_pipeline.sh: that would re-run the speed
# benchmark, which we concluded against (see reports/speed_bench.json).
cd "$HOME/Documents/ML_SOTA" || exit 0

# real work in flight? leave it alone
pgrep -f "train_seg.py|predict_to_coco.py|coco_eval_report.py" >/dev/null && exit 0
pgrep -f "run_seg_ablation" >/dev/null && exit 0

# everything finished?
[ -f reports/ablation_S4_valid_segm.json ] && exit 0

if [ ! -f reports/ablation_S2_valid_segm.json ]; then
  echo "[$(date '+%F %T')] queue 1 down, relaunching" >> logs/watchdog.log
  setsid nohup bash run_seg_ablation.sh >> logs/seg_ablation.log 2>&1 < /dev/null &
else
  echo "[$(date '+%F %T')] queue 2 down, relaunching" >> logs/watchdog.log
  setsid nohup bash run_seg_ablation2.sh >> logs/seg_ablation2.log 2>&1 < /dev/null &
fi
