#!/usr/bin/env bash
# Relaunch the pipeline if it is not running. The queues are resumable (any arm
# whose report exists is skipped), so restarting is always safe and never
# repeats finished work.
cd "$HOME/Documents/ML_SOTA" || exit 0
# already alive? nothing to do
pgrep -f "run_pipeline.sh" >/dev/null && exit 0
# real work in flight (started by the pipeline) counts as alive
pgrep -f "main.py --output_dir|train_seg.py|bench_train_speed|export_dino_preds" >/dev/null && exit 0
# everything finished? stop relaunching
[ -f reports/ablation_S4_valid_segm.json ] && exit 0
echo "[$(date '+%F %T')] pipeline down, relaunching" >> logs/watchdog.log
setsid nohup bash run_pipeline.sh >> logs/pipeline.log 2>&1 < /dev/null &
