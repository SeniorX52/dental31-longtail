#!/usr/bin/env bash
# Keep the whole delivery chain alive across crashes and reboots:
#
#   seg ablation (S0..S2) -> seg copy-paste arms (S3,S4) -> P2 final 100ep+test
#   -> DINO ablation (A0..A5,A2p)
#
# Every stage is resumable (finished work is skipped; partial training resumes
# from last.pt / checkpoint.pth), so relaunching is always safe. Stages are
# launched in pipeline order; each script itself waits for the GPU to free.
cd "$HOME/Documents/ML_SOTA" || exit 0
log() { echo "[$(date '+%F %T')] $*" >> logs/watchdog.log; }

# NIGHT WINDOW: only COLD-START new work between 21:00 and 08:00 (the owner uses
# the machine during the day). Work already running is never touched.
#
# CHAIN CONTINUATION: a stage that finishes at, say, 10:52 would otherwise leave
# the GPU idle until 21:00 -- ~10 h lost in the middle of a live chain. So if the
# pipeline produced output within CHAIN_WINDOW_H hours, treat it as in-flight and
# launch the next stage regardless of the hour. A genuinely cold machine (nothing
# produced for hours, e.g. the owner stopped everything to game) still waits for
# night. Override either way with:  bash watchdog.sh force
CHAIN_WINDOW_H=${CHAIN_WINDOW_H:-3}
if [ "$1" != "force" ]; then
  H=$(date +%H)
  if [ "$H" -ge 8 ] && [ "$H" -lt 21 ]; then
    # most recent pipeline artifact (reports or run results), in minutes ago
    recent=$(find reports runs -type f \( -name '*.json' -o -name 'results.csv' -o -name 'log.txt' \) \
               -newermt "-${CHAIN_WINDOW_H} hours" 2>/dev/null | head -1)
    if [ -z "$recent" ]; then
      exit 0            # cold: respect the night window
    fi
    log "daytime, but chain active within ${CHAIN_WINDOW_H}h -- continuing"
  fi
fi

# anything already running (or queued and waiting)? leave it alone
pgrep -f "train_seg.py|predict_to_coco|coco_eval_report|main.py --output_dir|export_dino_preds" >/dev/null && exit 0
pgrep -f "run_seg_ablation|final_seg_run|run_dino_ablation|final_dino_run" >/dev/null && exit 0

if [ ! -f reports/ablation_S2_valid_segm.json ]; then
  log "stage 1 (seg ablation) down, relaunching"
  setsid nohup bash run_seg_ablation.sh >> logs/seg_ablation.log 2>&1 < /dev/null &
elif [ ! -f reports/ablation_S4_valid_segm.json ]; then
  log "stage 2 (copy-paste arms) down, relaunching"
  setsid nohup bash run_seg_ablation2_v2.sh >> logs/seg_ablation2.log 2>&1 < /dev/null &
elif [ ! -f reports/final_seg_test_segm.json ]; then
  log "stage 3 (P2 final) down, relaunching"
  setsid nohup bash final_seg_run.sh >> logs/final_seg.log 2>&1 < /dev/null &
elif [ ! -f reports/dino_ablation_A2p_valid_bbox.json ]; then
  log "stage 4 (DINO ablation) down, relaunching"
  setsid nohup bash run_dino_ablation.sh >> logs/dino_ablation.log 2>&1 < /dev/null &
elif [ ! -f reports/final_dino_test_bbox.json ]; then
  log "stage 5 (P1 final test eval) down, relaunching"
  setsid nohup bash final_dino_run.sh >> logs/final_dino.log 2>&1 < /dev/null &
fi
