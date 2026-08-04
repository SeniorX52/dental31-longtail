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

# PAUSE SWITCH. While this file exists nothing is started, so the machine can
# be shut down without the watchdog relaunching a job seconds later. Every
# stage resumes from its own last checkpoint, so pausing costs at most the
# epoch in progress.
#   pause:   touch PAUSE
#   resume:  rm PAUSE
if [ -f PAUSE ]; then
  exit 0
fi

# NIGHT WINDOW: only COLD-START new work between 21:00 and 08:00, since the
# workstation is in interactive use during the day. Work already running is
# never touched.
#
# CHAIN CONTINUATION: a stage that finishes at, say, 10:52 would otherwise leave
# the GPU idle until 21:00 -- ~10 h lost in the middle of a live chain. So if the
# pipeline produced output within CHAIN_WINDOW_H hours, treat it as in-flight and
# launch the next stage regardless of the hour. A genuinely cold machine (nothing
# produced for hours, because the queue was stopped by hand) still waits for
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
pgrep -f "run_seg_ablation|final_seg_run|run_dino_ablation|final_dino_run|run_seg_completion" >/dev/null && exit 0

if [ ! -f reports/ablation_S2_valid_segm.json ]; then
  log "stage 1 (seg ablation) down, relaunching"
  setsid nohup bash run_seg_ablation.sh >> logs/seg_ablation.log 2>&1 < /dev/null &
elif [ ! -f reports/ablation_S4_valid_segm.json ]; then
  log "stage 2 (copy-paste arms) down, relaunching"
  setsid nohup bash run_seg_ablation2_v2.sh >> logs/seg_ablation2.log 2>&1 < /dev/null &
elif [ ! -f reports/final_seg_test_segm.json ]; then
  log "stage 3 (P2 final) down, relaunching"
  setsid nohup bash final_seg_run.sh >> logs/final_seg.log 2>&1 < /dev/null &
elif [ ! -f reports/dino_abl_C1_valid_bbox.json ]; then
  # stage 4 = the FROZEN detection matrix D1..D7 + the class-balanced control
  # C1. Gated on C1 rather than on the last arm in the script so the final test
  # evaluation can start as soon as the deliverable matrix is complete; the
  # leave-one-out arms L1..L3 are picked up afterwards by stage 6.
  log "stage 4 (DINO frozen matrix D1-D7,C1) down, relaunching"
  setsid nohup bash run_dino_ablation_v2.sh >> logs/dino_ablation_v2.log 2>&1 < /dev/null &
elif [ ! -f reports/final_dino_test_bbox.json ]; then
  log "stage 5 (P1 final test eval) down, relaunching"
  setsid nohup bash final_dino_run.sh >> logs/final_dino.log 2>&1 < /dev/null &
elif [ ! -f reports/ablation_S2_s2024_valid_segm.json ]; then
  # stage 6 = the segmentation paper's remaining evidence, in priority order
  # inside the script: the isolated boundary cell, the published comparators
  # (soft Dice, Kervadec, Tversky, Focal Tversky), the leave-one-out cells,
  # cross-backbone transfer, then seed replicates. Ranked above the detection
  # leave-one-out arms because beating stock BCE is a necessary control but not
  # a claim -- without the comparators there is no segmentation paper.
  log "stage 6 (seg comparators, cross-backbone, seed replicates) down, relaunching"
  setsid nohup bash run_seg_completion.sh >> logs/seg_completion.log 2>&1 < /dev/null &
elif [ ! -f reports/dino_abl_L3_valid_bbox.json ]; then
  # stage 7 = detection leave-one-out complement. Same script; run_arm skips
  # every arm that already has a report, so this resumes at L1. Last because
  # D2/D3/D4 already give the individual-component readings.
  log "stage 7 (DINO leave-one-out L1-L3) down, relaunching"
  setsid nohup bash run_dino_ablation_v2.sh >> logs/dino_ablation_v2.log 2>&1 < /dev/null &
fi
