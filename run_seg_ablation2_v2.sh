#!/usr/bin/env bash
# Queue 2, v2 — copy-paste arms (S3, S4), crash-tolerant.
#
# v2 exists because the PC was shut down twice mid-S3 and each watchdog
# relaunch of the old queue retrained the arm from epoch 0 (35 epochs burned
# across two dead partials). This version:
#
#   * RESUMES the most-advanced partial run via ultralytics resume
#     (optimizer state + epoch counter restored) instead of starting fresh
#   * skips training entirely when a run already reached the epoch target
#   * picks weights by MOST COMPLETED EPOCHS, never by mtime
#   * scoring failures are non-fatal
#
# The original run_seg_ablation2.sh is left untouched because bash reads
# running scripts lazily — editing it while an instance executes corrupts that
# instance. The watchdog launches v2 from now on; an already-running old
# instance finishes its current arm harmlessly (reports are idempotent).
#
# Launch:
#   setsid nohup bash run_seg_ablation2_v2.sh > logs/seg_ablation2.log 2>&1 </dev/null &
set -o pipefail

ROOT="$HOME/Documents/ML_SOTA"
cd "$ROOT"
mkdir -p logs reports preds runs
source "$HOME/miniconda3/bin/activate" dental
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

EPOCHS=${EPOCHS:-50}
BATCH=${BATCH:-8}
IMGSZ=${IMGSZ:-640}
SEED=${SEED:-42}
MODEL=${MODEL:-yolov8x-seg.pt}
DATA="$ROOT/data_clean_cp/data.yaml"            # augmented TRAIN
GT="$ROOT/data_clean/annotations/instances_valid.json"   # ORIGINAL valid
IMAGES="$ROOT/data_clean/valid/images"
TRAIN_JSON="$ROOT/data_clean/annotations/instances_train.json"

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
step()  { echo; echo "=== [$(stamp)] $* ==="; }

# best_run <arm> -> "epochs<TAB>dir" of the most-advanced run directory
best_run() {
  local best="" bestn=-1 d n
  for d in runs/*/abl_${1} runs/*/abl_${1}-*; do
    [ -d "$d" ] || continue
    n=$(( $(wc -l < "$d/results.csv" 2>/dev/null || echo 1) - 1 ))
    [ "$n" -lt 0 ] && n=0
    if [ "$n" -gt "$bestn" ]; then bestn=$n; best="$d"; fi
  done
  [ -n "$best" ] && printf '%s\t%s\n' "$bestn" "$best"
}

run_arm() {
  local name="$1" wscheme="$2" bw="$3"
  local report="reports/ablation_${name}_valid_segm"
  [ -f "${report}.json" ] && { echo "[$(stamp)] $name already scored, skipping"; return 0; }

  local info n dir
  info=$(best_run "$name")
  n=$(echo "$info" | cut -f1); dir=$(echo "$info" | cut -f2)

  if [ -n "$dir" ] && [ "${n:-0}" -ge "$EPOCHS" ]; then
    echo "[$(stamp)] $name already trained ($dir, $n ep) -- scoring only"
  elif [ -n "$dir" ] && [ -f "$dir/weights/last.pt" ] && [ "${n:-0}" -ge 2 ]; then
    step "arm $name — RESUMING $dir from epoch $n"
    python yolov8_seg_longtail/train_seg.py \
      --data "$DATA" --model "$MODEL" \
      --train-labels "$ROOT/data_clean_cp/train/labels" \
      --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" --seed "$SEED" \
      --weights "$wscheme" --boundary-weight "$bw" \
      --resume "$dir/weights/last.pt" \
      --name "abl_${name}" 2>&1 | tail -15
  else
    step "arm $name  (weights=$wscheme  boundary=$bw  +copy-paste)"
    python yolov8_seg_longtail/train_seg.py \
      --data "$DATA" --model "$MODEL" \
      --train-labels "$ROOT/data_clean_cp/train/labels" \
      --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" --seed "$SEED" \
      --weights "$wscheme" --boundary-weight "$bw" \
      ${EXTRA} --name "abl_${name}" 2>&1 | tail -15
  fi

  info=$(best_run "$name"); dir=$(echo "$info" | cut -f2)
  local W="$dir/weights/best.pt"
  [ -f "$W" ] || { echo "[$(stamp)] $name: no weights, skipping scoring"; return 0; }

  step "score $name ($W)"
  if ! python yolov8_seg_longtail/predict_to_coco.py \
        --weights "$W" --gt "$GT" --images "$IMAGES" \
        --out "preds/ablation_${name}_valid.json" \
        --imgsz "$IMGSZ" --conf 0.001 --seed "$SEED" 2>&1 | tail -3; then
    echo "[$(stamp)] $name: export failed, continuing"; return 0
  fi
  python eval/coco_eval_report.py --gt "$GT" \
    --dt "preds/ablation_${name}_valid.json" --train-json "$TRAIN_JSON" \
    --iou-type segm --out "$report" 2>&1 | tail -4 \
    || echo "[$(stamp)] $name: eval failed, continuing"
}

step "waiting for queue 1 / GPU"
while pgrep -f "run_seg_ablation.sh|run_seg_ablation2.sh|train_seg.py|predict_to_coco" >/dev/null 2>&1 \
   || nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; do
  sleep 300
done
sleep 20

BEST=$(cat reports/_best_s1.txt 2>/dev/null || echo S1c)
case "$BEST" in S1a) WS=0.9 ;; S1b) WS=0.99 ;; *) WS=invsqrt ;; esac
echo "weighting scheme from S1 sweep: $WS"

run_arm S3 "$WS" 0
run_arm S4 "$WS" 0.5

step "full table"
python3 _internal/update_results.py 2>/dev/null || true
step "QUEUE 2 (v2) DONE"
