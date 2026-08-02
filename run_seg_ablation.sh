#!/usr/bin/env bash
# Project-2 ablation queue (arms S0..S2). See yolov8_seg_longtail/ABLATION_PLAN.md
#
# Robustness rules, each learned from a failure that cost ~3 h of GPU time:
#
#  * PYTHONPATH must include the repo root. predict_to_coco.py imports
#    yolov8_seg_longtail.train_seg to resolve the model classes pickled inside
#    each checkpoint; without it the weights cannot be loaded at all.
#  * An arm that already has weights is NEVER retrained. Previously a failure
#    in the scoring step killed the queue, the watchdog relaunched it, and a
#    finished 50-epoch arm was trained again from zero.
#  * Scoring failures are non-fatal. One unscoreable arm must not take the
#    whole queue down with it.
#  * The weights glob tolerates ultralytics' "-2" suffixes (abl_S0-2), which
#    appear whenever a run directory already exists.
#
# Launch:
#   EXTRA="--cache ram --channels-last" \
#   setsid nohup bash run_seg_ablation.sh > logs/seg_ablation.log 2>&1 </dev/null &
set -o pipefail          # NOT -e: a failed arm must not abort the queue

ROOT="$HOME/Documents/ML_SOTA"
cd "$ROOT"
mkdir -p logs reports preds runs
source "$HOME/miniconda3/bin/activate" dental
export PYTHONPATH="$ROOT:$PYTHONPATH"

EPOCHS=${EPOCHS:-50}
BATCH=${BATCH:-8}
IMGSZ=${IMGSZ:-640}
SEED=${SEED:-42}
MODEL=${MODEL:-yolov8x-seg.pt}
DATA="$ROOT/data_clean/data.yaml"
GT="$ROOT/data_clean/annotations/instances_valid.json"
IMAGES="$ROOT/data_clean/valid/images"
TRAIN_JSON="$ROOT/data_clean/annotations/instances_train.json"

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
step()  { echo; echo "=== [$(stamp)] $* ==="; }

wait_for_gpu() {
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; do
    echo "[$(stamp)] GPU busy, waiting..."; sleep 300
  done
  sleep 20
}

# Pick the run with the MOST COMPLETED EPOCHS, not the newest. When an arm is
# retrained after a failure ultralytics creates abl_<name>-2, and that partial
# rerun is *newer* than the finished original -- selecting by mtime would
# silently put an undertrained model into the ablation table.
find_weights() {
  local best="" bestn=-1 d n
  for d in runs/*/abl_${1} runs/*/abl_${1}-*; do
    [ -f "$d/weights/best.pt" ] || continue
    n=$(( $(wc -l < "$d/results.csv" 2>/dev/null || echo 1) - 1 ))
    if [ "$n" -gt "$bestn" ]; then bestn=$n; best="$d/weights/best.pt"; fi
  done
  [ -n "$best" ] && echo "$best"
}

# run_arm <name> <weights-scheme> <boundary-weight>
run_arm() {
  local name="$1" wscheme="$2" bw="$3"
  local report="reports/ablation_${name}_valid_segm"

  if [ -f "${report}.json" ]; then
    echo "[$(stamp)] $name already scored, skipping"; return 0
  fi

  local W
  W=$(find_weights "$name")
  if [ -n "$W" ]; then
    echo "[$(stamp)] $name already trained ($W) -- scoring only"
  else
    step "arm $name  (weights=$wscheme  boundary=$bw)"
    python yolov8_seg_longtail/train_seg.py \
      --data "$DATA" --model "$MODEL" \
      --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" --seed "$SEED" \
      --weights "$wscheme" --boundary-weight "$bw" \
      ${EXTRA} --name "abl_${name}" 2>&1 | tail -20
    W=$(find_weights "$name")
    [ -z "$W" ] && { echo "[$(stamp)] $name: training produced no weights"; return 0; }
  fi

  step "score $name"
  if ! python yolov8_seg_longtail/predict_to_coco.py \
        --weights "$W" --gt "$GT" --images "$IMAGES" \
        --out "preds/ablation_${name}_valid.json" \
        --imgsz "$IMGSZ" --conf 0.001 --seed "$SEED" 2>&1 | tail -3; then
    echo "[$(stamp)] $name: export failed, continuing"; return 0
  fi
  if ! python eval/coco_eval_report.py --gt "$GT" \
        --dt "preds/ablation_${name}_valid.json" --train-json "$TRAIN_JSON" \
        --iou-type segm --out "$report" 2>&1 | tail -4; then
    echo "[$(stamp)] $name: eval failed, continuing"
  fi
}

step "waiting for the GPU"
wait_for_gpu
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA unavailable')" \
  || { echo "CUDA unavailable, aborting"; exit 1; }

run_arm S0  none     0
run_arm S1a 0.9      0
run_arm S1b 0.99     0
run_arm S1c invsqrt  0

step "S1 sweep result"
python - <<'PY'
import json, glob, os
rows = []
for p in sorted(glob.glob("reports/ablation_S1*_valid_segm.json")):
    d = json.load(open(p))
    rows.append((os.path.basename(p).split("_")[1], d["coco_stats"]["mAP"], d["group_AP"]))
print("%-6s %8s %9s %9s %9s" % ("arm","mAP","head","mid","tail"))
for a, m, g in rows:
    print("%-6s %8.4f %9.4f %9.4f %9.4f"
          % (a, m, g.get("head",0), g.get("mid",0), g.get("tail",0)))
if rows:
    best = max(rows, key=lambda r: r[1])
    open("reports/_best_s1.txt","w").write(best[0])
    print("\nbest by mAP:", best[0])
PY

BEST=$(cat reports/_best_s1.txt 2>/dev/null || echo S1b)
case "$BEST" in S1a) WS=0.9 ;; S1c) WS=invsqrt ;; *) WS=0.99 ;; esac
echo "carrying '$WS' into S2"
run_arm S2 "$WS" 0.5

step "QUEUE 1 DONE"
