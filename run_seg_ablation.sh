#!/usr/bin/env bash
# Project-2 ablation queue. Runs each arm to completion, scores it on VALID,
# and moves on. Designed to be launched while something else still owns the
# GPU: it waits for the GPU to go idle first, so it can be queued immediately
# behind the DINO baseline and the card never sits unused.
#
# Resumable: an arm whose report already exists is skipped, so the queue can be
# killed and relaunched without losing finished work.
#
# Arms (see yolov8_seg_longtail/ABLATION_PLAN.md):
#   S0   none                       50-epoch reference
#   S1a  beta=0.9                   weighting strength
#   S1b  beta=0.99                  weighting strength
#   S1c  invsqrt                    alternative weighting
#   S2   <best S1> + boundary
#   S3   <best S1> + copy-paste     (queued separately once data is generated)
#   S4   <best S1> + boundary + copy-paste
#
# Launch:
#   setsid nohup bash run_seg_ablation.sh > logs/seg_ablation.log 2>&1 </dev/null &
set -eo pipefail

ROOT="$HOME/Documents/ML_SOTA"
cd "$ROOT"
mkdir -p logs reports preds runs
source "$HOME/miniconda3/bin/activate" dental

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
  # Wait until the card is genuinely free. Checking compute-apps alone is not
  # enough: the DINO job releases the GPU between training and its prediction
  # export, and starting here in that gap would put two jobs on a 16 GB card.
  # So also wait for the whole DINO pipeline to exit.
  while true; do
    local busy=0
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]' && busy=1
    pgrep -f "run_dino_baseline.sh|main.py --output_dir|export_dino_preds" >/dev/null 2>&1 && busy=1
    [ "$busy" -eq 0 ] && break
    echo "[$(stamp)] GPU/pipeline busy, waiting..."
    sleep 300
  done
  sleep 30   # let VRAM actually drain before allocating
  echo "[$(stamp)] GPU free"
}

# run_arm <name> <weights-scheme> <boundary-weight>
run_arm() {
  local name="$1" wscheme="$2" bw="$3"
  local report="reports/ablation_${name}_valid_segm"
  if [ -f "${report}.json" ]; then
    echo "[$(stamp)] $name already scored, skipping"
    return 0
  fi

  step "arm $name  (weights=$wscheme  boundary=$bw)"
  python yolov8_seg_longtail/train_seg.py \
    --data "$DATA" --model "$MODEL" \
    --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" --seed "$SEED" \
    --weights "$wscheme" --boundary-weight "$bw" \
    --name "abl_${name}" 2>&1 | tail -25

  local W
  W=$(ls -t runs/*/abl_${name}/weights/best.pt runs/abl_${name}/weights/best.pt 2>/dev/null | head -1)
  if [ -z "$W" ]; then echo "[$(stamp)] $name: no weights produced, skipping scoring"; return 0; fi

  python yolov8_seg_longtail/predict_to_coco.py \
    --weights "$W" --gt "$GT" --images "$IMAGES" \
    --out "preds/ablation_${name}_valid.json" \
    --imgsz "$IMGSZ" --conf 0.001 --seed "$SEED" 2>&1 | tail -3

  python eval/coco_eval_report.py --gt "$GT" \
    --dt "preds/ablation_${name}_valid.json" --train-json "$TRAIN_JSON" \
    --iou-type segm --out "$report" 2>&1 | tail -4
}

step "0 waiting for the GPU"
wait_for_gpu
python - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("CUDA unavailable -- aborting")
print("GPU:", torch.cuda.get_device_name(0))
PY

run_arm S0  none     0
run_arm S1a 0.9      0
run_arm S1b 0.99     0
run_arm S1c invsqrt  0

step "S1 comparison (pick the best weighting before spending runs on S2/S4)"
python - <<'PY'
import json, glob, os
rows = []
for p in sorted(glob.glob("reports/ablation_S*_valid_segm.json")):
    d = json.load(open(p))
    rows.append((os.path.basename(p).split("_")[1], d["coco_stats"]["mAP"],
                 d["coco_stats"]["AP50"], d["group_AP"]))
print("%-6s %8s %8s %9s %9s %9s" % ("arm","mAP","AP50","head","mid","tail"))
for a, m, a50, g in rows:
    print("%-6s %8.4f %8.4f %9.4f %9.4f %9.4f"
          % (a, m, a50, g.get("head", 0), g.get("mid", 0), g.get("tail", 0)))
if rows:
    best = max(rows, key=lambda r: r[1])
    print("\nbest by mAP:", best[0])
    open("reports/_best_s1.txt", "w").write(best[0])
PY

BEST=$(cat reports/_best_s1.txt 2>/dev/null || echo "0.99")
case "$BEST" in
  S1a) WS=0.9 ;; S1b) WS=0.99 ;; S1c) WS=invsqrt ;; *) WS=0.99 ;;
esac
echo "carrying weighting scheme '$WS' into S2"

run_arm S2 "$WS" 0.5

step "DONE"
