#!/usr/bin/env bash
# Second half of the Project-2 ablation: the copy-paste arms.
#
# Split from run_seg_ablation.sh deliberately -- bash reads a script lazily as
# it executes, so editing the first queue while it runs would corrupt it.
# This one waits for that queue to finish, reads which weighting scheme won,
# and runs the arms that need the augmented training set.
#
#   S3  <best weighting> + copy-paste
#   S4  <best weighting> + copy-paste + boundary
#
# Evaluation always uses the ORIGINAL valid split: data_clean_cp only augments
# train, so every arm is scored on identical, real data.
#
# Launch:
#   setsid nohup bash run_seg_ablation2.sh > logs/seg_ablation2.log 2>&1 </dev/null &
set -o pipefail   # NOT -e: a failed arm must not abort the queue

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
DATA="$ROOT/data_clean_cp/data.yaml"          # augmented TRAIN, original valid
GT="$ROOT/data_clean/annotations/instances_valid.json"
IMAGES="$ROOT/data_clean/valid/images"
TRAIN_JSON="$ROOT/data_clean/annotations/instances_train.json"

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
step()  { echo; echo "=== [$(stamp)] $* ==="; }

step "0 waiting for queue 1 and the GPU"
while pgrep -f "run_seg_ablation.sh|run_dino_baseline.sh|main.py --output_dir|train_seg.py|export_dino_preds" >/dev/null 2>&1 \
   || nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; do
  echo "[$(stamp)] busy, waiting..."
  sleep 300
done
sleep 30
echo "[$(stamp)] free"

BEST=$(cat reports/_best_s1.txt 2>/dev/null || echo "S1b")
case "$BEST" in
  S1a) WS=0.9 ;; S1c) WS=invsqrt ;; *) WS=0.99 ;;
esac
echo "using weighting scheme '$WS' (winner of the S1 sweep: $BEST)"

run_arm() {
  local name="$1" wscheme="$2" bw="$3"
  local report="reports/ablation_${name}_valid_segm"
  if [ -f "${report}.json" ]; then echo "[$(stamp)] $name done, skipping"; return 0; fi
  step "arm $name  (weights=$wscheme  boundary=$bw  +copy-paste)"
  python yolov8_seg_longtail/train_seg.py \
    --data "$DATA" --model "$MODEL" \
    --train-labels "$ROOT/data_clean_cp/train/labels" \
    --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" --seed "$SEED" \
    --weights "$wscheme" --boundary-weight "$bw" \
    ${EXTRA} \
    --name "abl_${name}" 2>&1 | tail -25
  local W
  W=$(ls -t runs/*/abl_${name}*/weights/best.pt 2>/dev/null | head -1)
  [ -z "$W" ] && { echo "no weights for $name"; return 0; }
  python yolov8_seg_longtail/predict_to_coco.py \
    --weights "$W" --gt "$GT" --images "$IMAGES" \
    --out "preds/ablation_${name}_valid.json" \
    --imgsz "$IMGSZ" --conf 0.001 --seed "$SEED" 2>&1 | tail -3
  python eval/coco_eval_report.py --gt "$GT" \
    --dt "preds/ablation_${name}_valid.json" --train-json "$TRAIN_JSON" \
    --iou-type segm --out "$report" 2>&1 | tail -4
}

run_arm S3 "$WS" 0
run_arm S4 "$WS" 0.5

step "full ablation table"
python - <<'PY'
import json, glob, os
rows = []
for p in sorted(glob.glob("reports/ablation_S*_valid_segm.json")):
    d = json.load(open(p))
    rows.append((os.path.basename(p).split("_")[1], d["coco_stats"]["mAP"],
                 d["coco_stats"]["AP50"], d["coco_stats"]["AP75"], d["group_AP"]))
print("%-6s %8s %8s %8s %9s %9s %9s" % ("arm","mAP","AP50","AP75","head","mid","tail"))
for a, m, a50, a75, g in rows:
    print("%-6s %8.4f %8.4f %8.4f %9.4f %9.4f %9.4f"
          % (a, m, a50, a75, g.get("head",0), g.get("mid",0), g.get("tail",0)))
PY

step "DONE"
