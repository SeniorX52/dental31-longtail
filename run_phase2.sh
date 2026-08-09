#!/usr/bin/env bash
# Phase 2: the learning curve, then a second architecture generation.
#
# WHY THE LEARNING CURVE IS FIRST. Two unrelated architectures land within
# 0.7 pp of each other on this corpus (DINO-DETR 0.1625, YOLOv8x box 0.1551)
# where the same families are several AP apart on COCO. Convergence across
# families is the signature of a DATA-limited regime, but it is circumstantial;
# a learning curve measures it. Train the identical S0 configuration on 25, 50
# and 75 percent of the training images and read the slope against the 100
# percent point we already hold (abl_S0, 0.1055). If the curve has flattened by
# 100 percent, more data will not help and the ceiling is the labels or the
# task, which the 911 universally-missed pathology annotations already suggest.
# If it is still climbing, the honest answer to "what beats the baseline" is
# more data, with a measured slope attached. Either result is decisive, and at
# 2.9 h per full run the three points cost about 4.4 h in total because a
# quarter of the data trains in roughly a quarter of the time.
#
# Subsets are stratified on the rarest class each image carries, so every
# fraction keeps all 31 classes present and the experiment does not confound
# data quantity with class coverage. Validation and test are the untouched
# data_clean directories in every case.
#
# THEN a second architecture generation. yolo11x-seg is three years newer than
# yolov8x-seg and is a drop-in under the same trainer, verified on CPU before
# queueing: it builds, loads 1071/1077 pretrained tensors and produces a finite
# loss through our criterion. It answers whether architecture generation buys
# anything here at all, which no arm so far has tested.
#
# WAITING WITHOUT RACING. Two drivers are already in flight, and a third that
# merely waited for a free GPU would start the instant the 1280 probe exits,
# colliding with the Mask DINO driver that is waiting for exactly the same
# signal. So this script waits for BOTH conditions: no python owning a training
# script, AND no earlier driver process still alive. The driver check reads
# argv[1] rather than matching the whole command line, because `pgrep -f` also
# matches any diagnostic shell that merely mentions the script name; that
# self-match has produced four false "still running" readings in this project,
# the last of which cost twelve hours of idle GPU.
#
# Usage:  nohup ./run_phase2.sh > logs/phase2.log 2>&1 &

cd "$HOME/Documents/ML_SOTA" || exit 1
source "$HOME/miniconda3/bin/activate" dental
set -u
export PYTHONPATH="$PWD:$PYTHONPATH"
mkdir -p logs reports preds

stamp() { date '+%F %T'; }
finished() { python tools/run_finished.py "runs/segment/$1" >/dev/null 2>&1; }

# a python process that owns one of our training scripts
trainer_running() {
  local p a
  for p in $(pgrep -x python 2>/dev/null); do
    [ -r "/proc/$p/cmdline" ] || continue
    a=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    case "$a" in
      *train_seg.py*|*train_dental.py*|*predict_to_coco*) return 0 ;;
    esac
  done
  return 1
}

# an EARLIER queue driver, identified by argv[1] so a diagnostic shell that
# merely mentions the filename cannot masquerade as one
earlier_driver_running() {
  local p a1
  for p in $(pgrep -x bash 2>/dev/null); do
    [ -r "/proc/$p/cmdline" ] || continue
    a1=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | sed -n '2p')
    case "$a1" in
      ./run_hr1280.sh|*/run_hr1280.sh|./run_hr1600.sh|*/run_hr1600.sh|\
      ./run_compound.sh|*/run_compound.sh|\
      ./run_maskdino.sh|*/run_maskdino.sh) return 0 ;;
    esac
  done
  return 1
}

wait_turn() {
  local w=0
  while trainer_running || earlier_driver_running; do
    [ $((w % 1800)) -eq 0 ] && echo "[$(stamp)] waiting for the queue ahead (${w}s)"
    sleep 120; w=$((w + 120))
  done
  sleep 60
  echo "[$(stamp)] queue clear after ${w}s"
}

# Held identical to abl_S0 except for the one variable under test, so the
# comparison against its 0.1055 is clean.
train_arm() {           # $1=tag $2=data.yaml $3=weights [$4=imgsz $5=batch $6=epochs]
  local tag="$1" data="$2" model="$3" sz="${4:-640}" bs="${5:-8}" ep="${6:-50}"
  if finished "$tag"; then echo "[$(stamp)] $tag already complete"; return 0; fi
  local RESUME=()
  [ -f "runs/segment/$tag/weights/last.pt" ] && \
    RESUME=(--resume "runs/segment/$tag/weights/last.pt")
  echo "[$(stamp)] === $tag  (data $data, model $model) ==="
  local CACHE=(--cache ram)
  [ "$sz" -gt 640 ] && CACHE=()      # a 1280 RAM cache does not fit
  python yolov8_seg_longtail/train_seg.py \
      --data "$data" --model "$model" --nc 31 \
      --epochs "$ep" --imgsz "$sz" --batch "$bs" --seed 42 "${CACHE[@]}" \
      --channels-last --weights none --boundary-weight 0 \
      --name "$tag" "${RESUME[@]}" 2>&1 | tail -15
  finished "$tag" && echo "[$(stamp)] $tag finished" \
                  || echo "[$(stamp)] *** $tag did NOT finish"
}

score() {
  local tag="$1" dt="preds/ablation_$1_valid.json"
  finished "$tag" || { echo "[$(stamp)] $tag unfinished, not scoring"; return 0; }
  [ -f "$dt" ] || python yolov8_seg_longtail/predict_to_coco.py \
      --weights "runs/segment/$tag/weights/best.pt" \
      --gt data_clean/annotations/instances_valid.json \
      --images data_clean/valid/images --out "$dt" \
      --imgsz "${2:-640}" --conf 0.001 --seed 42 2>&1 | tail -2
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
      --dt "$dt" --train-json data_clean/annotations/instances_train.json \
      --iou-type segm --out "reports/eval_${tag}_valid" 2>&1 | tail -3
}

echo "[$(stamp)] phase 2 queued behind the 1280 probe and Mask DINO"
wait_turn

for f in 25 50 75; do
  train_arm "abl_LC${f}" "$PWD/data_frac${f}/data.yaml" yolov8x-seg.pt
  score     "abl_LC${f}"
done

# Moved from 640 to 1280. Testing a newer architecture at the resolution we
# have just shown handicaps the model would measure the handicap rather than the
# architecture: the same backbone gained +14.2 % relative going 640 -> 1280.
# Thirty epochs because every arm here peaks by epoch 26 and best.pt is compared;
# batch 2 because 1280 does not fit at batch 8.
train_arm abl_YOLO11x_1280 "$PWD/data_clean/data.yaml" yolo11x-seg.pt 1280 2 30
score     abl_YOLO11x_1280 1280

echo "[$(stamp)] === learning curve ==="
python - <<'PY'
import json, os
pts = [("25%", "reports/eval_abl_LC25_valid.json"),
       ("50%", "reports/eval_abl_LC50_valid.json"),
       ("75%", "reports/eval_abl_LC75_valid.json"),
       ("100%", "reports/ablation_S0_valid_segm.json")]
xs, ys = [], []
for lab, f in pts:
    if os.path.exists(f):
        m = json.load(open(f))["coco_stats"]["mAP"]
        xs.append(lab); ys.append(m)
        print("  %-5s segm mAP %.4f" % (lab, m))
if len(ys) >= 2:
    slope = (ys[-1] - ys[-2]) * 100
    print("  slope over the last step: %+.2f pp" % slope)
    print("  reading: a slope near zero means more data will not help and the")
    print("  ceiling is the labels or the task; a clearly positive slope means")
    print("  more data is the lever, and this is its measured size.")
PY
echo "[$(stamp)] done"
