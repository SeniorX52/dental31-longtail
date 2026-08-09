#!/usr/bin/env bash
# Native-resolution probe: does the resolution lever keep paying, or has it saturated?
#
# WHY. The 1280 probe is the first thing in this project that worked and whose
# mechanism survives scrutiny. Against the client's recipe reproduced on the
# clean split it gives +1.50 pp segm mAP (+14.2 % relative), and critically the
# gain is NOT detection-side: box mAP moved +0.17 pp against a +-0.75 pp noise
# floor, so the improvement has nowhere to come from except the masks. The
# AP50/AP75 split confirms it, +6.9 % at loose IoU against +47.4 % at strict
# IoU, which is what better localisation looks like and not what a ranking
# artefact looks like. On the classes that motivated the probe: caries +35 %,
# periapical lesion +29 %, root canal treatment +40 %.
#
# What that measures is one point on a curve. This run measures the second.
# The scans are natively 1615x840, so 1600 is essentially the physical ceiling:
# beyond it we would be upsampling every image and inventing detail. If 1600
# beats 1280 the lever is still paying and the native ceiling is the target; if
# it matches or falls back, 1280 has already captured the available detail and
# further resolution is wasted GPU. Either answer redirects the next week.
#
# BATCH 1, AND WHY THAT IS NOT A SECOND VARIABLE. At 1280 the probe peaked at
# 13.4 GB of 16.3 with batch 2. Pixels scale with the square of the side, so
# 1600 at batch 2 would need about 20.9 GB and cannot run on this card. At
# batch 1 it needs about 11.9 GB and fits. Ultralytics holds the effective
# optimizer batch at nbs=64 by accumulating gradients, so halving the loader
# batch doubles the accumulation and leaves the optimizer step identical; what
# actually changes is batch-norm statistics and wall-clock. That is a real but
# second-order difference and it is recorded here rather than hidden.
#
# TWENTY EPOCHS, NOT TWENTY-FIVE. The 1280 probe reached its best checkpoint at
# epoch 14 of 25 and never beat it afterwards, while its training loss kept
# falling at 0.020 per epoch: the corpus overfits a 71 M-parameter model long
# before the schedule ends. The baseline shows the same shape, peaking at epoch
# 26 of 50 and then LOSING 1.68 pp by epoch 50. Twenty epochs clears the
# observed peak with margin, and both arms are compared on best.pt, so the
# schedule length does not enter the comparison.
#
# Usage:  nohup ./run_hr1600.sh > logs/hr1600.log 2>&1 &

cd "$HOME/Documents/ML_SOTA" || exit 1
source "$HOME/miniconda3/bin/activate" dental
set -u
export PYTHONPATH="$PWD:$PYTHONPATH"
mkdir -p logs reports preds

TAG=abl_HR1600ft
IMGSZ=1600
EPOCHS=20
CONF=0.15
stamp() { date '+%F %T'; }
finished() { python tools/run_finished.py "runs/segment/$1" >/dev/null 2>&1; }

trainer_running() {
  local p a
  for p in $(pgrep -x python 2>/dev/null); do
    [ -r "/proc/$p/cmdline" ] || continue
    a=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    case "$a" in *train_seg.py*|*train_dental.py*|*predict_to_coco*) return 0 ;; esac
  done
  return 1
}
# argv[1] test: `pgrep -f` also matches any shell that merely names the script,
# a self-match that has produced five wrong readings in this project.
earlier_driver_running() {
  local p a1
  for p in $(pgrep -x bash 2>/dev/null); do
    [ -r "/proc/$p/cmdline" ] || continue
    a1=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | sed -n '2p')
    case "$a1" in ./run_maskdino.sh|*/run_maskdino.sh) return 0 ;; esac
  done
  return 1
}

echo "[$(stamp)] native-resolution probe queued behind Mask DINO"
w=0
while trainer_running || earlier_driver_running; do
  [ $((w % 1800)) -eq 0 ] && echo "[$(stamp)] waiting (${w}s)"
  sleep 120; w=$((w + 120))
done
sleep 60
echo "[$(stamp)] GPU free after ${w}s"

# The 11.9 GB estimate for 1600 at batch 1 is extrapolated from the 13.4 GB
# measured at 1280 batch 2, not observed, and the card has 16.3 GB. If it is
# wrong the run dies and a seven-hour slot is wasted unattended, so the launch
# falls back one step to 1440 (about 10.0 GB by the same estimate) and only
# gives up if that fails too. 1440 is still well above the 1280 point and still
# answers the question the probe is asking.
run_at() {              # $1 = imgsz
  local sz="$1" RESUME=()
  [ -f "runs/segment/$TAG/weights/last.pt" ] && \
    RESUME=(--resume "runs/segment/$TAG/weights/last.pt")
  echo "[$(stamp)] === $TAG: S0 weights, imgsz $sz, $EPOCHS epochs, batch 1 ==="
  python yolov8_seg_longtail/train_seg.py \
      --data "$PWD/data_clean/data.yaml" \
      --model runs/segment/abl_S0/weights/best.pt --nc 31 \
      --epochs "$EPOCHS" --imgsz "$sz" --batch 1 --seed 42 \
      --channels-last --weights none --boundary-weight 0 \
      --name "$TAG" "${RESUME[@]}" > "logs/${TAG}_train.log" 2>&1
  tail -12 "logs/${TAG}_train.log"
}

if ! finished "$TAG"; then
  run_at "$IMGSZ"
  if ! finished "$TAG" && grep -qiE "out of memory|CUDA out of memory" "logs/${TAG}_train.log" 2>/dev/null; then
    echo "[$(stamp)] OOM at $IMGSZ; falling back to 1440"
    rm -rf "runs/segment/$TAG"
    IMGSZ=1440
    run_at "$IMGSZ"
  fi
fi

if finished "$TAG"; then
  echo "[$(stamp)] scoring $TAG at its own resolution"
  dt="preds/ablation_${TAG}_valid.json"
  [ -f "$dt" ] || python yolov8_seg_longtail/predict_to_coco.py \
      --weights "runs/segment/$TAG/weights/best.pt" \
      --gt data_clean/annotations/instances_valid.json \
      --images data_clean/valid/images --out "$dt" \
      --imgsz "$IMGSZ" --conf 0.001 --seed 42 2>&1 | tail -3
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
      --dt "$dt" --train-json data_clean/annotations/instances_train.json \
      --iou-type segm --out "reports/eval_${TAG}_valid" 2>&1 | tail -4
  # the box control: if the gain is real it must NOT show up here
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
      --dt "$dt" --train-json data_clean/annotations/instances_train.json \
      --iou-type bbox --out "reports/bboxchk_${TAG}_valid" 2>&1 | tail -3
  PYTHONPATH="$PWD/eval:$PYTHONPATH" python eval/paired_contour.py \
      --gt data_clean/annotations/instances_valid.json \
      --dt-a preds/ablation_S0_valid.json --label-a S0 \
      --dt-b "$dt" --label-b "$TAG" --conf "$CONF" --boot 500 \
      --out "reports/paired_contour_S0_${TAG}_valid" 2>&1 | tail -8

  echo "[$(stamp)] === the resolution curve ==="
  python - <<'PY'
import json, os
pts = [("640  baseline", "reports/ablation_S0_valid_segm.json"),
       ("1280 probe",    "reports/eval_abl_HR1280ft_valid.json"),
       ("1600 native",   "reports/eval_abl_HR1600ft_valid.json")]
base = None
for lab, f in pts:
    if not os.path.exists(f):
        print("  %-14s not scored" % lab); continue
    s = json.load(open(f))["coco_stats"]
    if base is None: base = s["mAP"]
    print("  %-14s segm mAP %.4f (%+.2f pp)  AP75 %.4f" % (lab, s["mAP"], (s["mAP"]-base)*100, s["AP75"]))
a = "reports/eval_abl_HR1280ft_valid.json"; b = "reports/eval_abl_HR1600ft_valid.json"
if os.path.exists(a) and os.path.exists(b):
    d = (json.load(open(b))["coco_stats"]["mAP"] - json.load(open(a))["coco_stats"]["mAP"]) * 100
    print("  1280 -> 1600: %+.2f pp" % d)
    print("  reading: clearly positive means the lever is still paying and native")
    print("  resolution is the target; flat or negative means 1280 already captured")
    print("  the available detail and further resolution is wasted GPU.")
PY
else
  echo "[$(stamp)] *** $TAG did not finish; not scoring"
fi
echo "[$(stamp)] done"
