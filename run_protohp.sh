#!/usr/bin/env bash
# Retry the high-resolution prototype cell that the validator bug killed.
#
# abl_HR1280hp started on 10 Aug at 13:21 and died at 14:03 at its first
# validation, having trained without complaint. The driver read the first
# failure as an out-of-memory error and retried at batch 1, which hit the same
# wall for the same reason: ultralytics 8.4.108 hardcodes the input/4 prototype
# layout in its segment validator and ignores mask_ratio, so it compared ground
# truth at 328x176 against predictions at 656x352. yolov8_seg_longtail/
# proto_scale_patch.py parameterises both places by the stride actually in use.
#
# The cell itself is unchanged and still worth running. XP3 raised the prototype
# grid at 640 input and came out 1.30 pp WORSE, which established the law that
# detail must exist in the input before a higher-resolution head can represent
# it. At 1280 the input HAS the detail, so the same lever should now cut the
# other way. The paired contour test gives it a precise target: the 1280 model's
# masks overlap better than the baseline's (IoU separably +0.0048) while their
# contours sit worse (boundary F separably -0.0096), and contour placement is
# exactly what prototype resolution buys.
#
# It runs BEFORE run_bestof.sh so the combination arm can consider the result.
#
# Usage:  nohup ./run_protohp.sh > logs/protohp.log 2>&1 &

cd "$HOME/Documents/ML_SOTA" || exit 1
source "$HOME/miniconda3/bin/activate" dental
set -u
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs reports preds

TAG=abl_HR1280hp
CONF=0.15
stamp() { date '+%F %T'; }
finished() { python tools/run_finished.py "runs/segment/$1" >/dev/null 2>&1; }

# Identify other drivers by argv[1] from /proc, never `pgrep -f`: that pattern
# matches this script itself and has stalled this queue five times.
trainer_running() {
  local p a
  for p in $(pgrep -x python 2>/dev/null); do
    [ -r "/proc/$p/cmdline" ] || continue
    a=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    case "$a" in *train_seg.py*|*train_dental.py*|*tooth_stage2.py*|*predict_to_coco*) return 0 ;; esac
  done
  return 1
}
wait_on() {
  local w=0 busy p a1 s
  while :; do
    busy=0
    trainer_running && busy=1
    for p in $(pgrep -x bash 2>/dev/null); do
      [ "$p" = "$$" ] && continue
      [ -r "/proc/$p/cmdline" ] || continue
      a1=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | sed -n '2p')
      for s in "$@"; do case "$a1" in ./$s|*/$s) busy=1;; esac; done
    done
    [ "$busy" = 0 ] && break
    [ $((w % 1800)) -eq 0 ] && echo "[$(stamp)] waiting on the queue (${w}s)"
    sleep 120; w=$((w + 120))
  done
  sleep 60
  echo "[$(stamp)] queue clear after ${w}s"
}

echo "[$(stamp)] proto-scale retry queued"
wait_on run_final.sh run_extras.sh


# The failed attempt left a run directory with no results.csv. Leaving it there
# would make ultralytics resume from a checkpoint that does not exist.
if ! finished "$TAG" && [ ! -f "runs/segment/$TAG/weights/last.pt" ]; then
  [ -d "runs/segment/$TAG" ] && { echo "[$(stamp)] clearing the failed attempt"; rm -rf "runs/segment/$TAG"; }
fi

if finished "$TAG"; then
  echo "[$(stamp)] $TAG already complete"
else
  RESUME=()
  [ -f "runs/segment/$TAG/weights/last.pt" ] && RESUME=(--resume "runs/segment/$TAG/weights/last.pt")
  echo "[$(stamp)] === $TAG (S0 init, 1280, 25ep, --proto-scale 2) ==="
  python yolov8_seg_longtail/train_seg.py \
      --data "$PWD/data_clean/data.yaml" \
      --model runs/segment/abl_S0/weights/best.pt --nc 31 \
      --epochs 25 --imgsz 1280 --batch 2 --seed 42 \
      --channels-last --weights none --boundary-weight 0 \
      --proto-scale 2 --name "$TAG" "${RESUME[@]}" > "logs/${TAG}_train.log" 2>&1
  tail -8 "logs/${TAG}_train.log"
  # the input/2 prototype grid is 4x the activations of the stock head
  if ! finished "$TAG" && grep -qiE "out of memory" "logs/${TAG}_train.log"; then
    echo "[$(stamp)] $TAG OOM at batch 2; retrying at batch 1"
    rm -rf "runs/segment/$TAG"
    python yolov8_seg_longtail/train_seg.py \
        --data "$PWD/data_clean/data.yaml" \
        --model runs/segment/abl_S0/weights/best.pt --nc 31 \
        --epochs 25 --imgsz 1280 --batch 1 --seed 42 \
        --channels-last --weights none --boundary-weight 0 \
        --proto-scale 2 --name "$TAG" > "logs/${TAG}_train.log" 2>&1
    tail -8 "logs/${TAG}_train.log"
  fi
  # distinguish the two failure modes explicitly this time
  if ! finished "$TAG"; then
    echo "[$(stamp)] *** $TAG did NOT finish. Last error:"
    grep -iE 'error|exception|cannot be multiplied' "logs/${TAG}_train.log" | tail -3 | sed 's/^/      /'
  else
    echo "[$(stamp)] $TAG finished its schedule"
  fi
fi

if finished "$TAG"; then
  dt="preds/ablation_${TAG}_valid.json"
  [ -f "$dt" ] || python yolov8_seg_longtail/predict_to_coco.py \
      --weights "runs/segment/$TAG/weights/best.pt" \
      --gt data_clean/annotations/instances_valid.json \
      --images data_clean/valid/images --out "$dt" \
      --imgsz 1280 --conf 0.001 --seed 42 2>&1 | tail -2
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
      --dt "$dt" --train-json data_clean/annotations/instances_train.json \
      --iou-type segm --out "reports/eval_${TAG}_valid" 2>&1 | tail -3
  # boundary F is the metric this cell is aimed at, so the paired-on-intersection
  # comparison is the result, not the mAP
  PYTHONPATH="$PWD/eval:${PYTHONPATH:-}" python eval/paired_contour.py \
      --gt data_clean/annotations/instances_valid.json \
      --dt-a preds/ablation_abl_HR1280ft_valid.json --label-a HR1280ft \
      --dt-b "$dt" --label-b "$TAG" --conf "$CONF" --boot 500 \
      --out "reports/paired_contour_HR1280ft_${TAG}_valid" 2>&1 | tail -8
fi
echo "[$(stamp)] done"
