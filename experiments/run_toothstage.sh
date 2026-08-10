#!/usr/bin/env bash
# PRIORITY 3: the tooth-conditioned two-stage pipeline, v1.
#
# The domain-correct inductive bias: a finding is a property of a TOOTH
# (DENTEX hierarchy, arXiv:2305.19112; HierarchicalDet, arXiv:2303.06500).
# Stage one, a single-class tooth detector trained on the DENTEX enumeration
# sets (yolov8s, deliberately small per law L4). Stage two, a ResNet-18
# multi-label head over tooth crops, supervised by mapping our lesion labels
# to their enclosing tooth, which converts unreliable outlines into reliable
# tooth-level bits and outputs the granularity the external check already
# scores at 73.6 to 84.1 percent precision.
cd "$HOME/Documents/ML_SOTA" || exit 1
source "$HOME/miniconda3/bin/activate" dental
set -u
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs reports preds weights
stamp() { date '+%F %T'; }
finished() { python tools/run_finished.py "runs/segment/$1" >/dev/null 2>&1; }
trainer_running() {
  local p a
  for p in $(pgrep -x python 2>/dev/null); do
    [ -r "/proc/$p/cmdline" ] || continue
    a=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    case "$a" in *train_seg.py*|*train_dental.py*|*predict_to_coco*|*tooth_stage2.py*) return 0 ;; esac
  done
  return 1
}
wait_on() {  # names of earlier driver scripts as args
  local w=0 busy p a1 s
  while :; do
    busy=0
    trainer_running && busy=1
    for p in $(pgrep -x bash 2>/dev/null); do
      [ -r "/proc/$p/cmdline" ] || continue
      a1=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | sed -n '2p')
      for s in "$@"; do case "$a1" in ./$s|*/$s) busy=1;; esac; done
    done
    [ "$busy" = 0 ] && break
    [ $((w % 1800)) -eq 0 ] && echo "[$(stamp)] waiting on queue ahead (${w}s)"
    sleep 120; w=$((w + 120))
  done
  sleep 60
  echo "[$(stamp)] queue clear after ${w}s"
}
train_ft() {  # tag init imgsz batch epochs seed data extra...
  local tag="$1" init="$2" sz="$3" bs="$4" ep="$5" seed="$6" data="$7"; shift 7
  if finished "$tag"; then echo "[$(stamp)] $tag already complete"; return 0; fi
  local RESUME=()
  [ -f "runs/segment/$tag/weights/last.pt" ] && RESUME=(--resume "runs/segment/$tag/weights/last.pt")
  echo "[$(stamp)] === $tag (init $(basename "$init"), $sz px, ${ep}ep, seed $seed, $*) ==="
  python yolov8_seg_longtail/train_seg.py \
      --data "$data" --model "$init" --nc 31 \
      --epochs "$ep" --imgsz "$sz" --batch "$bs" --seed "$seed" \
      --channels-last --weights none --boundary-weight 0 \
      --name "$tag" "$@" "${RESUME[@]}" > "logs/${tag}_train.log" 2>&1
  tail -6 "logs/${tag}_train.log"
  finished "$tag" && echo "[$(stamp)] $tag finished" || echo "[$(stamp)] *** $tag did NOT finish"
}
score() {  # tag imgsz
  local tag="$1" sz="$2" dt="preds/ablation_$1_valid.json"
  finished "$tag" || return 0
  [ -f "$dt" ] || python yolov8_seg_longtail/predict_to_coco.py \
      --weights "runs/segment/$tag/weights/best.pt" \
      --gt data_clean/annotations/instances_valid.json \
      --images data_clean/valid/images --out "$dt" \
      --imgsz "$sz" --conf 0.001 --seed 42 2>&1 | tail -2
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
      --dt "$dt" --train-json data_clean/annotations/instances_train.json \
      --iou-type segm --out "reports/eval_${tag}_valid" 2>&1 | tail -3
}
echo "[$(stamp)] tooth-stage driver queued"
wait_on run_maskdino.sh run_hr1600.sh run_compound.sh run_phase2.sh run_k2seeds.sh run_labelnoise.sh run_selftrain.sh

[ -d data_tooth/train ] || python tools/build_tooth_corpus.py \
    --dentex /media/mostafa/EGYPT_SSD/dental31/dentex/extracted --out data_tooth

if ! finished tooth_det; then
  python yolov8_seg_longtail/train_seg.py \
      --data "$PWD/data_tooth/data.yaml" --model yolov8s-seg.pt --nc 1 \
      --epochs 15 --imgsz 1024 --batch 8 --seed 42 \
      --channels-last --weights none --boundary-weight 0 \
      --name tooth_det > logs/tooth_det_train.log 2>&1
  tail -6 logs/tooth_det_train.log
fi
finished tooth_det || { echo "[$(stamp)] tooth detector did not finish"; exit 1; }

for split in train valid; do
  [ -f "reports/tooth_crops_${split}.json" ] || python tools/tooth_stage2.py crops \
      --teeth runs/segment/tooth_det/weights/best.pt \
      --gt "data_clean/annotations/instances_${split}.json" \
      --images "data_clean/${split}/images" \
      --out "reports/tooth_crops_${split}.json" 2>&1 | tail -3
done

python tools/tooth_stage2.py train \
    --train reports/tooth_crops_train.json \
    --val reports/tooth_crops_valid.json \
    --epochs 10 --model-out weights/tooth_stage2_resnet18.pt \
    --report reports/toothstage_valid 2>&1 | tail -16
echo "[$(stamp)] done"
