#!/usr/bin/env bash
# PRIORITY 1: label-noise-robust training at 1280.
#
# Attacks the measured binding constraint (L6): 911 validation pathology
# annotations invisible to fifteen models, 2.7 percent recovered by the model
# that fixed caries, so the labels bound the pathology classes, and an
# unreliable label set poisons training in BOTH directions.
#
# Two mechanisms, each with a source:
#   loss side  --bg-gate 0.5: unassigned anchors OUTSIDE every annotated box
#     whose best class exceeds 0.5 stop being trained toward background.
#     Threshold from Background Recalibration Loss (arXiv:2002.05274, t=0.5);
#     the inside-annotated-box exemption follows Soft Sampling
#     (arXiv:1806.06986); ignoring rather than encouraging is the conservative
#     half of both. Verified in a unit test: gate fires only on confident
#     negatives, other loss terms bit-identical.
#   label side  tools/denoise_labels.py: annotations that BOTH a 640 model and
#     the 1280 model cannot fit ON THE TRAINING SPLIT THEY TRAINED ON are
#     dropped (SparseDet's framing, arXiv:2201.04620: untrustworthy regions
#     should not supervise).
#
# Two cells so the mechanisms are attributable:
#   abl_GATE1280  gate only          (data unchanged)
#   abl_DN1280    gate + denoised GT
# Both fine-tune S0 at 1280 for 25 epochs, mirroring abl_HR1280ft exactly, so
# 0.1204 is the reference with one-and-two variables added.
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
echo "[$(stamp)] label-noise driver queued"
wait_on run_final.sh run_extras.sh run_protohp.sh


# 1. train-split scans (an annotation unfittable by the model that TRAINED on
#    it is the strongest cheap label-error signal we can compute)
[ -f preds/trainscan_S0.json ] || python yolov8_seg_longtail/predict_to_coco.py \
    --weights runs/segment/abl_S0/weights/best.pt \
    --gt data_clean/annotations/instances_train.json \
    --images data_clean/train/images --out preds/trainscan_S0.json \
    --imgsz 640 --conf 0.001 --seed 42 2>&1 | tail -2
[ -f preds/trainscan_HR1280.json ] || python yolov8_seg_longtail/predict_to_coco.py \
    --weights runs/segment/abl_HR1280ft/weights/best.pt \
    --gt data_clean/annotations/instances_train.json \
    --images data_clean/train/images --out preds/trainscan_HR1280.json \
    --imgsz 1280 --conf 0.001 --seed 42 2>&1 | tail -2

python tools/universal_misses.py \
    --gt data_clean/annotations/instances_train.json \
    --preds 'preds/trainscan_*.json' \
    --classes 'Caries,Bone Loss,Periapical lesion' \
    --out reports/universal_misses_train

python tools/denoise_labels.py --src data_clean \
    --suspects reports/universal_misses_train.json \
    --gt data_clean/annotations/instances_train.json \
    --out data_clean_dn

train_ft abl_GATE1280 runs/segment/abl_S0/weights/best.pt 1280 2 25 42 "$PWD/data_clean/data.yaml" --bg-gate 0.5
score    abl_GATE1280 1280
train_ft abl_DN1280   runs/segment/abl_S0/weights/best.pt 1280 2 25 42 "$PWD/data_clean_dn/data.yaml" --bg-gate 0.5
score    abl_DN1280   1280

echo "[$(stamp)] === verdict (reference abl_HR1280ft = same recipe, no noise handling) ==="
python - <<'PY'
import json, os
def m(f): return json.load(open(f))["coco_stats"]["mAP"] if os.path.exists(f) else None
ref = m("reports/eval_abl_HR1280ft_valid.json")
for lab, f in (("HR1280 reference", "reports/eval_abl_HR1280ft_valid.json"),
               ("+ bg-gate",        "reports/eval_abl_GATE1280_valid.json"),
               ("+ gate + denoise", "reports/eval_abl_DN1280_valid.json")):
    v = m(f)
    print("  %-18s %s" % (lab, f"{v:.4f} ({(v-ref)*100:+.2f} pp)" if v and ref else "n/a"))
PY
echo "[$(stamp)] done"
