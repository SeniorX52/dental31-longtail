#!/usr/bin/env bash
# Two things the evaluation protocol requires that no run has ever produced.
#
# 1. THE PUBLISHED COMPARATORS. "Comparisons with the strongest relevant
#    existing boundary-aware or class-imbalance methods, not only the models'
#    default settings". Four were implemented on 04 Aug and
#    none produced a number, because ultralytics 8.4.108's `crop_mask` zeroes
#    outside the box with two IN-PLACE multiplies and returns the same tensor.
#    Cropping `pred_mask.sigmoid()` therefore destroys a value autograd needs
#    and every comparator died in backward. Our own band term was untouched
#    because it crops the output of a subtraction, whose backward does not need
#    its output. Fixed by `crop_to_box` in train_seg.py, reproduced and verified
#    on CPU before queueing.
#
#    Two arms, not four: soft Dice is the canonical region-based comparator and
#    Kervadec is the canonical boundary-aware one, so between them they cover
#    both families the protocol requires. Tversky and Focal Tversky are the same
#    family as Dice with a tuned asymmetry and add little for another six GPU
#    hours we do not have before Wednesday.
#
#    Reference is abl_S0: weights none, boundary-weight 0, identical in every
#    other respect, so only the auxiliary term differs.
#
# 2. CAPACITY DOWN AT HIGH RESOLUTION. Law L4 says the model is oversized for
#    the data -- runs peak at epoch 14-26 then decline, 100-epoch runs lose to
#    50-epoch ones, and two unrelated architectures converge within 0.7 pp.
#    Every capacity experiment so far went UP and none helped. yolov8l-seg is
#    44 M parameters against yolov8x-seg's 71 M; at the resolution that works,
#    the smaller model should peak later and possibly higher. It is the one
#    capacity direction the evidence actually points at, it has never been run,
#    and it doubles as a second backbone for the transfer requirement.
#
# Usage:  nohup ./run_extras.sh > logs/extras.log 2>&1 &

cd "$HOME/Documents/ML_SOTA" || exit 1
source "$HOME/miniconda3/bin/activate" dental
set -u
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs reports preds

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

score_valid() {                # $1 tag  $2 imgsz
  local tag="$1" sz="$2" dt="preds/ablation_${1}_valid.json"
  finished "$tag" || { echo "[$(stamp)] $tag unfinished, not scoring"; return 0; }
  [ -f "$dt" ] || python yolov8_seg_longtail/predict_to_coco.py \
      --weights "runs/segment/$tag/weights/best.pt" \
      --gt data_clean/annotations/instances_valid.json \
      --images data_clean/valid/images --out "$dt" \
      --imgsz "$sz" --conf 0.001 --seed 42 2>&1 | tail -2
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
      --dt "$dt" --train-json data_clean/annotations/instances_train.json \
      --iou-type segm --out "reports/eval_${tag}_valid" 2>&1 | tail -3
  # the comparators are boundary claims, so they get the paired-on-intersection
  # test against the same reference, not just mAP
  PYTHONPATH="$PWD/eval:${PYTHONPATH:-}" python eval/paired_contour.py \
      --gt data_clean/annotations/instances_valid.json \
      --dt-a preds/ablation_S0_valid.json --label-a S0 \
      --dt-b "$dt" --label-b "$tag" --conf "$CONF" --boot 500 \
      --out "reports/paired_contour_S0_${tag}_valid" 2>&1 | tail -6
}

echo "[$(stamp)] extras queued behind the deliverable runs"
wait_on run_final.sh

# ------------------------------------------- 1. capacity down at 1280 ---
# Runs first: it is the cheaper arm and the one that could still change the
# headline result, whereas the comparators can only close a gap.
TAG=abl_L1280
if finished "$TAG"; then
  echo "[$(stamp)] $TAG already complete"
else
  RESUME=()
  [ -f "runs/segment/$TAG/weights/last.pt" ] && RESUME=(--resume "runs/segment/$TAG/weights/last.pt")
  echo "[$(stamp)] === $TAG: yolov8l-seg (44 M) at 1280, 30 ep, seed 42 ==="
  python yolov8_seg_longtail/train_seg.py \
      --data "$PWD/data_clean/data.yaml" --model yolov8l-seg.pt --nc 31 \
      --epochs 30 --imgsz 1280 --batch 2 --seed 42 \
      --channels-last --weights none --boundary-weight 0 \
      --name "$TAG" "${RESUME[@]}" > "logs/${TAG}_train.log" 2>&1
  tail -6 "logs/${TAG}_train.log"
  finished "$TAG" && echo "[$(stamp)] $TAG finished" || echo "[$(stamp)] *** $TAG did NOT finish"
fi
score_valid "$TAG" 1280

# ------------------------------------------------ 2. the comparators ---
# Matched to abl_S0 exactly: 50 epochs, 640, batch 8, seed 42, cache ram,
# no class weighting. Only --mask-aux differs, at the same weight our own
# band term was given.
cmp_arm() {                    # $1 tag  $2 aux
  local tag="$1" aux="$2"
  if finished "$tag"; then echo "[$(stamp)] $tag already complete"; return 0; fi
  local RESUME=()
  [ -f "runs/segment/$tag/weights/last.pt" ] && RESUME=(--resume "runs/segment/$tag/weights/last.pt")
  echo "[$(stamp)] === $tag: BCE + $aux, 50 ep, 640 ==="
  python yolov8_seg_longtail/train_seg.py \
      --data "$PWD/data_clean/data.yaml" --model yolov8x-seg.pt --nc 31 \
      --epochs 50 --imgsz 640 --batch 8 --seed 42 --cache ram \
      --channels-last --weights none --boundary-weight 0.5 --mask-aux "$aux" \
      --name "$tag" "${RESUME[@]}" > "logs/${tag}_train.log" 2>&1
  tail -6 "logs/${tag}_train.log"
  # the failure mode this arm previously hit is a backward-pass crash, so say
  # explicitly whether it recurred rather than reporting a silent non-result
  if ! finished "$tag"; then
    echo "[$(stamp)] *** $tag did NOT finish. Looking for the old defect:"
    grep -iE 'inplace|in-place|RuntimeError' "logs/${tag}_train.log" | tail -3 | sed 's/^/      /'
  else
    echo "[$(stamp)] $tag finished its schedule"
  fi
}

cmp_arm abl_CMPdice dice
score_valid abl_CMPdice 640
cmp_arm abl_CMPkerv kervadec
score_valid abl_CMPkerv 640

echo "[$(stamp)] === comparator table (validation, vs abl_S0) ==="
python - <<'PY'
import json, os
def m(f):
    try: return json.load(open(f))["coco_stats"]
    except Exception: return None
s0 = m("reports/ablation_S0_valid_segm.json")
print("  %-22s %9s %9s %9s" % ("arm", "segm mAP", "AP50", "AP75"))
if s0: print("  %-22s %9.4f %9.4f %9.4f" % ("S0 (BCE only)", s0["mAP"], s0["AP50"], s0["AP75"]))
for lab, tag in (("+ soft Dice", "abl_CMPdice"), ("+ Kervadec boundary", "abl_CMPkerv"),
                 ("+ our band term", "abl_S2")):
    d = m("reports/eval_%s_valid.json" % tag) or m("reports/ablation_%s_valid_segm.json" % tag.replace("abl_", ""))
    if d: print("  %-22s %9.4f %9.4f %9.4f" % (lab, d["mAP"], d["AP50"], d["AP75"]))
    else: print("  %-22s %9s" % (lab, "n/a"))
l = m("reports/eval_abl_L1280_valid.json")
if l: print("\n  yolov8l-seg @1280 (capacity down): segm mAP %.4f" % l["mAP"])
PY
echo "[$(stamp)] done"
