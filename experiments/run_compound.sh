#!/usr/bin/env bash
# The compound cells: confirm the resolution result properly, then stack the
# levers that plausibly compose with it.
#
# WHY THESE FOUR, AND IN THIS ORDER.
#
# Cells 1-2, from-scratch replicates of the winner. The 1280 result (+1.50 pp,
# +14.2 % relative) carries exactly two open confounds: it fine-tunes from the
# converged baseline so it has seen extra epochs, and it is a single seed. Two
# from-scratch runs at seeds 42 and 1337 close BOTH at once: same COCO
# initialisation as the baseline itself, thirty epochs against the baseline's
# fifty (conservative, since every arm here peaks by epoch 26 and the baseline
# then LOSES 1.68 pp by epoch 50), one at each seed. If both land above the
# baseline by a clear margin the headline claim is settled; if they scatter
# back, the fine-tune was doing the work and the claim dies here rather than in
# a headline result. The training size adapts to whichever of 1280/1600 the
# native-resolution probe crowns, read from its report at runtime.
#
# Cell 3, high-resolution prototypes AT high input resolution. XP3 raised the
# prototype grid at 640 input and came out -1.30 pp: the head had more cells
# than the input had detail. That negative established the law "detail must
# exist in the input before a higher-resolution head can represent it" -- which
# cuts the other way now that the input HAS the detail. At 1280 the stock
# prototype grid is 320 (representable Dice 0.9492 by our own ceiling table);
# proto-scale 2 lifts it to 640 (0.9901). The paired contour test gives this
# cell a precise target: the 1280 model's masks overlap better (IoU separably
# +0.0048) but their contours sit WORSE (boundary F separably -0.0096), and
# contour placement is exactly what prototype resolution buys. If boundary F
# recovers while IoU holds, the two resolution levers compose.
#
# Cell 4, the widened coefficient head at 1280. K2's +0.64 pp at 640 never
# resolved its mechanism, but it is the only other arm above the noise floor
# and it touches a different part of the head than resolution does. One cell
# answers whether the two compose or overlap.
#
# Cells 3-4 fine-tune from S0 with the identical recipe to abl_HR1280ft, so
# 0.1204 is their clean reference with one variable added each.
#
# QUEUE DISCIPLINE. Waits for the Mask DINO and native-resolution drivers by
# argv[1] (a `pgrep -f` pattern would match any shell that mentions the script
# name; that self-match has produced five wrong readings in this project).
# run_phase2.sh and run_k2seeds.sh are rewired to wait for THIS script, so the
# chain is maskdino -> hr1600 -> compound -> phase2 -> k2seeds.
#
# Usage:  nohup ./run_compound.sh > logs/compound.log 2>&1 &

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
earlier_driver_running() {
  local p a1
  for p in $(pgrep -x bash 2>/dev/null); do
    [ -r "/proc/$p/cmdline" ] || continue
    a1=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | sed -n '2p')
    case "$a1" in
      ./run_maskdino.sh|*/run_maskdino.sh|\
      ./run_hr1600.sh|*/run_hr1600.sh) return 0 ;;
    esac
  done
  return 1
}

echo "[$(stamp)] compound cells queued behind Mask DINO and the 1600 probe"
w=0
while trainer_running || earlier_driver_running; do
  [ $((w % 1800)) -eq 0 ] && echo "[$(stamp)] waiting (${w}s)"
  sleep 120; w=$((w + 120))
done
sleep 60
echo "[$(stamp)] queue clear after ${w}s"

# ------------------------------------------------------------------ sizing ---
# The from-scratch cells train at whichever resolution the probes crowned.
# 1600 must beat 1280 by more than 0.30 pp to be chosen: the epoch-to-epoch
# jitter of a single run is ~0.6 pp peak-to-peak, so a smaller margin is not a
# decision, and 1280 at batch 2 is the cheaper, better-understood setting.
read -r SCRATCH_SZ SCRATCH_BS <<< "$(python - <<'PY'
import json, os
def m(f):
    return json.load(open(f))["coco_stats"]["mAP"] if os.path.exists(f) else None
a, b = m("reports/eval_abl_HR1280ft_valid.json"), m("reports/eval_abl_HR1600ft_valid.json")
if a is not None and b is not None and (b - a) > 0.003:
    print("1600 1")
else:
    print("1280 2")
PY
)"
echo "[$(stamp)] from-scratch cells will train at $SCRATCH_SZ (batch $SCRATCH_BS)"

train_ft() {            # $1 tag  $2 init  $3 imgsz  $4 batch  $5 epochs  $6 seed  extra...
  local tag="$1" init="$2" sz="$3" bs="$4" ep="$5" seed="$6"; shift 6
  if finished "$tag"; then echo "[$(stamp)] $tag already complete"; return 0; fi
  local RESUME=()
  [ -f "runs/segment/$tag/weights/last.pt" ] && \
    RESUME=(--resume "runs/segment/$tag/weights/last.pt")
  echo "[$(stamp)] === $tag  (init $(basename "$init"), imgsz $sz, ${ep}ep, seed $seed, extra: $*) ==="
  python yolov8_seg_longtail/train_seg.py \
      --data "$PWD/data_clean/data.yaml" --model "$init" --nc 31 \
      --epochs "$ep" --imgsz "$sz" --batch "$bs" --seed "$seed" \
      --channels-last --weights none --boundary-weight 0 \
      --name "$tag" "$@" "${RESUME[@]}" > "logs/${tag}_train.log" 2>&1
  tail -8 "logs/${tag}_train.log"
  # 1280 at batch 2 is measured at 13.4 GB; the extra-head cells could exceed it
  if ! finished "$tag" && grep -qiE "out of memory" "logs/${tag}_train.log"; then
    echo "[$(stamp)] $tag OOM at batch $bs; retrying at batch 1"
    rm -rf "runs/segment/$tag"
    python yolov8_seg_longtail/train_seg.py \
        --data "$PWD/data_clean/data.yaml" --model "$init" --nc 31 \
        --epochs "$ep" --imgsz "$sz" --batch 1 --seed "$seed" \
        --channels-last --weights none --boundary-weight 0 \
        --name "$tag" "$@" > "logs/${tag}_train.log" 2>&1
    tail -8 "logs/${tag}_train.log"
  fi
  finished "$tag" && echo "[$(stamp)] $tag finished" || echo "[$(stamp)] *** $tag did NOT finish"
}

score() {               # $1 tag  $2 imgsz
  local tag="$1" sz="$2" dt="preds/ablation_$1_valid.json"
  finished "$tag" || { echo "[$(stamp)] $tag unfinished, not scoring"; return 0; }
  [ -f "$dt" ] || python yolov8_seg_longtail/predict_to_coco.py \
      --weights "runs/segment/$tag/weights/best.pt" \
      --gt data_clean/annotations/instances_valid.json \
      --images data_clean/valid/images --out "$dt" \
      --imgsz "$sz" --conf 0.001 --seed 42 2>&1 | tail -2
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
      --dt "$dt" --train-json data_clean/annotations/instances_train.json \
      --iou-type segm --out "reports/eval_${tag}_valid" 2>&1 | tail -3
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
      --dt "$dt" --train-json data_clean/annotations/instances_train.json \
      --iou-type bbox --out "reports/bboxchk_${tag}_valid" 2>&1 | tail -2
  PYTHONPATH="$PWD/eval:${PYTHONPATH:-}" python eval/paired_contour.py \
      --gt data_clean/annotations/instances_valid.json \
      --dt-a preds/ablation_S0_valid.json --label-a S0 \
      --dt-b "$dt" --label-b "$tag" --conf "$CONF" --boot 500 \
      --out "reports/paired_contour_S0_${tag}_valid" 2>&1 | tail -6
}

# cells 1-2: the confirmation pair (same COCO init as the baseline itself)
train_ft abl_SCRATCH_s42   yolov8x-seg.pt "$SCRATCH_SZ" "$SCRATCH_BS" 30 42
score    abl_SCRATCH_s42   "$SCRATCH_SZ"
train_ft abl_SCRATCH_s1337 yolov8x-seg.pt "$SCRATCH_SZ" "$SCRATCH_BS" 30 1337
score    abl_SCRATCH_s1337 "$SCRATCH_SZ"

# cell 3: high-res prototypes now that the input has the detail (XP3 retried under L2)
train_ft abl_HR1280hp runs/segment/abl_S0/weights/best.pt 1280 2 25 42 --proto-scale 2
score    abl_HR1280hp 1280

# cell 4: the widened coefficient head at 1280 (does K2 compose with resolution?)
train_ft abl_HR1280w  runs/segment/abl_S0/weights/best.pt 1280 2 25 42 --coeff-width 256
score    abl_HR1280w  1280

echo "[$(stamp)] === verdicts ==="
python - <<'PY'
import json, os, statistics as st
def m(f, k="mAP"):
    return json.load(open(f))["coco_stats"][k] if os.path.exists(f) else None
S0 = m("reports/ablation_S0_valid_segm.json")
HR = m("reports/eval_abl_HR1280ft_valid.json")
s42 = m("reports/eval_abl_SCRATCH_s42_valid.json")
s13 = m("reports/eval_abl_SCRATCH_s1337_valid.json")
print("  baseline S0 (640, 50ep)        : %.4f" % S0)
print("  HR1280 fine-tune (reference)   : %s" % (f"{HR:.4f}" if HR else "n/a"))
for lab, v in (("SCRATCH seed 42", s42), ("SCRATCH seed 1337", s13)):
    print("  %-30s : %s" % (lab, f"{v:.4f}  ({(v-S0)*100:+.2f} pp vs S0)" if v else "n/a"))
if s42 and s13:
    mu = st.mean([s42, s13]); d = (mu - S0) * 100
    print("  mean of the pair %.4f -> %+.2f pp vs S0" % (mu, d))
    print("  RESOLUTION CLAIM %s: from-scratch, fewer epochs than the baseline,"
          % ("CONFIRMED" if d > 0.42 else "NOT CONFIRMED"))
    print("  two seeds%s clear of the +-0.21 pp floor."
          % (" both" if min(s42, s13) - S0 > 0.0021 else " not both"))
for lab, f, ref, rl in (("hi-res protos @1280", "reports/eval_abl_HR1280hp_valid.json", HR, "HR1280"),
                        ("wide head @1280",     "reports/eval_abl_HR1280w_valid.json",  HR, "HR1280")):
    v = m(f)
    if v and ref:
        print("  %-30s : %.4f  (%+.2f pp vs %s)" % (lab, v, (v-ref)*100, rl))
PY
echo "[$(stamp)] done"
