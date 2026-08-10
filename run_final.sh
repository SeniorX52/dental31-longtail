#!/usr/bin/env bash
# The deliverable queue, reordered against the Wednesday submission.
#
# WHY THIS EXISTS. The chain was built when the open question was "does anything
# work". That question is answered: training at native resolution beats the
# baseline at two seeds, +23.2 % and +18.7 % relative on validation. What is left
# is not discovery, it is the three things the contract and the client's own
# 04 Aug methodology message require, none of which were scheduled:
#
#   1. A TEST-split number for the method. Everything we have quoted is
#      validation, measured against abl_S0. On test abl_S0 scores 0.1007 while
#      the CONTRACTUAL baseline -- 100 epochs at 640, deliberately given the
#      longer schedule so the method cannot win on training length -- scores
#      0.1051. So the reference behind our headline sits BELOW the bar we have
#      to clear, and no run has ever been scored against that bar. This is the
#      single number the deliverable turns on and it costs 40 minutes.
#
#   2. Cross-backbone transfer. "Evidence that the improvement transfers across
#      more than one suitable backbone" is an explicit requirement. yolo11x-seg
#      at 1280 is that evidence and it was queued behind two learning-curve
#      cells that answer a different question.
#
#   3. A third seed. BASELINE_PROTOCOL.md principle 4 promises final tables
#      averaged over three seeds with the spread reported. Two is not three.
#      This is NOT the K2 seed work that was killed today: that replicated a
#      lever already known to die at 1280. This replicates the winner.
#
# ORDER IS DELIBERATE. The test evaluation runs FIRST because it is 40 minutes
# and it decides whether there is a deliverable at all. Cross-backbone second
# because it is a stated requirement. The third seed last because it strengthens
# a claim that two seeds already support.
#
# TEST DISCIPLINE. Test is touched once per reported configuration and never for
# selection. Only the arms that go in the final table are scored here: the two
# confirmed seeds of the method. Every other arm stays on validation. Scoring
# candidates on test to pick between them would convert the frozen split into a
# selection set, which is exactly what the client's message forbids.
#
# Usage:  nohup ./run_final.sh > logs/final.log 2>&1 &

cd "$HOME/Documents/ML_SOTA" || exit 1
source "$HOME/miniconda3/bin/activate" dental
set -u
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs reports preds

stamp() { date '+%F %T'; }
finished() { python tools/run_finished.py "runs/segment/$1" >/dev/null 2>&1; }

GT_T=data_clean/annotations/instances_test.json
IM_T=data_clean/test/images
GT_V=data_clean/annotations/instances_valid.json
TRJ=data_clean/annotations/instances_train.json

# ------------------------------------------------------- 1. TEST evaluation ---
# The two confirmed seeds, at the resolution they were trained at.
score_test() {                 # $1 tag  $2 imgsz
  local tag="$1" sz="$2" dt="preds/test_${1}.json"
  finished "$tag" || { echo "[$(stamp)] $tag unfinished, not scoring on test"; return 0; }
  echo "[$(stamp)] TEST: $tag at imgsz $sz"
  [ -f "$dt" ] || python yolov8_seg_longtail/predict_to_coco.py \
      --weights "runs/segment/$tag/weights/best.pt" \
      --gt "$GT_T" --images "$IM_T" --out "$dt" \
      --imgsz "$sz" --conf 0.001 --seed 42 2>&1 | tail -2
  python eval/coco_eval_report.py --gt "$GT_T" --dt "$dt" --train-json "$TRJ" \
      --iou-type segm --out "reports/final_${tag}_test_segm" 2>&1 | tail -3
  python eval/coco_eval_report.py --gt "$GT_T" --dt "$dt" --train-json "$TRJ" \
      --iou-type bbox --out "reports/final_${tag}_test_bbox" 2>&1 | tail -2
}

echo "[$(stamp)] === stage 1: test-split evaluation of the method ==="
score_test abl_SCRATCH_s42   1280
score_test abl_SCRATCH_s1337 1280

echo "[$(stamp)] --- the comparison that decides the deliverable ---"
python - <<'PY'
import json, os
def m(f):
    try: return json.load(open(f))["coco_stats"]
    except Exception: return None
base = m("reports_egypc/baseline_yolov8x_clean_test_segm.json")
s0   = m("reports/baseline_yolo_test.json")
rows = [(t, m("reports/final_abl_SCRATCH_%s_test_segm.json" % s))
        for t, s in (("seed 42", "s42"), ("seed 1337", "s1337"))]
if base:
    print("  CONTRACTUAL baseline (100 ep, 640)   segm mAP %.4f" % base["mAP"])
if s0:
    print("  abl_S0 ablation reference (50 ep)     segm mAP %.4f" % s0["mAP"])
vals = []
for lab, d in rows:
    if not d: print("  %-36s n/a" % lab); continue
    vals.append(d["mAP"])
    delta = (d["mAP"] - base["mAP"]) * 100 if base else 0
    print("  method, %-28s segm mAP %.4f  (%+.2f pp vs contractual)"
          % (lab, d["mAP"], delta))
if vals and base:
    mu = sum(vals) / len(vals)
    print("  mean of seeds                         segm mAP %.4f  (%+.2f pp, %+.1f %% relative)"
          % (mu, (mu - base["mAP"]) * 100, (mu - base["mAP"]) / base["mAP"] * 100))
    print("  VERDICT: %s the contractual baseline on the frozen test split."
          % ("BEATS" if mu > base["mAP"] else "DOES NOT BEAT"))
PY

# --------------------------------------------- 2. cross-backbone transfer ---
# Reported on VALIDATION, like every other comparison. It is transfer evidence
# for the finding, not a candidate competing for the final slot, so it has no
# business on the test split.
train_arm() {                  # $1 tag  $2 init  $3 imgsz  $4 batch  $5 epochs  $6 seed
  local tag="$1" init="$2" sz="$3" bs="$4" ep="$5" seed="$6"
  if finished "$tag"; then echo "[$(stamp)] $tag already complete"; return 0; fi
  local RESUME=()
  [ -f "runs/segment/$tag/weights/last.pt" ] && RESUME=(--resume "runs/segment/$tag/weights/last.pt")
  echo "[$(stamp)] === $tag ($(basename "$init"), $sz, ${ep}ep, seed $seed) ==="
  python yolov8_seg_longtail/train_seg.py \
      --data "$PWD/data_clean/data.yaml" --model "$init" --nc 31 \
      --epochs "$ep" --imgsz "$sz" --batch "$bs" --seed "$seed" \
      --channels-last --weights none --boundary-weight 0 \
      --name "$tag" "${RESUME[@]}" > "logs/${tag}_train.log" 2>&1
  tail -6 "logs/${tag}_train.log"
  if ! finished "$tag" && grep -qiE "out of memory" "logs/${tag}_train.log"; then
    echo "[$(stamp)] $tag OOM at batch $bs; retrying at batch 1"
    rm -rf "runs/segment/$tag"
    python yolov8_seg_longtail/train_seg.py \
        --data "$PWD/data_clean/data.yaml" --model "$init" --nc 31 \
        --epochs "$ep" --imgsz "$sz" --batch 1 --seed "$seed" \
        --channels-last --weights none --boundary-weight 0 \
        --name "$tag" > "logs/${tag}_train.log" 2>&1
    tail -6 "logs/${tag}_train.log"
  fi
  finished "$tag" && echo "[$(stamp)] $tag finished" || echo "[$(stamp)] *** $tag did NOT finish"
}

score_valid() {                # $1 tag  $2 imgsz
  local tag="$1" sz="$2" dt="preds/ablation_${1}_valid.json"
  finished "$tag" || { echo "[$(stamp)] $tag unfinished, not scoring"; return 0; }
  [ -f "$dt" ] || python yolov8_seg_longtail/predict_to_coco.py \
      --weights "runs/segment/$tag/weights/best.pt" \
      --gt "$GT_V" --images data_clean/valid/images --out "$dt" \
      --imgsz "$sz" --conf 0.001 --seed 42 2>&1 | tail -2
  python eval/coco_eval_report.py --gt "$GT_V" --dt "$dt" --train-json "$TRJ" \
      --iou-type segm --out "reports/eval_${tag}_valid" 2>&1 | tail -3
}

echo "[$(stamp)] === stage 2: cross-backbone transfer (yolo11x-seg at 1280) ==="
train_arm  abl_YOLO11x_1280 yolo11x-seg.pt 1280 2 30 42
score_valid abl_YOLO11x_1280 1280

# ------------------------------------------------------------ 3. third seed ---
echo "[$(stamp)] === stage 3: third seed of the method ==="
train_arm  abl_SCRATCH_s2024 yolov8x-seg.pt 1280 2 30 2024
score_valid abl_SCRATCH_s2024 1280
score_test  abl_SCRATCH_s2024 1280

echo "[$(stamp)] === three-seed table (validation), as the protocol promises ==="
python - <<'PY'
import json, os, statistics as st
def m(f):
    try: return json.load(open(f))["coco_stats"]["mAP"]
    except Exception: return None
s0 = m("reports/ablation_S0_valid_segm.json")
seeds = [(s, m("reports/eval_abl_SCRATCH_%s_valid.json" % s)) for s in ("s42", "s1337", "s2024")]
vals = [v for _, v in seeds if v is not None]
for s, v in seeds:
    print("  %-8s %s" % (s, ("%.4f" % v) if v else "n/a"))
if len(vals) >= 2:
    mu = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    print("  mean %.4f  sd %.4f  (n=%d)" % (mu, sd, len(vals)))
    if s0: print("  vs abl_S0 %.4f : %+.2f pp (%+.1f %% relative)"
                 % (s0, (mu - s0) * 100, (mu - s0) / s0 * 100))
x = m("reports/eval_abl_YOLO11x_1280_valid.json")
if x: print("  cross-backbone yolo11x-seg @1280: %.4f" % x)
PY
echo "[$(stamp)] done"
