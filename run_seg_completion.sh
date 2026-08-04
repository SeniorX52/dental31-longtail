#!/usr/bin/env bash
# Project-2 (segmentation): completion cells, published comparators,
# cross-backbone transfer, and seed replicates -- in priority order.
#
# WHY THIS EXISTS
# The original seg grid had the same defect as the detection grid: it was
# cumulative. Every arm carrying the boundary term also
# carried inverse-sqrt class weighting, so "S2 - S0" attributed the weighting
# change to the boundary loss. Correcting the isolation (S2 - S1c, both
# invsqrt, only boundary_weight differing) moved the head figure from +3.7 % to
# +3.0 %; the AP75 headline was unaffected.
#
# Beyond fixing that, beating stock YOLOv8-seg is a necessary control, not a
# sufficient claim. A method paper has to beat region-based AND existing
# boundary-aware losses, and the improvement should transfer across more than
# one backbone. That is what the CMP and XB arms below are for.
#
# ORDERING is by what the paper cannot be written without:
#   1. SB          -- boundary ALONE on the plain baseline. Every comparator is
#                     measured against this cell, so it goes first.
#   2. CMPdice     -- soft Dice, the canonical region-based loss
#      CMPkerv     -- Kervadec boundary loss, the canonical boundary-aware one
#   3. SCP, SNW    -- the remaining individual / leave-one-out cells
#   4. CMPtver     -- Tversky and Focal Tversky, the tuned region-based family
#      CMPftver
#   5. XB*         -- the same contrast on a second backbone (yolov8l-seg)
#   6. seed replicates on the isolated boundary contrast
# A partial run therefore still yields a writable paper; only the tail is lost.
#
# MATCHED BUDGET, AND WHAT THAT DOES NOT MEAN
# Every comparator runs at the SAME auxiliary weight (0.5) as our band term,
# 50 epochs, seed 42, identical everything else. No loss receives a per-loss
# weight sweep -- including ours. So this is a matched-budget comparison at one
# configuration each, NOT a best-effort-per-method comparison. The write-up
# says exactly that rather than describing the competitors as "tuned": no
# per-loss sweep was run, and claiming otherwise would misrepresent the
# protocol.
#
# Determinism was verified exact on this pipeline (runs/segment/abl_S0 and
# abl_S0-2 are the same configuration at the same seed and agree to six
# decimals on every reported metric), so run-to-run noise is zero and
# seed-to-seed variation is the ONLY source of spread. Replicates are run on
# S1c and S2 -- the pair that isolates the boundary term -- because variance is
# only useful on the contrast being claimed.
#
# Launch: setsid nohup bash run_seg_completion.sh > logs/seg_completion.log 2>&1 </dev/null &
set -o pipefail          # NOT -e: one failed arm must not abort the queue

ROOT="$HOME/Documents/ML_SOTA"
cd "$ROOT"
mkdir -p logs reports preds runs
source "$HOME/miniconda3/bin/activate" dental
export PYTHONPATH="$ROOT:$PYTHONPATH"

EPOCHS=50
BATCH=8
IMGSZ=640

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
step()  { echo; echo "=== [$(stamp)] $* ==="; }

gpu_busy() {
  local p c
  for p in $(pgrep -f "train_seg\.py|main\.py --output_dir|predict_to_coco\.py|export_dino_preds\.py" 2>/dev/null); do
    c=$(ps -o comm= -p "$p" 2>/dev/null)
    case "$c" in python*) return 0 ;; esac
  done
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]' && return 0
  return 1
}

# pick the run directory with the most completed epochs (ultralytics appends
# "-2", "-3" when a directory already exists, and the newest is often an
# abandoned partial rather than the finished run)
# Progress is the LAST EPOCH NUMBER in results.csv, not the row count -- see the
# same note in final_seg_run.sh. A run repaired after a crash can be one row
# short of the epoch it reaches, and counting rows would make the completion
# guard reject a finished run. Identical for an uninterrupted run.
run_epoch() {
  [ -f "$1/results.csv" ] || { echo 0; return; }
  awk -F, 'NR==1{for(i=1;i<=NF;i++){gsub(/^[ \t]+|[ \t]+$/,"",$i); if($i=="epoch") c=i} next}
           c && $c+0>m {m=$c+0} END{printf "%d", m+0}' "$1/results.csv"
}

best_run() {
  local best="" bestn=-1 d n
  for d in runs/segment/${1} runs/segment/${1}-*; do
    [ -d "$d" ] || continue
    n=$(run_epoch "$d"); [ "${n:-0}" -lt 0 ] && n=0
    if [ "${n:-0}" -gt "$bestn" ]; then bestn=$n; best="$d"; fi
  done
  [ -n "$best" ] && printf '%s\t%s\n' "$bestn" "$best"
}

# run_arm <tag> <weights> <boundary> <copypaste> <seed> <mask_aux> <model>
run_arm() {
  local tag="$1" ws="$2" bw="$3" cp="$4" seed="$5" aux="${6:-band}" mdl="${7:-yolov8x-seg.pt}"
  local report="reports/ablation_${tag}_valid_segm"
  [ -f "${report}.json" ] && { echo "[$(stamp)] $tag already scored, skipping"; return 0; }

  local DATA TL
  if [ "$cp" = "1" ]; then
    DATA="$ROOT/data_clean_cp/data.yaml"; TL="--train-labels $ROOT/data_clean_cp/train/labels"
  else
    DATA="$ROOT/data_clean/data.yaml"; TL=""
  fi

  local NAME="abl_${tag}"
  local info n dir
  info=$(best_run "$NAME"); n=$(echo "$info" | cut -f1); dir=$(echo "$info" | cut -f2)

  if [ -n "$dir" ] && [ "${n:-0}" -ge "$EPOCHS" ]; then
    echo "[$(stamp)] $tag already trained ($dir)"
  elif [ -n "$dir" ] && [ -f "$dir/weights/last.pt" ] && [ "${n:-0}" -ge 2 ]; then
    step "resuming $tag from epoch $n"
    python yolov8_seg_longtail/train_seg.py --data "$DATA" --model "$mdl" $TL \
      --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" --seed "$seed" \
      --cache ram --channels-last --weights "$ws" --boundary-weight "$bw" \
      --mask-aux "$aux" --resume "$dir/weights/last.pt" --name "$NAME" 2>&1 | tail -12
  else
    step "arm $tag  (model=$mdl weights=$ws aux=$aux w=$bw copy-paste=$cp seed=$seed)  ${EPOCHS} ep"
    python yolov8_seg_longtail/train_seg.py --data "$DATA" --model "$mdl" $TL \
      --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" --seed "$seed" \
      --cache ram --channels-last --weights "$ws" --boundary-weight "$bw" \
      --mask-aux "$aux" --name "$NAME" 2>&1 | tail -12
  fi

  info=$(best_run "$NAME"); n=$(echo "$info" | cut -f1); dir=$(echo "$info" | cut -f2)
  local W="$dir/weights/best.pt"
  [ -f "$W" ] || { echo "[$(stamp)] $tag: no weights produced, skipping"; return 0; }

  # COMPLETION GUARD -- see final_seg_run.sh for why. A killed run leaves a
  # usable best.pt, and scoring it writes a partial arm's numbers into the
  # ablation table permanently, because a scored arm is skipped on relaunch.
  if [ "${n:-0}" -lt "$EPOCHS" ]; then
    echo "[$(stamp)] $tag reached only ${n:-0}/$EPOCHS epochs -- REFUSING to score;"
    echo "            relaunch to resume rather than recording a partial arm."
    return 0
  fi

  step "score $tag on VALID  (verified ${n}/$EPOCHS epochs)"
  python yolov8_seg_longtail/predict_to_coco.py \
    --weights "$W" --gt data_clean/annotations/instances_valid.json \
    --images data_clean/valid/images --out "preds/ablation_${tag}_valid.json" \
    --imgsz "$IMGSZ" --conf 0.001 --seed 42 2>&1 | tail -3 || return 0
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
    --dt "preds/ablation_${tag}_valid.json" \
    --train-json data_clean/annotations/instances_train.json \
    --iou-type segm --out "$report" 2>&1 | tail -4 \
    || echo "[$(stamp)] $tag: eval failed, continuing"
}

step "waiting for the GPU"
while gpu_busy; do sleep 300; done
sleep 20

#            tag        weights  bw   cp seed  aux            model
# 1. the isolated boundary cell -- the reference every comparator is measured against
run_arm      SB         none     0.5  0  42    band           yolov8x-seg.pt
# 2. the two canonical competitors
run_arm      CMPdice    none     0.5  0  42    dice           yolov8x-seg.pt
run_arm      CMPkerv    none     0.5  0  42    kervadec       yolov8x-seg.pt
# 3. remaining individual / leave-one-out cells
run_arm      SCP        none     0    1  42    band           yolov8x-seg.pt
run_arm      SNW        none     0.5  1  42    band           yolov8x-seg.pt
# 4. the tuned region-based family
run_arm      CMPtver    none     0.5  0  42    tversky        yolov8x-seg.pt
run_arm      CMPftver   none     0.5  0  42    focal_tversky  yolov8x-seg.pt
# 5. cross-backbone transfer: same contrast, second backbone
run_arm      XBbase     none     0    0  42    none           yolov8l-seg.pt
run_arm      XBband     none     0.5  0  42    band           yolov8l-seg.pt
# 6. seed replicates on the isolated boundary contrast
run_arm      S1c_s1337  invsqrt  0    0  1337  band           yolov8x-seg.pt
run_arm      S2_s1337   invsqrt  0.5  0  1337  band           yolov8x-seg.pt
run_arm      S1c_s2024  invsqrt  0    0  2024  band           yolov8x-seg.pt
run_arm      S2_s2024   invsqrt  0.5  0  2024  band           yolov8x-seg.pt

step "segmentation results table"
python3 - <<'PY'
import json, glob, os
import statistics as st

CFG = {
 "S0":  ("none,    bw=0",              "baseline (stock BCE)"),
 "S1c": ("invsqrt, bw=0",              "+ weighting only"),
 "SB":  ("none,    band 0.5",          "+ BOUNDARY only  <- our objective"),
 "SCP": ("none,    bw=0, +cp",         "+ copy-paste only"),
 "S2":  ("invsqrt, band 0.5",          "weighting + boundary"),
 "S3":  ("invsqrt, bw=0, +cp",         "complete - boundary"),
 "SNW": ("none,    band 0.5, +cp",     "complete - weighting"),
 "S4":  ("invsqrt, band 0.5, +cp",     "COMPLETE METHOD"),
 "CMPdice":  ("none, dice 0.5",        "vs soft Dice (Milletari 2016)"),
 "CMPtver":  ("none, tversky 0.5",     "vs Tversky (Salehi 2017)"),
 "CMPftver": ("none, focalTversky 0.5","vs Focal Tversky (Abraham 2019)"),
 "CMPkerv":  ("none, kervadec 0.5",    "vs Boundary loss (Kervadec 2018)"),
 "XBbase":   ("yolov8l, bw=0",         "second backbone, baseline"),
 "XBband":   ("yolov8l, band 0.5",     "second backbone, + boundary"),
}
R = {}
for p in glob.glob("reports/ablation_*_valid_segm.json"):
    tag = os.path.basename(p)[len("ablation_"):-len("_valid_segm.json")]
    d = json.load(open(p)); s, g = d["coco_stats"], d["group_AP"]
    R[tag] = (s["mAP"], s["AP50"], s["AP75"], g.get("head",0), g.get("tail",0))

print("%-9s %-22s %-30s %8s %8s %8s %8s"
      % ("arm","configuration","meaning","mAP","AP50","AP75","head"))
for a, (cfg, mean) in CFG.items():
    if a in R:
        print("%-9s %-22s %-30s %8.4f %8.4f %8.4f %8.4f" % ((a,cfg,mean)+R[a][:4]))

if "S0" in R:
    b = R["S0"]
    print("\nISOLATED contribution of each component (vs the S0 baseline, pp):")
    for a, lab in (("S1c","weighting"), ("SB","boundary"), ("SCP","copy-paste")):
        if a in R:
            r = R[a]
            print("  %-11s mAP %+.2f  AP50 %+.2f  AP75 %+.2f  head %+.2f"
                  % (lab, 100*(r[0]-b[0]), 100*(r[1]-b[1]), 100*(r[2]-b[2]), 100*(r[3]-b[3])))

if "SB" in R:
    o = R["SB"]
    print("\nOURS vs PUBLISHED COMPARATORS (all at aux weight 0.5, no per-loss tuning):")
    print("  %-26s %8s %8s %8s   %s" % ("loss","mAP","AP75","head","AP75 vs ours"))
    print("  %-26s %8.4f %8.4f %8.4f   %s" % ("band-Dice (ours)", o[0], o[2], o[3], "--"))
    for a, lab in (("CMPdice","soft Dice"), ("CMPtver","Tversky"),
                   ("CMPftver","Focal Tversky"), ("CMPkerv","Boundary (Kervadec)")):
        if a in R:
            r = R[a]
            print("  %-26s %8.4f %8.4f %8.4f   %+.2f pp"
                  % (lab, r[0], r[2], r[3], 100*(o[2]-r[2])))

if "XBbase" in R and "XBband" in R:
    b, o = R["XBbase"], R["XBband"]
    print("\nCROSS-BACKBONE (yolov8l-seg): boundary effect %+.2f pp AP75, %+.2f pp head"
          % (100*(o[2]-b[2]), 100*(o[3]-b[3])))
    if "S0" in R and "SB" in R:
        print("  for comparison, on yolov8x-seg: %+.2f pp AP75, %+.2f pp head"
              % (100*(R["SB"][2]-R["S0"][2]), 100*(R["SB"][3]-R["S0"][3])))

print("\nSEED VARIANCE (50 ep, valid; determinism is exact, so this is seed-to-seed only):")
for base in ("S1c", "S2"):
    seeds = [t for t in R if t == base or t.startswith(base + "_s")]
    if len(seeds) < 2:
        print("  %-4s only %d seed(s) so far" % (base, len(seeds))); continue
    print("  %-4s n=%d" % (base, len(seeds)))
    for i, kname in enumerate(("mAP","AP50","AP75","head")):
        v = [R[t][i] for t in seeds]
        m, sd = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
        half = 1.96*sd/(len(v)**0.5)
        print("      %-5s mean %.4f  sd %.4f  95%% CI [%.4f, %.4f]"
              % (kname, m, sd, m-half, m+half))
a1 = sorted(t for t in R if t.startswith("S1c"))
a2 = sorted(t for t in R if t.startswith("S2") and not t.startswith("S2_s") or t.startswith("S2_s"))
a2 = sorted(t for t in R if t == "S2" or t.startswith("S2_s"))
if len(a1) > 1 and len(a2) > 1 and len(a1) == len(a2):
    d = [R[b][2] - R[a][2] for a, b in zip(a1, a2)]
    print("\n  PAIRED boundary effect on AP75 across seeds: %s"
          % ", ".join("%+.2f pp" % (100*x) for x in d))
    if len(d) > 1:
        print("  mean %+.2f pp, sd %.2f pp -> %s"
              % (100*st.mean(d), 100*st.stdev(d),
                 "effect exceeds seed noise" if abs(st.mean(d)) > 2*st.stdev(d)
                 else "NOT separable from seed noise"))
PY
step "SEG COMPLETION DONE"
