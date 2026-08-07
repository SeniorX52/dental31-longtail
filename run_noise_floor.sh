#!/usr/bin/env bash
# Measure the noise floor: the same configuration at three seeds.
#
# WHY THIS IS NOW THE MOST USEFUL REMAINING RUN
# Every intervention tried on this dataset has landed within roughly half a
# percentage point of the baseline once per-class contributions are decomposed.
# The segmentation arms span 0.1029-0.1065 mAP across seven configurations; the
# detection arms that did not diverge sit within 0.5 pp of theirs. Those spreads
# are currently compared against nothing, because every arm is a single seed.
#
# Without a variance estimate the honest statement is "we cannot tell these
# apart", which is weak. With one, the statement becomes "the seed-to-seed
# standard deviation is X, and no intervention exceeded it", which is a
# measurement and a genuine contribution to anyone else working on this data.
#
# Determinism was verified exact on this pipeline: runs/segment/abl_S0 and
# abl_S0-2 are the same configuration at the same seed, trained independently,
# and agree to six decimal places on every reported metric. So run-to-run noise
# is zero and seed choice is the ONLY source of spread. Three seeds of the
# reference configuration therefore measure exactly the quantity needed.
#
# S0 is the arm to replicate, not S2 or SB, because it is the reference every
# other arm is compared against. Its spread is the yardstick for the whole grid.
#
# Settings are identical to the existing abl_S0 run (seed 42), which supplies
# the third point: 50 epochs, imgsz 640, batch 8, cache=ram, channels_last,
# deterministic. Only the seed differs.
#
# Launch: setsid nohup bash run_noise_floor.sh > logs/noise_floor.log 2>&1 </dev/null &
set -o pipefail

ROOT="$HOME/Documents/ML_SOTA"
cd "$ROOT"
mkdir -p logs reports preds runs
source "$HOME/miniconda3/bin/activate" dental
export PYTHONPATH="$ROOT:$PYTHONPATH"

EPOCHS=50; BATCH=8; IMGSZ=640
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
step()  { echo; echo "=== [$(stamp)] $* ==="; }

gpu_busy() {
  local p c
  for p in $(pgrep -f "train_seg\.py|main\.py --output_dir|predict_to_coco\.py|export_dino_preds\.py" 2>/dev/null); do
    c=$(ps -o comm= -p "$p" 2>/dev/null); case "$c" in python*) return 0 ;; esac
  done
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]' && return 0
  return 1
}
run_epoch() {
  [ -f "$1/results.csv" ] || { echo 0; return; }
  awk -F, 'NR==1{for(i=1;i<=NF;i++){gsub(/^[ \t]+|[ \t]+$/,"",$i); if($i=="epoch") c=i} next}
           c && $c+0>m {m=$c+0} END{printf "%d", m+0}' "$1/results.csv"
}
best_run() {
  local best="" bestn=-1 d n
  for d in runs/segment/${1} runs/segment/${1}-*; do
    [ -d "$d" ] || continue
    n=$(run_epoch "$d"); [ "${n:-0}" -gt "$bestn" ] && { bestn=$n; best="$d"; }
  done
  [ -n "$best" ] && printf '%s\t%s\n' "$bestn" "$best"
}

seed_arm() {
  # NOTE: separate assignments. `local a="$1" b="${a}x"` does not work --
  # the shell expands every word on the line before `local` binds anything, so
  # b would be built from an empty a. That silently produced one shared arm
  # name for both seeds, which would have collided into a single report and
  # yielded no variance estimate at all.
  local seed="$1"
  local tag="S0_s${seed}"
  local report="reports/ablation_${tag}_valid_segm"
  [ -f "${report}.json" ] && { echo "[$(stamp)] $tag already scored"; return 0; }
  local NAME="abl_${tag}" info n dir
  info=$(best_run "$NAME"); n=$(echo "$info"|cut -f1); dir=$(echo "$info"|cut -f2)
  if [ -n "$dir" ] && [ "${n:-0}" -ge "$EPOCHS" ]; then
    echo "[$(stamp)] $tag already trained"
  else
    local resume=""
    [ -n "$dir" ] && [ -f "$dir/weights/last.pt" ] && [ "${n:-0}" -ge 2 ] && resume="--resume $dir/weights/last.pt"
    step "$tag : reference configuration, seed $seed ${resume:+[resuming from $n]}"
    python yolov8_seg_longtail/train_seg.py --data data_clean/data.yaml \
      --model yolov8x-seg.pt --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" \
      --seed "$seed" --cache ram --channels-last --weights none --boundary-weight 0 \
      $resume --name "$NAME" 2>&1 | tail -12
  fi
  info=$(best_run "$NAME"); n=$(echo "$info"|cut -f1); dir=$(echo "$info"|cut -f2)
  local W="$dir/weights/best.pt"
  [ -f "$W" ] || { echo "[$(stamp)] $tag: no weights"; return 0; }
  if [ "${n:-0}" -lt "$EPOCHS" ] && ! python3 tools/run_finished.py "$dir" 2>/dev/null; then
    echo "[$(stamp)] $tag only ${n:-0}/$EPOCHS -- refusing to score"; return 0
  fi
  step "score $tag on VALID"
  python yolov8_seg_longtail/predict_to_coco.py --weights "$W" \
    --gt data_clean/annotations/instances_valid.json \
    --images data_clean/valid/images --out "preds/ablation_${tag}_valid.json" \
    --imgsz "$IMGSZ" --conf 0.001 --seed 42 2>&1 | tail -3 || return 0
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
    --dt "preds/ablation_${tag}_valid.json" \
    --train-json data_clean/annotations/instances_train.json \
    --iou-type segm --out "$report" 2>&1 | tail -4 || true
}

step "waiting for the GPU"
while gpu_busy; do sleep 300; done
sleep 20

seed_arm 1337
seed_arm 2024

step "NOISE FLOOR"
python3 - <<'PY'
import json, os, glob
import statistics as st
paths=[("seed 42","reports/ablation_S0_valid_segm.json"),
       ("seed 1337","reports/ablation_S0_s1337_valid_segm.json"),
       ("seed 2024","reports/ablation_S0_s2024_valid_segm.json")]
vals={}
print("  identical configuration, three seeds:\n")
print("  %-10s %8s %8s %8s %8s %8s"%("seed","mAP","AP50","AP75","head","tail"))
for n,p in paths:
    if not os.path.exists(p): print("  %-10s %8s"%(n,"pending")); continue
    d=json.load(open(p)); s,g=d["coco_stats"],d["group_AP"]
    v=(s["mAP"],s["AP50"],s["AP75"],g.get("head",0),g.get("tail",0))
    vals[n]=v
    print("  %-10s %8.4f %8.4f %8.4f %8.4f %8.4f"%((n,)+v))
if len(vals)>=2:
    print("\n  %-10s %8s %8s %8s %8s %8s"%("","mAP","AP50","AP75","head","tail"))
    for lab,fn in (("mean",st.mean),("sd",lambda x: st.stdev(x) if len(x)>1 else 0.0)):
        print("  %-10s"%lab, " ".join("%8.4f"%fn([v[i] for v in vals.values()]) for i in range(5)))
    sd=[st.stdev([v[i] for v in vals.values()]) if len(vals)>1 else 0 for i in range(5)]
    print("\n  NOISE FLOOR (2 sd, percentage points):")
    for i,k in enumerate(("mAP","AP50","AP75","head","tail")):
        print("    %-6s +/- %.2f pp"%(k,200*sd[i]))
    print("\n  Any effect smaller than this cannot be distinguished from seed choice.")
    print("  Measured intervention effects for comparison:")
    print("    segmentation SB vs baseline (test)  +0.21 pp mAP")
    print("    detection CRT vs baseline (test)    +0.53 pp mAP, +0.01 pp excluding")
    print("                                        one 2-instance class")
PY
step "NOISE FLOOR DONE"
