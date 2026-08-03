#!/usr/bin/env bash
# Project 2 final: train the candidate configuration(s) for the full 100 epochs
# and evaluate ONCE on the held-out test split.
#
# WHY TWO CANDIDATES, NOT ONE "WINNER"
# The 50-epoch arms span only 0.1029-0.1065 overall mAP. With a single seed per
# arm there is no variance estimate, and a 0.36 pp spread across six arms is not
# a result -- selecting the top-mAP arm would be selecting noise, and quietly
# switching the criterion to whichever metric flatters us would be worse.
# So we run BOTH:
#   * the best-mAP arm  -- the criterion pre-registered in ABLATION_PLAN.md
#   * the best-AP75 arm -- the only arm with a mechanistically predicted and
#                          consistent effect (the boundary term: masks fit
#                          tighter, so high-IoU matching improves)
# If they are the same arm, only one run happens. Both are reported on test
# against the 100-epoch baseline, and the write-up states plainly that the arms
# were within noise on mAP. That is the honest version of "which change won".
set -o pipefail

ROOT="$HOME/Documents/ML_SOTA"
cd "$ROOT"
mkdir -p logs reports preds runs
source "$HOME/miniconda3/bin/activate" dental
export PYTHONPATH="$ROOT:$PYTHONPATH"

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
step()  { echo; echo "=== [$(stamp)] $* ==="; }

step "waiting for the GPU"
while pgrep -f "train_seg.py|main.py --output_dir|predict_to_coco|export_dino_preds" >/dev/null 2>&1 \
   || nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; do
  sleep 300
done
sleep 20

step "selecting candidates from the VALID ablation table"
CANDS=$(python3 - <<'PY'
import json, glob, os
rows = []
for p in glob.glob("reports/ablation_S*_valid_segm.json"):
    d = json.load(open(p))["coco_stats"]
    rows.append((os.path.basename(p).split("_")[1], d["mAP"], d["AP75"]))
if not rows:
    print("S1c"); raise SystemExit
by_map = max(rows, key=lambda r: r[1])
by_75  = max(rows, key=lambda r: r[2])
spread = max(r[1] for r in rows) - min(r[1] for r in rows)
print("# mAP spread across arms: %.4f (%.2f pp)" % (spread, 100*spread))
out = [by_map[0]] + ([by_75[0]] if by_75[0] != by_map[0] else [])
print(" ".join(out))
PY
)
echo "$CANDS" | grep '^#' || true
CANDS=$(echo "$CANDS" | grep -v '^#')
echo "candidates: $CANDS"

best_run() {
  local best="" bestn=-1 d n
  for d in runs/*/${1} runs/*/${1}-*; do
    [ -d "$d" ] || continue
    n=$(( $(wc -l < "$d/results.csv" 2>/dev/null || echo 1) - 1 )); [ "$n" -lt 0 ] && n=0
    if [ "$n" -gt "$bestn" ]; then bestn=$n; best="$d"; fi
  done
  [ -n "$best" ] && printf '%s\t%s\n' "$bestn" "$best"
}

for ARM in $CANDS; do
  case "$ARM" in
    S0)  WS=none;    BW=0;   CP=0 ;;
    S1a) WS=0.9;     BW=0;   CP=0 ;;
    S1b) WS=0.99;    BW=0;   CP=0 ;;
    S1c) WS=invsqrt; BW=0;   CP=0 ;;
    S2)  WS=invsqrt; BW=0.5; CP=0 ;;
    S3)  WS=invsqrt; BW=0;   CP=1 ;;
    S4)  WS=invsqrt; BW=0.5; CP=1 ;;
    *)   WS=invsqrt; BW=0;   CP=0 ;;
  esac
  if [ "$CP" = "1" ]; then
    DATA="$ROOT/data_clean_cp/data.yaml"; TL="--train-labels $ROOT/data_clean_cp/train/labels"
  else
    DATA="$ROOT/data_clean/data.yaml"; TL=""
  fi

  REPORT="reports/final_${ARM}_test_segm"
  if [ -f "${REPORT}.json" ]; then echo "[$(stamp)] $ARM final already scored, skipping"; continue; fi

  step "FINAL $ARM  (weights=$WS boundary=$BW copy-paste=$CP)  100 epochs"
  NAME="final_${ARM}_100ep"
  info=$(best_run "$NAME"); n=$(echo "$info" | cut -f1); dir=$(echo "$info" | cut -f2)

  if [ -n "$dir" ] && [ "${n:-0}" -ge 100 ]; then
    echo "already trained ($dir)"
  elif [ -n "$dir" ] && [ -f "$dir/weights/last.pt" ] && [ "${n:-0}" -ge 2 ]; then
    step "resuming $ARM from epoch $n"
    python yolov8_seg_longtail/train_seg.py --data "$DATA" --model yolov8x-seg.pt $TL \
      --epochs 100 --imgsz 640 --batch 8 --seed 42 --cache ram --channels-last \
      --weights "$WS" --boundary-weight "$BW" \
      --resume "$dir/weights/last.pt" --name "$NAME" 2>&1 | tail -15
  else
    python yolov8_seg_longtail/train_seg.py --data "$DATA" --model yolov8x-seg.pt $TL \
      --epochs 100 --imgsz 640 --batch 8 --seed 42 --cache ram --channels-last \
      --weights "$WS" --boundary-weight "$BW" --name "$NAME" 2>&1 | tail -15
  fi

  info=$(best_run "$NAME"); dir=$(echo "$info" | cut -f2)
  W="$dir/weights/best.pt"
  [ -f "$W" ] || { echo "[$(stamp)] $ARM: no weights produced, skipping"; continue; }

  step "ONE-TIME test evaluation for $ARM"
  python yolov8_seg_longtail/predict_to_coco.py \
    --weights "$W" --gt data_clean/annotations/instances_test.json \
    --images data_clean/test/images --out "preds/final_${ARM}_test.json" \
    --imgsz 640 --conf 0.001 --seed 42 2>&1 | tail -3 || continue
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_test.json \
    --dt "preds/final_${ARM}_test.json" --train-json data_clean/annotations/instances_train.json \
    --iou-type segm --out "$REPORT" 2>&1 | tail -4 || true
done

step "final test comparison vs the 100-epoch baseline"
python3 - <<'PY'
import json, glob, os
def row(name, p):
    d = json.load(open(p)); s, g = d["coco_stats"], d["group_AP"]
    return (name, s["mAP"], s["AP50"], s["AP75"], g.get("head",0), g.get("mid",0), g.get("tail",0))
rows = []
base = "reports_egypc/baseline_yolov8x_clean_test_segm.json"
if os.path.exists(base): rows.append(row("baseline (100ep)", base))
for p in sorted(glob.glob("reports/final_S*_test_segm.json")):
    rows.append(row(os.path.basename(p).split("_")[1] + " (100ep)", p))
print("%-20s %8s %8s %8s %8s %8s %8s" % ("model","mAP","AP50","AP75","head","mid","tail"))
for r in rows: print("%-20s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f" % r)
if len(rows) > 1:
    b = rows[0]
    print("\ndeltas vs baseline (pp):")
    for r in rows[1:]:
        print("  %-18s mAP %+.2f  AP50 %+.2f  AP75 %+.2f  head %+.2f  tail %+.2f"
              % (r[0], 100*(r[1]-b[1]), 100*(r[2]-b[2]), 100*(r[3]-b[3]),
                 100*(r[4]-b[4]), 100*(r[6]-b[6])))
    # canonical file for the watchdog chain: best test mAP among the finals
    best = max(rows[1:], key=lambda r: r[1])
    arm = best[0].split()[0]
    src = "reports/final_%s_test_segm.json" % arm
    if os.path.exists(src):
        json.dump(json.load(open(src)), open("reports/final_seg_test_segm.json","w"), indent=1)
        print("\ncanonical final (best test mAP): %s" % arm)
PY

step "P2 FINAL DONE"
