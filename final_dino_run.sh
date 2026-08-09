#!/usr/bin/env bash
# Project-1 FINAL: evaluate the detection method ONCE on the held-out test split.
#
# No retraining. Every arm in the frozen matrix is already a full 12-epoch
# (official 1x) run on the identical budget as the baseline, so the arm's own
# checkpoint IS the final model.
#
# WHICH ARM GETS THE TEST NUMBER
# D5 -- frequency-awareness applied consistently across classification loss,
# Hungarian matching and denoising -- is the PRE-REGISTERED proposed method. It
# is evaluated on test whether or not it happened to top the validation table,
# because reporting only the arm that won on validation, and calling that the
# method, converts a selection effect into a headline result.
#
# The best-on-validation arm is ALSO evaluated when it differs from D5, and
# both numbers are reported next to the baseline. If they disagree, that
# disagreement is the finding and it goes in the write-up. This mirrors
# final_seg_run.sh, which runs both the best-mAP and best-AP75 candidates for
# the same reason.
#
# Launch:  setsid nohup bash final_dino_run.sh > logs/final_dino.log 2>&1 </dev/null &
set -o pipefail

ROOT="$HOME/Documents/ML_SOTA"
DINO="$HOME/DINO"
cd "$ROOT"
source "$HOME/miniconda3/bin/activate" dental
export PYTHONPATH="$ROOT:$DINO:${PYTHONPATH:-}"

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
step()  { echo; echo "=== [$(stamp)] $* ==="; }

# gpu_busy: true only when a REAL python training/eval process is alive, or the
# GPU has a compute app attached. Filtering by comm (python*) rather than
# matching full command lines stops a monitoring command that merely mentions a
# script name from counting as "busy" and stalling the queue.
gpu_busy() {
  local p c
  for p in $(pgrep -f "train_seg\.py|main\.py --output_dir|predict_to_coco\.py|export_dino_preds\.py" 2>/dev/null); do
    c=$(ps -o comm= -p "$p" 2>/dev/null)
    case "$c" in python*) return 0 ;; esac
  done
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]' && return 0
  return 1
}

[ -f reports/final_dino_test_bbox.json ] && { echo "final dino already scored"; exit 0; }

step "waiting for the DINO ablation / GPU"
while gpu_busy; do
  sleep 300
done
sleep 20

# candidate list: the pre-registered method, plus the valid-table winner if different
CANDS=$(python3 - <<'PY'
import json, glob, os
best, name = -1, None
for p in glob.glob("reports/dino_abl_*_valid_bbox.json"):
    a = os.path.basename(p).split("_")[2]
    if a == "D1":
        continue                       # the baseline is the reference, not a candidate
    m = json.load(open(p))["coco_stats"]["mAP"]
    if m > best:
        best, name = m, a
# D5 was the pre-registered method and it failed badly; it is still evaluated
# so the negative result is reported on test rather than quietly dropped.
out = ["D5"] + ([name] if name and name != "D5" else [])
print(" ".join(out))
PY
)
echo "candidates: $CANDS"

ARM_OPTS_PY='
ARM_OPTS = {
 "D1": "",
 "D2": "lt_la_loss=True",
 "D3": "lt_la_cost=True",
 "D4": "lt_freq_dn=True",
 "D5": "lt_la_loss=True lt_la_cost=True lt_freq_dn=True",
 "D6": "lt_la_loss=True lt_la_cost=True lt_freq_dn=True lt_rfs=True",
 "D7": "lt_la_loss=True lt_la_cost=True lt_freq_dn=True lt_clahe=True",
 "C1": "lt_cb_loss=True",
 "L1": "lt_la_cost=True lt_freq_dn=True",
 "L2": "lt_la_loss=True lt_freq_dn=True",
 "L3": "lt_la_loss=True lt_la_cost=True",
 "T05": "lt_la_loss=True lt_la_cost=True lt_freq_dn=True lt_tau=0.5",
 "T025": "lt_la_loss=True lt_la_cost=True lt_freq_dn=True lt_tau=0.25",
 # CRT ships a retrained classifier inside its own checkpoint; nothing is
 # switched on at inference, so it needs no options at all
 "CRT": "",
}'

for ARM in $CANDS; do
  REPORT="reports/final_dino_${ARM}_test_bbox"
  [ -f "${REPORT}.json" ] && { echo "[$(stamp)] $ARM already scored on test, skipping"; continue; }

  OPTS=$(python3 -c "$ARM_OPTS_PY
print(ARM_OPTS.get('$ARM',''))")
  CKPT="$ROOT/runs/dino_abl/$ARM/checkpoint.pth"
  [ "$ARM" = "D1" ] && CKPT="$ROOT/runs/dino_baseline/checkpoint.pth"
  if [ ! -f "$CKPT" ]; then echo "[$(stamp)] $ARM checkpoint missing: $CKPT"; continue; fi
  FLAGS=""; case "$OPTS" in *lt_clahe=True*) FLAGS="--clahe" ;; esac

  step "ONE-TIME test evaluation of $ARM  (opts: ${OPTS:-none})"
  python dino_longtail/export_dino_preds.py \
    --dino-root "$DINO" --config "$DINO/config/DINO/DINO_4scale.py" \
    --checkpoint "$CKPT" --coco-path "$ROOT/data_coco" --split test2017 \
    --gt data_clean/annotations/instances_test.json \
    --out "preds/final_dino_${ARM}_test.json" $FLAGS \
    --options num_classes=32 dn_labelbook_size=32 $OPTS 2>&1 | tail -3 || continue
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_test.json \
    --dt "preds/final_dino_${ARM}_test.json" \
    --train-json data_clean/annotations/instances_train.json \
    --iou-type bbox --out "$REPORT" 2>&1 | tail -4 || true
done

step "final test comparison vs the baseline"
python3 - <<'PY'
import json, glob, os, shutil
def row(name, p):
    d = json.load(open(p)); s, g = d["coco_stats"], d["group_AP"]
    return (name, s["mAP"], s["AP50"], s["AP75"],
            g.get("head", 0), g.get("mid", 0), g.get("tail", 0))
rows = []
base = "reports/baseline_dino_clean_test_bbox.json"
if os.path.exists(base):
    rows.append(row("D1 baseline", base))
for p in sorted(glob.glob("reports/final_dino_*_test_bbox.json")):
    arm = os.path.basename(p).split("_")[2]
    rows.append(row(arm, p))
print("%-16s %8s %8s %8s %8s %8s %8s"
      % ("model", "mAP", "AP50", "AP75", "head", "mid", "tail"))
for r in rows:
    print("%-16s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f" % r)
if len(rows) > 1:
    b = rows[0]
    print("\ndeltas vs baseline (percentage points):")
    for r in rows[1:]:
        print("  %-14s mAP %+.2f  AP50 %+.2f  AP75 %+.2f  head %+.2f  tail %+.2f"
              % (r[0], 100*(r[1]-b[1]), 100*(r[2]-b[2]), 100*(r[3]-b[3]),
                 100*(r[4]-b[4]), 100*(r[6]-b[6])))
# canonical file for the watchdog chain: the PRE-REGISTERED method, not the
# best number on the page
src = "reports/final_dino_D5_test_bbox.json"
if os.path.exists(src):
    shutil.copyfile(src, "reports/final_dino_test_bbox.json")
    print("\ncanonical final = D5 (pre-registered unified method)")
elif len(rows) > 1:
    print("\nD5 missing; no canonical final written")
PY

step "PROJECT 1 FINAL DONE"
