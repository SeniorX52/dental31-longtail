#!/usr/bin/env bash
# Two seed replicates of the one arm that beat the reproduced baseline.
#
# WHY THIS IS THE DECISIVE RUN. abl_K2_cv4wide reaches 0.1118 segm mAP against
# the 0.1055 of abl_S0, the reference recipe reproduced on the clean split at the
# same 50-epoch budget with one variable changed. That is +0.64 pp, or +6.0 %
# relative, and the noise floor is +-0.21 pp at two standard deviations. Three
# times the floor looks decisive until you notice what the floor measures: it
# was computed from three seeds of the BASELINE, so it describes the baseline's
# spread, not K2's. K2 itself is a single draw. An arm can sit three floors above
# a reference and still be a lucky initialisation, and this project has already
# retracted one result that looked stronger than that.
#
# Two more seeds settle it. If 1337 and 2024 both land near 0.1118 the effect
# replicates and the claim is defensible; if they scatter back toward 0.1055 the
# single seed was noise and the claim dies here rather than in a deliverable.
#
# The arm is also worth settling because its mechanism is unresolved. Paired on
# the 5352 cases where both models emit a mask, K2's masks are NOT better: Dice
# -0.0016 with the interval spanning zero, and boundary F -0.0043, separable and
# favouring the reference. So the mAP moved while the thing it is supposed to
# measure did not. Replication tells us whether there is anything to explain.
#
# WAITING WITHOUT RACING. Three drivers are already queued, and any fourth that
# waited only for a free GPU would collide with whichever of them wakes first.
# This one waits until no training python is running AND no earlier driver is
# alive at all. The driver test reads argv[1] rather than matching the command
# line: `pgrep -f` and `pkill -f` both also match any shell that merely mentions
# the script name, a self-match that has now produced five wrong readings in
# this project, one of them costing twelve hours of idle GPU and one of them
# killing the command that issued it.
#
# Usage:  nohup ./run_k2seeds.sh > logs/k2seeds.log 2>&1 &

cd "$HOME/Documents/ML_SOTA" || exit 1
source "$HOME/miniconda3/bin/activate" dental
set -u
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs reports preds

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
      ./run_hr1280.sh|*/run_hr1280.sh|\
      ./run_compound.sh|*/run_compound.sh|\
      ./run_maskdino.sh|*/run_maskdino.sh|\
      ./run_phase2.sh|*/run_phase2.sh) return 0 ;;
    esac
  done
  return 1
}

echo "[$(stamp)] K2 seed replicates queued behind the three drivers ahead"
w=0
while trainer_running || earlier_driver_running; do
  [ $((w % 1800)) -eq 0 ] && echo "[$(stamp)] waiting (${w}s)"
  sleep 120; w=$((w + 120))
done
sleep 60
echo "[$(stamp)] queue clear after ${w}s"

# Identical to abl_K2_cv4wide in every respect except --seed.
for s in 1337 2024; do
  tag="abl_K2_s${s}"
  if finished "$tag"; then echo "[$(stamp)] $tag already complete"; continue; fi
  RESUME=()
  [ -f "runs/segment/$tag/weights/last.pt" ] && \
    RESUME=(--resume "runs/segment/$tag/weights/last.pt")
  echo "[$(stamp)] === $tag ==="
  python yolov8_seg_longtail/train_seg.py \
      --data "$PWD/data_clean/data.yaml" --model yolov8x-seg.pt --nc 31 \
      --epochs 50 --imgsz 640 --batch 8 --seed "$s" --cache ram \
      --channels-last --weights none --boundary-weight 0 \
      --coeff-width 256 --name "$tag" "${RESUME[@]}" 2>&1 | tail -15
  finished "$tag" || { echo "[$(stamp)] *** $tag did NOT finish"; continue; }

  dt="preds/ablation_${tag}_valid.json"
  [ -f "$dt" ] || python yolov8_seg_longtail/predict_to_coco.py \
      --weights "runs/segment/$tag/weights/best.pt" \
      --gt data_clean/annotations/instances_valid.json \
      --images data_clean/valid/images --out "$dt" \
      --imgsz 640 --conf 0.001 --seed 42 2>&1 | tail -2
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
      --dt "$dt" --train-json data_clean/annotations/instances_train.json \
      --iou-type segm --out "reports/eval_${tag}_valid" 2>&1 | tail -3
done

echo "[$(stamp)] === verdict ==="
python - <<'PY'
import json, os, statistics as st
def m(f):
    return json.load(open(f))["coco_stats"]["mAP"] if os.path.exists(f) else None
base = [m("reports/ablation_%s_valid_segm.json" % t)
        for t in ("S0", "S0_s1337", "S0_s2024")]
k2 = [m("reports/eval_abl_K2_cv4wide_valid.json"),
      m("reports/eval_abl_K2_s1337_valid.json"),
      m("reports/eval_abl_K2_s2024_valid.json")]
base = [b for b in base if b]; k2 = [k for k in k2 if k]
print("  baseline seeds:", [round(b, 4) for b in base])
print("  K2 seeds      :", [round(k, 4) for k in k2])
if len(k2) >= 2 and len(base) >= 2:
    db, dk = st.mean(base), st.mean(k2)
    sb, sk = st.stdev(base), st.stdev(k2)
    print("  baseline %.4f +- %.4f | K2 %.4f +- %.4f" % (db, sb, dk, sk))
    print("  effect %+.2f pp; pooled 2sd about +-%.2f pp"
          % ((dk - db) * 100, 2 * ((sb ** 2 + sk ** 2) / 2) ** 0.5 * 100))
    print("  REPLICATES" if (dk - db) > 2 * ((sb ** 2 + sk ** 2) / 2) ** 0.5
          else "  DOES NOT SEPARATE from the baseline once K2's own spread is counted")
PY
echo "[$(stamp)] done"
