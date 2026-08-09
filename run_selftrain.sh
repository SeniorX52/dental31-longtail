#!/usr/bin/env bash
# PRIORITY 2: self-training on ~2,900 unlabeled DENTEX panoramics.
#
# STAC recipe (Sohn et al., arXiv:2005.04757): teacher pseudo-labels unlabeled
# domain images at tau = 0.9, student trains on labeled + pseudo-labeled
# together (our mixing corresponds to lambda_u = 1, STAC's conservative end;
# STAC reports +4.8 to +5.9 mAP on COCO low-label protocols). Inference on the
# unlabeled pool runs at imgsz 2176, the size that matches DENTEX's apparent
# object scale to ours (2744 * 1280/1615), per the scale analysis in
# reports/dentex_toothlevel*. Deviation from STAC, stated: their student adds
# color+geometry+Cutout strong augmentation; ours uses the pipeline's standard
# mosaic+HSV+flip, so any gain here is a floor on what the full recipe offers.
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
echo "[$(stamp)] self-training driver queued"
wait_on run_maskdino.sh run_hr1600.sh run_compound.sh run_phase2.sh run_k2seeds.sh run_labelnoise.sh

D=/media/mostafa/EGYPT_SSD/dental31/dentex/extracted/training_data
U=data_selftrain/unlabeled_images
mkdir -p "$U"
for d in "$D/unlabelled/xrays" "$D/quadrant/xrays" "$D/quadrant_enumeration/xrays"; do
  [ -d "$d" ] || continue
  for f in "$d"/*; do
    t="$U/$(basename "$(dirname "$(dirname "$f")")")__$(basename "$f")"
    [ -e "$t" ] || ln -s "$f" "$t"
  done
done
echo "[$(stamp)] unlabeled pool: $(ls "$U" | wc -l) images"

# teacher = best confirmed high-res model at run time
TEACHER=$(python - <<'PY'
import json, os
def m(f): return json.load(open(f))["coco_stats"]["mAP"] if os.path.exists(f) else 0
cands = {"runs/segment/abl_HR1280ft/weights/best.pt": m("reports/eval_abl_HR1280ft_valid.json"),
         "runs/segment/abl_SCRATCH_s42/weights/best.pt": m("reports/eval_abl_SCRATCH_s42_valid.json")}
best = max(cands, key=cands.get)
print(best if os.path.exists(best) else "runs/segment/abl_HR1280ft/weights/best.pt")
PY
)
echo "[$(stamp)] teacher: $TEACHER"

[ -f data_selftrain/skel.json ] || python tools/pseudo_label.py index \
    --images "$U" --like data_clean/annotations/instances_valid.json \
    --out data_selftrain/skel.json
[ -f preds/pseudo_teacher.json ] || python yolov8_seg_longtail/predict_to_coco.py \
    --weights "$TEACHER" --gt data_selftrain/skel.json --images "$U" \
    --out preds/pseudo_teacher.json --imgsz 2176 --conf 0.05 --seed 42 2>&1 | tail -2
python tools/pseudo_label.py labels --preds preds/pseudo_teacher.json \
    --skel data_selftrain/skel.json --tau 0.9 \
    --out-labels data_selftrain/train/labels

# merged corpus: our clean train + the pseudo-labeled pool
mkdir -p data_selftrain/train/images
for f in data_clean/train/images/*; do
  t="data_selftrain/train/images/$(basename "$f")"; [ -e "$t" ] || ln -s "$(readlink -f "$f")" "$t"
done
for f in data_clean/train/labels/*; do
  t="data_selftrain/train/labels/$(basename "$f")"; [ -e "$t" ] || ln -s "$(readlink -f "$f")" "$t"
done
for f in "$U"/*; do
  t="data_selftrain/train/images/$(basename "$f")"; [ -e "$t" ] || ln -s "$(readlink -f "$f")" "$t"
done
python - <<'PY'
import yaml, os
yaml.safe_dump({"path": os.path.abspath("data_selftrain"),
  "train": "train/images",
  "val": os.path.abspath("data_clean/valid/images"),
  "test": os.path.abspath("data_clean/test/images"),
  "names": yaml.safe_load(open("data_clean/data.yaml"))["names"]},
  open("data_selftrain/data.yaml","w"), sort_keys=False)
print("  data_selftrain/data.yaml written")
PY

train_ft abl_ST1280 runs/segment/abl_S0/weights/best.pt 1280 2 25 42 "$PWD/data_selftrain/data.yaml"
score    abl_ST1280 1280
echo "[$(stamp)] === verdict vs abl_HR1280ft (same recipe, no pseudo data) ==="
python - <<'PY'
import json, os
def m(f): return json.load(open(f))["coco_stats"]["mAP"] if os.path.exists(f) else None
a, b = m("reports/eval_abl_HR1280ft_valid.json"), m("reports/eval_abl_ST1280_valid.json")
if a and b: print("  HR1280 %.4f -> self-trained %.4f  (%+.2f pp)" % (a, b, (b-a)*100))
PY
echo "[$(stamp)] done"
