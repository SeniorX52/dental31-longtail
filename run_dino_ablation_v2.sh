#!/usr/bin/env bash
# Project-1 (detection) ablation -- FROZEN MATRIX.
#
# This replaces run_dino_ablation.sh, whose grid was cumulative (every arm
# after A0 also carried repeat-factor sampling) and therefore could not
# attribute anything to the individual components: "RFS+loss" minus "RFS" only
# isolates the loss *in the presence of RFS*, and oversampling is a data-level
# intervention that interacts with all three.
#
# The matrix below is frozen before any of it runs. The claim under
# test is that frequency-awareness applied CONSISTENTLY across classification
# loss, Hungarian matching and denoising beats applying it in one place and
# beats ordinary loss reweighting. So each component is measured alone, the
# unified method is measured, and oversampling / contrast enhancement are
# supporting ablations layered on top rather than headline contributions.
#
#   D1  standard DINO-DETR                              (baseline, already trained)
#   D2  frequency-aware classification loss only
#   D3  frequency-aware matching cost only
#   D4  frequency-aware denoising only
#   D5  unified: loss + matching + denoising            <- the proposed method
#   D6  D5 + rare-class image oversampling (RFS)
#   D7  D5 + contrast enhancement (CLAHE)
#   C1  conventional class-balanced reweighting          <- the "is this just
#                                                           reweighting?" control
#   L1  D5 minus loss      (matching + denoising)
#   L2  D5 minus matching  (loss + denoising)
#   L3  D5 minus denoising (loss + matching)
#
# D1-D7 + C1 are the frozen deliverable and run first; L1-L3 are the
# leave-one-out complement and run only if the GPU frees up in time. Ordering
# is deliberate: a partial run still yields the complete frozen matrix.
#
# MATCHED BUDGET. Every arm gets identical settings -- 12 epochs (official DINO
# 1x), lr_drop 11, batch_size 2, seed 42, same initialization checkpoint, same
# input resolution, same evaluation protocol. No arm receives any
# hyperparameter search, including the proposed method: tau is fixed at its
# 1.0 default and beta at 0.99 throughout. Budgets are matched at "one
# configuration per cell, no search", which is stated in the write-up rather
# than left for a reviewer to ask about.
#
# Launch:  setsid nohup bash run_dino_ablation_v2.sh > logs/dino_ablation_v2.log 2>&1 </dev/null &
set -o pipefail

ROOT="$HOME/Documents/ML_SOTA"
DINO="$HOME/DINO"
cd "$ROOT"
mkdir -p logs reports preds runs/dino_abl
source "$HOME/miniconda3/bin/activate" dental
export PYTHONPATH="$ROOT:$DINO:$PYTHONPATH"

GT_VALID="$ROOT/data_clean/annotations/instances_valid.json"
# save_checkpoint_interval=100 is a DISK fix, not a training change. The DINO
# config ships with interval 1, so every epoch writes a 561 MB checkpoint and
# one 12-epoch arm costs 7.4 GB. Eleven arms would need ~81 GB against 59 GB
# free, and the queue would die part-way through with the disk full.
#
# It is accuracy-neutral: checkpoint.pth is still written every epoch (so
# --resume is unaffected), and the protocol selects the LAST epoch, never an
# intermediate one. All this suppresses is the per-epoch archive copies, taking
# an arm from ~7.4 GB to ~1.7 GB.
BASE_OPTS="num_classes=32 dn_labelbook_size=32 batch_size=2 epochs=12 lr_drop=11 save_checkpoint_interval=100"

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
step()  { echo; echo "=== [$(stamp)] $* ==="; }

# gpu_busy: true only when a REAL python training/eval process is alive, or the
# GPU has a compute app attached. Filtering by the process's comm (python*)
# rather than matching `pgrep -f` against full command lines means a monitoring
# command that merely mentions a script name no longer counts as "busy" and
# stalls the queue for another 300 s.
gpu_busy() {
  local p c
  for p in $(pgrep -f "train_seg\.py|main\.py --output_dir|predict_to_coco\.py|export_dino_preds\.py" 2>/dev/null); do
    c=$(ps -o comm= -p "$p" 2>/dev/null)
    case "$c" in python*) return 0 ;; esac
  done
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]' && return 0
  return 1
}

# run_arm <name> <extra options...>   ("-" = score-only, uses the existing ckpt)
run_arm() {
  local name="$1"; shift
  local opts="$*"
  [ "$opts" = "-" ] && opts=""
  local report="reports/dino_abl_${name}_valid_bbox"
  [ -f "${report}.json" ] && { echo "[$(stamp)] $name already scored, skipping"; return 0; }

  local outdir="$ROOT/runs/dino_abl/$name"
  local ckpt="$outdir/checkpoint.pth"
  local export_flags=""
  case "$opts" in *lt_clahe=True*) export_flags="--clahe" ;; esac

  if [ "$name" = "D1" ]; then
    # the frozen-matrix baseline IS the already-trained stock fine-tune;
    # retraining it would only add seed noise to the reference cell
    ckpt="$ROOT/runs/dino_baseline/checkpoint.pth"
    [ -f "$ckpt" ] || { echo "D1 baseline checkpoint missing"; return 0; }
  else
    mkdir -p "$outdir"
    local resume=""
    [ -f "$ckpt" ] && resume="--resume $ckpt"
    step "arm $name  ($opts) ${resume:+[resuming]}"
    # record the exact configuration next to the weights, so the matched-budget
    # claim is checkable from the artifacts and not just from this script
    printf '%s\n' "arm=$name" "opts=$opts" "base=$BASE_OPTS" "seed=42" \
      > "$outdir/arm_config.txt"
    ( cd "$DINO" && python main.py \
        --output_dir "$outdir" -c config/DINO/DINO_4scale.py \
        --coco_path "$ROOT/data_coco" \
        --pretrain_model_path "$ROOT/weights/checkpoint0033_4scale.pth" \
        --finetune_ignore class_embed label_enc \
        --amp --seed 42 $resume \
        --eval_every "${EVAL_EVERY:-4}" \
        --options $BASE_OPTS $opts 2>&1 | tail -12 )
    [ -f "$ckpt" ] || { echo "[$(stamp)] $name: no checkpoint produced"; return 0; }
  fi

  step "score $name on VALID"
  if ! python dino_longtail/export_dino_preds.py \
        --dino-root "$DINO" --config "$DINO/config/DINO/DINO_4scale.py" \
        --checkpoint "$ckpt" --coco-path "$ROOT/data_coco" --split val2017 \
        --gt "$GT_VALID" --out "preds/dino_abl_${name}_valid.json" $export_flags \
        --options $BASE_OPTS $opts 2>&1 | tail -3; then
    echo "[$(stamp)] $name: export failed, continuing"; return 0
  fi
  python eval/coco_eval_report.py --gt "$GT_VALID" \
    --dt "preds/dino_abl_${name}_valid.json" \
    --train-json data_clean/annotations/instances_train.json \
    --iou-type bbox --out "$report" 2>&1 | tail -4 \
    || echo "[$(stamp)] $name: eval failed, continuing"
}

step "waiting for the segmentation pipeline / GPU"
while gpu_busy; do
  sleep 300
done
sleep 20

# ---- the frozen matrix (deliverable) ----
run_arm D1  -
run_arm D2  lt_la_loss=True
run_arm D3  lt_la_cost=True
run_arm D4  lt_freq_dn=True
run_arm D5  lt_la_loss=True lt_la_cost=True lt_freq_dn=True
run_arm D6  lt_la_loss=True lt_la_cost=True lt_freq_dn=True lt_rfs=True
run_arm D7  lt_la_loss=True lt_la_cost=True lt_freq_dn=True lt_clahe=True
run_arm C1  lt_cb_loss=True

# ---- leave-one-out complement (runs only if there is time) ----
run_arm L1  lt_la_cost=True lt_freq_dn=True
run_arm L2  lt_la_loss=True lt_freq_dn=True
run_arm L3  lt_la_loss=True lt_la_cost=True

step "detection ablation table (VALID)"
python3 - <<'PY'
import json, glob, os
LABEL = {
    "D1": "standard DINO-DETR",
    "D2": "freq-aware loss only",
    "D3": "freq-aware matching only",
    "D4": "freq-aware denoising only",
    "D5": "UNIFIED (loss+match+dn)",
    "D6": "unified + oversampling",
    "D7": "unified + CLAHE",
    "C1": "class-balanced control",
    "L1": "unified - loss",
    "L2": "unified - matching",
    "L3": "unified - denoising",
}
ORDER = ["D1","D2","D3","D4","D5","D6","D7","C1","L1","L2","L3"]
rows = {}
for p in glob.glob("reports/dino_abl_*_valid_bbox.json"):
    arm = os.path.basename(p).split("_")[2]
    d = json.load(open(p)); g = d["group_AP"]; s = d["coco_stats"]
    rows[arm] = (s["mAP"], s["AP50"], s["AP75"],
                 g.get("head", 0), g.get("mid", 0), g.get("tail", 0))
print("%-4s %-26s %8s %8s %8s %8s %8s %8s"
      % ("arm", "configuration", "mAP", "AP50", "AP75", "head", "mid", "tail"))
for a in ORDER:
    if a in rows:
        print("%-4s %-26s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f" % ((a, LABEL[a]) + rows[a]))
missing = [a for a in ORDER if a not in rows]
if missing:
    print("\nnot yet run: %s" % " ".join(missing))
if "D1" in rows:
    b = rows["D1"]
    print("\ndeltas vs D1 (percentage points):")
    for a in ORDER[1:]:
        if a in rows:
            r = rows[a]
            print("  %-4s %-26s mAP %+.2f  AP50 %+.2f  AP75 %+.2f  head %+.2f  tail %+.2f"
                  % (a, LABEL[a], 100*(r[0]-b[0]), 100*(r[1]-b[1]), 100*(r[2]-b[2]),
                     100*(r[3]-b[3]), 100*(r[5]-b[5])))
    if "D5" in rows and "C1" in rows:
        u, c = rows["D5"], rows["C1"]
        print("\nthe control question -- unified vs ordinary reweighting:")
        print("  D5 - C1:  mAP %+.2f  AP75 %+.2f  tail %+.2f pp"
              % (100*(u[0]-c[0]), 100*(u[2]-c[2]), 100*(u[5]-c[5])))
PY
step "QUEUE DINO v2 DONE"
