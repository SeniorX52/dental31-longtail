#!/usr/bin/env bash
# Scale the logit adjustment to the imbalance: a tau sweep on the unified arm.
#
# WHY
# The unified arm D5 (frequency-aware classification loss + matching cost +
# denoising, all at tau = 1.0) does not merely fail to help. On validation it
# is catastrophic:
#
#   arm                 mAP      AP50     AP75      mid      tail
#   D1 baseline      0.1625    0.3080   0.1551   0.2118    0.0368
#   D5 unified       0.1020    0.1998   0.0939   0.0959    0.0000
#
# -6.05 pp mAP, and tail AP collapses to exactly zero -- the opposite of the
# intended effect. The two single-component arms that carry the same adjustment,
# D2 (loss only) and D3 (matching only), diverged outright with inf and NaN cost
# matrices in the Hungarian matcher.
#
# The cause is the magnitude, not the idea. Logit adjustment subtracts
# tau * log p(c) from each class logit. On this dataset:
#
#   rarest present class   n = 1       log-prior -11.47   shift +11.47 at tau=1
#   most common class      n = 34318   log-prior  -1.03   shift  +1.03
#   spread across classes                                        10.44
#
# A +11.47 shift saturates the sigmoid for every query on that class. Menon et
# al. (2021) calibrated tau = 1.0 on CIFAR-LT and ImageNet-LT, whose imbalance
# is of order 100:1 to 1000:1. This dataset is 34,320:1, so the standard setting
# is far outside the regime it was validated in.
#
# tau = 1.0 was used everywhere because the protocol deliberately gave no arm a
# hyperparameter search, which keeps budgets matched. That is the right default
# when the setting is in-distribution; here it is the defect.
#
# WHAT THIS MEASURES
# The same unified configuration at tau = 0.5 and tau = 0.25, changing nothing
# else. If the method recovers, the finding is that the adjustment must be
# scaled to the imbalance and the standard value does not transfer -- which is a
# reportable result about the method's applicability, not a rescue. If it does
# not recover, the unified treatment simply fails on this data and the write-up
# says so.
#
# Budgets stay matched: 12 epochs, seed 42, same initialisation, same everything
# except tau. The sweep is disclosed as a sweep; D5 at tau = 1.0 remains in the
# frozen matrix exactly as it was run.
#
# Launch: setsid nohup bash run_dino_tau.sh > logs/dino_tau.log 2>&1 </dev/null &
set -o pipefail

ROOT="$HOME/Documents/ML_SOTA"
DINO="$HOME/DINO"
cd "$ROOT"
mkdir -p logs reports preds runs/dino_abl
source "$HOME/miniconda3/bin/activate" dental
export PYTHONPATH="$ROOT:$DINO:$PYTHONPATH"

GT_VALID="$ROOT/data_clean/annotations/instances_valid.json"
BASE_OPTS="num_classes=32 dn_labelbook_size=32 batch_size=2 epochs=12 lr_drop=11 save_checkpoint_interval=100"
EPOCHS_TARGET=12
UNIFIED="lt_la_loss=True lt_la_cost=True lt_freq_dn=True"

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

run_tau() {
  local tau="$1" name="T$2"
  local report="reports/dino_abl_${name}_valid_bbox"
  [ -f "${report}.json" ] && { echo "[$(stamp)] $name already scored, skipping"; return 0; }

  local outdir="$ROOT/runs/dino_abl/$name"
  local ckpt="$outdir/checkpoint.pth"
  mkdir -p "$outdir"
  local resume=""; [ -f "$ckpt" ] && resume="--resume $ckpt"
  step "arm $name : unified at tau=$tau ${resume:+[resuming]}"
  printf '%s\n' "arm=$name" "opts=$UNIFIED lt_tau=$tau" "base=$BASE_OPTS" "seed=42" \
    > "$outdir/arm_config.txt"
  ( cd "$DINO" && python main.py \
      --output_dir "$outdir" -c config/DINO/DINO_4scale.py \
      --coco_path "$ROOT/data_coco" \
      --pretrain_model_path "$ROOT/weights/checkpoint0033_4scale.pth" \
      --finetune_ignore class_embed label_enc \
      --amp --seed 42 $resume --eval_every 4 \
      --options $BASE_OPTS $UNIFIED lt_tau=$tau 2>&1 | tail -12 )

  local done_ep
  done_ep=$(wc -l < "$outdir/log.txt" 2>/dev/null || echo 0)
  if [ "${done_ep:-0}" -lt "$EPOCHS_TARGET" ]; then
    echo "[$(stamp)] $name reached only ${done_ep:-0}/$EPOCHS_TARGET epochs -- REFUSING to score."
    echo "            A divergence here is itself a result: record it, do not score it."
    return 0
  fi

  step "score $name on VALID (verified ${done_ep}/$EPOCHS_TARGET)"
  python dino_longtail/export_dino_preds.py \
    --dino-root "$DINO" --config "$DINO/config/DINO/DINO_4scale.py" \
    --checkpoint "$ckpt" --coco-path "$ROOT/data_coco" --split val2017 \
    --gt "$GT_VALID" --out "preds/dino_abl_${name}_valid.json" \
    --options $BASE_OPTS $UNIFIED lt_tau=$tau 2>&1 | tail -3 || return 0
  python eval/coco_eval_report.py --gt "$GT_VALID" \
    --dt "preds/dino_abl_${name}_valid.json" \
    --train-json data_clean/annotations/instances_train.json \
    --iou-type bbox --out "$report" 2>&1 | tail -4 || true
}

step "waiting for the GPU"
while gpu_busy; do sleep 300; done
sleep 20

run_tau 0.50 05
run_tau 0.25 025

step "TAU SWEEP RESULT"
python3 - <<'PY'
import json, os
def g(p):
    d=json.load(open(p)); s,q=d["coco_stats"],d["group_AP"]
    return s["mAP"],s["AP50"],s["AP75"],q.get("head",0),q.get("mid",0),q.get("tail",0)
rows=[("D1  baseline","reports/dino_abl_D1_valid_bbox.json"),
      ("D5  unified tau=1.00","reports/dino_abl_D5_valid_bbox.json"),
      ("T05 unified tau=0.50","reports/dino_abl_T05_valid_bbox.json"),
      ("T025 unified tau=0.25","reports/dino_abl_T025_valid_bbox.json")]
print("  %-22s %8s %8s %8s %8s %8s"%("arm","mAP","AP50","AP75","mid","tail"))
base=None
for n,p in rows:
    if os.path.exists(p):
        v=g(p)
        if base is None: base=v
        print("  %-22s %8.4f %8.4f %8.4f %8.4f %8.4f"%(n,v[0],v[1],v[2],v[4],v[5]))
    else:
        print("  %-22s %8s"%(n,"— not run"))
if base:
    print("\n  deltas vs baseline (pp):")
    for n,p in rows[1:]:
        if os.path.exists(p):
            v=g(p)
            print("    %-22s mAP %+.2f  AP50 %+.2f  AP75 %+.2f  mid %+.2f  tail %+.2f"
                  %(n,100*(v[0]-base[0]),100*(v[1]-base[1]),100*(v[2]-base[2]),
                    100*(v[4]-base[4]),100*(v[5]-base[5])))
PY
step "TAU SWEEP DONE"
