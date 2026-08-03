#!/usr/bin/env bash
# Project-1 (detection) ablation queue. Grid from dino_longtail/INTEGRATION.md:
#
#   A0   baseline (already trained)          score-only on VALID
#   A1   + repeat-factor sampling
#   A2   A1 + logit-adjusted loss
#   A3   A2 + logit-adjusted matching cost
#   A4   A3 + frequency-aware denoising
#   A5   A4 + CLAHE
#   A2p  RFS + adjusted COST only            isolates the matching-consistency claim
#
# Each arm: 12 epochs (official DINO 1x), seed 42, resumable via DINO's own
# --resume from checkpoint.pth. Arms are scored on VALID by the shared scorer;
# the final configuration alone will be evaluated on TEST.
#
# Launch:  setsid nohup bash run_dino_ablation.sh > logs/dino_ablation.log 2>&1 </dev/null &
set -o pipefail

ROOT="$HOME/Documents/ML_SOTA"
DINO="$HOME/DINO"
cd "$ROOT"
mkdir -p logs reports preds runs/dino_abl
source "$HOME/miniconda3/bin/activate" dental
export PYTHONPATH="$ROOT:$DINO:$PYTHONPATH"

GT_VALID="$ROOT/data_clean/annotations/instances_valid.json"
BASE_OPTS="num_classes=32 dn_labelbook_size=32 batch_size=2 epochs=12 lr_drop=11"

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
step()  { echo; echo "=== [$(stamp)] $* ==="; }

# run_arm <name> <extra options...>   ("-" = none / score-only uses existing ckpt)
run_arm() {
  local name="$1"; shift
  local opts="$*"
  local report="reports/dino_ablation_${name}_valid_bbox"
  [ -f "${report}.json" ] && { echo "[$(stamp)] $name already scored, skipping"; return 0; }

  local outdir="$ROOT/runs/dino_abl/$name"
  local ckpt="$outdir/checkpoint.pth"
  local export_flags=""
  case "$opts" in *lt_clahe=True*) export_flags="--clahe" ;; esac

  if [ "$name" = "A0" ]; then
    ckpt="$ROOT/runs/dino_baseline/checkpoint.pth"
    [ -f "$ckpt" ] || { echo "A0 baseline checkpoint missing"; return 0; }
  else
    mkdir -p "$outdir"
    local resume=""
    [ -f "$ckpt" ] && resume="--resume $ckpt"
    step "arm $name  ($opts) ${resume:+[resuming]}"
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
while pgrep -f "run_seg_ablation|final_seg_run|train_seg.py|predict_to_coco" >/dev/null 2>&1 \
   || nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; do
  sleep 300
done
sleep 20

run_arm A0  -
run_arm A1  lt_rfs=True
run_arm A2  lt_rfs=True lt_la_loss=True
run_arm A3  lt_rfs=True lt_la_loss=True lt_la_cost=True
run_arm A4  lt_rfs=True lt_la_loss=True lt_la_cost=True lt_freq_dn=True
run_arm A5  lt_rfs=True lt_la_loss=True lt_la_cost=True lt_freq_dn=True lt_clahe=True
run_arm A2p lt_rfs=True lt_la_cost=True

step "detection ablation table"
python3 - <<'PY'
import json, glob, os
rows=[]
for p in sorted(glob.glob("reports/dino_ablation_A*_valid_bbox.json")):
    d=json.load(open(p)); g=d["group_AP"]
    rows.append((os.path.basename(p).split("_")[2], d["coco_stats"]["mAP"],
                 d["coco_stats"]["AP50"], g.get("head",0), g.get("mid",0), g.get("tail",0)))
print("%-5s %8s %8s %9s %9s %9s" % ("arm","mAP","AP50","head","mid","tail"))
for r in rows: print("%-5s %8.4f %8.4f %9.4f %9.4f %9.4f" % r)
PY
step "QUEUE DINO DONE"
