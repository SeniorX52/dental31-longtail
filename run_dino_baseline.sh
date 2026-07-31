#!/usr/bin/env bash
# Project-1 baseline: fine-tune the stock DINO-DETR COCO checkpoint to the
# dental classes on the frozen leakage-free split, then score it through the
# same COCO reporter used for Project 2.
#
# Detached launch (survives SSH/session loss):
#   setsid nohup bash run_dino_baseline.sh > logs/dino_baseline.log 2>&1 </dev/null &
set -eo pipefail

ROOT="$HOME/Documents/ML_SOTA"
DINO="$HOME/DINO"
cd "$ROOT"
mkdir -p logs reports preds runs/dino_baseline
source "$HOME/miniconda3/bin/activate" dental

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
step()  { echo; echo "=== [$(stamp)] $* ==="; }

step "0/3 environment"
python - <<'PY'
import torch, sys
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("CUDA unavailable -- enroll the MOK key or disable Secure Boot first")
print("device:", torch.cuda.get_device_name(0),
      "| %.1f GB" % (torch.cuda.get_device_properties(0).total_memory/1e9))
import MultiScaleDeformableAttention  # noqa: F401
print("deformable-attention op: OK")
PY

step "1/3 fine-tune DINO-DETR (12 epochs)"
# num_classes=32, not 31: DINO uses the raw category_id as the label
# (datasets/coco.py -> classes = [obj["category_id"] ...], no remapping), and
# our ids run 1..31, so the head needs max_id+1 outputs. Same reason COCO-DETR
# uses 91 for 80 classes.
#
# --finetune_ignore class_embed label_enc is the verified minimal cover: it
# drops exactly the 27 COCO class-head tensors and loads the other 599.
# Passing "transformer" would also match the whole encoder/decoder stack.
cd "$DINO"
python main.py \
  --output_dir "$ROOT/runs/dino_baseline" \
  -c config/DINO/DINO_4scale.py \
  --coco_path "$ROOT/data_coco" \
  --pretrain_model_path "$ROOT/weights/checkpoint0033_4scale.pth" \
  --finetune_ignore class_embed label_enc \
  --amp --seed 42 \
  --options num_classes=32 dn_labelbook_size=32 batch_size=2 epochs=12 lr_drop=11 \
  2>&1 | tail -60
cd "$ROOT"

step "2/3 export test-split detections"
CKPT="runs/dino_baseline/checkpoint.pth"
[ -f "$CKPT" ] || CKPT=$(ls -t runs/dino_baseline/checkpoint*.pth 2>/dev/null | head -1)
echo "checkpoint: $CKPT"
PYTHONPATH="$DINO" python dino_longtail/export_dino_preds.py \
  --dino-root "$DINO" --config "$DINO/config/DINO/DINO_4scale.py" \
  --checkpoint "$CKPT" --coco-path "$ROOT/data_coco" --split test2017 \
  --gt data_clean/annotations/instances_test.json \
  --out preds/dino_baseline_test.json \
  --options num_classes=32 dn_labelbook_size=32 2>&1 | tail -10

step "3/3 score with the shared reporter"
python eval/coco_eval_report.py \
  --gt data_clean/annotations/instances_test.json \
  --dt preds/dino_baseline_test.json \
  --train-json data_clean/annotations/instances_train.json \
  --iou-type bbox --out reports/baseline_dino_clean_test_bbox 2>&1 | tail -8

step "DONE"
