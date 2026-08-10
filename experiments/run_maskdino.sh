#!/usr/bin/env bash
# P1: Mask DINO R50 on the dental corpus, queued behind the 1280 probe.
#
# The family swap the evidence points at. The oracle fit put 85 percent of the
# mask deficit in coefficient prediction; adding capacity (K2) and adding a
# direct signal (K1b) both failed, so the 32-prototype linear composition is
# the indicted mechanism, not its inputs. Mask DINO predicts each mask as a
# query embedding dotted with a pixel embedding map, no shared low-rank basis,
# and on COCO its R50 config reports 51.5 box / 46.1 mask, above both DINO-R50
# (the detection baseline here) and YOLOv8x-seg masks. One model, both
# projects' deliverables.
#
# Schedule: 12-epoch equivalent. 9,752 train images at IMS_PER_BATCH 2 gives
# 4,876 iters/epoch, 58,500 total, LR step at 53,600 (the 11-epoch mark). LR is
# the paper's 1e-4 linearly scaled by 2/16 to 1.25e-5. COCO-pretrained
# initialisation; NUM_CLASSES changed to 31, the checkpoint's 80-way class
# heads are shape-mismatched and detectron2 skips them, which is the intended
# behaviour, everything else transfers.
#
# SMOKE TEST FIRST. 40 iters before committing 8 hours: catches OOM, mapper
# and registration failures while they cost seconds. Their bs16 config ran
# bs2-per-GPU on 32 GB V100s; this card has 16 GB, so OOM is a live risk. On
# smoke OOM the fallback drops LSJ crop size 1024 -> 800 and retries once.
#
# Usage:  nohup ./run_maskdino.sh > logs/maskdino.log 2>&1 &
cd "$HOME/MaskDINO" || exit 1
source "$HOME/miniconda3/bin/activate" dental
set -u
export PYTHONPATH="$HOME/MaskDINO:${PYTHONPATH:-}"
export DETECTRON2_DATASETS=/nonexistent   # everything is registered explicitly

ML=$HOME/Documents/ML_SOTA
CFG=configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml
OUT=output/dental_r50
mkdir -p "$OUT" "$ML/logs"

stamp() { date '+%F %T'; }

COMMON_OPTS=(
  MODEL.WEIGHTS weights/maskdino_r50_coco_instance.pth
  MODEL.SEM_SEG_HEAD.NUM_CLASSES 31
  DATASETS.TRAIN '("dental_train",)'
  DATASETS.TEST  '("dental_valid",)'
  SOLVER.IMS_PER_BATCH 2
  SOLVER.BASE_LR 0.0000125
  SOLVER.MAX_ITER 58500
  SOLVER.STEPS '(53600,)'
  SOLVER.CHECKPOINT_PERIOD 5000
  TEST.EVAL_PERIOD 10000
  DATALOADER.NUM_WORKERS 4
  OUTPUT_DIR "$OUT"
)

gpu_busy() {
  local p c
  for p in $(pgrep -f "train_seg\.py|main\.py --output_dir|predict_to_coco|train_dental\.py" 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    c=$(ps -o comm= -p "$p" 2>/dev/null)
    case "$c" in python*|pt_data*) return 0 ;; esac
  done
  return 1
}

echo "[$(stamp)] waiting for the GPU (1280 probe finishes first)"
while gpu_busy; do sleep 120; done
sleep 60
echo "[$(stamp)] GPU free"

smoke() {  # $@ = extra cfg overrides
  echo "[$(stamp)] smoke test, 40 iters: $*"
  python train_dental.py --config-file "$CFG" --num-gpus 1 \
    "${COMMON_OPTS[@]}" SOLVER.MAX_ITER 40 TEST.EVAL_PERIOD 0 \
    OUTPUT_DIR output/smoke "$@" > "$ML/logs/maskdino_smoke.log" 2>&1
}

EXTRA=()
if smoke; then
  echo "[$(stamp)] smoke passed at IMAGE_SIZE 1024"
elif grep -qi "out of memory" "$ML/logs/maskdino_smoke.log"; then
  echo "[$(stamp)] OOM at 1024; retrying smoke at INPUT.IMAGE_SIZE 800"
  EXTRA=(INPUT.IMAGE_SIZE 800)
  if smoke "${EXTRA[@]}"; then
    echo "[$(stamp)] smoke passed at 800"
  else
    echo "[$(stamp)] smoke FAILED at 800 too; see logs/maskdino_smoke.log"; exit 1
  fi
else
  echo "[$(stamp)] smoke FAILED (not OOM); see logs/maskdino_smoke.log"; exit 1
fi
rm -rf output/smoke

echo "[$(stamp)] === full run: 58,500 iters (~12 epochs) ==="
python train_dental.py --config-file "$CFG" --num-gpus 1 \
  "${COMMON_OPTS[@]}" "${EXTRA[@]}" 2>&1 | tail -30

echo "[$(stamp)] training exited; exporting predictions for the shared eval"
# detectron2's evaluator writes COCO-format instances with the ORIGINAL
# category ids, so the project's own scoring pipeline consumes them directly.
python train_dental.py --config-file "$CFG" --num-gpus 1 --eval-only \
  "${COMMON_OPTS[@]}" "${EXTRA[@]}" \
  MODEL.WEIGHTS "$OUT/model_final.pth" 2>&1 | tail -15

PRED="$OUT/inference/coco_instances_results.json"
if [ -f "$PRED" ]; then
  cp "$PRED" "$ML/preds/maskdino_r50_valid.json"
  cd "$ML"
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
    --dt preds/maskdino_r50_valid.json \
    --train-json data_clean/annotations/instances_train.json \
    --iou-type segm --out reports/eval_maskdino_r50_valid_segm 2>&1 | tail -4
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
    --dt preds/maskdino_r50_valid.json \
    --train-json data_clean/annotations/instances_train.json \
    --iou-type bbox --out reports/eval_maskdino_r50_valid_bbox 2>&1 | tail -4
  PYTHONPATH="$ML/eval:${PYTHONPATH:-}" python eval/paired_contour.py \
    --gt data_clean/annotations/instances_valid.json \
    --dt-a preds/ablation_S0_valid.json --label-a S0 \
    --dt-b preds/maskdino_r50_valid.json --label-b maskdino_r50 \
    --conf 0.15 --boot 500 \
    --out reports/paired_contour_S0_maskdino_r50_valid 2>&1 | tail -8
else
  echo "[$(stamp)] *** no predictions at $PRED; eval-only step failed"
fi
echo "[$(stamp)] done"
