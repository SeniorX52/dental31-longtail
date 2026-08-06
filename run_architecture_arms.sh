#!/usr/bin/env bash
# Two architectural / training-strategy changes, each aimed at a MEASURED
# failure rather than an assumed one.
#
# Everything tried so far changed a loss term. Those changes all landed inside
# the noise floor on segmentation, and made detection dramatically worse. The
# two arms here are different in kind.
#
# ---------------------------------------------------------------------------
# ARM 1 (segmentation): prototype head fed from P2 instead of P3
#
# Measured problem. YOLOv8-seg rebuilds every mask on a 160x160 prototype grid.
# The median instance here is 6 px on that grid. Round-tripping the ground
# truth through it (tools/mask_resolution_ceiling.py) shows 17.7 % of instances
# cannot reach IoU 0.75 whatever the model does, with root canal treatment
# capped at 0.622 and caries at 0.721 -- both below the threshold. The model's
# own behaviour agrees: mask AP50 is 87 % of box AP50, mask AP75 only 49 %.
#
# The change. Proto is built from layer 15 (P3, stride 8, 80x80). Layer 2 (P2,
# stride 4, 160x160) carries genuine high-resolution detail and is unused by the
# head. Feeding the head from P2 gives 320x320 prototypes from real stride-4
# features rather than upsampled coarse ones. This is the established fix for
# small objects in this architecture family; PointRend, RefineMask and Mask
# Transfiner attack the same coarse-mask failure in two-stage detectors.
#
# Capacity does not increase: -878,432 parameters against stock, because cv1
# shrinks (160 vs 320 input channels), cv3 shrinks (64 vs 256), and the
# transposed upsample is replaced by parameter-free sub-pixel convolution. A
# gain cannot be attributed to a bigger model.
#
# XP2 is compared against XP3, an otherwise identical stock-prototype run at the
# same budget with the same validation setting, so the feature level is the only
# difference. Both use --no-val because ultralytics' own validator derives mask
# resolutions in a way that assumes the stock grid; scoring is done afterwards
# by predict_to_coco.py, which does not use that path.
#
# ---------------------------------------------------------------------------
# ARM 2 (detection): decoupled classifier retraining
#
# Measured problem. Every arm that put frequency-awareness inside the
# representation loss failed. D2 and D3 diverged with inf/NaN cost matrices;
# D5, D6 and D7 trained stably to a solution 6 pp WORSE than the baseline with
# tail AP at exactly 0.0000. The cause is magnitude: tau=1.0 puts a +11.47
# logit shift on the rarest class against +1.03 on the most common.
#
# The change. Kang et al. (ICLR 2020) showed representation learning and
# classifier learning should be decoupled: learn features on the natural
# distribution, then retrain ONLY the classifier under class-balanced sampling.
# CRT starts from the trained baseline, freezes everything except class_embed
# and label_enc, and retrains under repeat-factor sampling. Nothing is added to
# the loss and the matcher is untouched, so the failure mode that killed D2/D3
# cannot arise.
#
# Launch: setsid nohup bash run_architecture_arms.sh > logs/arch_arms.log 2>&1 </dev/null &
set -o pipefail

ROOT="$HOME/Documents/ML_SOTA"
DINO="$HOME/DINO"
cd "$ROOT"
mkdir -p logs reports preds runs runs/dino_abl
source "$HOME/miniconda3/bin/activate" dental
export PYTHONPATH="$ROOT:$DINO:$PYTHONPATH"

EPOCHS=50
BATCH=8
IMGSZ=640
CONF=0.15

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

# seg_arm <tag> <proto-src>
seg_arm() {
  local tag="$1" src="$2"
  local report="reports/ablation_${tag}_valid_segm"
  [ -f "${report}.json" ] && { echo "[$(stamp)] $tag already scored"; return 0; }
  local NAME="abl_${tag}" info n dir
  info=$(best_run "$NAME"); n=$(echo "$info"|cut -f1); dir=$(echo "$info"|cut -f2)

  if [ -n "$dir" ] && [ "${n:-0}" -ge "$EPOCHS" ]; then
    echo "[$(stamp)] $tag already trained ($dir)"
  else
    local resume=""
    [ -n "$dir" ] && [ -f "$dir/weights/last.pt" ] && [ "${n:-0}" -ge 2 ] && resume="--resume $dir/weights/last.pt"
    step "$tag : prototype head from ${src^^}, ${EPOCHS} ep ${resume:+[resuming from $n]}"
    python yolov8_seg_longtail/train_seg.py --data data_clean/data.yaml \
      --model yolov8x-seg.pt --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" \
      --seed 42 --cache ram --weights none --boundary-weight 0 \
      --proto-src "$src" --no-val $resume --name "$NAME" 2>&1 | tail -14
  fi

  info=$(best_run "$NAME"); n=$(echo "$info"|cut -f1); dir=$(echo "$info"|cut -f2)
  # --no-val means there is no best.pt; last.pt is the model
  local W="$dir/weights/last.pt"
  [ -f "$W" ] || { echo "[$(stamp)] $tag: no weights"; return 0; }
  if [ "${n:-0}" -lt "$EPOCHS" ] && ! python3 tools/run_finished.py "$dir" 2>/dev/null; then
    echo "[$(stamp)] $tag reached only ${n:-0}/$EPOCHS -- refusing to score"; return 0
  fi

  step "score $tag on VALID (${n}/$EPOCHS, last.pt)"
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

# ---- ARM 1: P2 prototypes, with its own matched stock control --------------
seg_arm XP3 p3      # control: stock prototype source, --no-val, last.pt
seg_arm XP2 p2      # the change: P2-fed prototypes at input/2

step "SEGMENTATION: P2 prototypes vs stock, matched budget and protocol"
python3 - <<'PY'
import json, os
def g(p):
    d=json.load(open(p)); s,q=d["coco_stats"],d["group_AP"]
    return s["mAP"],s["AP50"],s["AP75"],q.get("head",0),q.get("mid",0),q.get("tail",0)
a,b="reports/ablation_XP3_valid_segm.json","reports/ablation_XP2_valid_segm.json"
if os.path.exists(a) and os.path.exists(b):
    x,y=g(a),g(b)
    print("  %-28s %8s %8s %8s %8s %8s"%("arm","mAP","AP50","AP75","head","mid"))
    print("  %-28s %8.4f %8.4f %8.4f %8.4f %8.4f"%(("XP3 stock protos input/4",)+x[:5]))
    print("  %-28s %8.4f %8.4f %8.4f %8.4f %8.4f"%(("XP2 P2-fed protos input/2",)+y[:5]))
    print("\n  delta (pp):  mAP %+.2f  AP50 %+.2f  AP75 %+.2f  head %+.2f  mid %+.2f"
          %tuple(100*(y[i]-x[i]) for i in range(5)))
    print("\n  AP75 is the metric the ceiling caps: 17.7%% of instances are")
    print("  unreachable at input/4 versus 5.4%% at input/2.")
else:
    print("  one or both arms missing")
PY

# ---- ARM 2: decoupled classifier retraining for detection ------------------
CRT_OUT="$ROOT/runs/dino_abl/CRT"
CRT_REPORT="reports/dino_abl_CRT_valid_bbox"
if [ ! -f "${CRT_REPORT}.json" ]; then
  mkdir -p "$CRT_OUT"
  resume=""; [ -f "$CRT_OUT/checkpoint.pth" ] && resume="--resume $CRT_OUT/checkpoint.pth"
  step "CRT : classifier-only retrain from the trained baseline, balanced sampling"
  ( cd "$DINO" && python main.py \
      --output_dir "$CRT_OUT" -c config/DINO/DINO_4scale.py \
      --coco_path "$ROOT/data_coco" \
      --pretrain_model_path "$ROOT/runs/dino_baseline/checkpoint.pth" \
      --amp --seed 42 $resume --eval_every 4 \
      --options num_classes=32 dn_labelbook_size=32 batch_size=2 epochs=6 \
                lr_drop=5 save_checkpoint_interval=100 lt_crt=True lt_rfs=True 2>&1 | tail -12 )

  done_ep=$(wc -l < "$CRT_OUT/log.txt" 2>/dev/null || echo 0)
  if [ "${done_ep:-0}" -lt 6 ]; then
    echo "[$(stamp)] CRT reached only ${done_ep:-0}/6 epochs -- refusing to score"
  else
    step "score CRT on VALID (${done_ep}/6)"
    python dino_longtail/export_dino_preds.py \
      --dino-root "$DINO" --config "$DINO/config/DINO/DINO_4scale.py" \
      --checkpoint "$CRT_OUT/checkpoint.pth" --coco-path "$ROOT/data_coco" \
      --split val2017 --gt data_clean/annotations/instances_valid.json \
      --out preds/dino_abl_CRT_valid.json \
      --options num_classes=32 dn_labelbook_size=32 2>&1 | tail -3 && \
    python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
      --dt preds/dino_abl_CRT_valid.json \
      --train-json data_clean/annotations/instances_train.json \
      --iou-type bbox --out "$CRT_REPORT" 2>&1 | tail -4
  fi
fi

step "DETECTION: decoupled retraining against everything else tried"
python3 - <<'PY'
import json, os
def g(p):
    d=json.load(open(p)); s,q=d["coco_stats"],d["group_AP"]
    return s["mAP"],s["AP50"],s["AP75"],q.get("mid",0),q.get("tail",0)
rows=[("D1  baseline","reports/dino_abl_D1_valid_bbox.json"),
      ("C1  plain reweighting","reports/dino_abl_C1_valid_bbox.json"),
      ("D5  unified tau=1.0","reports/dino_abl_D5_valid_bbox.json"),
      ("T05 unified tau=0.5","reports/dino_abl_T05_valid_bbox.json"),
      ("CRT classifier retrain","reports/dino_abl_CRT_valid_bbox.json")]
print("  %-24s %8s %8s %8s %8s %8s"%("arm","mAP","AP50","AP75","mid","tail"))
base=None
for n,p in rows:
    if os.path.exists(p):
        v=g(p)
        if base is None: base=v
        print("  %-24s %8.4f %8.4f %8.4f %8.4f %8.4f"%((n,)+v))
    else:
        print("  %-24s %8s"%(n,"— not run"))
if base:
    print("\n  deltas vs baseline (pp):")
    for n,p in rows[1:]:
        if os.path.exists(p):
            v=g(p)
            print("    %-24s mAP %+.2f  AP50 %+.2f  AP75 %+.2f  mid %+.2f  tail %+.2f"
                  %((n,)+tuple(100*(v[i]-base[i]) for i in range(5))))
PY
step "ARCHITECTURE ARMS DONE"
