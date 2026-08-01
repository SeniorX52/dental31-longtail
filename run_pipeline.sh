#!/usr/bin/env bash
# Master queue: wait for the GPU, benchmark accuracy-neutral speedups, then run
# the whole Project-2 ablation with the winning settings applied UNIFORMLY.
#
# Uniformly matters. A speed setting that is applied to some arms and not
# others would confound the ablation, so the benchmark runs once, up front, and
# every arm (including the S0 reference) then trains under identical settings.
#
# Launch:
#   setsid nohup bash run_pipeline.sh > logs/pipeline.log 2>&1 </dev/null &
set -eo pipefail

ROOT="$HOME/Documents/ML_SOTA"
cd "$ROOT"
mkdir -p logs reports preds runs
source "$HOME/miniconda3/bin/activate" dental

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
step()  { echo; echo "=== [$(stamp)] $* ==="; }

step "0 waiting for the GPU to go idle"
while pgrep -f "run_dino_baseline.sh|main.py --output_dir|export_dino_preds" >/dev/null 2>&1 \
   || nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; do
  sleep 300
done
sleep 30
echo "[$(stamp)] GPU free"

step "1 benchmark accuracy-neutral speedups"
if [ ! -f reports/speed_bench.json ]; then
  python tools/bench_train_speed.py \
    --data "$ROOT/data_clean/data.yaml" --model yolov8x-seg.pt \
    --fraction 0.10 --epochs 4 --batch 8 --imgsz 640 --seed 42 \
    --out reports/speed_bench 2>&1 | tail -30
else
  echo "benchmark already done, reusing reports/speed_bench.json"
fi

# Translate the verified winners into ultralytics flags. Only configs that were
# BOTH faster and loss-equivalent are adopted; 'nondet' is deliberately excluded
# even if it wins, because it trades the bit-exact reproducibility the protocol
# promises the client.
SPEED_ARGS=$(python - <<'PY'
import json, os
p = "reports/speed_bench.json"
args = []
if os.path.exists(p):
    d = json.load(open(p))
    for r in d["results"]:
        if r["config"] in ("ref", "nondet"):
            continue
        if r["equivalent"] and r["speedup"] > 1.02:
            for k, v in r["overrides"].items():
                args.append("--%s=%s" % (k.replace("_", "-"), v))
print(" ".join(args))
PY
)
echo "adopted speed settings: ${SPEED_ARGS:-<none>}"
echo "$SPEED_ARGS" > reports/_speed_args.txt

step "2 segmentation ablation (uniform settings across every arm)"
EXTRA="$SPEED_ARGS" bash run_seg_ablation.sh 2>&1 | tail -40

step "3 copy-paste arms"
EXTRA="$SPEED_ARGS" bash run_seg_ablation2.sh 2>&1 | tail -40

step "PIPELINE DONE"
