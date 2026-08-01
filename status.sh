#!/usr/bin/env bash
# One-screen status, sized for a phone. Run:  ~/Documents/ML_SOTA/status.sh
cd "$HOME/Documents/ML_SOTA" 2>/dev/null || exit 1

echo "== $(date '+%a %H:%M') =="
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null \
  | awk -F, '{printf "GPU %s  %s  %s\n",$1,$2,$3}'

RUN=$(pgrep -af "main.py --output_dir|train_seg.py|bench_train_speed|export_dino_preds|predict_to_coco" \
      | grep -v pgrep | head -1)
if [ -n "$RUN" ]; then
  PID=$(echo "$RUN" | awk '{print $1}')
  case "$RUN" in
    *main.py*)        WHAT="DINO training" ;;
    *bench_train*)    WHAT="speed benchmark" ;;
    *train_seg*)      WHAT="seg arm $(echo "$RUN" | grep -oE 'abl_[A-Za-z0-9]+' | head -1)" ;;
    *export_dino*)    WHAT="DINO export" ;;
    *predict_to_coco*) WHAT="scoring" ;;
    *)                WHAT="running" ;;
  esac
  echo "NOW: $WHAT  ($(ps -o etime= -p $PID | tr -d ' '))"
else
  pgrep -f "run_pipeline.sh" >/dev/null && echo "NOW: idle, pipeline waiting" || echo "NOW: NOTHING RUNNING"
fi

echo
echo "-- DINO baseline --"
python3 - <<'PY' 2>/dev/null || echo "  no epochs yet"
import json, os
p = "runs/dino_baseline/log.txt"
rows = [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
if not rows: raise SystemExit(1)
d = rows[-1]; b = d["test_coco_eval_bbox"]
print("  %d/12 epochs | last mAP %.4f" % (len(rows), b[0]))
best = max(rows, key=lambda r: r["test_coco_eval_bbox"][0])
print("  best epoch %d mAP %.4f" % (best["epoch"], best["test_coco_eval_bbox"][0]))
PY
[ -f reports/baseline_dino_clean_test_bbox.json ] && python3 -c "
import json; s=json.load(open('reports/baseline_dino_clean_test_bbox.json'))
print('  TEST mAP %.4f AP50 %.4f' % (s['coco_stats']['mAP'], s['coco_stats']['AP50']))" 2>/dev/null

echo
echo "-- speed benchmark --"
if [ -f reports/speed_bench.json ]; then
  python3 -c "
import json
for r in json.load(open('reports/speed_bench.json'))['results']:
    print('  %-8s %.2fx  %s' % (r['config'], r['speedup'], 'ok' if r['equivalent'] else 'CHANGES LOSS'))" 2>/dev/null
else
  echo "  not run yet"
fi

echo
echo "-- seg ablation --"
n=0
for f in reports/ablation_S*_valid_segm.json; do
  [ -e "$f" ] || continue
  n=$((n+1))
  python3 -c "
import json,sys,os
d=json.load(open('$f')); g=d['group_AP']
print('  %-5s mAP %.4f  tail %.4f' % (os.path.basename('$f').split('_')[1], d['coco_stats']['mAP'], g.get('tail',0)))" 2>/dev/null
done
[ "$n" -eq 0 ] && echo "  0/7 arms done"

echo
df -h /home/mostafa | awk 'NR==2{print "disk free "$4"  used "$5}'
echo "log: tail -5 logs/pipeline.log"
