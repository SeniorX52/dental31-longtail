#!/usr/bin/env bash
# The last two cells in the chain: a multi-scale probe, then a combination arm
# that is assembled at run time from whatever actually worked.
#
# WHY A COMBINATION ARM AT ALL. Every cell in this project so far changes ONE
# thing against a fixed reference, which is what makes each result readable but
# also means nothing has ever been stacked. Resolution is currently the only
# lever clear of the +-0.21 pp noise floor (0.1055 -> 0.1204, +14.2 % relative).
# If the widened head, the prototype grid, the background gate, the denoised
# labels or the pseudo-labels each add something on top of it, no run in the
# queue would show it, because each is measured alone against the same 1280
# reference. This cell is the only one that can find a compounding effect.
#
# HOW THE LEVERS ARE CHOSEN. Not by me picking favourites before the runs
# finish. The selector below reads each arm's own eval report and admits a lever
# only if it beats ITS OWN reference by more than 0.21 pp, the 2sd noise floor
# measured from three seeds on this corpus. An arm that is merely non-negative
# does not get in: stacking things that did nothing is how a combination cell
# turns into an unreadable result. If no lever qualifies the arm is skipped
# rather than run as a duplicate of the winner.
#
# The denoised-label arm is judged against abl_GATE1280, not against the 1280
# reference, because abl_DN1280 is gate PLUS denoise and only its marginal
# contribution is attributable to the label surgery.
#
# WHY MULTI-SCALE IS SEPARATE AND FIRST. It is the one lever with a diagnosis
# rather than a hope behind it. The 1280 model transfers to DENTEX at 32.5 %
# tooth-level recall against the baseline's 37.9 %, and re-testing it at a
# scale-matched 2176 made recall WORSE, not better. That rules out the "the
# external test is mis-specified" reading and leaves the one where the model
# narrowed its scale envelope while gaining on the in-domain distribution.
# Jittering the input size during training acts exactly there. It runs first so
# that, if it wins, the combination arm can include it.
#
# Usage:  nohup ./run_bestof.sh > logs/bestof.log 2>&1 &

cd "$HOME/Documents/ML_SOTA" || exit 1
source "$HOME/miniconda3/bin/activate" dental
set -u
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs reports preds

CONF=0.15
FLOOR=0.0021          # 2sd segm-mAP noise floor, measured from three seeds
stamp() { date '+%F %T'; }
finished() { python tools/run_finished.py "runs/segment/$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------- queueing ---
# Identify other drivers by argv[1] read from /proc, never by `pgrep -f`. A
# `pgrep -f run_bestof` matches this very script, any editor with the file open
# and any shell that merely mentions the name; that self-match stalled this
# queue five separate times, once costing about twelve hours of idle GPU.
trainer_running() {
  local p a
  for p in $(pgrep -x python 2>/dev/null); do
    [ -r "/proc/$p/cmdline" ] || continue
    a=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    case "$a" in *train_seg.py*|*train_dental.py*|*tooth_stage2.py*|*predict_to_coco*) return 0 ;; esac
  done
  return 1
}
wait_on() {
  local w=0 busy p a1 s
  while :; do
    busy=0
    trainer_running && busy=1
    for p in $(pgrep -x bash 2>/dev/null); do
      [ "$p" = "$$" ] && continue
      [ -r "/proc/$p/cmdline" ] || continue
      a1=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | sed -n '2p')
      for s in "$@"; do case "$a1" in ./$s|*/$s) busy=1;; esac; done
    done
    [ "$busy" = 0 ] && break
    [ $((w % 1800)) -eq 0 ] && echo "[$(stamp)] waiting on the queue (${w}s)"
    sleep 120; w=$((w + 120))
  done
  sleep 60
  echo "[$(stamp)] queue clear after ${w}s"
}

train_ft() {   # tag init imgsz batch epochs seed data extra...
  local tag="$1" init="$2" sz="$3" bs="$4" ep="$5" seed="$6" data="$7"; shift 7
  if finished "$tag"; then echo "[$(stamp)] $tag already complete"; return 0; fi
  local RESUME=()
  [ -f "runs/segment/$tag/weights/last.pt" ] && \
    RESUME=(--resume "runs/segment/$tag/weights/last.pt")
  echo "[$(stamp)] === $tag (init $(basename "$init"), $sz, ${ep}ep, extra: $*) ==="
  python yolov8_seg_longtail/train_seg.py \
      --data "$data" --model "$init" --nc 31 \
      --epochs "$ep" --imgsz "$sz" --batch "$bs" --seed "$seed" \
      --channels-last --weights none --boundary-weight 0 \
      --name "$tag" "$@" "${RESUME[@]}" > "logs/${tag}_train.log" 2>&1
  tail -8 "logs/${tag}_train.log"
  # 1280/batch2 measures 13.4 GB of the 16 GB card; stacked heads can exceed it
  if ! finished "$tag" && grep -qiE "out of memory" "logs/${tag}_train.log"; then
    echo "[$(stamp)] $tag OOM at batch $bs; retrying at batch 1"
    rm -rf "runs/segment/$tag"
    python yolov8_seg_longtail/train_seg.py \
        --data "$data" --model "$init" --nc 31 \
        --epochs "$ep" --imgsz "$sz" --batch 1 --seed "$seed" \
        --channels-last --weights none --boundary-weight 0 \
        --name "$tag" "$@" > "logs/${tag}_train.log" 2>&1
    tail -8 "logs/${tag}_train.log"
  fi
  finished "$tag" && echo "[$(stamp)] $tag finished" || echo "[$(stamp)] *** $tag did NOT finish"
}

score() {      # tag imgsz
  local tag="$1" sz="$2" dt="preds/ablation_$1_valid.json"
  finished "$tag" || { echo "[$(stamp)] $tag unfinished, not scoring"; return 0; }
  [ -f "$dt" ] || python yolov8_seg_longtail/predict_to_coco.py \
      --weights "runs/segment/$tag/weights/best.pt" \
      --gt data_clean/annotations/instances_valid.json \
      --images data_clean/valid/images --out "$dt" \
      --imgsz "$sz" --conf 0.001 --seed 42 2>&1 | tail -2
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
      --dt "$dt" --train-json data_clean/annotations/instances_train.json \
      --iou-type segm --out "reports/eval_${tag}_valid" 2>&1 | tail -3
  python eval/coco_eval_report.py --gt data_clean/annotations/instances_valid.json \
      --dt "$dt" --train-json data_clean/annotations/instances_train.json \
      --iou-type bbox --out "reports/bboxchk_${tag}_valid" 2>&1 | tail -2
  # COCO mAP alone cannot judge these: HD95 and ASSD average only over cases
  # where both masks are non-empty, so an arm that predicts less gets an easier
  # denominator. Every arm goes through the paired-on-intersection test.
  PYTHONPATH="$PWD/eval:${PYTHONPATH:-}" python eval/paired_contour.py \
      --gt data_clean/annotations/instances_valid.json \
      --dt-a preds/ablation_S0_valid.json --label-a S0 \
      --dt-b "$dt" --label-b "$tag" --conf "$CONF" --boot 500 \
      --out "reports/paired_contour_S0_${tag}_valid" 2>&1 | tail -6
}

echo "[$(stamp)] best-of arm queued behind the entire chain"
wait_on run_maskdino.sh run_hr1600.sh run_compound.sh run_phase2.sh \
        run_k2seeds.sh run_labelnoise.sh run_selftrain.sh run_toothstage.sh \
        run_protohp.sh

# ------------------------------------------------------------ base config ---
# The strongest confirmed single model becomes the starting point. Resolution,
# initialisation and backbone all come from whichever arm actually won, so this
# does not assume the 1280 fine-tune stays on top.
read -r BASE_SZ BASE_BS BASE_INIT BASE_TAG <<< "$(python - <<'PY'
import json, os
def m(f):
    try: return json.load(open(f))["coco_stats"]["mAP"]
    except Exception: return None
S0 = "runs/segment/abl_S0/weights/best.pt"
cand = [
    ("reports/eval_abl_HR1280ft_valid.json",     1280, 2, S0,               "HR1280ft"),
    ("reports/eval_abl_HR1600ft_valid.json",     1600, 1, S0,               "HR1600ft"),
    ("reports/eval_abl_SCRATCH_s42_valid.json",  None, None, "yolov8x-seg.pt","SCRATCH_s42"),
    ("reports/eval_abl_SCRATCH_s1337_valid.json",None, None, "yolov8x-seg.pt","SCRATCH_s1337"),
    ("reports/eval_abl_YOLO11x_1280_valid.json", 1280, 2, "yolo11x-seg.pt", "YOLO11x_1280"),
]
best, bv = None, -1
for f, sz, bs, init, tag in cand:
    v = m(f)
    if v is None or v <= bv: continue
    if sz is None:   # the scratch cells adopt whichever size their driver chose
        sz, bs = (1600, 1) if m("reports/eval_abl_HR1600ft_valid.json") and \
            m("reports/eval_abl_HR1600ft_valid.json") - (m("reports/eval_abl_HR1280ft_valid.json") or 0) > 0.003 \
            else (1280, 2)
    best, bv = (sz, bs, init, tag), v
print(*(best if best else (1280, 2, S0, "HR1280ft")))
PY
)"
echo "[$(stamp)] base: $BASE_TAG at imgsz $BASE_SZ batch $BASE_BS, init $(basename "$BASE_INIT")"

# ------------------------------------------------------------- cell 1: MS ---
train_ft abl_MS "$BASE_INIT" "$BASE_SZ" "$BASE_BS" 25 42 "$PWD/data_clean/data.yaml" --multi-scale
score    abl_MS "$BASE_SZ"

# -------------------------------------------------------- lever selection ---
# Each lever must clear the noise floor against its own reference. Printed with
# the deltas so the log shows why each one was admitted or rejected.
mapfile -t SEL < <(python - <<PY
import json, os
FLOOR = $FLOOR
def m(f):
    try: return json.load(open("reports/eval_%s_valid.json" % f))["coco_stats"]["mAP"]
    except Exception: return None
ref  = m("abl_$BASE_TAG")
hr   = m("abl_HR1280ft")
gate = m("abl_GATE1280")
flags, data, notes = [], None, []
def judge(name, arm, base, then):
    v, b = m(arm), base
    if v is None or b is None:
        notes.append("  %-22s no report, skipped" % name); return
    d = (v - b) * 100
    if v - b > FLOOR:
        notes.append("  %-22s %+.2f pp  ADMITTED" % (name, d)); then()
    else:
        notes.append("  %-22s %+.2f pp  rejected (inside floor)" % (name, d))
judge("hi-res prototypes", "abl_HR1280hp", hr,   lambda: flags.extend(["--proto-scale","2"]))
judge("wide coeff head",   "abl_HR1280w",  hr,   lambda: flags.extend(["--coeff-width","256"]))
judge("background gate",   "abl_GATE1280", hr,   lambda: flags.extend(["--bg-gate","0.5"]))
judge("multi-scale",       "abl_MS",       ref,  lambda: flags.append("--multi-scale"))
# data variants are mutually exclusive: take the better of denoise / self-train,
# and only if it beats the reference it was itself measured against
cands = []
if m("abl_DN1280") is not None and gate is not None and m("abl_DN1280") - gate > FLOOR:
    cands.append((m("abl_DN1280"), "data_clean_dn", "denoised labels"))
if m("abl_ST1280") is not None and hr is not None and m("abl_ST1280") - hr > FLOOR:
    cands.append((m("abl_ST1280"), "data_selftrain", "pseudo-labels"))
if cands:
    cands.sort(reverse=True); _, data, why = cands[0]
    notes.append("  %-22s chosen: %s" % ("data variant", why))
else:
    notes.append("  %-22s none qualified, using data_clean" % "data variant")
open("reports/bestof_selection.txt","w").write("\n".join(notes) + "\n")
print("\n".join(notes))
print("@@FLAGS@@ " + " ".join(flags))
print("@@DATA@@ "  + (data or "data_clean"))
PY
)
printf '%s\n' "${SEL[@]}" | grep -v '^@@'
EXTRA=$(printf '%s\n' "${SEL[@]}" | sed -n 's/^@@FLAGS@@ //p')
DATASET=$(printf '%s\n' "${SEL[@]}" | sed -n 's/^@@DATA@@ //p')

# --------------------------------------------------------- cell 2: BESTOF ---
if [ -z "${EXTRA// /}" ] && [ "$DATASET" = "data_clean" ]; then
  echo "[$(stamp)] no lever cleared the noise floor on top of $BASE_TAG."
  echo "          A combination cell would just re-run the winner, so it is skipped."
  echo "          That is itself the result: the levers do not compose."
else
  echo "[$(stamp)] combination: $BASE_TAG + [$EXTRA] on $DATASET"
  # shellcheck disable=SC2086
  train_ft abl_BESTOF "$BASE_INIT" "$BASE_SZ" "$BASE_BS" 25 42 "$PWD/$DATASET/data.yaml" $EXTRA
  score    abl_BESTOF "$BASE_SZ"
fi

echo "[$(stamp)] === final table ==="
python - <<'PY'
import json, os, glob
def m(f):
    try: return json.load(open(f))["coco_stats"]["mAP"]
    except Exception: return None
S0 = m("reports/ablation_S0_valid_segm.json") or m("reports/eval_abl_S0_valid.json")
rows = []
for f in sorted(glob.glob("reports/eval_abl_*_valid.json")):
    tag = os.path.basename(f)[len("eval_"):-len("_valid.json")]
    v = m(f)
    if v is not None: rows.append((v, tag))
rows.sort(reverse=True)
print("  %-26s %8s %10s" % ("run", "segm mAP", "vs S0"))
for v, tag in rows:
    d = "%+.2f pp" % ((v - S0) * 100) if S0 else "n/a"
    star = "  <-- best" if (v, tag) == rows[0] else ""
    print("  %-26s %8.4f %10s%s" % (tag, v, d, star))
if S0: print("\n  baseline abl_S0 = %.4f ; noise floor +-0.21 pp (2sd, three seeds)" % S0)
PY
echo "[$(stamp)] done -- whole chain complete"
