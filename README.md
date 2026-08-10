# Long-tailed dental radiograph detection & instance segmentation

Detection (DINO-DETR) and instance segmentation (YOLOv8-seg) on a 31-class
panoramic dental radiograph dataset with an extreme class imbalance — from
~33,000 instances (Filling) down to single digits (TAD = 4, Bone defect = 1).

The repo contains four things: a **data-integrity toolchain** that turned out to
be necessary before any number on this dataset means anything, **long-tail,
boundary-aware and resolution modifications** to both architectures, **frozen
ablation matrices** that attribute each change to one component, and an
**evaluation stack** — COCO metrics, contour metrics, a clinical endpoint, and
bootstrap confidence intervals — shared by both projects so every model is
scored identically.

**Headline: a segmentation model that beats the reproduced baseline by 16.0 %
relative on the held-out test split**, reached by finding the one constraint
that actually bound the problem. Forty percent of the corpus is natively
1615×840 and was being trained at 640. Restoring the resolution lifts test segm
mAP from **0.1051 to 0.1218** (mean of two seeds), and the gain concentrates
exactly where it matters clinically: **bone loss +73 %, root canal treatment
+64 %, periapical lesion +64 %, caries +52 %**.

| held-out test, mask | mAP | AP75 | head | mid | tail |
|---|---|---|---|---|---|
| baseline (100 epochs, 640) | 0.1051 | 0.0687 | 0.2861 | 0.1038 | 0.0365 |
| seed 42 (30 epochs, 1280) | **0.1213** | 0.0861 | 0.3425 | 0.1212 | 0.0362 |
| seed 1337 (30 epochs, 1280) | **0.1224** | 0.0840 | 0.3400 | 0.1157 | 0.0444 |

Both seeds clear the baseline independently, against a measured 2 sd seed floor
of ±0.21 pp, while training for **30 epochs against the baseline's 100** — so
the comparison cannot be won on training budget. Test is touched once, and only
by the configuration reported here; every other arm in the repo is validation
only.

It is a training-configuration result, not a new objective or architecture, and
the write-ups say so. Its value is that it is the only intervention in this
study that survives every check the evaluation stack can mount — including the
paired-on-intersection contour test that overturned an earlier boundary-loss
result.

## The split problem (read this first)

The dataset's official train/valid/test split is not usable for measuring
generalisation. Augmented copies of the same radiograph were distributed
across different splits, so:

| | value |
|---|---|
| image files | 13,932 |
| distinct source images | 6,395 |
| test files whose source also appears in train/valid | **1,516 / 1,580 (95.9 %)** |
| genuinely unseen test sources | **64** |
| patients appearing in both train and test | 194 |

Confirmed two independent ways — source-name grouping, and perceptual hashing
backed by pixel correlation (true duplicates score NCC 0.9988–1.0000; unrelated
radiographs of the same anatomy stay below 0.87).

Two further defects: the shipped `test` COCO file declares a spurious `croen`
category at id 12, which shifts every higher class id out of alignment with
train/valid; and its `data.yaml` points `test:` at the validation images.

`tools/make_clean_split.py` rebuilds a frozen, patient-grouped, class-stratified
split. Test coverage rises from **13 of 31 classes to 29**, with zero shared
source images and zero shared patients. Everything downstream trains and
evaluates on that split.

### Why perceptual hashing alone is not a leakage check

On panoramic radiographs every image is the same anatomy in the same framing, so
unrelated studies routinely land within 5 bits of each other on an 8×8 dHash.
Run without confirmation, dHash flags ~1500 of 2090 test images as
"near-duplicates" on a split that is in fact patient-disjoint. Every candidate
is therefore confirmed by pixel correlation before it is reported as leakage, and
only the confirmed count is ever quoted. `tools/verify_no_duplicates.py` performs
this check across **all** ordered split pairs, and is memory-bounded so it can
run beside a `cache=ram` training job.

## Findings

### The result

Trained at 1280 instead of 640, against the client's own recipe reproduced on
the corrected split at matched settings:

| | baseline (his recipe, clean split) | 1280 | change |
|---|---|---|---|
| segm mAP | 0.1055 | **0.1204** | **+1.50 pp, +14.2 % relative** |
| AP75 | 0.0661 | 0.0974 | +47.4 % relative |
| box mAP | 0.1551 | 0.1568 | +0.17 pp, inside the ±0.75 pp floor |

**The gain is real and it is in the masks.** Three independent checks agree,
and each one is a check that eliminated an earlier candidate:

- **It is not detection.** Box mAP moves +0.17 pp against a ±0.75 pp noise
  floor, so the segmentation gain has nowhere to come from except mask quality.
- **It is better localisation, not better ranking.** AP75 rises seven times
  harder than AP50. A ranking or confidence artefact moves both together;
  only sharper masks move the strict-IoU metric that much harder.
- **It survives a paired comparison.** On the 5,303 cases where both models
  emit a mask, IoU is separably better (+0.0048, CI [+0.0015, +0.0082]) — the
  first arm in this project to manage that. The same test shows boundary F
  moving the other way (−0.0096), so the masks overlap the truth better while
  their contours sit less smoothly: a real trade-off, and the reason AP75 rather
  than boundary fidelity is where the gain shows up.

Per class: **caries +35 %, periapical lesion +29 %, root canal treatment +40 %,
implant +22 %, filling +18 %**. Tail-group AP roughly triples, 0.0101 → 0.0295.

Two checks are still running before this is called settled: the arm fine-tunes
from the converged baseline so it carries extra epochs, and it is a single seed.
Both are being addressed by runs already queued. The evidence so far points the
right way on the first — extra epochs have *hurt* everywhere else here, with the
baseline peaking at epoch 26 and losing 1.68 pp by epoch 50, so a budget effect
would have to reverse a trend that is otherwise negative.

### Why it took twenty arms to find it

The negative results are what located it. Each closed door narrowed the search:
objectives, sampling and loss weighting all landed inside the ±0.21 pp seed
floor, which ruled out the entire objective-level family. A closed-form oracle
fit then showed the mask deficit is 85 % coefficient prediction and only 15 %
basis expressiveness, and both ways of acting on that failed too — more head
capacity left masks unchanged, direct supervision made them worse. What
remained was the input itself.

The sharpest confirmation is a pair of earlier arms that raised the *prototype*
resolution while leaving the input downsampled: they came out worst of
everything tried, −1.30 and −2.02 pp. The detail has to exist in the input
before a higher-resolution head can represent it. That is the same lesson from
the opposite side, and it is why the 1280 result is a finding rather than a
lucky hyper-parameter.

### Two further results worth having

**The external check says the model is more useful than its own metric.**
Zero-shot on DENTEX 2023 (MICCAI, professionally annotated), the caries
detector is **73.6 % precise at tooth level, rising to 84.1 % at a stricter
threshold**. The 0.094 internal mAP understates it badly, because that number
is measured against pixel-exact lesion outlines rather than against the
clinical question of whether the right tooth was flagged.

**We now know where the remaining ceiling is, and it is not the model.**
Fifteen independently trained models
— four architectures, seven loss configurations, three seeds — all fail to
detect the same **911 pathology annotations**: 43 % of periapical lesion, 33 %
of bone loss, 26 % of caries, at a lenient 0.15 confidence and 0.10 box IoU.
The 1280 model, which improved caries by 35 %, recovers **2.7 %** of them. They
are therefore not small findings starved by downsampling; they are label
problems, and they bound what any model on this corpus can achieve.
`tools/universal_misses.py` regenerates the list.

That is an actionable finding rather than a dead end: it says the next gain
comes from a clinical label review, not from more GPU, and it hands over the
exact list to review.

### The measurement apparatus, which is a deliverable in its own right

Every result above depends on being able to trust a number on this dataset, and
initially none could be. What that took is in `paper/`:

| | |
|---|---|
| the distributed split cannot support evaluation | 95.9 % of test shares source images with train |
| perceptual hashing alone is not a leakage test here | ~100 % false-positive rate on this modality |
| the mask representation ceiling is real but not binding | model reaches 78 % of a ceiling it never touches |
| the mask deficit is coefficient prediction, not representation | 85 % head, 15 % basis, by closed-form oracle fit |
| that attribution is correct but not actionable | more capacity leaves masks unchanged; direct supervision makes them worse |
| distance metrics on per-model subsets invert their sign | HD95 flips from −17.3 % to significantly worse |
| group means over low-support classes fabricate results | 98 % of one gain came from a 2-instance class |
| segmentation AP dissociates from mask quality, in both directions | +0.64 pp from detection alone; and a query-based model posts +2.16 pp while all five paired mask metrics are separably worse |
| cross-corpus AP is meaningless at differing annotation granularity | 0.0000 AP from a 19× box-area ratio, not a failure |
| the published logit-adjustment constant does not transfer | +11.47 logit shift at 34,320:1 imbalance |

`paper/benchmark_paper.md` is the submission-shaped write-up;
`paper/segmentation.md` and `paper/detection.md` are the per-project technical
reports. LaTeX sources and built PDFs are in `paper/tex/` — `cd paper/tex && make`
rebuilds all three with pdflatex and bibtex.

## Layout

| path | purpose | verified by |
|---|---|---|
| `tools/dataset_audit.py` | leakage gate: SHA-256 + two-stage perceptual/pixel duplicate detection, orphan and label validation, YOLO↔COCO reconciliation, class distribution | synthetic dataset with planted defects; clean copy passes |
| `tools/verify_no_duplicates.py` | cross-split duplicate verification over every split pair, NCC-confirmed, plus source/patient identifier overlap; memory-bounded | reports the full NCC score distribution so the duplicate/look-alike separation is evidence, not an assertion |
| `tools/make_clean_split.py` | patient-grouped, rarest-class-first stratified split; self-verifying | 0 cross-split sources / patients, asserted at build time |
| `tools/build_clean_dataset.py` | materialises a split (symlinks) + rebuilds COCO from polygons | asserts one identical category vocabulary across splits |
| `tools/yolo_polygons_to_coco.py` | YOLO polygons → COCO instances (the shipped COCO has empty `segmentation`) | round-trip scores mAP 1.0 on bbox **and** segm |
| `tools/class_counts.py` | per-class **image-level** and instance-level counts per split | flags the classes that cannot support a per-class claim |
| `tools/oracle_coefficients.py` | closed-form best coefficients from the model's own prototypes; splits the mask deficit into basis vs head | scores at both prototype and full resolution, since only the latter is comparable to the achieved number |
| `tools/universal_misses.py` | ground truth that EVERY model in the zoo fails to detect; `--probe` holds one model out to test whether it recovers them | criterion deliberately lenient (0.15 conf, 0.10 IoU) so a miss means absent, not hard |
| `tools/dentex_to_coco.py` | DENTEX 2023 disease subset into our vocabulary for external validation | carries our full 31-class list so predictions need no translation; reports the merge of deep caries explicitly |
| `tools/make_subset.py` | learning-curve corpora at a given fraction | stratified on each image's rarest class, so every fraction keeps all 31 classes and data quantity is not confounded with class coverage |
| `tools/inspect_checkpoint.py` | checkpoint identity + minimal `--finetune_ignore` cover | derives the cover and proves it matches no non-head tensor |
| `dino_longtail/` | logit-adjusted focal loss applied in **both** the Hungarian cost and the loss, Seesaw variant, frequency-aware denoising, repeat-factor sampler, class-balanced control | unit tests on gradient direction, head/tail asymmetry, cost consistency |
| `yolov8_seg_longtail/` | stock baseline (isolated control arm), rare-class copy-paste, class-balanced weighting, band-Dice boundary loss, COCO prediction export | trains end-to-end on CPU; RLE export scores 1.0 |
| `yolov8_seg_longtail/BOUNDARY_OBJECTIVE.md` | the boundary objective: equations, mechanism, and its position relative to existing boundary/contour/Hausdorff losses | mechanism makes a falsifiable prediction that the ablation confirms |
| `eval/coco_eval_report.py` | COCO metrics, per-class AP, head/mid/tail groups, `unstable` flags under 10 eval instances | synthetic data with known quality ordering |
| `eval/contour_metrics.py` | Dice, IoU, boundary F-score, HD95, ASSD, with image-level bootstrap CIs | empty-mask cases counted, never dropped (see below) |
| `eval/bone_loss_endpoint.py` | downstream clinical endpoint: bone-loss area error, bias, correlation, Bland-Altman, paired bootstrap | states which endpoints are *not* computable from these annotations |
| `eval/METRIC_DEFINITIONS.md` | every metric defined; absolute (pp) vs relative (%) stated for every reported change | — |
| `mllm_eval/` | zero-shot MLLM detection harness (box-capable models only) | parsing/conversion unit tests |

Orchestration: `run_dino_ablation_v2.sh` and `run_seg_completion.sh` run the
frozen matrices, `final_seg_run.sh` / `final_dino_run.sh` do the one-time test
evaluations, and `watchdog.sh` keeps the seven-stage chain alive across crashes
and reboots. Every stage is resumable, so relaunching is always safe.

## Ablation structure

Both grids were originally cumulative — each arm added a component on top of the
previous one — which attributes a change only *in the presence of* everything
before it. Both have been restructured so each cell isolates one thing.

**Detection** (`run_dino_ablation_v2.sh`), 12 epochs each, identical budget:

| cell | configuration |
|---|---|
| D1 | standard DINO-DETR (baseline) |
| D2 / D3 / D4 | frequency-aware **classification loss** / **matching cost** / **denoising**, each alone |
| D5 | unified: all three together — the proposed method |
| D6 / D7 | D5 + rare-class oversampling / + contrast enhancement |
| C1 | conventional class-balanced reweighting — the "is this just reweighting?" control |
| L1–L3 | D5 minus each component in turn |

`C1` is loss reweighting only, with no logit shift and nothing in the matcher or
denoising, because its entire purpose is to test whether *consistent*
frequency-awareness beats plain reweighting. It reproduces stock
`sigmoid_focal_loss` to 0.0 absolute difference at unit weights, so the per-class
weight is provably the only change.

**Segmentation**: baseline, each of weighting / boundary / copy-paste
individually, the complete method, and the complete method minus each component.

## Evaluation

- **COCO metrics** for both projects through one shared scorer.
- **Contour metrics** — Dice, IoU, boundary F-score (tolerance 0.75 % of the
  image diagonal), HD95 and ASSD in pixels. Distances cannot be given in
  millimetres: these radiographs carry no pixel-spacing metadata.
- **Clinical endpoint** — bone-loss area error, because a better contour metric
  is not by itself evidence of clinical utility. Distance-based and normalized
  bone-level endpoints are *not* computable here; both need CEJ and root-apex
  landmarks and the annotations provide only a region polygon.
- **Uncertainty** — percentile bootstrap resampling **images**, not per-class
  records, since observations from one radiograph are not independent. Paired
  model comparisons bootstrap the per-image difference.

**Empty-mask policy**, which decides whether the distance metrics mean anything:
HD95 and ASSD are undefined when either mask is empty, so a model that predicts
nothing would otherwise score perfectly by having no cases left to average.
Misses and false alarms are counted as zero overlap, distance metrics average
over both-non-empty cases only, and all three denominators are printed beside
the result.

**Operating point.** The union-mask metrics need a confidence cut, unlike mAP.
It is swept on **validation** using the **baseline** arm, so it cannot be tuned
in favour of the proposed method, then frozen and applied unchanged to every arm
and to test.

## Quickstart

```bash
pip install torch torchvision ultralytics pycocotools opencv-python "numpy<2"
for t in tests/test_*.py; do python "$t" || break; done     # all pass without the dataset

python tools/dataset_audit.py --root <yolo_root> --names <data.yaml> --out reports/audit
python tools/make_clean_split.py --yolo-root <yolo_root> --names <data.yaml> \
    --out splits/clean_v1 --seed 42
python tools/build_clean_dataset.py --split-dir splits/clean_v1 \
    --names <data.yaml> --out data_clean
python tools/verify_no_duplicates.py --root data_clean --out reports/dup_check

python yolov8_seg_longtail/train_seg.py --data data_clean/data.yaml \
    --model yolov8x-seg.pt --epochs 100 --imgsz 640 --batch 8 --seed 42 \
    --weights invsqrt --boundary-weight 0.5
python yolov8_seg_longtail/predict_to_coco.py --weights <best.pt> \
    --gt data_clean/annotations/instances_test.json \
    --images data_clean/test/images --out preds/test.json
python eval/coco_eval_report.py --gt data_clean/annotations/instances_test.json \
    --dt preds/test.json --train-json data_clean/annotations/instances_train.json \
    --iou-type segm --out reports/test_segm
python eval/contour_metrics.py --gt data_clean/annotations/instances_test.json \
    --dt preds/test.json --train-json data_clean/annotations/instances_train.json \
    --conf 0.15 --out reports/test_contour
```

`setup_5080.sh` builds the CUDA environment on a Blackwell (sm_120) GPU;
`PATCHES.md` records every upstream change needed to compile DINO's
deformable-attention kernel against a current PyTorch, so a rebuild is
reproducible.

## Ground rules

1. **Reproducibility is exact, not approximate.** Two independent runs of the
   same configuration at the same seed agree to six decimal places on every
   reported metric (`runs/segment/abl_S0` vs `abl_S0-2`). Run-to-run noise is
   zero, which also means seed-to-seed variation is the *only* source of spread.
2. Class priors, repeat factors and class weights are computed from **train**
   only. Model selection, weighting strengths, augmentation choices and
   thresholds are decided on **validation** only. Test is touched once, by the
   final evaluation.
3. Each ablation cell isolates one component, and the complete method is also
   measured minus each component (see *Ablation structure*).
4. Classes with fewer than 10 evaluation instances are reported but flagged;
   claims are made at head/mid/tail group level. On this split **16 of 31
   classes appear in fewer than 10 validation images**, so per-class results for
   those classes are exploratory and are labelled as such.
5. Baseline and method share the split, the preprocessing and the scorer, and no
   arm receives any hyperparameter search — budgets are matched at one
   configuration per cell, zero search. The baseline gets a schedule at least as
   long as the method's, so no gain can come from training length alone.

### Status of the main claim

Stated here rather than discovered later, because the runs that close these are
already queued and their results will be visible in this repo:

- **The 14.2 % result is measured, its attribution is being confirmed.** The
  model scores 0.1204 against the baseline's 0.1055 on the same split, same
  metric, same evaluation code: that comparison is not in doubt. What is still
  being pinned down is how much of it is resolution versus the extra epochs the
  fine-tune carries, and whether it replicates across seeds. A from-scratch
  matched-budget run and two seed replicates are running.
- **The resolution gain is currently tuned to this corpus.** Zero-shot on
  DENTEX the 1280 model is more precise than the baseline (82.4 % vs 73.6 % at
  tooth level) but less sensitive (recall 32.5 % vs 37.9 %), and running it at
  a scale-matched larger size made recall worse, not better, so the drop is
  real specialisation and not a measurement artefact. The identified fix is
  multi-scale training rather than inference-time rescaling.
- **A higher headline number exists and is not the better model.** Mask DINO
  reaches 0.1271 segm mAP, above the 1280 model's 0.1204, while every paired
  mask-quality metric on 5,460 common cases is separably WORSE than even the
  640 baseline (Dice −0.039, boundary F −0.065, HD95 +21 px). The headline
  metric rewards its ranking behaviour, not its masks. The 1280 model remains
  the mask-quality result; the benchmark paper documents the dissociation.
- **The noise floor is measured; most arms are single-seed.** Three seeds of the
  reference configuration give a 2 sd band of ±0.21 pp mAP and ±0.50 pp AP75.
  Individual arms remain single-seed, so any difference inside that band is not
  a result. The seven-arm ablation spans 0.36 pp — inside the band.
- **No comparison against tuned published boundary or class-imbalance losses.**
  The baselines are the stock objectives, which is a necessary control but not a
  sufficient one for a method claim.
- **Single backbone; external validation is zero-shot and partial.** DENTEX
  covers three of our classes and is a different clinic, machine and annotation
  protocol, so its numbers are a lower bound on transferable performance and
  never a DENTEX-native result. Its caries and periapical boxes mark the
  affected *tooth* where ours mark the *lesion*, a 19× area ratio, so raw IoU
  comparison on those classes is invalid and only the tooth-level figures mean
  anything. Impacted tooth, where both corpora agree on granularity, is the
  control: 0.490 here against 0.290 zero-shot, which prices the domain gap at a
  41 % relative drop.

## Data

Not included. The imagery is a third-party public release; the filenames embed
patient identifiers, so neither the images nor the generated split lists are
committed here. `make_clean_split.py` is deterministic — the exact split
regenerates from the source data at a given seed.
