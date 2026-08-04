# Long-tailed dental radiograph detection & instance segmentation

Detection (DINO-DETR) and instance segmentation (YOLOv8-seg) on a 31-class
panoramic dental radiograph dataset with an extreme class imbalance — from
~33,000 instances (Filling) down to single digits (TAD = 4, Bone defect = 1).

The repo contains four things: a **data-integrity toolchain** that turned out to
be necessary before any number on this dataset means anything, **long-tail and
boundary-aware modifications** to both architectures, **frozen ablation
matrices** that attribute each change to one component, and an **evaluation
stack** — COCO metrics, contour metrics, a clinical endpoint, and bootstrap
confidence intervals — shared by both projects so every model is scored
identically.

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

## Layout

| path | purpose | verified by |
|---|---|---|
| `tools/dataset_audit.py` | leakage gate: SHA-256 + two-stage perceptual/pixel duplicate detection, orphan and label validation, YOLO↔COCO reconciliation, class distribution | synthetic dataset with planted defects; clean copy passes |
| `tools/verify_no_duplicates.py` | cross-split duplicate verification over every split pair, NCC-confirmed, plus source/patient identifier overlap; memory-bounded | reports the full NCC score distribution so the duplicate/look-alike separation is evidence, not an assertion |
| `tools/make_clean_split.py` | patient-grouped, rarest-class-first stratified split; self-verifying | 0 cross-split sources / patients, asserted at build time |
| `tools/build_clean_dataset.py` | materialises a split (symlinks) + rebuilds COCO from polygons | asserts one identical category vocabulary across splits |
| `tools/yolo_polygons_to_coco.py` | YOLO polygons → COCO instances (the shipped COCO has empty `segmentation`) | round-trip scores mAP 1.0 on bbox **and** segm |
| `tools/class_counts.py` | per-class **image-level** and instance-level counts per split | flags the classes that cannot support a per-class claim |
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

### Current limitations

Stated here rather than discovered later:

- **Final tables are single-seed.** Seed replicates are queued for the isolated
  boundary contrast; the detection side remains single-seed. Any claim resting
  on a difference smaller than the seed spread is not yet supported.
- **No comparison against tuned published boundary or class-imbalance losses.**
  The baselines are the stock objectives, which is a necessary control but not a
  sufficient one for a method claim.
- **Single backbone, single dataset.** No cross-backbone transfer and no
  external validation set, so no long-tail claim is made in any headline.

## Data

Not included. The imagery is a third-party public release; the filenames embed
patient identifiers, so neither the images nor the generated split lists are
committed here. `make_clean_split.py` is deterministic — the exact split
regenerates from the source data at a given seed.
