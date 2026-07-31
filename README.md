# Long-tailed dental radiograph detection & instance segmentation

Detection (DINO-DETR) and instance segmentation (YOLOv8-seg) on a 31-class
panoramic dental radiograph dataset with an extreme class imbalance — from
~33,000 instances (Filling) down to single digits (TAD = 4, Bone defect = 1).

The repo contains three things: a **data-integrity toolchain** that turned out
to be necessary before any number on this dataset means anything, **long-tail
modifications** to both architectures, and a **single shared evaluator** so
every model is scored the same way.

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

## Layout

| path | purpose | verified by |
|---|---|---|
| `tools/dataset_audit.py` | leakage gate: SHA-256 + two-stage perceptual/pixel duplicate detection, orphan and label validation, YOLO↔COCO reconciliation, class distribution | synthetic dataset with planted defects; clean copy passes |
| `tools/make_clean_split.py` | patient-grouped, rarest-class-first stratified split; self-verifying | 0 cross-split sources / patients, asserted at build time |
| `tools/build_clean_dataset.py` | materialises a split (symlinks) + rebuilds COCO from polygons | asserts one identical category vocabulary across splits |
| `tools/yolo_polygons_to_coco.py` | YOLO polygons → COCO instances (the shipped COCO has empty `segmentation`) | round-trip scores mAP 1.0 on bbox **and** segm |
| `tools/inspect_checkpoint.py` | checkpoint identity + minimal `--finetune_ignore` cover | derives the cover and proves it matches no non-head tensor |
| `dino_longtail/` | logit-adjusted focal loss applied in **both** the Hungarian cost and the loss, Seesaw variant, repeat-factor sampler, integration notes | unit tests on gradient direction, head/tail asymmetry, cost consistency |
| `yolov8_seg_longtail/` | stock baseline (isolated control arm), rare-class copy-paste, class-balanced + boundary-aware mask loss, COCO prediction export | trains end-to-end on CPU; RLE export scores 1.0 |
| `eval/coco_eval_report.py` | the one scorer: COCO metrics, per-class AP, head/mid/tail groups, `unstable` flags under 10 eval instances | synthetic data with known quality ordering |
| `mllm_eval/` | zero-shot MLLM detection harness (box-capable models only) | parsing/conversion unit tests |

## Quickstart

```bash
pip install torch torchvision ultralytics pycocotools opencv-python "numpy<2"
for t in tests/test_*.py; do python "$t" || break; done     # all pass without the dataset

python tools/dataset_audit.py --root <yolo_root> --names <data.yaml> --out reports/audit
python tools/make_clean_split.py --yolo-root <yolo_root> --names <data.yaml> \
    --out splits/clean_v1 --seed 42
python tools/build_clean_dataset.py --split-dir splits/clean_v1 \
    --names <data.yaml> --out data_clean

python yolov8_seg_longtail/train_baseline.py --data data_clean/data.yaml \
    --model yolov8x-seg.pt --epochs 100 --imgsz 640 --batch 8 --seed 42
python yolov8_seg_longtail/predict_to_coco.py --weights <best.pt> \
    --gt data_clean/annotations/instances_test.json \
    --images data_clean/test/images --out preds/test.json
python eval/coco_eval_report.py --gt data_clean/annotations/instances_test.json \
    --dt preds/test.json --train-json data_clean/annotations/instances_train.json \
    --iou-type segm --out reports/test_segm
```

`setup_5080.sh` builds the CUDA environment on a Blackwell (sm_120) GPU;
`PATCHES.md` records every upstream change needed to compile DINO's
deformable-attention kernel against a current PyTorch, so a rebuild is
reproducible.

## Ground rules

1. Fixed seeds throughout; final tables averaged over 3 seeds.
2. Class priors, repeat factors and class weights are computed from **train**
   only. Test is touched once, by the final evaluation.
3. Each ablation arm changes exactly one thing.
4. Classes with fewer than 10 evaluation instances are reported but flagged;
   claims are made at head/mid/tail group level.
5. Baseline and method share the split, the preprocessing and the scorer. The
   baseline gets a schedule at least as long as the method's, so no gain can
   come from training length alone.

## Data

Not included. The imagery is a third-party public release; the filenames embed
patient identifiers, so neither the images nor the generated split lists are
committed here. `make_clean_split.py` is deterministic — the exact split
regenerates from the source data at a given seed.
