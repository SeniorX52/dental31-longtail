# Trained weights

Eight checkpoints: four for the instance-segmentation project and four for
the detection project. Every file has been loaded in a clean Python process
with this repository removed from the path, which is the condition you will
be in. Checksums are in `checksums.json`.

## Loading

Segmentation (`pip install ultralytics`, nothing from this repo needed):

```python
from ultralytics import YOLO
m = YOLO("segmentation/method_1280_seed42.pt")
r = m.predict("radiograph.jpg", imgsz=1280, conf=0.15)   # imgsz matters, see below
```

Detection checkpoints are DINO-DETR state dicts and load with the official
IDEA-Research DINO repository:

```python
import torch
ck = torch.load("detection/dino_baseline_r50_12ep.pth", map_location="cpu")
model.load_state_dict(ck["model"])   # DINO_4scale.py, num_classes=32
```

**Use the image size each model was trained at.** The 1280 models lose
accuracy at 640, and that is the whole point of the result: resolution is
what the improvement is made of.

## What is in the package

| file | project | trained at | test score | role |
|---|---|---|---|---|
| `segmentation/baseline_yolov8x_640_100ep.pt` | 2 | 640, 100 epochs | 0.1051 segm mAP | baseline the method must beat |
| `segmentation/method_1280_seed42.pt` | 2 | 1280, 30 epochs | 0.1213 segm mAP | DELIVERABLE - the model that beats the baseline |
| `segmentation/method_1280_seed1337.pt` | 2 | 1280, 30 epochs | 0.1224 segm mAP | DELIVERABLE - independent seed replicate |
| `segmentation/reference_640_50ep.pt` | 2 | 640, 50 epochs | 0.1007 segm mAP | the 50-epoch reference every ablation arm is measured against |
| `detection/dino_baseline_r50_12ep.pth` | 1 | short side 480-800, 12 epochs | 0.157 box mAP | baseline |
| `detection/dino_crt.pth` | 1 | short side 480-800, classifier retrain | 0.1623 box mAP | only detection arm above baseline on test (+0.53 pp), but 98 percent of the gain is one class with 2 test instances |
| `detection/dino_d4_freqaware_denoising.pth` | 1 | short side 480-800, 12 epochs | 0.1633 box mAP (valid) | frequency-aware denoising only; the only frozen-matrix arm above baseline on validation (+0.08 pp, inside noise) |
| `detection/dino_d5_unified.pth` | 1 | short side 480-800, 12 epochs | 0.102 box mAP (valid) | the proposed unified method. Shipped because it is the evidence for the negative result: 6.05 pp below its own baseline with tail AP exactly 0.0000 |

## Provenance and integrity

| file | bytes | sha256 (first 16) | load check |
|---|---|---|---|
| `segmentation/baseline_yolov8x_640_100ep.pt` | 144,018,180 | `bdfad124933636ea` | OK (SegmentationModel) |
| `segmentation/method_1280_seed42.pt` | 144,034,942 | `fa40b64fbd6eb8d8` | OK (SegmentationModel) |
| `segmentation/method_1280_seed1337.pt` | 144,036,276 | `8be05f025c7d29e7` | OK (SegmentationModel) |
| `segmentation/reference_640_50ep.pt` | 144,038,654 | `cd3c7bc506aaf6b7` | OK (SegmentationModel) |
| `detection/dino_baseline_r50_12ep.pth` | 188,035,691 | `f3193dc7cb898806` | OK (626 tensors, 48.5 M params) |
| `detection/dino_crt.pth` | 188,028,075 | `329778061db87c43` | OK (626 tensors, 48.5 M params) |
| `detection/dino_d4_freqaware_denoising.pth` | 188,038,539 | `83fcf6501c5cae33` | OK (626 tensors, 48.5 M params) |
| `detection/dino_d5_unified.pth` | 188,031,947 | `b0a7de1c5b73504e` | OK (626 tensors, 48.5 M params) |

## Two things done to these files, and why

**The segmentation checkpoints were re-exported.** They were trained through
`yolov8_seg_longtail/train_seg.py`, whose model class is defined in a script
running as `__main__`. ultralytics pickles the whole module into the
checkpoint, so the class name is baked in and `YOLO(best.pt)` fails on any
machine without this repository:

```
AttributeError: Can't get attribute 'LongTailSegModel' on <module '__main__'>
```

That class is a pure subclass of ultralytics' `SegmentationModel` overriding
one training-only method, so re-pointing the class and dropping the training
attributes gives a stock checkpoint with the same parameters. This was
verified numerically rather than assumed: the network's output on a fixed
input is **bit-identical** before and after, for all three re-exported files.
`tools/package_weights.py` performs and re-checks this.

**The detection checkpoints were slimmed.** Training checkpoints carry
optimizer and lr_scheduler state, about two thirds of each file and useless
for inference. Keeping `model`, `epoch` and `args` takes the four detection
files from 1.9 GB to 0.7 GB. If you want to *resume* DINO training rather
than run it, ask and the full checkpoints can be supplied.

## Reproducing these numbers

```bash
# segmentation, the deliverable model
python yolov8_seg_longtail/predict_to_coco.py \
    --weights segmentation/method_1280_seed42.pt \
    --gt data_clean/annotations/instances_test.json \
    --images data_clean/test/images --out preds.json \
    --imgsz 1280 --conf 0.001 --seed 42
python eval/coco_eval_report.py --gt data_clean/annotations/instances_test.json \
    --dt preds.json --train-json data_clean/annotations/instances_train.json \
    --iou-type segm --out report
```

Both segmentation deliverable models were trained from the same COCO
initialisation as the baseline, for 30 epochs against the baseline's 100, at
seeds 42 and 1337. Everything is seeded and `deterministic=True`.

## What these weights do and do not support

The two 1280 models beat the baseline on the frozen test split: 0.1213 and
0.1224 segm mAP against 0.1051, +16.0 % relative at the mean. The gain is
from input resolution, not from the boundary or class-weighting objectives
in this repository, which are nulls and are reported as such.

The detection checkpoints do **not** contain an improvement. `dino_d5_unified`
is the proposed method and it scores 6.05 pp *below* its own baseline with
tail AP at exactly 0.0000. It ships because it is the evidence for that
negative result. `dino_crt` is +0.53 pp on test, but 98 % of that gain comes
from a single class with two test instances, so it is an anecdote and is not
presented as a result.
