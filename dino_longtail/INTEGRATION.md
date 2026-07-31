# Integrating the long-tail modules into DINO-DETR

Target repo: [IDEA-Research/DINO](https://github.com/IDEA-Research/DINO), initialized
from `checkpoint0033_4scale.pth` (ResNet-50, 4-scale, 900 queries, COCO).

## 0. 31-class fine-tune baseline (no novelty yet — this is the control)

1. Register the dental dataset in `datasets/coco.py` (paths to the converted
   JSONs from `tools/yolo_polygons_to_coco.py`).
2. Set `num_classes = 32` in the config. DINO uses the raw `category_id` as
   the label (`datasets/coco.py`: `classes = [obj["category_id"] ...]`, no
   remapping), so with ids 1..31 the head needs 32 outputs; index 0 is unused.
   This is the same reason COCO-DETR uses 91 for 80 classes. DINO builds `class_embed` from
   `num_classes`; the COCO 91-class embedding in the checkpoint mismatches, so
   load with `--pretrain_model_path checkpoint0033_4scale.pth --finetune_ignore
   label_enc.weight class_embed` (the repo's documented fine-tune path). All
   other weights (backbone, transformer, bbox heads) transfer.
3. Baseline schedule on a single 16 GB GPU: 12 epochs, lr 1e-4 (1e-5 backbone),
   drop at epoch 11, batch 1 + grad-accumulation 2, shorter-side 800 with the
   repo's default multi-scale augmentation. Seed everything (`--seed 42`;
   see `eval/coco_eval_report.py` for the seeding checklist).

## 1. Long-tail classification loss (`losses.py`)

Patch `models/dino/dino.py :: SetCriterion.loss_labels`:

```python
# once, in SetCriterion.__init__:
priors = class_priors_from_coco(train_json, num_classes)      # TRAIN json only
self.register_buffer("log_priors", priors.log())

# in loss_labels, replace sigmoid_focal_loss(...) with:
loss_ce = logit_adjusted_focal_loss(
    src_logits, target_classes_onehot, self.log_priors, tau=1.0,
    alpha=self.focal_alpha, gamma=2.0, reduction="none")
```

`tau` sweep: {0.5, 1.0, 1.5}. `SeesawBCE` is the alternative arm of the same
ablation slot (swap, don't stack).

## 2. Matching-cost consistency (`losses.py`)

Patch `models/dino/matcher.py :: HungarianMatcher.forward`: replace the
focal `cost_class` block with `logit_adjusted_cost_class(out_prob_logits,
tgt_ids, log_priors, tau)`, passing raw logits (pre-sigmoid) instead of
`out_prob`. Keep `tau` identical to the loss. Ablate: loss-only vs
cost-only vs both — the hypothesis is that "both" is what moves tail AP,
and this isolates the claim.

## 3. Frequency-aware denoising queries

DINO's contrastive denoising (`models/dino/dn_components.py ::
prepare_for_cdn`) samples GT labels uniformly when building noised queries,
so tail classes almost never appear in the denoising task. Change the
label-flip distribution: when flipping a label (probability
`label_noise_ratio`), draw the fake label from `p(c)^-0.5` (normalized
inverse-sqrt frequency) instead of uniform, and duplicate tail-class GT
groups with an extra dn group when present in the batch. This gives the
model denoising practice precisely on the classes it sees least. This is
the most novel of the three changes and the one to write up carefully.

## 4. CLAHE for low-contrast lesions

Add to `datasets/transforms.py` before normalization, train and eval:

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# apply on the L channel of LAB, or directly on grayscale radiographs
```

Ablate on/off; expected effect concentrated on Caries / Periapical lesion AP.

## 5. Repeat-factor sampling (`repeat_factor.py`)

In `main.py`, replace the train `RandomSampler` with `RepeatFactorSampler`
built from `image_repeat_factors(train_json, t=0.001)`, mapping image ids to
dataset indices in dataset order; call `sampler.set_epoch(epoch)` each epoch.
Sweep `t`: {0.001, 0.01}.

## Ablation grid (run in this order, one seed first, 3 seeds for the final table)

| run | RFS | LA-loss | LA-cost | freq-DN | CLAHE |
|-----|-----|---------|---------|---------|-------|
| A0 baseline | – | – | – | – | – |
| A1 | x | – | – | – | – |
| A2 | x | x | – | – | – |
| A3 | x | x | x | – | – |
| A4 | x | x | x | x | – |
| A5 (full) | x | x | x | x | x |
| A2' (cost-consistency isolation) | x | – | x | – | – |

Report: COCO mAP / AP50 / AP75, per-class AP, and grouped head (>5k), mid
(100–5k), tail (<100 instances) AP. Classes with <10 test instances
(Bone defect = 1, TAD = 4, Fracture teeth = 9) are reported in the tail
group only — single-class claims at those counts are statistically
meaningless and we say so in the write-up rather than pretending otherwise.
