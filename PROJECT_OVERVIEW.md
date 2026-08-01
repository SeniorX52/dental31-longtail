# What this project actually is

A plain-language summary. Assumes you know YOLO and sequence models; explains
everything else from there.

---

## The one-sentence version

We have ~14,000 dental panoramic X-rays labelled with 31 kinds of findings, and
we need two models — one that draws boxes, one that draws outlines — to work
noticeably better than the standard off-the-shelf versions, especially on the
findings that are rare.

## The data

Panoramic dental X-rays: the wide single image that shows the whole jaw, both
arches, all teeth at once.

Every image is labelled with polygons around 31 kinds of things. Three groups:

- **Treatments already done** — Filling, Crown, Implant, Root Canal Treatment
- **Problems** — Caries (decay), Periapical lesion (infection at the root tip),
  Bone Loss, Cyst, Fracture
- **Anatomy** — Mandibular Canal (the nerve channel), Maxillary sinus

### The thing that makes it hard

The classes are wildly unbalanced:

```
Filling               34,320 instances
impacted tooth        19,582
Root Canal Treatment  13,390
...
TAD                        2
Root resorption            2
bone defect                1
```

That is a **34,320 : 1** ratio between the most and least common class.

This is the whole difficulty. A model trained normally learns to be excellent
at Filling and effectively blind to bone defect, because ignoring a class that
appears once costs it almost nothing in the loss. Standard training has no
reason to care about the rare classes, and clinically the rare ones often
matter more.

This is called the **long tail** problem: a few classes with huge counts, a
long tail of classes with almost none.

## The two models

### Project 2 — YOLOv8-seg (you know this one)

Same YOLO you're familiar with, but the segmentation variant: instead of only a
box, it outputs a mask — the actual outline of the finding. Useful here because
"how much bone loss" is an area question, not a box question.

### Project 1 — DINO-DETR (the less familiar one)

A different family of detector. The mental model:

**YOLO** divides the image into a grid, has each cell propose boxes, then runs
non-maximum suppression afterwards to delete duplicate detections.

**DETR** skips all of that. It carries a fixed set of 900 learned "queries" —
think of them as 900 slots, each of which will either claim one object or
declare itself empty. Attention lets the slots see the image and see each
other, so they negotiate: if two slots latch onto the same tooth, one backs
off. No grid, no anchors, no NMS.

Training uses **Hungarian matching**: each step, the model's 900 predictions are
optimally paired one-to-one with the real objects, and only the matched pairs
get a loss. That pairing step matters a lot later — it's where one of our
changes lives.

**DINO** is DETR plus a training trick called denoising: alongside the real
task, it feeds in deliberately corrupted copies of the ground-truth boxes and
asks the model to repair them. It's a free extra learning signal that makes
DETR converge in 12 epochs instead of 500.

## What "better" means here

The metric is **mAP** (mean Average Precision). Two properties matter:

1. It's a **mean over classes** — every class counts equally, whether it has
   34,000 examples or 1. So the rare classes have nowhere to hide.
2. It rewards both finding things and outlining them tightly.

We report it three ways: overall, per class, and grouped:

- **head** — over 5,000 training instances
- **mid** — 100 to 5,000
- **tail** — under 100

The grouping exists because a single class with 2 test instances gives a
meaningless number on its own, but the tail *group* is a real, quotable signal.

## Where we currently stand

Stock YOLOv8x-seg, trained properly on a clean split, on held-out test data:

| | overall | head | mid | tail |
|---|---|---|---|---|
| box mAP | 0.153 | 0.384 | 0.159 | 0.058 |
| mask mAP | 0.105 | 0.286 | 0.104 | 0.037 |

Read that as: **decent on common findings, nearly blind on rare ones.**
Head 0.286 versus tail 0.037 is roughly an 8× gap. Closing that gap is the job.

The overall 0.105 looks low, but it's a mean over 31 classes where 18 score
near zero. Individually the common classes are fine — impacted tooth 0.58,
Crown 0.53.

## The problem we found in the data

The dataset ships with a train/test split that cannot measure anything.

Whoever built it applied augmentation (rotations, crops, colour shifts)
**before** splitting. So augmented copies of the same X-ray ended up in both
the training set and the test set. Concretely:

- 13,932 image files, but only **6,395 distinct source X-rays**
- **1,516 of the 1,580 test images** have their source sitting in training
- Only **64 test images** are genuinely unseen
- 194 patients appear on both sides

Testing a model on images it trained on measures memorisation, not skill. Every
number ever produced on this split — including the baseline that shipped with
the dataset — is inflated.

We rebuilt the split, keeping each patient entirely on one side. Test coverage
also went from 13 of 31 classes to 29. Everything is trained and measured on
that clean split.

## What we're actually changing

Four ideas, each switchable so we can prove which one did what.

**1. Make the loss care about rare classes.** Weight each class in the loss by
how rare it is. The catch: everyone uses a standard strength (β=0.999) copied
from papers with 100:1 imbalance. We measured it on our 34,320:1 data and found
it would set the head classes' weight to 0.009 — effectively deleting them —
while gaining the tail nothing, because the tail weight saturates. So we use
β=0.99 and test a gentler alternative. That measurement is a real finding.

**2. Fix a mismatch in DETR's matching step.** If the loss is rebalanced toward
rare classes but the Hungarian matching still uses unbalanced scores, the two
disagree: matching assigns slots as if nothing changed, while the loss pulls
the other way. We apply the same rebalancing in *both* places. This is the most
novel piece.

**3. Teach the denoising task about rare classes.** DINO's denoising currently
picks corrupted labels uniformly, so rare classes almost never show up in it.
We bias that sampling toward rare classes — free extra practice exactly where
the model is weakest.

**4. Make masks hug edges.** The baseline's AP75 (0.069) is far below its AP50
(0.259), meaning it finds things but outlines them loosely. We add a loss term
that specifically penalises boundary error, which should matter most for thin
structures like the Mandibular Canal.

Plus rare-class copy-paste: physically paste rare findings into other training
images so the model sees `bone defect` more than once.

## How we prove it

Ablation. Train one arm per change, all under identical conditions, and report
the table. The point isn't "the number went up" — it's showing *which* change
moved it and by how much. That's what the client explicitly asked for, and it's
what a reviewer would demand.

Rules we hold to:

- Baseline and method share the split, the preprocessing, and the scorer
- The baseline gets a schedule at least as long as the method, so nothing wins
  by training longer
- Fixed seeds; final numbers averaged over 3 seeds
- The test set is touched **once**, at the end. All development uses validation

## Why this is worth doing

Not really about dentistry. Long-tailed data is the normal case in medical
imaging — common conditions are common, and the dangerous ones are rare almost
by definition. A model that only works on frequent findings is the least useful
kind of model. Everything here is about the rare end.
