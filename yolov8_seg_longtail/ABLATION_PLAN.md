# Project 2 (segmentation) — method changes and ablation plan

Baseline: stock `yolov8x-seg`, 100 epochs, batch 8, imgsz 640, seed 42, on the
frozen leakage-free split. Scored on the held-out test split with pycocotools:

| | mAP | AP50 | AP75 | head | mid | tail |
|---|---|---|---|---|---|---|
| box | 0.153 | 0.297 | 0.140 | 0.384 | 0.159 | 0.058 |
| mask | 0.105 | 0.259 | 0.069 | 0.286 | 0.104 | 0.037 |

Tail mask AP of **0.037** is the number the method has to move. Head is already
0.286, so gains there are limited; the headroom is mid and tail.

---

## Choosing the class-balance strength (measured, not copied)

Effective-number weights are `w_c = (1 - β) / (1 - β^{n_c})`, normalised to
mean 1. β = 0.999 is the value carried over from CIFAR-LT papers, but those
have a ~100:1 imbalance. Ours is **34,320:1**, which changes the picture
completely. Measured on our training split:

| β | weight ratio | head w | mid w | tail w | tail/head |
|---|---|---|---|---|---|
| 0.9 | 10× | 0.529 | 0.529 | 1.502 | 2.8 |
| **0.99** | 100× | 0.083 | 0.092 | 1.972 | 23.8 |
| 0.999 | 1000× | 0.009 | 0.032 | 2.040 | 234 |
| 0.9999 | 9675× | 0.001 | 0.027 | 2.046 | 1606 |
| inverse-sqrt | 185× | 0.040 | 0.220 | 1.892 | 47.5 |

**The tail weight saturates at ~2.0 beyond β = 0.99, while the head weight keeps
collapsing** (0.083 → 0.009 → 0.001). Raising β past 0.99 therefore buys the
tail nothing and simply deletes the head classes from the loss. β = 0.999 —
the default anyone would reach for — would have wrecked overall mAP for no
tail benefit.

So the sweep is **β ∈ {0.9, 0.99}** plus an inverse-sqrt arm, which is
interesting because it treats the *mid* group far less harshly (0.220 vs 0.092)
while still giving the tail ~1.9. Mid classes hold most of the recoverable
headroom, so this may beat effective-number weighting outright.

## The changes

1. **Class-balanced classification loss** — per-class weights on the BCE term,
   via the `model.class_weights` hook ultralytics already honours.
2. **Boundary-aware mask loss** — adds a soft-boundary Dice term to the
   per-instance mask BCE. Boundary bands come from a differentiable
   morphological gradient (dilate − erode via max-pool). Rationale: mask mAP on
   thin, elongated structures (Mandibular Canal, Bone Loss) is dominated by
   boundary error, which plain BCE under-penalises. Note AP75 (0.069) is far
   below AP50 (0.259) — the model finds objects but localises masks loosely,
   which is exactly what this targets.
3. **Rare-class copy-paste** — offline, manifest-audited augmentation that
   pastes instances of classes under 100 training instances into other images.
   Applied to the training split only, never to valid or test.

## Grid

All ablation arms run at **50 epochs** including their own 50-epoch baseline,
so every comparison is like-for-like; the winning configuration is then
retrained at 100 epochs against the 100-epoch baseline above. Running the
ablation at half schedule is what makes the grid affordable, and comparing it
against a half-schedule baseline is what keeps it honest.

| run | class-balanced | boundary | copy-paste | purpose |
|---|---|---|---|---|
| S0 | – | – | – | 50-epoch reference |
| S1a | β=0.9 | – | – | weighting strength |
| S1b | β=0.99 | – | – | weighting strength |
| S1c | inv-sqrt | – | – | alternative weighting |
| S2 | best of S1 | x | – | isolates the boundary term |
| S3 | best of S1 | – | x | isolates copy-paste |
| S4 | best of S1 | x | x | combined |

Final: S4 (or whichever wins) at 100 epochs, 3 seeds, versus the 100-epoch
baseline, reported on test.

## Reporting

Every arm is scored by `eval/coco_eval_report.py` on the **valid** split during
the ablation; only the final configuration is scored on test. Report mask mAP,
AP50, AP75 and head/mid/tail group AP. Classes with under 10 evaluation
instances are reported but flagged, and claims are made at group level.
