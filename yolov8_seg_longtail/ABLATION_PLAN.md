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
| S2 | inv-sqrt | x | – | complete method minus copy-paste |
| S3 | inv-sqrt | – | x | complete method minus boundary |
| S4 | inv-sqrt | x | x | complete method |

### Correction: this grid is cumulative, and the completion arms fix it

As originally written, every arm carrying the boundary term also carried
inverse-sqrt weighting. `S2 − S0` therefore measures **weighting plus
boundary**, not the boundary term, and the row above previously described it as
"isolates the boundary term", which it does not.

Two consequences, both recorded rather than quietly corrected:

- The valid isolation of the boundary term among these arms is **`S2 − S1c`**,
  where both carry inverse-sqrt weighting and only `boundary_weight` differs.
  Recomputed on that contrast the boundary effect is AP75 **+0.70 pp /
  +10.7 %** and head AP **+0.85 pp / +3.0 %**. The AP75 figure is unchanged from
  the cumulative contrast; the head figure was overstated at +3.7 % and is
  **+3.0 %** correctly isolated.
- The grid has no "baseline plus boundary alone" cell at all. That cell (`SB`),
  together with copy-paste alone (`SCP`) and complete-minus-weighting (`SNW`),
  is added by `run_seg_completion.sh`, which also adds published comparators and
  a second backbone.

| run | weighting | boundary | copy-paste | purpose |
|---|---|---|---|---|
| SB | – | x | – | **boundary alone** on the plain baseline |
| SCP | – | – | x | copy-paste alone |
| SNW | – | x | x | complete method minus weighting |

Final: the pre-registered configuration at 100 epochs versus the 100-epoch
baseline, reported on test. Final runs are **single-seed**; seed replicates are
run at the 50-epoch ablation budget on the `S1c`/`S2` pair, which is the
contrast the boundary claim rests on.

## Reporting

Every arm is scored by `eval/coco_eval_report.py` on the **valid** split during
the ablation; only the final configuration is scored on test. Report mask mAP,
AP50, AP75 and head/mid/tail group AP. Classes with under 10 evaluation
instances are reported but flagged, and claims are made at group level.
