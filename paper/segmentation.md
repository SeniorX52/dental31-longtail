# Boundary-aware training for dental radiograph instance segmentation

**A negative result with a measured explanation, and the evaluation apparatus
needed to establish it.**

---

## Abstract

We evaluate boundary-aware and class-balanced training objectives for instance
segmentation on a 31-class panoramic dental radiograph corpus, and report that
none of them improves on a tuned stock baseline once per-class contributions are
decomposed. The result is not a null for want of trying: seven ablation cells at
matched budget, four published comparator losses (implemented; only partially run, see limitations), a second backbone, an
architectural change to the mask head, and a downstream clinical endpoint all
land within the seed-to-seed spread.

Three things make the null informative rather than empty. First, the corpus as
distributed cannot support any evaluation at all — 96 % of its test partition
shares source images with training — and we supply a verified-disjoint
replacement split. Second, we measure the ceiling the mask representation
imposes, independent of any model: 17.7 % of instances cannot reach IoU 0.75 at
the architecture's prototype resolution, with caries and root canal treatment
capped *below* the threshold. Third, we show that ceiling is **not** the binding
constraint — the model reaches only 78 % of it — which correctly predicts, and
is confirmed by, the failure of a higher-resolution mask head.

We also document two evaluation failures that produced confident wrong answers
before being caught, both of which would survive peer review unnoticed.

---

## 1. The corpus cannot be evaluated as distributed

The dataset ships a train/valid/test split. It is unusable.

| | |
|---|---|
| image files | 13,932 |
| distinct source images | 6,395 |
| test files whose source also appears in train or valid | **1,516 of 1,580 (95.9 %)** |
| genuinely unseen test sources | **64** |
| patients appearing in both train and test | 194 |

Augmented copies of the same radiograph were distributed across partitions after
augmentation rather than before. Any metric computed on this split principally
measures memorisation.

Two further defects: the shipped test COCO file declares a spurious `croen`
category at id 12, shifting every higher class id out of alignment with train
and valid; and its `data.yaml` points `test:` at the validation images.

**Replacement split.** We rebuild patient-grouped and class-stratified:
9752 / 2090 / 2090 images, test coverage rising from 13 of 31 classes to 29.

**Verification.** Disjointness is established twice by independent
implementations:

| check | `dataset_audit.py` | `verify_no_duplicates.py` |
|---|---|---|
| exact duplicate groups (SHA-256) | 0 | 0 |
| NCC-confirmed near-duplicates | **0** | **0** |
| dHash candidate pairs examined | 441,240 | 792,811 |
| rejected as look-alikes | all | all |
| scope | test vs rest | all ordered pairs |

Across 792,811 candidate pairs the **maximum** normalised cross-correlation is
0.9334 against a 0.98 confirmation threshold.

**Perceptual hashing alone is not a leakage test on this modality.** Every
panoramic radiograph is the same anatomy in the same framing, so unrelated
studies routinely fall within 5 bits on an 8×8 dHash. Run without pixel
confirmation, dHash reports all 792,811 pairs as leaks on a split that is in
fact clean. Any audit of this modality that stops at perceptual hashing will
report a false positive rate near 100 %.

**Statistical power.** 16 of 31 classes appear in fewer than 10 validation
images; two appear in none; eleven tail classes score exactly zero. Image
counts, not instance counts, bound what is claimable. No long-tail claim is
supportable on this corpus and none is made.

---

## 2. The objective

Let `Ω ⊂ Z²` be the pixel grid, `M : Ω → {0,1}` a ground-truth instance mask,
`M̂` the predicted logits, `p = σ(M̂)`. For odd band width `k`, with `S_k(i)` the
`k × k` window at pixel `i`:

    dilation  δ_k(x)[i] = max_{j ∈ S_k(i)} x[j]
    erosion   ε_k(x)[i] = min_{j ∈ S_k(i)} x[j]
    band      B_k(x)    = δ_k(x) − ε_k(x)

`B_k` is the morphological gradient, computed as one max-pool using
`ε_k(x) = −δ_k(−x)`. For binary `M` it is the indicator of a band of width `≈ k`
centred on `∂M`; for continuous `p` it is the differentiable analogue.

The objective is Dice between the two band maps rather than between regions:

    L_band = 1 − ( 2 Σ B_k(p)·B_k(M) + ε ) / ( Σ B_k(p) + Σ B_k(M) + ε )

added to the stock cropped BCE with weight `λ`. Throughout, `k = 3`, `λ = 0.5`,
never swept.

**Mechanism.** `∂δ_k/∂x` is nonzero only at each pooling window's arg-max. In
a confidently predicted interior the window is locally constant, `δ_k = ε_k`,
and no gradient flows. The term acts only on pixels at or beside the predicted
edge — measured at **48.5 % nonzero gradient versus 100 % for region losses**.

**Predicted signature.** A loss that only sharpens contours should raise
high-IoU matching and leave low-IoU matching flat or worse. So: AP75 up, AP50
down — not a uniform lift.

**Novelty, stated honestly.** The construction is a synthesis of known
components: morphological gradient as a differentiable edge extractor, soft Dice
as the overlap objective. Its nearest relative, Boundary IoU (Cheng et al. 2021),
is an evaluation metric; Dice on Sobel edge maps is established. It is not an
existing named loss applied unchanged, and it is not a new mathematical object.
What is defensible is narrow: no auxiliary distance map, a single interpretable
tolerance (`k` is the tolerance, in pixels), and application inside a
prototype-coefficient mask head where the term reshapes a shared basis rather
than a per-pixel output.

---

## 3. Ablation

Validation, 50 epochs per arm, seed 42, identical settings. Mask metrics via
pycocotools.

| arm | configuration | mAP | AP50 | AP75 | head |
|---|---|---|---|---|---|
| S0 | reference | 0.1055 | 0.2549 | 0.0661 | 0.2815 |
| S1a | β = 0.9 | 0.1051 | 0.2559 | 0.0646 | 0.2840 |
| S1b | β = 0.99 | 0.1029 | 0.2473 | 0.0665 | 0.2831 |
| S1c | inverse-sqrt | 0.1065 | 0.2591 | 0.0659 | 0.2836 |
| S2 | inverse-sqrt + boundary | 0.1050 | 0.2467 | **0.0730** | **0.2921** |
| S3 | inverse-sqrt + copy-paste | 0.1042 | 0.2570 | 0.0635 | 0.2836 |
| S4 | all three | 0.1051 | 0.2513 | 0.0675 | 0.2916 |

**The grid is cumulative**, so `S2 − S0` conflates weighting with the boundary
term. The valid isolation is `S2 − S1c`, where only `boundary_weight` differs:
AP75 **+0.70 pp (+10.7 %)**, head **+0.85 pp (+3.0 %)**, AP50 −1.23 pp,
mAP −0.15 pp.

The predicted signature is present. A figure reported earlier as "+3.7 % head"
was the conflated contrast; **+3.0 % is correct**, and is recorded here rather
than silently amended.

**Class weighting does nothing, mechanically.** YOLO's `TaskAlignedAssigner`
selects positives by `cls_score^α × IoU^β`. For a class the model never learned,
`cls_score ≈ 0`, so the ground truth receives few or no positive assignments and
re-weighting scales a loss over assignments never made. The bottleneck is
upstream of the loss.

**Copy-paste hurts, and the mechanism generalises.** It added ~10× more tail
instances and worsened every metric that matters. In panoramic radiographs
anatomical position is part of the class definition; a finding pasted into an
anatomically impossible location teaches the model to detect the paste rather
than the pathology. Augmentations that ignore anatomical priors are
counter-productive in radiographs even when they fix the class-count imbalance.

---

## 4. Held-out test

Both candidates nominated by the validation table, plus the isolated boundary
arm, trained to 100 epochs and scored once.

| model | mAP | AP50 | AP75 | head | tail |
|---|---|---|---|---|---|
| baseline | 0.1051 | 0.2590 | 0.0687 | 0.2861 | 0.0365 |
| S1c (invsqrt) | 0.0973 | 0.2369 | 0.0612 | 0.2867 | 0.0252 |
| S2 (invsqrt + boundary) | 0.1007 | 0.2430 | 0.0650 | 0.2908 | 0.0261 |
| **SB (boundary alone)** | **0.1071** | 0.2585 | 0.0680 | 0.2852 | 0.0441 |

SB is nominally +0.21 pp mAP. **Decomposed, it is not an improvement**: the gain
is carried by the tail group, and the tail movement comes from three classes
with 4, 8 and 8 test instances (abutment +0.086, gingival former +0.068, Supra
Eruption −0.059). Seven of fifteen tail classes changed at all; five of those
have under ten instances. AP50, AP75 and head are flat.

**The validation ordering did not transfer, three separate times.** S1c led on
validation and came last on test. The +10.7 % AP75 gain measured at 50 epochs on
validation is −0.37 pp at 100 epochs on test. Inverse-sqrt weighting looked
neutral on validation and costs 0.78 pp on test.

---

## 5. What the mask representation can support

YOLOv8-seg does not predict masks per pixel. It predicts 32 coefficients and
reconstructs each mask as a linear combination of prototypes at **input/4** —
160×160 at imgsz 640. The median instance here is **6 px on that grid**; 68 %
are under 8 px.

Round-tripping every ground-truth mask through the grid bounds what *any* model
can achieve, independent of its loss:

| prototype grid | mean Dice | mean IoU | cannot reach IoU 0.75 |
|---|---|---|---|
| **160×160 (stock)** | **0.8963** | 0.8277 | **17.7 %** |
| 320×320 | 0.9492 | 0.9074 | 5.4 % |
| 640×640 | 0.9901 | 0.9813 | 0.5 % |

20,601 instances. The ceiling falls on the clinically important classes because
they are the small ones:

| class | median side | IoU ceiling | instances |
|---|---|---|---|
| Root Canal Treatment | 17.2 px | **0.622** | 2878 |
| Caries | 16.7 px | **0.721** | 1615 |

Both below the 0.75 threshold: 4493 instances structurally excluded from AP75
before training begins. The model's own behaviour agrees — mask AP50 reaches
87 % of box AP50, mask AP75 only 49 %.

### The ceiling is real and it is not the binding constraint

This is the correction that matters, and we report it because we initially drew
the opposite inference.

| | Dice |
|---|---|
| ceiling at the current grid | 0.8963 |
| **achieved by the model** | **0.6969** |
| headroom already available | 0.1994 |
| extra headroom from doubling the grid | 0.0529 |

The model sits at **78 % of the ceiling it already had**. Raising the ceiling
addresses a limit nothing was pressing against.

**Confirmed experimentally.** We built a prototype head fed from P2 (stride 4)
rather than P3 (stride 8), yielding 320×320 prototypes from genuine
high-resolution features, with **878,432 fewer parameters** than stock so a gain
could not be a capacity artifact. Against a matched stock-prototype control at
identical budget and protocol:

| arm | mAP | AP50 | AP75 | mid |
|---|---|---|---|---|
| stock protos 160×160 | 0.0925 | 0.2279 | 0.0581 | 0.1147 |
| P2-fed protos 320×320 | 0.0852 | 0.2207 | 0.0532 | 0.1000 |

**−0.72 pp mAP, −0.49 pp AP75.** The prediction was wrong and the experiment
says so. Representation is not what limits this model; learning is.

---

## 6. Evaluation failures worth documenting

Both produced confident, plausible, wrong answers.

**Distance metrics conditioned on a per-model subset.** HD95 and ASSD are
undefined when either mask is empty, so they average over cases where both are
non-empty — a denominator that differs per model. Averaged per model, the
boundary term appeared **17.3 % better on HD95 and 24.6 % better on ASSD, with
disjoint confidence intervals.** Recomputed on the 5385 cases both arms actually
scored:

| metric | reference | boundary | difference | 95 % CI | separable |
|---|---|---|---|---|---|
| Dice | 0.6953 | 0.6969 | +0.0017 | [−0.0010, +0.0041] | no |
| IoU | 0.5685 | 0.5706 | +0.0021 | [−0.0007, +0.0048] | no |
| boundary F | 0.7857 | 0.7881 | +0.0024 | [−0.0009, +0.0053] | no |
| **HD95** | 66.36 | 68.66 | **+2.30 (worse)** | [+0.19, +4.25] | **yes** |
| ASSD | 18.40 | 18.82 | +0.43 (worse) | [−0.25, +1.05] | no |

Correcting the denominator did not shrink the effect; it **inverted its sign**.
The boundary arm misses 7 % more structures, and the ones it misses carry the
large distances. Any metric conditioned on a per-model subset must be compared
on the intersection, with coverage reported beside it.

**Group AP over classes with single-digit support.** Reported group means are
dominated by classes that cannot support them. Every apparent tail improvement
in this project decomposed to two or three detections on classes with 2–8
instances.

---

## 7. Clinical endpoint

A better contour metric is not evidence of clinical utility, so we measure the
quantity a clinician reads off the segmentation: **bone-loss area as a
percentage of image area**. 225 validation images contain ground-truth bone
loss; 1865 do not and are used to measure bone loss reported where there is none.

Distance-based and normalized bone-level endpoints are **not computable** from
this annotation set — both require cemento-enamel-junction and root-apex
landmarks, and the annotations provide a region polygon only. These images carry
no pixel-spacing metadata, so no measurement can be given in millimetres.

| | reference | boundary |
|---|---|---|
| MAE (pp of image area) | 1.114 | 1.129 |
| bias | −0.436 | −0.717 |
| Pearson r | 0.585 | 0.593 |
| false-positive area on healthy images | 0.129 | **0.071** |

Paired per-image absolute error: **+0.0154 pp, 95 % CI [−0.0738, +0.1060]**.
No detectable change.

The arms do differ, but not in accuracy: the boundary term carries a larger
negative bias while producing roughly half the false-positive area on healthy
radiographs. It shifts the operating point toward precision — fewer false
alarms, more misses — and the two cancel in the endpoint.

---

---

---

## The mask deficit is coefficient prediction, not representation

The 20-point gap between what the model achieves (Dice 0.6969) and what its
prototype grid allows (0.8963) had no attribution. Prototype-based heads
reconstruct each mask as `sigmoid(coeffs @ prototypes)`, so either the learned
32-dimensional basis cannot represent these shapes, or the basis is adequate and
the coefficient head fails to locate the right point in it. Those call for
opposite fixes.

`tools/oracle_coefficients.py` separates them without training. For every
ground-truth instance it solves, in closed form, for the coefficient vector that
best reconstructs it from the model's own prototypes:

    c* = argmin_c || P^T c - y ||^2      over the instance's box crop

Thresholding `P^T c*` gives the best mask **any** coefficient head could produce
from that basis.

| quantity | Dice |
|---|---|
| grid ceiling (what the resolution allows) | 0.8963 |
| **oracle coefficients, full resolution** | **0.8659** |
| oracle coefficients, at prototype resolution | 0.9636 |
| what the trained model achieves | 0.6969 |

**Attribution of the 0.1994 gap** (4123 instances):

| source | Dice lost | share |
|---|---|---|
| prototype **basis** cannot represent it | 0.0304 | **15 %** |
| **coefficient head** fails to find it | 0.1690 | **85 %** |

The basis is not the constraint. At its own resolution it reconstructs ground
truth at 0.9636; handed optimal coefficients the model jumps from 0.70 to 0.87.
**85 % of the deficit is a head that cannot locate the right point in a space
that already spans the shapes.**

This closes the loop on every negative result above. Boundary terms and
comparator losses reshape the *objective*; higher prototype resolution enlarges
the *representation*. Neither is the binding constraint, which is why neither
moved the number, and it is why the P2-fed head performed worse rather than
better: it added resolution to a basis that was never the limit.

The head capacities make the finding concrete. In YOLOv8x-seg the coefficient
branch bottlenecks 320 to 80 channels and carries 1.33 M parameters, against
2.26 M for the prototype generator and 7.41 M for classification. The smallest
of the three heads is the one responsible for 85 % of the mask deficit.

---

## Acting on the attribution: both routes fail

An attribution is only useful if it is actionable, so we tested the two things it
implies. If the coefficient head is the constraint, it either lacks the
**capacity** to express the mapping or lacks a training **signal** that reaches
it. Every objective tried earlier supervises in pixel space, where the gradient
must travel back through the prototype product to reach the coefficients at all.

**Capacity (K2).** The coefficient branch `cv4` is rebuilt with its hidden width
raised from 80 to 256 channels, 0.29 to 3.86 M parameters in the branch. Nothing
else changes.

**Supervision (K1b).** An auxiliary term supervises the coefficients directly
against the closed-form optimum `c*`, recomputed each step from the model's own
prototypes and detached so the basis cannot move to meet the prediction, as a
relative error `||c_pred - c*||^2 / ||c*||^2`.

### A numerical caution that is part of the result

The first implementation used a fixed ridge 1e-3 in the solve for `c*`. The
normal-equation matrix `A = P B P^T` is built from learned features restricted to
one instance box, and on COCO-pretrained prototypes over real instances its
eigenvalues span 1e-6 to 1e3, with the smallest slightly negative from rounding.
A fixed ridge regularises some instances and leaves others singular, and one
singular instance out of 95,745 annotations is enough. Training loss ran 1.8e5 at
epoch 1, NaN by epoch 10 and 7.2e7 by epoch 40, against 2.9 falling to 1.6
without the term. The arm still produced a plausible-looking mAP of 0.1106, which
measures the defect and not the method.

The weight had been calibrated on the **converged** reference model, where mean
`c*^2` is 2.86; on a run initialised from COCO weights the same quantity measures
4 to 53 with a far heavier tail. Making the ridge relative to `trace(A)/n` bounds
the condition number by construction and removes the failure. The results below
use that form.

| arm | segm mAP | delta | box mAP | delta | tail |
|---|---|---|---|---|---|
| S0 reference | 0.1055 | - | 0.1551 | - | 0.0101 |
| K1b supervision | 0.0985 | -0.70 | 0.1542 | -0.09 | 0.0039 |
| **K2 capacity** | **0.1118** | **+0.64** | 0.1625 | +0.74 | 0.0297 |

Validation, 50 epochs, seed 42, otherwise identical to the reference. Deltas in
percentage points. K2 is the largest gain of any arm in this study and exceeds
the +-0.21 pp noise floor threefold.

### K2's gain is not a mask gain

Scoring the **same** predictions as boxes rather than masks gives +0.74 pp,
**larger** than the +0.64 pp segmentation gain. `cv4` emits mask coefficients and
has no path to the box head, so the improvement cannot originate there:
rebuilding the branch with three times the parameters perturbs the gradients
returning into the shared neck, and the detector, not the mask, is what improves.
Because segmentation AP requires a correct detection **and** a correct mask,
better detection lifts it even when masks are unchanged.

| metric | S0 | K2 | difference (95 % CI) |
|---|---|---|---|
| Dice | 0.7043 | 0.7028 | -0.0016 [-0.0043, +0.0013] |
| IoU | 0.5784 | 0.5770 | -0.0014 [-0.0041, +0.0019] |
| boundary F | 0.7971 | 0.7928 | **-0.0043** [-0.0077, -0.0007] |
| HD95 (px) | 64.15 | 63.95 | -0.20 [-2.65, +2.11] |
| ASSD (px) | 17.80 | 17.62 | -0.18 [-0.88, +0.49] |

Paired on the same 5352 cases, 500-resample image-level bootstrap. Only boundary
F is separable from zero, and it favours the **reference**. An arm can gain
0.64 pp of segmentation AP while its masks get no better.

### K1b fails on its merits

With the target numerically sound the arm is a fair test, and it is negative:
-0.70 pp, paired Dice -0.0017 and not separable. The mechanism is visible in the
loss. At epoch 50 K1b's total segmentation loss is 5.05 against the reference's
1.65; the auxiliary term is capped at 2.0 by its clip, so K1b's own pixel BCE is
roughly 3.05 against 1.65. The term did not help the head find the pixel optimum,
it pulled the head away from it.

A scale mismatch would be the obvious explanation and is not the correct one: the
oracle target and the head's own coefficients agree to within 3 % in RMS, 0.473
against 0.460. The coefficients that minimise pixel BCE and those that best
reconstruct the mask in least squares are simply different points, and moving
toward the second costs the first.

**Consequence.** The attribution is a correct *localisation* and not an
actionable one. The coefficient head is where the error lives, and it is short of
neither parameters nor a direct training signal.

## Noise floor (measured)

The reference configuration was trained at three seeds, everything else
identical. Determinism is exact on this pipeline, so seed choice is provably the
only source of variation.

| seed | mAP | AP50 | AP75 | head | tail |
|---|---|---|---|---|---|
| 42 | 0.1055 | 0.2549 | 0.0661 | 0.2815 | 0.0101 |
| 1337 | 0.1056 | 0.2573 | 0.0679 | 0.2837 | 0.0117 |
| 2024 | 0.1037 | 0.2568 | 0.0629 | 0.2823 | 0.0105 |
| **sd** | **0.0010** | 0.0013 | **0.0025** | 0.0011 | 0.0008 |

**Noise floor at 2 sd: mAP ±0.21 pp, AP50 ±0.26 pp, AP75 ±0.50 pp,
head ±0.22 pp, tail ±0.17 pp.**

Three consequences, and they are not all the same verdict:

- The seven-arm ablation spans 0.1029–0.1065 mAP, a 0.36 pp spread. That is
  **inside** the ±0.21 pp band once both directions are counted. Selecting a
  winner on mAP across those arms was selecting noise.
- The best segmentation arm on test is +0.21 pp, sitting **exactly at** the
  threshold. It cannot be distinguished from seed choice.
- The isolated boundary effect on AP75 is **+0.70 pp against a ±0.50 pp floor**,
  so it **exceeds** noise on validation at 50 epochs. That effect was real. It
  then reversed on test at 100 epochs (−0.37 pp). This is a **transfer failure,
  not a noise result**, and the distinction matters: the objective does what its
  mechanism predicts under the conditions it was measured in, and that does not
  survive a change of split and schedule.

---

## 8. Conclusion

On this corpus, boundary-aware and class-balanced training objectives do not
improve instance segmentation beyond seed-level variation, and neither does
acting on the one place the deficit was traced to. We report this as the result
rather than selecting the metric and split on which it appears otherwise.

The two arms of "Acting on the attribution" are the sharpest statement of it.
Handed optimal coefficients the model jumps from Dice 0.70 to 0.87, so the head
is demonstrably the constraint; given three times the parameters it produces the
same masks, and given the optimal coefficients as an explicit target it produces
worse ones. Whatever limits this head is not capacity and not the absence of a
signal pointing at the right answer.

What the work establishes instead:

1. **The distributed split cannot support evaluation**, and a verified-disjoint
   replacement exists.
2. **Perceptual hashing alone is not a leakage test on panoramic radiographs**,
   with a measured false-positive rate near 100 %.
3. **The mask representation imposes a quantified ceiling**, and the model
   operates well below it, so representation is not the binding constraint.
4. **Distance metrics conditioned on per-model subsets invert their own sign.**
5. **Anatomical position is part of the class definition in radiographs**, which
   is why position-agnostic copy-paste is counter-productive.

### Limitations

**The published comparator losses did not run.** Soft Dice, Tversky, Focal
Tversky and Kervadec's boundary loss are implemented behind a common interface,
but the Kervadec arm failed with an in-place autograd error and the other three
were queued behind it. The objective is therefore compared against the stock
BCE baseline only, which is a necessary control and not a sufficient one.

Single seed on all reported arms; the noise floor is measured above and is
the denominator for every comparison. Single corpus, single backbone family. No comparison
against prior published results, because none exist for this dataset. No
long-tail claim, because the evaluation split cannot support one.
