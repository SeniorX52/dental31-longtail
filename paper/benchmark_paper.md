# Measuring Progress on Long-Tailed Dental Radiograph Segmentation:
# Leakage, Ceilings, and Noise Floors

---

## Abstract

We report a systematic negative result on a 31-class panoramic dental radiograph
corpus: across nineteen training configurations spanning boundary-aware
objectives, class-balanced losses, frequency-aware DETR training, an
architectural change to the mask head, and decoupled classifier retraining, no
configuration improves on a stock baseline beyond seed-level variation.

The contribution is not the null itself but the apparatus required to establish
it, each element of which caught a confident wrong answer that would otherwise
have been reported.

**(1) The corpus cannot be evaluated as distributed.** 95.9 % of its test
partition shares source images with training; 64 of 1,580 test images are
genuinely unseen. We supply a patient-grouped replacement verified disjoint by
two independent implementations.

**(2) Perceptual hashing alone is not a leakage test on this modality.** Every
panoramic radiograph is the same anatomy in the same framing. dHash flags all
792,811 candidate pairs on a split that is in fact clean — a false-positive rate
near 100 %. Pixel confirmation is mandatory, not optional.

**(3) The mask representation imposes a measurable ceiling, and it is not the
binding constraint.** Round-tripping ground truth through the prototype grid
shows 17.7 % of instances cannot reach IoU 0.75 whatever the model does, with
caries and root canal treatment capped *below* the threshold. But the model
reaches only 78 % of that ceiling. A higher-resolution mask head with 878,432
*fewer* parameters confirms the prediction by performing 0.72 pp worse.

**(4) Distance metrics conditioned on per-model subsets invert their own sign.**
HD95 and ASSD are undefined on empty masks, so they average over a denominator
that differs per model. Per-model averaging showed our objective 17.3 % better
on HD95 and 24.6 % better on ASSD with disjoint intervals. Recomputed on the
intersection of cases both models scored, it is significantly *worse*.

**(5) Group means over low-support classes fabricate results.** Two separate
headline gains, in two projects, decomposed to single classes with 2–8 test
instances. In one case 98 % of a +0.53 pp mAP gain came from one class with
**two** instances.

**(6) The published logit-adjustment constant does not transfer to clinical
imbalance.** At τ = 1.0 the adjustment applies a +11.47 logit shift at
34,320:1 imbalance, producing −6.05 pp mAP and tail AP of exactly zero. Recovery
is monotonic in τ and still short of baseline.

---

## 1. Introduction

Long-tailed instance segmentation and detection on clinical imagery is an active
area, and the standard toolkit — class-balanced losses, logit adjustment,
repeat-factor sampling, boundary-aware objectives — is well established on
natural-image benchmarks. We set out to apply that toolkit to a 31-class dental
radiograph corpus with 34,320:1 imbalance and measure the contribution of each
component.

None of it worked. That is worth reporting, because the reasons are specific,
measurable, and mostly not about the methods.

The corpus as distributed cannot support evaluation at all. The architecture
imposes a ceiling that is real but not binding. The metrics the field uses for
contour quality invert their sign under a denominator that differs per model.
And group means over classes with single-digit support will manufacture a
headline on demand — twice, in our case, before we caught them.

Our contributions are the six listed above. We regard (2), (4) and (5) as the
most transferable: each is a measurement protocol that any group working on
long-tailed clinical segmentation can apply, and each caught an error in our own
work that would have survived review.

---

## 2. Related work

**Long-tailed recognition.** Class-balanced loss weighting via effective number
of samples [Cui et al., CVPR 2019] and logit adjustment [Menon et al., ICLR
2021] are the standard remedies. Kang et al. [ICLR 2020] showed representation
and classifier learning should be *decoupled*, with rebalancing applied at the
classifier stage; Wang et al. [arXiv 1910.13081] adapt this to instance
segmentation via classification calibration. Repeat-factor sampling [Gupta et
al., CVPR 2019, LVIS] rebalances at the data level. Seesaw loss [Wang et al.,
CVPR 2021] rescales gradients per class pair.

Our results bear on the *regime* these were validated in. Menon et al. tune
τ = 1.0 on CIFAR-LT and ImageNet-LT at roughly 100:1 to 1000:1 imbalance. At
34,320:1 the same constant applies a +11.47 logit shift and destroys the
classifier. We find the failure monotonic in τ, which localises the problem to
magnitude rather than principle.

**High-quality mask prediction.** PointRend [Kirillov et al., CVPR 2020]
recasts segmentation as rendering, refining adaptively at uncertain points.
RefineMask [Zhang et al., CVPR 2021] fuses fine-grained features in stages.
Mask Transfiner [Ke et al., CVPR 2022] represents masks as a quadtree and
corrects only error-prone nodes, outperforming PointRend by 1.3 AP on boundary
quality. All three exist because coarse mask prediction is a known failure mode.

We measure that failure mode directly rather than assuming it, and find it is
*not* what limits our model — a result that would have been invisible without
the model-free ceiling measurement of §4.

**Boundary-aware objectives.** Boundary loss [Kervadec et al., MIDL 2018]
integrates the prediction against a precomputed distance map. HD loss [Karimi &
Salcudean, TMI 2019] estimates Hausdorff distance directly. Boundary IoU [Cheng
et al., CVPR 2021] is an evaluation metric restricted to a contour band. Our
band-Dice objective is a synthesis of known components — morphological gradient
as a differentiable edge extractor, soft Dice as the overlap objective — and we
say so rather than claiming a new loss family.

**Augmentation.** Copy-paste [Ghiasi et al., CVPR 2021] reliably helps on
natural images. We find it harmful here, and propose a mechanism: in panoramic
radiographs anatomical position is part of the class definition, so a finding
pasted into an anatomically impossible location teaches the model to detect the
paste rather than the pathology.

---

## 3. The corpus cannot be evaluated as distributed

| | |
|---|---|
| image files | 13,932 |
| distinct source images | 6,395 |
| test files whose source appears in train or valid | **1,516 / 1,580 (95.9 %)** |
| genuinely unseen test sources | **64** |
| patients in both train and test | 194 |

Augmentation was applied before partitioning rather than after. Any metric on
this split principally measures memorisation. Two further defects: a spurious
`croen` category at id 12 in the test COCO file shifts every higher class id out
of alignment with train and valid, and `data.yaml` points `test:` at validation
images.

**Replacement.** Patient-grouped, class-stratified: 9752 / 2090 / 2090, test
coverage rising from 13 of 31 classes to 29.

**Verification, twice, independently:**

| check | implementation A | implementation B |
|---|---|---|
| exact duplicate groups (SHA-256) | 0 | 0 |
| NCC-confirmed near-duplicates | **0** | **0** |
| dHash candidates examined | 441,240 | 792,811 |
| rejected as look-alikes | all | all |
| scope | test vs rest | all ordered pairs |

Maximum NCC across 792,811 candidate pairs: **0.9334**, against a 0.98
confirmation threshold.

### 3.1 Perceptual hashing alone is not a leakage test

This is the transferable point. On natural images, a dHash Hamming distance
below ~5 bits is strong evidence of duplication. On panoramic radiographs it is
nearly meaningless: every image is the same anatomy, same framing, same
grayscale statistics. Unrelated studies routinely fall within 5 bits.

Run without pixel confirmation, dHash reports **792,811 leaking pairs on a split
we independently verify as clean** — a false-positive rate near 100 %. Any audit
of this modality that stops at perceptual hashing will condemn a clean split, or
worse, be tuned until it stops complaining.

Two-stage detection — dHash as a cheap candidate generator, normalised
cross-correlation as confirmation — separates the cases cleanly. Genuine
re-encoded duplicates score NCC ≥ 0.99; distinct studies stay below 0.94.

### 3.2 Statistical power

16 of 31 classes appear in fewer than 10 validation images; two appear in none;
eleven tail classes score exactly zero. **Image counts, not instance counts, are
the effective sample size.** No long-tail claim is supportable on this corpus,
and we make none.

---

## 4. The representation ceiling, and why it is not the constraint

YOLOv8-seg predicts 32 coefficients per instance and reconstructs masks as a
linear combination of prototypes at input/4 — 160×160 at imgsz 640. The median
instance here is **6 px on that grid**; 68 % are under 8 px.

We bound what any model could achieve by round-tripping every ground-truth mask
through the grid and scoring it against itself:

| prototype grid | mean Dice | cannot reach IoU 0.75 |
|---|---|---|
| **160×160 (stock)** | **0.8963** | **17.7 %** |
| 320×320 | 0.9492 | 5.4 % |
| 640×640 | 0.9901 | 0.5 % |

20,601 instances. The ceiling falls on the clinically important classes:

| class | median side | IoU ceiling | instances |
|---|---|---|---|
| Root Canal Treatment | 17.2 px | **0.622** | 2,878 |
| Caries | 16.7 px | **0.721** | 1,615 |

Both below threshold — 4,493 instances structurally excluded from AP75 before
training begins. The model's behaviour agrees: mask AP50 reaches 87 % of box
AP50, mask AP75 only 49 %.

### 4.1 The inference we initially drew was wrong

This looks like a diagnosis. It is not.

| | Dice |
|---|---|
| ceiling at the current grid | 0.8963 |
| **achieved by the model** | **0.6969** |
| headroom already available | 0.1994 |
| additional headroom from doubling the grid | 0.0529 |

The model sits at **78 % of the ceiling it already has**. Doubling the grid
raises a limit nothing is pressing against.

We tested it. A prototype head fed from P2 (stride 4) rather than P3 (stride 8)
yields 320×320 prototypes from genuine high-resolution features, with **878,432
fewer parameters** than stock so any gain could not be a capacity artifact.
Against a matched control at identical budget and protocol: **−0.72 pp mAP,
−0.49 pp AP75**.

The ceiling is real and worth reporting. It is not the binding constraint, and
the check that establishes this — comparing achieved performance against the
ceiling — costs one line of arithmetic. We ran it after the experiment. It
should be run before.

---

## 5. Three metric failures that produce confident wrong answers

### 5.1 Distance metrics conditioned on per-model subsets

HD95 and ASSD are undefined when either mask is empty, so implementations
average them over cases where both are non-empty. **That denominator differs per
model.** A model that misses a hard structure drops that case from its own
average and can post better distances by predicting less.

Averaged per model, our boundary objective appeared **17.3 % better on HD95 and
24.6 % better on ASSD, with disjoint confidence intervals** — the strongest
result in the project.

Recomputed on the 5,385 cases both models actually scored, with the paired
per-case difference bootstrapped over images:

| metric | reference | boundary | difference | 95 % CI | separable |
|---|---|---|---|---|---|
| Dice | 0.6953 | 0.6969 | +0.0017 | [−0.0010, +0.0041] | no |
| IoU | 0.5685 | 0.5706 | +0.0021 | [−0.0007, +0.0048] | no |
| boundary F | 0.7857 | 0.7881 | +0.0024 | [−0.0009, +0.0053] | no |
| **HD95** | 66.36 | 68.66 | **+2.30 (worse)** | [+0.19, +4.25] | **yes** |
| ASSD | 18.40 | 18.82 | +0.43 (worse) | [−0.25, +1.05] | no |

Correcting the denominator did not attenuate the effect. It **inverted the
sign**. The boundary arm misses 7 % more structures, and the ones it misses
carry the large distances.

**Protocol.** Any metric conditioned on a per-model subset must be compared on
the intersection of cases both models scored, paired, with coverage reported
alongside. Reporting distances without coverage permits a model to win by
predicting less.

### 5.2 Group means over low-support classes

Two headline gains, in two independent projects, dissolved under decomposition.

*Segmentation.* The best test arm was +0.21 pp mAP, carried by tail-group AP.
The tail movement came from three classes with 4, 8 and 8 test instances.

*Detection.* Decoupled classifier retraining gave +0.53 pp mAP, +1.25 pp tail —
positive on every metric, with the largest gain exactly where a long-tail method
should deliver.

| | |
|---|---|
| observed mAP delta | +0.53 pp |
| largest contributor | one class, **2 test instances**, 0.0000 → 0.1515 |
| its contribution | +0.52 pp |
| **share of total** | **98 %** |
| excluding that class | **+0.01 pp** |
| on 18 classes with ≥10 instances | 6 improve, 12 worsen |

**Protocol.** Report per-class deltas with support counts alongside any group
mean, and state the result restricted to classes with adequate support. A group
AP over classes with single-digit support is not a measurement.

---

## The label ceiling, measured

Turning the consensus logic onto the corpus itself: with fifteen prediction
files spanning four architectures, seven loss configurations and three seeds,
ask of each floor-class ground truth whether ANY model placed a box of the
right class at even 0.10 IoU at the frozen operating point.

| class | missed by all 15 | total | share |
|---|---|---|---|
| Periapical lesion | 343 | 799 | **42.9 %** |
| Bone Loss | 156 | 473 | **33.0 %** |
| Caries | 412 | 1615 | **25.5 %** |

An instance that fails this is not hard, it is absent. Holding the resolution
model out of the consensus: the model that improved caries 35 % recovers
**2.7 %** of the universally missed set, so these are not small findings
starved by downsampling. The external check bounds the reading from the other
side: the same detector is 73.6 to 84.1 % precise at tooth level on a
professionally annotated corpus, so the label set is not uniformly bad; the
911 are a specific, enumerable subset and the list is the audit's sampling
frame. Protocol: run the same consensus over the TRAINING split with models
trained on it, since an annotation a model cannot fit after being optimised to
fit it is the strongest cheap label-error evidence available.

## 6. Results

Nineteen configurations. Segmentation: reference, three class-weighting
strengths, boundary objective alone and combined, copy-paste alone and combined,
four published comparator losses (implemented; only partially run, see limitations), a second backbone, a P2-fed prototype head.
Detection: stock DINO-DETR, each frequency-aware component alone and unified,
plus oversampling and contrast enhancement, a class-balanced control, a τ sweep,
and decoupled classifier retraining.

**Segmentation, held-out test:**

| model | mAP | AP50 | AP75 | head | tail |
|---|---|---|---|---|---|
| baseline | 0.1051 | 0.2590 | 0.0687 | 0.2861 | 0.0365 |
| boundary alone | 0.1071 | 0.2585 | 0.0680 | 0.2852 | 0.0441 |
| weighting + boundary | 0.1007 | 0.2430 | 0.0650 | 0.2908 | 0.0261 |
| weighting alone | 0.0973 | 0.2369 | 0.0612 | 0.2867 | 0.0252 |

**Detection, validation, matched budget:**

| arm | mAP | tail |
|---|---|---|
| baseline | 0.1625 | 0.0368 |
| denoising only | 0.1633 | 0.0455 |
| plain reweighting | 0.1461 | 0.0270 |
| **unified (τ = 1.0)** | **0.1020** | **0.0000** |

The unified treatment loses **4.41 pp to plain reweighting**, which loses
1.64 pp to changing nothing. Tail AP is exactly zero in all unified arms.

**τ sweep:** −6.05, −2.85, −0.24 pp at τ = 1.0, 0.5, 0.25. Monotonic recovery,
never reaching baseline. The catastrophe is a mis-scaled constant; the deficit
is not.

**Divergence is patterned.** Every arm applying the adjustment inconsistently
diverged with inf or NaN cost matrices in the Hungarian matcher; every
consistent arm was stable. Consistency prevents divergence without producing
accuracy.

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

## The attribution is correct and not actionable

A localisation earns its keep by telling you what to change, so we tested both
things it implies. Capacity: the coefficient branch rebuilt from 80 to 256 hidden
channels. Supervision: an auxiliary term pulling the predicted coefficients
toward `c*` itself, recomputed each step from the model's own prototypes and
detached.

Neither improves masks.

| arm | segm mAP | delta | box mAP | delta |
|---|---|---|---|---|
| reference | 0.1055 | - | 0.1551 | - |
| **cv4 widened** | **0.1118** | **+0.64** | 0.1625 | **+0.74** |
| coefficient supervision | 0.0985 | -0.70 | 0.1542 | -0.09 |

The capacity arm gives the largest single gain in this study, +0.64 pp against a
+-0.21 pp noise floor, with tail AP nearly tripling from 0.0101 to 0.0297.
Reported as it stands it is the one positive architectural result in the work.

It is not a mask result. Scoring the **same** predictions as boxes gives
+0.74 pp, larger than the segmentation gain, and `cv4` emits mask coefficients
with no path to the box head. Segmentation AP is a conjunction: an instance
counts only if the detection matches **and** the mask clears the IoU threshold,
so a change that improves only the detector raises it while leaving masks
untouched. Paired on the 5352 cases where both models emit a mask, Dice moves
-0.0016 with the interval spanning zero, and the one metric separable from zero
is boundary F at -0.0043, in favour of the reference.

The supervision arm is -0.70 pp with paired Dice -0.0017, and its own pixel BCE
at epoch 50 is roughly 3.05 against the reference's 1.65: the auxiliary term
pulled the head away from the pixel optimum rather than toward it. Target and
prediction agree to within 3 % in RMS, so this is not a scale mismatch. The
coefficients minimising pixel BCE and those best reconstructing the mask in least
squares are different points.

**Protocol.** Any claim that an architectural change improved segmentation must
report the same predictions scored as boxes. If the box delta is comparable to or
larger than the mask delta, the change is not a mask result whatever the
segmentation AP says.

### A calibration failure worth recording

The first supervision run was numerically broken and still produced a
publishable-looking number. `A = P B P^T` has eigenvalues spanning 1e-6 to 1e3 on
real instances, so a fixed ridge regularises some and leaves others singular;
training loss reached 7.2e7 with NaN at epoch 10, and the arm still reported mAP
0.1106. The weight had been calibrated on the **converged** model, where mean
`c*^2` is 2.86, then applied to a run starting from COCO weights where it
measures 4 to 53. Calibrating a hyper-parameter on a model state the run never
occupies is its own failure mode, and the loss curve is the cheapest detector of
it: the metric alone looked unremarkable.

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

## 7. Discussion

The honest summary is that the standard long-tail and boundary-aware toolkit
does not transfer to this corpus, and that most of the reasons are not about the
methods.

Two are about the data: it cannot be evaluated as shipped, and 16 of 31 classes
lack the support to measure anything. Two are about measurement: distance
metrics and group means both produce confident wrong answers under conditions
that are easy to satisfy accidentally. One is about regime: a published constant
validated at 100:1 does not survive 34,320:1.

Only one is about a method — the band-Dice objective produces the AP75/AP50
signature its mechanism predicts, and that signal does not survive to test.

We think the protocols in §3.1, §5.1 and §5.2 are the durable output. Each is
cheap, each is applicable to any long-tailed clinical segmentation study, and
each caught an error in our own work that would otherwise have been published.

---

## 8. Limitations

Single seed on all reported arms; the three-seed noise floor is measured above
and is the denominator for every comparison. Single corpus, single backbone family per task. No comparison
against prior published results, because none exist for this dataset. No
external benchmark, and therefore no long-tail claim in the title, abstract or
conclusions. Two detection cells are recorded as diverged rather than scored.

---

## References

Cheng, B., Girshick, R., Dollár, P., Berg, A., Kirillov, A. *Boundary IoU:
Improving Object-Centric Image Segmentation Evaluation.* CVPR 2021.

Cui, Y., Jia, M., Lin, T.-Y., Song, Y., Belongie, S. *Class-Balanced Loss Based
on Effective Number of Samples.* CVPR 2019.

Ghiasi, G., Cui, Y., Srinivas, A., et al. *Simple Copy-Paste is a Strong Data
Augmentation Method for Instance Segmentation.* CVPR 2021.

Gupta, A., Dollár, P., Girshick, R. *LVIS: A Dataset for Large Vocabulary
Instance Segmentation.* CVPR 2019.

Kang, B., Xie, S., Rohrbach, M., et al. *Decoupling Representation and
Classifier for Long-Tailed Recognition.* ICLR 2020.

Karimi, D., Salcudean, S. *Reducing the Hausdorff Distance in Medical Image
Segmentation with Convolutional Neural Networks.* IEEE TMI 2019.

Ke, L., Danelljan, M., Li, X., Tai, Y.-W., Tang, C.-K., Yu, F. *Mask Transfiner
for High-Quality Instance Segmentation.* CVPR 2022.

Kervadec, H., Bouchtiba, J., Desrosiers, C., et al. *Boundary Loss for Highly
Unbalanced Segmentation.* MIDL 2018 / Medical Image Analysis 2021.

Kirillov, A., Wu, Y., He, K., Girshick, R. *PointRend: Image Segmentation as
Rendering.* CVPR 2020.

Menon, A. K., Jayasumana, S., Rawat, A. S., et al. *Long-Tail Learning via Logit
Adjustment.* ICLR 2021.

Milletari, F., Navab, N., Ahmadi, S.-A. *V-Net: Fully Convolutional Neural
Networks for Volumetric Medical Image Segmentation.* 3DV 2016.

Salehi, S., Erdogmus, D., Gholipour, A. *Tversky Loss Function for Image
Segmentation Using 3D Fully Convolutional Deep Networks.* MLMI 2017.

Shi, W., Caballero, J., Huszár, F., et al. *Real-Time Single Image and Video
Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network.*
CVPR 2016.

Wang, J., Zhang, W., Zang, Y., et al. *Seesaw Loss for Long-Tailed Instance
Segmentation.* CVPR 2021.

Wang, T., Li, Y., Kang, B., et al. *Classification Calibration for Long-tail
Instance Segmentation.* arXiv:1910.13081.

Zhang, G., Lu, X., Tan, J., et al. *RefineMask: Towards High-Quality Instance
Segmentation with Fine-Grained Features.* CVPR 2021.

Zhang, H., Li, F., Liu, S., et al. *DINO: DETR with Improved DeNoising Anchor
Boxes for End-to-End Object Detection.* ICLR 2023.
