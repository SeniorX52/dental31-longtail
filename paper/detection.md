# Frequency-aware DETR training at extreme class imbalance

**Why the standard recipe fails at 34,320:1, with the frozen ablation that
establishes it.**

---

## Abstract

Logit adjustment is a standard remedy for long-tailed recognition, and applying
it consistently across a DETR-family model's classification loss, Hungarian
matching cost and denoising task is a natural extension. We report that on a
31-class dental radiograph corpus with 34,320:1 imbalance it fails
catastrophically: the unified treatment is 6.05 pp mAP below a stock baseline
with tail AP at exactly zero, and loses 4.41 pp to conventional class-balanced
reweighting.

We identify the cause as magnitude rather than principle. At τ = 1.0 the
adjustment applies a **+11.47 logit shift** to the rarest class against +1.03 to
the most common. A τ sweep recovers monotonically — −6.05, −2.85, −0.24 pp at
τ = 1.0, 0.5, 0.25 — confirming the diagnosis while never clearing the baseline.

We further report a reproducible stability pattern: **every arm applying the
adjustment inconsistently diverged**, with inf or NaN cost matrices in the
matcher, while every consistent arm trained stably. Consistency prevents
divergence; it does not produce accuracy.

Finally we evaluate decoupled classifier retraining, which is the mechanism the
long-tail literature indicates and which structurally cannot exhibit the failure
above. It is nominally +0.53 pp mAP on test — but 98 % of that gain comes from a
single class with **two** test instances, and on the 18 classes with adequate
support, 6 improve and 12 worsen.

No configuration improves on the baseline beyond noise.

---

## 1. Setup

Stock DINO-DETR, ResNet-50 4-scale, initialised from the official COCO
checkpoint with a reinitialised 31-class embedding. `num_classes = 32` because
DINO indexes raw `category_id` without remapping, so ids 1..31 require 32
outputs.

All arms: 12 epochs (official 1× schedule), `lr_drop` 11, batch 2, seed 42,
identical initialisation and evaluation protocol. **No arm receives a
hyperparameter search.** That keeps budgets matched and comparable; §4 shows it
is also, in this instance, the defect.

Evaluated on the verified-disjoint split described in the companion segmentation
report: 9752 / 2090 / 2090 images, 0 exact duplicates and 0 NCC-confirmed
near-duplicates across 792,811 candidate pairs, 0 shared patients.

---

## 2. The claim under test

In DETR-family models the matcher decides which query is supervised for which
ground truth. If the classification loss is frequency-adjusted but the matching
cost is not, matching keeps assigning tail ground truth to queries whose
unadjusted scores look good while the loss optimises a different scale. The
hypothesis is that applying the adjustment **consistently** — across
classification loss, matching cost, and the denoising task — beats applying it
in one place, and beats ordinary loss reweighting.

The matrix was frozen before any arm ran.

| cell | configuration |
|---|---|
| D1 | standard DINO-DETR |
| D2 / D3 / D4 | frequency-aware **loss** / **matching** / **denoising**, each alone |
| D5 | unified: all three |
| D6 / D7 | D5 + rare-class oversampling / + contrast enhancement |
| C1 | conventional class-balanced reweighting (effective-number, Cui et al. 2019) |

C1 is the control that settles the claim. It applies per-class loss weights only
— no logit shift, nothing in the matcher or denoising — and was verified to
reproduce stock `sigmoid_focal_loss` to **0.0 absolute difference** at unit
weights, so the per-class weight is provably the only change.

---

## 3. Result

Validation, matched budget.

| arm | mAP | AP50 | AP75 | head | mid | tail |
|---|---|---|---|---|---|---|
| D1 baseline | **0.1625** | 0.3080 | 0.1551 | 0.3808 | **0.2118** | 0.0368 |
| D2 loss only | *diverged* | | | | | |
| D3 matching only | *diverged* | | | | | |
| D4 denoising only | 0.1633 | **0.3165** | **0.1572** | **0.3821** | 0.2029 | **0.0455** |
| **D5 unified** | 0.1020 | 0.1998 | 0.0939 | 0.3805 | 0.0959 | **0.0000** |
| D6 unified + oversampling | 0.1014 | 0.1988 | 0.0946 | 0.3808 | 0.0943 | 0.0000 |
| D7 unified + contrast | 0.1012 | 0.1993 | 0.0929 | 0.3733 | 0.0972 | 0.0000 |
| C1 plain reweighting | 0.1461 | 0.2910 | 0.1375 | 0.3707 | 0.1847 | 0.0270 |

**The decisive comparison, D5 − C1:**

    mAP −4.41   AP50 −9.12   AP75 −4.36   mid −8.87   tail −2.70  (pp)

The unified treatment loses decisively to plain class-balanced reweighting,
which in turn loses 1.64 pp to changing nothing. **Tail AP in every unified arm
is exactly 0.0000** — the metric the method targets is the one it destroys.
Layering oversampling or contrast enhancement changes nothing, because the base
configuration is already broken.

Only D4 — frequency-aware denoising alone — clears the baseline, by 0.08 pp mAP.
That is inside noise and carries no claim, though its +0.88 pp tail and +0.84 pp
AP50 are the only movement in the intended direction anywhere in the matrix.

**On held-out test**, D5 confirms: mAP 0.1001 against 0.1570, **−5.69 pp**, tail
0.0000 against 0.0598.

---

## 4. Diagnosis: magnitude, not principle

Logit adjustment subtracts `τ · log p(c)` from each class logit. On this
training set:

| | instances | log-prior | shift at τ = 1.0 |
|---|---|---|---|
| rarest present class | 1 | −11.47 | **+11.47** |
| most common class | 34,318 | −1.03 | +1.03 |
| **spread** | | | **10.44** |

A +11.47 shift saturates the sigmoid for every query on that class. Menon et al.
(2021) validated τ = 1.0 on CIFAR-LT and ImageNet-LT, whose imbalance is of
order 100:1 to 1000:1. This corpus is **34,320:1**, well outside that regime.

A sweep on the unified configuration, changing nothing else:

| τ | mAP | Δ vs baseline | tail |
|---|---|---|---|
| 1.00 | 0.1020 | −6.05 | 0.0000 |
| 0.50 | 0.1340 | −2.85 | 0.0135 |
| 0.25 | 0.1601 | −0.24 | 0.0346 |

Recovery is **monotonic in τ**, which confirms magnitude as the mechanism. It is
also incomplete: even at τ = 0.25 the method does not reach the baseline.
Correcting the constant removes the catastrophe, not the deficit.

**This is a transferability result about the published default.** τ = 1.0 is
used across the long-tail literature without restatement of the imbalance regime
it was tuned for. At clinical imbalance it is not merely suboptimal; it is
destructive.

---

## 5. Divergence is patterned

Four arms failed to complete, all with `inf` or `NaN` cost matrices at
`linear_sum_assignment` in the Hungarian matcher. Which arms failed is not
random:

| arm | adjustment applied to | outcome |
|---|---|---|
| D2 | loss only | diverged |
| D3 | matching only | diverged |
| L1 | matching + denoising | diverged at 9/12 |
| L2 | loss + denoising | diverged |
| D4 | denoising only | **stable** |
| D5, D6, D7 | loss + matching + denoising | **stable** |

Checkpoints on disk are numerically clean — no NaN or inf, max |w| 1.42e+01,
identical to arms that completed — so divergence develops during the run rather
than being inherited. Gradient clipping was active at 0.1 throughout.

When the loss carries a +11.47 shift and the matcher scores queries on
unadjusted probabilities, the two components optimise against different scales
and the assignment degenerates. **Consistency does prevent divergence.** It
simply does not produce accuracy at this τ.

We report this as an observation from crashes, not a designed stability
experiment: the arms were not built to test it, and each is a single seed. The
pattern is reproducible — D2 diverged on every attempt — and worth stating for
that reason.

---

## 6. Decoupled classifier retraining

The long-tail literature indicates rebalancing belongs at the **classifier**
stage, not inside representation learning (Kang et al., ICLR 2020; Wang et al.'s
classification calibration for instance segmentation). That is precisely the
opposite of what every arm above does, and it structurally cannot exhibit the
failure in §5: nothing is added to the loss, the matcher is untouched, and the
representation is frozen.

We freeze all parameters except `class_embed` and `label_enc`, start from the
trained baseline, and retrain for 6 epochs under repeat-factor sampling.

| model | mAP | AP50 | AP75 | head | tail |
|---|---|---|---|---|---|
| baseline (test) | 0.1570 | 0.3086 | 0.1345 | 0.3864 | 0.0598 |
| **CRT (test)** | **0.1623** | **0.3175** | **0.1375** | 0.3857 | **0.0724** |

+0.53 pp mAP, +0.88 pp AP50, +1.25 pp tail — positive on every metric, with the
largest gain exactly where a long-tail method should deliver.

**It does not survive decomposition.**

| | |
|---|---|
| observed mAP delta | +0.0053 (+0.53 pp) |
| largest single contributor | **Permanent Teeth**, **2 test instances**, 0.0000 → 0.1515 |
| its contribution to mAP | +0.0052 (+0.52 pp) |
| **share of the total gain** | **98 %** |
| mAP delta excluding that one class | **+0.0001 (+0.01 pp)** |
| on 18 classes with ≥10 test instances | 6 improve, 12 worsen |

One class with two instances accounts for essentially the whole result. CRT is
not a demonstrated improvement.

We report it because the same decomposition applied to the segmentation
companion's best arm produced the same verdict, and because a group mean over
classes with single-digit support will manufacture a headline on demand.

---

---

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

## 7. Conclusion

No configuration tested improves on stock DINO-DETR beyond noise. What the work
establishes:

1. **Consistent frequency-awareness loses to plain reweighting** by 4.41 pp, and
   plain reweighting loses to changing nothing. The pre-registered hypothesis is
   refuted by its own frozen matrix.
2. **The published τ = 1.0 does not transfer** to clinical imbalance. The failure
   is monotonic in τ and traceable to a +11.47 logit shift. Correcting it removes
   the catastrophe but not the deficit.
3. **Inconsistent application diverges reproducibly**; consistent application is
   stable but inaccurate.
4. **Decoupled classifier retraining is the right mechanism and still does not
   clear noise** on this corpus.
5. **Group means over low-support classes fabricate results.** Two separate
   headline gains, in two projects, dissolved to single classes with 2–8
   instances.

### Limitations

**The published comparator losses did not run.** Soft Dice, Tversky, Focal
Tversky and Kervadec's boundary loss are implemented behind a common interface,
but the Kervadec arm failed with an in-place autograd error and the other three
were queued behind it. The objective is therefore compared against the stock
BCE baseline only, which is a necessary control and not a sufficient one.

Single seed per arm; the noise floor is measured above and is the
denominator for every comparison here. Single corpus. No external benchmark, so
no long-tail claim appears in the title, abstract or conclusions. D2 and D3 are
recorded as diverged rather than scored, so the individual-component cells for
loss and matching are unavailable at τ = 1.0.
