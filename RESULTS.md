# Results

Every number measured so far, with the protocol that produced it. Numbers that
did not survive scrutiny are recorded alongside the corrected version rather
than removed, because two of them were reported before they were checked.

Test is touched once per project, at the end. Everything else is validation.

---

## 1. Data integrity

The rebuilt split is patient-grouped and frozen: **9752 / 2090 / 2090** images.

Cross-split disjointness was verified twice, by two independent
implementations, and they agree:

| check | `dataset_audit.py` | `verify_no_duplicates.py` |
|---|---|---|
| exact duplicate groups (SHA-256) | 0 | 0 |
| **NCC-confirmed near-duplicates** | **0** | **0** |
| dHash candidate pairs examined | 441,240 | 792,811 |
| rejected as look-alikes | all | all |
| source images in >1 split | — | 0 |
| patients in >1 split | — | 0 |
| scope | test vs rest | all ordered pairs |

Across 792,811 candidate pairs the **maximum** NCC is 0.9334 against a 0.98
confirmation threshold — nothing comes close to being a duplicate. This also
settles the method question: dHash alone would have reported all 792,811 as
leaks on a split that is in fact clean, so pixel confirmation is not optional
on this modality.

Within-split exact duplicates do exist (40 train, 6 valid, 11 test groups).
They are not leakage, but the effective test set is **2079 distinct images**,
not 2090.

### Statistical power

**16 of 31 classes appear in fewer than 10 validation images**; two appear in
none. Eleven tail classes score exactly 0.0000. Image counts, not instance
counts, are the effective sample size, and per-class results for those classes
are exploratory. Full table in `reports/class_counts_seg.md`.

---

## 2. Baselines, held-out test

| model | metric | mAP | AP50 | AP75 | head | mid | tail |
|---|---|---|---|---|---|---|---|
| DINO-DETR | bbox | 0.1570 | 0.3086 | 0.1345 | 0.3864 | 0.1677 | 0.0598 |
| YOLOv8x-seg | bbox | 0.1527 | 0.2971 | 0.1402 | 0.3837 | 0.1593 | 0.0583 |
| YOLOv8x-seg | **segm** | **0.1051** | 0.2590 | 0.0687 | 0.2861 | 0.1038 | 0.0365 |

These are the first trustworthy numbers on this dataset. The shipped split
could not produce valid ones: ~96 % of its test files had their source image in
train or valid, only 64 were genuinely unseen, and 194 patients appeared on
both sides.

No prior published result exists for this dataset to compare against —
verified across the origin repository, the mirrors, and the literature.

---

## 3. Segmentation ablation — validation, 50 epochs, mask metrics

| arm | configuration | mAP | AP50 | AP75 | head |
|---|---|---|---|---|---|
| S0 | reference (no weighting, no boundary) | 0.1055 | 0.2549 | 0.0661 | 0.2815 |
| S1a | β = 0.9 | 0.1051 | 0.2559 | 0.0646 | 0.2840 |
| S1b | β = 0.99 | 0.1029 | 0.2473 | 0.0665 | 0.2831 |
| S1c | inverse-sqrt | 0.1065 | 0.2591 | 0.0659 | 0.2836 |
| S2 | inverse-sqrt + boundary | 0.1050 | 0.2467 | **0.0730** | **0.2921** |
| S3 | inverse-sqrt + copy-paste | 0.1042 | 0.2570 | 0.0635 | 0.2836 |
| S4 | inverse-sqrt + boundary + copy-paste | 0.1051 | 0.2513 | 0.0675 | 0.2916 |

### Isolated contributions

This grid is cumulative — every arm carrying the boundary term also carries
inverse-sqrt weighting — so `S2 − S0` measures *both* changes. The valid
isolation is **`S2 − S1c`**, where only `boundary_weight` differs.

| contrast | metric | change |
|---|---|---|
| **boundary alone** (S2 − S1c) | AP75 | **+0.70 pp, +10.7 %** |
| | head AP | +0.85 pp, +3.0 % |
| | AP50 | −1.23 pp, −4.8 % |
| | mAP | −0.15 pp |
| **weighting alone** (S1c − S0) | AP75 | −0.01 pp |
| | mAP | +0.10 pp |
| **copy-paste** (S3 − S0) | AP75 | −0.25 pp |

**Correction on record.** The head figure was first computed as `S2 − S0`,
giving +1.05 pp / +3.7 %. Isolated correctly it is **+0.85 pp / +3.0 %**. The
AP75 headline is unaffected (+10.4 % conflated, +10.7 % isolated).

Class weighting does essentially nothing, and the mechanism is known: YOLO's
`TaskAlignedAssigner` selects positives by `cls_score^α × IoU^β`, so for a
class the model never learned `cls_score ≈ 0` and the ground truth receives few
or no positive assignments. Re-weighting then scales a loss over assignments
that were never made. The bottleneck is upstream of the loss.

Copy-paste made every metric that matters worse despite adding ~10× more tail
instances. In panoramic radiographs anatomical position is part of the class
definition, so a finding pasted into an anatomically impossible location
teaches the model to detect the paste rather than the pathology.

The arms span 0.1029–0.1065 mAP — **a 0.36 pp spread at one seed each**, with
no variance estimate. Selecting a winner on mAP would be selecting noise, which
is why the final stage runs both the best-mAP and best-AP75 candidates.

---

## 4. Contour metrics — where the boundary claim does not hold

Region and contour metrics on the isolated contrast, validation, confidence cut
0.15 (selected on the *baseline* arm so it cannot favour the method, then
frozen).

Averaged per model, the result looked decisive:

| metric | S1c | S2 | change |
|---|---|---|---|
| HD95 | 85.55 px | 70.75 px | −17.3 % |
| ASSD | 27.25 px | 20.55 px | −24.6 %, intervals disjoint |

**That result is an artifact and it does not survive.** HD95 and ASSD are
undefined when either mask is empty, so they average only over cases where both
masks are non-empty — and that denominator differs per model. S2 misses more
structures than S1c (605 vs 566), and the cases it misses are disproportionately
the hard ones carrying large distances. Removing them from its own average
produced the entire gap.

Recomputed on the **same 5385 cases** — every (image, class) pair where both
arms produced a non-empty mask — with the paired per-case difference
bootstrapped over images:

| metric | S1c | S2 | difference | 95 % CI | separable |
|---|---|---|---|---|---|
| Dice | 0.6953 | 0.6969 | +0.0017 | [−0.0010, +0.0041] | no |
| IoU | 0.5685 | 0.5706 | +0.0021 | [−0.0007, +0.0048] | no |
| boundary F | 0.7857 | 0.7881 | +0.0024 | [−0.0009, +0.0053] | no |
| **HD95** | 66.36 | 68.66 | **+2.30 (worse)** | [+0.19, +4.25] | **yes** |
| ASSD | 18.40 | 18.82 | +0.43 (worse) | [−0.25, +1.05] | no |

On equal footing the boundary term is **significantly worse on HD95**,
indistinguishable on Dice, IoU, boundary F and ASSD, and costs 7 % more misses.
Correcting the denominator did not shrink the effect — it inverted its sign.

**What survives.** The AP75 gain stands: pycocotools matches instances
internally across all confidences and has no per-model denominator to bias. But
it is now the only positive evidence, and the two protocols measure different
things — AP75 is instance-level across the full confidence range, the contour
metrics are per-class union masks at one frozen threshold. A gain in one with
none in the other is as consistent with better confidence ranking as with
better contours. The contour interpretation cannot be asserted while the direct
contour measurements decline to support it.

Any metric conditioned on a per-model subset must be compared on the
intersection, with coverage reported beside it. `eval/paired_contour.py` does
this.

---

## 5. Downstream clinical endpoint — bone-loss area error

A better contour metric is not itself evidence of clinical utility, so the
quantity a clinician would read off the segmentation is measured directly.

**Endpoint:** area of the `Bone Loss` region as a percentage of image area.
Distance-based and normalized bone-level endpoints are **not computable** from
this annotation set — both need cemento-enamel-junction and root-apex
landmarks, and the annotations provide a region polygon and nothing else. These
images also carry no pixel-spacing metadata, so no measurement can be given in
millimetres.

225 validation images contain ground-truth bone loss; the other 1865 are used to
measure bone loss reported where there is none.

| | S1c | S2 |
|---|---|---|
| MAE (pp of image area) | 1.114 | 1.129 |
| bias (signed) | −0.436 | −0.717 |
| Pearson r | 0.585 | 0.593 |
| false-positive area on healthy images | 0.129 | **0.071** |
| fraction of healthy images with a false positive | 6.6 % | **4.3 %** |

Paired per-image absolute error, S2 − S1c: **+0.0154 pp, 95 % CI
[−0.0738, +0.1060]**, better on 27.6 % of images. The interval spans zero.

**Verdict: no detectable change in bone-loss measurement error.**

The two arms do differ, but not in accuracy. S2 carries a larger negative bias
(−0.717 vs −0.436, i.e. it under-segments more) while producing roughly half
the false-positive area on healthy radiographs. Together with its higher miss
count in section 4, this is one coherent effect: **the boundary term shifts the
operating point toward precision** — fewer false alarms, more misses — and the
two cancel in the endpoint.

One limitation dominates both arms: median absolute percentage error is 100 %
for each, meaning that on at least half the diseased images the predicted bone
loss area is essentially zero. That is a property of the underlying detector at
this operating point, not of the loss being compared.

---

## 6. Training schedule — every arm overtrains

Validation mask mAP50-95 from the trainer's own per-epoch metrics (a shape, not
a number — the reportable figures come from the shared pycocotools scorer):

| run | budget | peak | at epoch | last epoch | drop |
|---|---|---|---|---|---|
| baseline | 100 | 0.11601 | 47 | 0.09397 | −19.0 % |
| S1c final | 100 | 0.11677 | 24 | 0.10134 | −13.2 % |
| S2 ablation | 50 | 0.12137 | 24 | 0.11337 | −6.6 % |

S2 peaks at 0.12137 on a 50-epoch schedule and 0.12135 on a 100-epoch one — the
same number. Every epoch past roughly 30 makes the model worse, for the baseline
as much as for the modified arms. This is a property of the dataset.

**The comparison is nonetheless sound.** Scoring uses `best.pt`, selected on
validation fitness during training, and the baseline was scored the same way.
Both sides report their validation-selected peak on a matched 100-epoch budget.
Had either been scored from the last epoch the comparison would be meaningless.

**Validation ordering has not transferred to test.** S1c's validation peak sat
0.7 % above the baseline's; its test mask mAP came in **below** baseline
(0.0973 vs 0.1051). The 50-epoch ablation taught the same lesson when its
best-mAP arm lost at 100 epochs.

---

## 7. Detection

| arm | split | mAP | AP50 | AP75 | head | tail |
|---|---|---|---|---|---|---|
| D1 standard DINO-DETR | valid | 0.1625 | 0.3080 | 0.1551 | 0.3808 | 0.0368 |

The frozen matrix (D2–D7 plus the class-balanced control C1) has not run. The
previous grid was cumulative — every arm after the baseline also carried
repeat-factor sampling — so it could not attribute anything to individual
components and was discarded rather than reported.

---

## 8. What is not claimed

- **No long-tail improvement.** The evaluation split has no statistical power
  for it: 16 of 31 classes occur in fewer than 10 validation images and eleven
  tail classes score exactly zero. No such claim appears in any title, abstract
  or conclusion.
- **No improvement in contour fidelity**, on the direct evidence in section 4.
- **No reduction in clinical measurement error**, on the evidence in section 5.
  The single positive result across the three independent measurements is the
  AP75 gain; the contour metrics and the clinical endpoint do not support it.
- **No comparison against prior published SOTA**, because none exists for this
  dataset.
- **Single seed** on all finals. Determinism is exact — two independent runs of
  the same configuration at the same seed agree to six decimals — so seed
  variation is the only source of spread, and it has not yet been measured.
