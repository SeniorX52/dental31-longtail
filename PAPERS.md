# Proposed papers: question, contribution, endpoint, and risk

One row per proposed paper, stating the evidence that exists rather than the
evidence anticipated. Two of the four are defined at the level of a genuine
methodological contribution. Two are not, and this document says so rather than
splitting a single contribution to reach a target count.

---

## Paper 1 — Boundary- and geometry-aware segmentation objective

| field | content |
|---|---|
| **Research question** | Does an objective that supervises the *contour band* rather than the region improve clinically meaningful outline fidelity in dental radiographs, and does that improvement survive against losses already published for the same purpose? |
| **Methodological contribution** | `L_band`: Dice computed between the morphological-gradient bands of prediction and ground truth, implemented as two stride-1 max-pools, applied inside a prototype-coefficient mask head. No distance map, no offline preprocessing, one interpretable hyperparameter (band width = tolerance in pixels). The gradient is provably sparse and contour-localized — measured at 48.5 % nonzero versus 100 % for region losses — so the term acts only where the contour is. Full definition in `yolov8_seg_longtail/BOUNDARY_OBJECTIVE.md`. |
| **Dataset** | 31-class panoramic radiographs, patient-grouped rebuilt split (9752 / 2090 / 2090). Partitions verified disjoint: 0 exact duplicates, 0 NCC-confirmed near-duplicates across 792,811 candidate pairs, 0 shared patients. |
| **Baselines** | Stock BCE mask loss (necessary control), and published comparators at matched budget: soft Dice (Milletari 2016), Tversky and Focal Tversky (Salehi 2017; Abraham & Khan 2019), Boundary loss (Kervadec 2018). Second backbone (yolov8l-seg) for transfer. |
| **Primary endpoint** | Contour fidelity: boundary F-score and HD95/ASSD, with Dice/IoU as region context, per class with image-level bootstrap CIs. **Secondary, clinical:** bone-loss area measurement error (MAE, bias, Bland-Altman), paired per-image against the baseline. |
| **Evidence today** | Isolated effect (S2 − S1c, only `boundary_weight` differs): AP75 **+0.70 pp / +10.7 %**, head AP **+0.85 pp / +3.0 %**, AP50 **−1.23 pp**. The AP50 cost is the *predicted* signature of contour sharpening, which is what makes this a mechanism result rather than a number. **But the reference config trained at three seeds gives an AP75 spread of 0.50 pp (sd 0.25 pp), so a +0.70 pp effect is 1.4× the observed spread from a single realisation of each arm.** It is not inside the noise, and it is not comfortably outside it either. |
| **Risk** | **Medium-high, and higher than it looked on 04 Aug.** (1) The loss form is a synthesis of known components — morphological gradient plus soft Dice — and its nearest relative, Boundary IoU, is an established metric; the novelty argument is application and mechanism, not a new mathematical object. (2) **The comparator arms were attempted and failed to run** — an in-place autograd error in the Kervadec implementation meant none of the four published comparators produced a number. This is not "not yet got to"; it is a known defect. "Beats existing boundary-aware losses" is therefore unsupported. (3) The effect size sits at 1.4× the seed spread on a single seed per arm, so replication is required before it is reportable as an effect at all. |
| **What would remove the risk** | A curvature- or scale-adaptive band width, making the objective a function of local contour geometry rather than a constant. Defined, not done. |

---

## Paper 2 — Unified frequency-aware detection

| field | content |
|---|---|
| **Research question** | Does applying class-frequency information *consistently* across the classification loss, the Hungarian matching cost and the denoising task beat applying it in any one place, and beat ordinary loss reweighting? |
| **Methodological contribution** | The consistency argument itself: in DETR-family models the matcher decides which query is supervised for which ground truth, so adjusting the loss while leaving the cost frequency-blind makes the two fight. The contribution is the joint treatment — logit-adjusted focal loss, the mirrored adjustment inside the matching cost, and frequency-aware sampling of denoising labels — plus the ablation that separates them. |
| **Dataset** | Same 31-class panoramic split, presented in COCO layout. **See the open issue below.** |
| **Baselines** | Standard DINO-DETR (stock focal loss is itself the focal-style control), and `C1`, conventional class-balanced reweighting (effective-number, Cui 2019) — verified to reproduce stock focal loss to 0.0 absolute difference at unit weights, so the per-class weight is provably the only change. |
| **Primary endpoint** | COCO mAP / AP50 / AP75 with head/mid/tail grouping, on the frozen test split. The decisive comparison is **D5 (unified) − C1 (plain reweighting)**: if that is not positive, the consistency claim fails regardless of the baseline delta. |
| **Evidence today** | **The matrix has run, and the decisive comparison came out negative.** Validation bbox mAP: D1 baseline 0.1625, C1 plain reweighting 0.1461, D5 unified 0.1020. **D5 − C1 = −4.41 pp**, D5 − D1 = −6.05 pp. The consistency claim fails on the criterion set for it above. The failure is specific and legible rather than noisy: D5, D6 and D7 all score tail AP **exactly 0.0000** while head AP is untouched at ~0.380, so the unified treatment is not degrading the model uniformly — it is erasing the tail it was built to serve. A τ sweep confirms this is a dose response, not instability: τ = 1.0 / 0.5 / 0.25 gives mAP 0.1020 / 0.1340 / 0.1601, monotonically approaching the baseline as the adjustment is removed. |
| **Risk** | **The paper as conceived is dead; what survives is the negative result.** (1) The claim it was built on is measured and false on this corpus. (2) The tail is not measurable on this split — 16 of 31 classes appear in fewer than 10 validation images — which is *also* the most likely explanation for the failure, and cannot be separated from it here. (3) The one positive arm, cRT decoupled classifier retraining, is +0.53 pp on test (0.1623 vs 0.1570) with tail 0.0598 → 0.0724, but **98 % of that gain is a single class with 2 test instances**, so it is an anecdote, not a result. The honest paper is now "consistent frequency adjustment erases the tail in DETR-family detectors on a corpus whose tail is unmeasurable", which needs a corpus with a measurable tail to be worth writing. |
| **What would remove the risk** | Validation on an established long-tail benchmark (LVIS — a separate training programme in its own right) or a larger external dental set. DENTEX 2023 supplies *cross-dataset transfer* evidence, which is valuable but is **not** a long-tail benchmark and should not be described as one. |

---

## Paper 3 — candidate: anatomy-aware augmentation for radiographs

**Not yet a contribution.** Stated as a candidate with what is missing.

| field | content |
|---|---|
| **Research question** | Why does copy-paste augmentation, which reliably helps on natural images, *hurt* on panoramic radiographs — and can an anatomically constrained version help? |
| **Finding in hand** | Copy-paste increased rare-class instances roughly tenfold (bone defect 1→11, TAD 2→22, brackets 94→804) and made every metric that matters *worse*: AP75 −0.25 pp alone, and it removes 0.55 pp of AP75 when stacked on the boundary term. Proposed mechanism: in radiographs anatomical position is part of the class definition, so a lesion pasted into an anatomically impossible location teaches the model to detect the paste, not the pathology. |
| **Why it is not a paper yet** | A negative result plus a plausible mechanism is a strong *section*, not a paper. It becomes one only with the positive counterpart: a placement model that respects anatomical priors, shown to recover the gain that naive pasting destroys. That has not been built. |
| **Risk** | **Very high as a standalone paper.** Right now this is the honest home for the finding: a results section inside Paper 1, or a workshop note. Promoting it to a fourth paper would be an artificial split of a single contribution. |

---

## Paper 4 — candidate: evaluation integrity for long-tailed clinical datasets

**Not yet a contribution, and probably belongs elsewhere.**

| field | content |
|---|---|
| **Research question** | What does it take to make a long-tail claim on a clinical dataset *measurable*, and how often is the published setup incapable of supporting one? |
| **Findings in hand** | (1) The shipped split was ~96 % contaminated — 1516 of 1580 test files had their source image in train/valid, only 64 genuinely unseen, 194 patients on both sides. (2) Perceptual hashing alone is not a leakage test on this modality: dHash flags 792,811 candidate pairs on a split that is in fact clean, and every one is rejected by pixel correlation (max NCC 0.9334 against a 0.98 threshold). (3) 16 of 31 classes occur in fewer than 10 validation images, so most per-class tail numbers are one or two detections. |
| **Why it is not a separate method paper** | It is methodology and benchmarking, not a method. Its natural home is the lab's **benchmark paper**, where it is genuinely valuable — it is the argument for why that benchmark's split is trustworthy. |
| **Risk** | **High as a standalone.** Recommend folding into the benchmark paper rather than counting it toward four. |

---

## Open issue that affects Papers 1 and 2 jointly

The two projects are **currently running on the same images**. Verified directly:
the filename sets in `data_clean` (segmentation) and `data_coco` (detection) are
identical across all three splits — `data_coco` is `data_clean` re-presented in
COCO layout.

Only one annotated dataset was ever received. The OrthoBench archive contains
8,299 images across 962 patients, but its labels are **Angle malocclusion class
(I / II / III) at patient level** — no boxes, no polygons, no per-object
annotations, and `split: unassigned` on every row. It supports a classification
task and cannot serve either project as specified.

This matters for the manuscripts because the instruction "do not imply that
YOLOv8x-seg and DINO-DETR were trained or evaluated on the same dataset unless
that was actually done" currently cuts the other way: they were. Either a second
annotated dataset is supplied, or both papers state plainly that they share a
corpus and differ in task and annotation type.

## Honest count

**Two** papers are defined at the level of a genuine methodological
contribution, and one of those (Paper 2) has no results yet. Papers 3 and 4 are
candidates whose contributions are not yet defined; on current evidence the
right move is to fold them into Paper 1 and the benchmark paper respectively.
Three strong papers is a better outcome than four thin ones, and dividing a
contribution artificially to reach a target count points the other way.
