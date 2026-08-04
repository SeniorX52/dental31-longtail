# Metric definitions, and the exact values behind the reported gains

Written because "ten percent better outline accuracy" is not a reportable
number: it does not say which metric, on which split, at which training budget,
against which arm, or whether the figure is absolute or relative. Every number
this project reports is defined here.

## 1. The two figures previously quoted, stated precisely

Both come from the **50-epoch ablation, validation split, mask (segm) metrics,
scored by pycocotools via `eval/coco_eval_report.py`, seed 42**.

The valid isolation of the boundary term is **`S2` vs `S1c`**. Both arms use
inverse-sqrt class weighting; they differ *only* in `boundary_weight`
(0.5 vs 0). Comparing against the `S0` reference instead folds the
class-weighting change into the boundary figure.

| quoted as | metric | arms | from | to | absolute | relative |
|---|---|---|---|---|---|---|
| "about 10 percent better mask tightness" | **AP75** (mask) | S2 − S1c | 0.0659 | 0.0730 | **+0.70 pp** | **+10.7 %** |
| "about 4 percent on the common classes" | **head-group AP** (mask) | S2 − S1c | 0.2836 | 0.2921 | **+0.85 pp** | **+3.0 %** |

**Correction.** The "about 4 percent" figure was computed as `S2 − S0`, which
gives +1.05 pp / +3.7 % but includes the effect of inverse-sqrt weighting.
Isolated correctly it is **+0.85 pp / +3.0 %**. The AP75 figure is essentially
unchanged by the correction (+0.69 pp / +10.4 % conflated, +0.70 pp / +10.7 %
isolated), so the headline stands; the head figure was overstated by 0.2 pp.

Full context for the same contrast:

| metric | S1c | S2 | absolute | relative |
|---|---|---|---|---|
| mAP | 0.1065 | 0.1050 | −0.15 pp | −1.4 % |
| AP50 | 0.2591 | 0.2467 | −1.23 pp | −4.8 % |
| AP75 | 0.0659 | 0.0730 | +0.70 pp | +10.7 % |
| head AP | 0.2836 | 0.2921 | +0.85 pp | +3.0 % |
| tail AP | 0.0117 | 0.0076 | −0.41 pp | not interpretable |

The AP50 decrease is not a defect to be explained away; it is the predicted
cost of sharpening contours and is part of the evidence for the mechanism
(see `yolov8_seg_longtail/BOUNDARY_OBJECTIVE.md`).

**Isolated effect of inverse-sqrt weighting alone** (`S1c − S0`): AP75 −0.01 pp,
mAP +0.10 pp, head +0.21 pp. It does essentially nothing.

## 2. Conventions

- **pp** = percentage points, an absolute difference between two rates.
  **%** = relative change, `(new − old) / old`. Every reported change states
  which it is. A change from 0.0659 to 0.0730 is +0.70 pp and +10.7 %.
- All AP figures are in the range 0–1 as pycocotools emits them; "pp" refers to
  the value multiplied by 100.
- No number is reported from the test split unless the arm was pre-registered
  for a test evaluation. Selection happens on validation only.

## 3. Detection and segmentation metrics (COCO family)

Computed by `eval/coco_eval_report.py`, a single shared scorer used for every
run in both projects so numbers are always comparable.

- **mAP** — AP averaged over IoU thresholds 0.50:0.05:0.95, then over classes.
- **AP50**, **AP75** — AP at a single IoU threshold. AP75 is the tightness-
  sensitive one: it requires 75 % overlap, which loose masks do not clear.
- **iou-type** `bbox` for detection, `segm` for segmentation. Segmentation
  results are always mask metrics unless explicitly labelled box.
- **Frequency groups**, defined by **train-split** instance count so the
  grouping never peeks at evaluation statistics:
  `head > 5000`, `mid 100–5000`, `tail < 100`.
  Group AP is the unweighted mean of the per-class APs in that group.
- **`unstable` flag** — set when a class has fewer than 10 instances in the
  evaluation split. Its per-class AP is printed for transparency but is not a
  claimable number.

## 4. Region and contour metrics

Computed by `eval/contour_metrics.py`. Protocol: for each image and class, all
ground-truth instances are unioned into one mask and all predictions above the
frozen confidence cut into another; metrics are computed between those masks.

Let `M` be the ground-truth mask, `P` the predicted mask, `∂` the 1-pixel inner
boundary, and `d(x, S)` the Euclidean distance from pixel `x` to set `S`.

- **Dice** = `2|M ∩ P| / (|M| + |P|)`
- **IoU** = `|M ∩ P| / |M ∪ P|`
- **Boundary F-score** — with tolerance `θ`:
  `precision = |{x ∈ ∂P : d(x, ∂M) ≤ θ}| / |∂P|`,
  `recall = |{x ∈ ∂M : d(x, ∂P) ≤ θ}| / |∂M|`, `F = 2PR / (P + R)`.
  `θ = 0.75 %` of the image diagonal (DAVIS convention), so the metric does not
  become easier on larger radiographs.
- **HD95** — `max` of the two directed 95th percentiles,
  `max(Q95{d(x, ∂M) : x ∈ ∂P}, Q95{d(x, ∂P) : x ∈ ∂M})`. The max is used rather
  than pooling both directions into one percentile, where a large one-sided
  error can hide behind the other side's mass.
- **ASSD** — the average symmetric surface distance, the mean of all directed
  boundary distances in both directions.
- Distances are in **pixels at native image resolution**. These panoramic
  images carry no pixel-spacing metadata, so no distance can be given in
  millimetres.

**Empty-mask handling**, which decides whether the distance metrics mean
anything: HD95 and ASSD are undefined when either mask is empty, so a model
predicting nothing would otherwise score perfectly by having no cases left to
average. This project never drops those cases silently:

| case | treatment |
|---|---|
| both non-empty | contributes to every metric |
| both empty | dropped — the class is simply absent |
| ground truth non-empty, prediction empty | **miss**: Dice, IoU, bF = 0 |
| ground truth empty, prediction non-empty | **false alarm**: Dice, IoU, bF = 0 |

HD95 and ASSD average over both-non-empty cases only, and `n_both`, `n_miss`
and `n_false_alarm` are printed beside them so the denominator is always visible.

**Confidence cut.** Union masks need a threshold, unlike mAP. It is selected on
the **validation** split using the **baseline** arm, so it cannot be tuned in
favour of the proposed method, then frozen and applied unchanged to every arm
and to the test split. The selected value is recorded in every output file.

## 5. Uncertainty

Bootstrap resamples **images** with replacement, not per-class records:
observations from the same radiograph are not independent, and resampling
records would understate the interval. Reported intervals are two-sided 95 %
percentile intervals. The resample count, alpha, seed and resampled unit are
recorded in every output file.

For paired model comparisons the **per-image difference** is bootstrapped,
which is the correct construction for "did the error decrease" and is far
tighter than comparing two independent intervals.

**Determinism.** This pipeline is exactly reproducible: `runs/segment/abl_S0`
and `runs/segment/abl_S0-2` are the same configuration at the same seed, trained
independently, and agree to six decimal places on every reported metric. Run-to-
run noise is therefore zero and **seed-to-seed variation is the only source of
spread**, which is why the seed replicates in `run_seg_completion.sh` are the
sole route to a variance estimate.

## 6. Clinical endpoint

Computed by `eval/bone_loss_endpoint.py`. Endpoint is the **area of the
`Bone Loss` region as a percentage of image area**, with error reported as MAE,
signed bias, median absolute percentage error, Pearson and Spearman
correlation, and Bland-Altman limits of agreement, plus the false-positive area
on radiographs with no ground-truth bone loss.

Distance-based and normalized-bone-level endpoints are **not computable** from
this annotation set: both require cemento-enamel-junction and root-apex
landmarks, and the annotations provide only a region polygon. This is a
limitation of the data, stated rather than worked around.
