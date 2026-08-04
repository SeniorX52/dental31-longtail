# The band-Dice boundary objective

Definition, mechanism, and where it sits relative to existing boundary,
contour and Hausdorff-style losses.

## 1. Definition

Let `Ω ⊂ Z²` be the pixel grid, `M : Ω → {0,1}` a ground-truth instance mask,
and `M̂ : Ω → R` the predicted mask logits, with `p = σ(M̂) ∈ [0,1]^Ω`.

For an odd band width `k`, with `S_k(i)` the `k × k` window centred on pixel `i`:

    dilation    δ_k(x)[i] = max_{j ∈ S_k(i)} x[j]
    erosion     ε_k(x)[i] = min_{j ∈ S_k(i)} x[j]
    band        B_k(x)    = δ_k(x) − ε_k(x)

`B_k` is the **morphological gradient**. Both operations are implemented as a
single max-pool with stride 1 and padding `⌊k/2⌋`, using the identity
`ε_k(x) = −δ_k(−x)`:

```python
dilated = F.max_pool2d(x, k, stride=1, padding=k // 2)
eroded  = -F.max_pool2d(-x, k, stride=1, padding=k // 2)
band    = dilated - eroded
```

For a binary mask, `B_k(M)[i] = 1` exactly when `S_k(i)` meets both `M` and its
complement, i.e. `B_k(M)` is the indicator of a band of width `≈ k` centred on
the contour `∂M`. For continuous `p` it is the differentiable analogue, with
magnitude equal to the local dynamic range of the predicted probability.

The objective is a Dice overlap **between the two band maps** rather than
between the region masks:

    L_band(M̂, M) = 1 − ( 2 · Σ_Ω B_k(p) ⊙ B_k(M) + ε ) / ( Σ_Ω B_k(p) + Σ_Ω B_k(M) + ε )

with `ε = 1` for smoothing. The complete per-instance mask loss adds it to the
stock cropped BCE term with weight `λ`:

    L_mask = L_BCE^crop(M̂, M) + λ · L_band(M̂, M)

Settings used throughout: `k = 3`, `λ = 0.5`. Both are fixed for every arm; no
sweep was run on either, so the comparison against the baseline is not
confounded by tuning budget.

## 2. Mechanism

The reason to state the mechanism rather than only the score is that it makes a
falsifiable prediction, and that prediction is what the ablation tests.

**The gradient is sparse and contour-localized.** `∂δ_k/∂x` is nonzero only at
the arg-max position of each pooling window, and `∂ε_k/∂x` only at the arg-min.
In the interior of a confidently predicted region the window is locally
constant, so `δ_k = ε_k`, the band is zero, and no boundary gradient flows. The
term therefore acts *only* on pixels that currently sit on or beside the
predicted edge, and leaves the interior to the BCE term.

**Predicted consequence.** A loss that only sharpens contours should improve
high-IoU matching and leave low-IoU matching alone or slightly worse: masks that
already cleared a loose overlap criterion gain nothing, while marginal
detections can be lost as the contour tightens. So the signature to look for is
**AP75 up, AP50 flat or down** — not a uniform lift.

**Observed** (50 epochs, validation, mask metrics, pycocotools).

The comparison that isolates the band term is **`S2` vs `S1c`**: both arms use
inverse-sqrt class weighting and differ *only* in `boundary_weight`
(0.5 vs 0). Comparing `S2` against the `S0` reference instead would fold the
class-weighting change into the boundary number and attribute both to the loss.

| metric | `S1c` (invsqrt, bw=0) | `S2` (invsqrt, bw=0.5) | change |
|---|---|---|---|
| AP75 | 0.0659 | 0.0730 | **+0.70 pp, +10.7 % relative** |
| AP50 | 0.2591 | 0.2467 | −1.23 pp, −4.8 % relative |
| head AP | 0.2836 | 0.2921 | **+0.85 pp, +3.0 % relative** |
| mAP | 0.1065 | 0.1050 | −0.15 pp, −1.4 % relative |
| tail AP | 0.0117 | 0.0076 | −0.41 pp (not interpretable, see below) |

The predicted signature is present and is *sharper* under the correct
isolation than under the `S2 − S0` contrast: AP75 up 10.7 %, AP50 down 4.8 %.
The gain is concentrated exactly where a tighter contour should pay and the
AP50 cost is the expected price, which is evidence for the stated mechanism
rather than a bare number.

The tail column is reported for completeness only. Eleven of fifteen tail
classes score exactly 0 on this split and sixteen of thirty-one classes occur
in fewer than ten validation images, so a tail delta of ±0.4 pp is one or two
detections changing and carries no information about the objective.

For reference, the contrast previously recorded in the working notes,
`S2 − S0`, gives AP75 +0.69 pp / +10.4 % and head +1.05 pp / +3.7 %. The
headline is materially unchanged, but the head figure was overstated by
0.2 pp because it included the class-weighting effect; **+3.0 % is the
correct isolated head improvement**. Isolated separately, inverse-sqrt
weighting alone (`S1c − S0`) moves AP75 by −0.01 pp — it does essentially
nothing, for the mechanical reason given in the ablation findings.

**Interaction with the prototype basis.** In YOLOv8-seg the instance mask is
reconstructed as a linear combination of shared prototypes,
`M̂ = Σ_n c_n · P_n`. Because `∂L_band/∂M̂` is supported on edge pixels, the
term back-propagates into `P` only through edge pixels, so it reshapes the
*shared prototype basis* toward edge-localizing components rather than only
adjusting per-instance coefficients. The effect is therefore not confined to the
instance that produced the gradient.

## 3. Position relative to existing losses

Stated plainly, because the distinction decides whether this is a methodological
contribution or an experimental result.

| prior work | mechanism | relation |
|---|---|---|
| **Boundary loss** (Kervadec et al., MIDL 2018; MedIA 2021) | region integral weighted by a **precomputed distance map** of the GT contour, `∫_Ω φ_G(q) s_θ(q) dq` | Different object. Needs an offline distance transform per mask and carries distance semantics. Band-Dice has neither. |
| **Hausdorff-distance loss** (Karimi & Salcudean, TMI 2019) | directly estimates HD, one variant approximating distance by iterated **erosions** | Nearest in spirit on the morphology side, but the quantity minimized is a distance estimate, not an overlap of edge sets. |
| **Active-contour loss** (Chen et al., CVPR 2019) | Mumford–Shah length + region terms, length from prediction gradients | Different functional; a length penalty, not a matching of two contours. |
| **Boundary IoU** (Cheng et al., CVPR 2021) | IoU restricted to pixels within `d` of the contour | **Closest relative, but it is an evaluation metric, not a loss.** Band-Dice is close to a differentiable soft-Dice analogue of it. |
| **Edge-Dice / boundary-aware Dice variants** | Dice on Sobel or Laplacian edge maps | Same family. Substituting a morphological gradient for a linear edge filter is a change of edge operator, not of objective class. |

### Honest assessment of novelty

The construction is a **synthesis of known components**: a morphological
gradient as a differentiable edge extractor, and soft Dice as an overlap
objective. It is not an existing named loss applied unchanged, and it is not a
new mathematical object either. The claims that survive scrutiny are narrow and
specific:

1. **No auxiliary computation.** Unlike the distance-map family it needs no
   offline distance transform, no per-mask preprocessing, and no extra memory
   beyond two pooling buffers. Cost is two max-pools per instance.
2. **A single interpretable tolerance.** The band width `k` *is* the tolerance,
   in pixels, and is the only hyperparameter. Distance-map methods bury the
   equivalent choice in the weighting function.
3. **Applied inside a prototype-coefficient mask head**, where the term shapes a
   shared basis rather than a per-pixel output. We are not aware of this
   combination being reported, and section 2 gives the mechanism by which it
   differs from applying an edge loss to a dense per-pixel decoder.
4. **A mechanism with a pre-registered, confirmed signature** (AP75 up, AP50
   down), plus a documented negative result for copy-paste augmentation under
   the same protocol.

What will **not** survive review is presenting the loss form itself as new
without the positioning above. On the current evidence this is best framed as a
targeted objective plus a mechanism study, not as a new loss family. Making the
mathematical contribution unambiguous would require going further — an
instance-scale-adaptive or curvature-weighted band, where the band width is a
function of local contour geometry rather than a constant — which is a defined
next step, not a claim we make today.

## 4. Reproduction

```
python yolov8_seg_longtail/train_seg.py \
    --data data_clean/data.yaml --model yolov8x-seg.pt \
    --epochs 50 --imgsz 640 --batch 8 --seed 42 \
    --weights invsqrt --boundary-weight 0.5
```

`--boundary-weight 0` recovers the stock objective exactly; the band term is the
only difference between the two arms.
