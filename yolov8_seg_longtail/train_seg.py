#!/usr/bin/env python3
"""Long-tail + boundary-aware training for YOLOv8/YOLO26-seg (ultralytics 8.4.x).

Two changes over the stock `yolo segment train` pipeline, each independently
switchable so the ablation isolates them:

1. Class-balanced classification loss (`--weights`): per-class weights from
   the TRAIN label files, injected via the `model.class_weights` hook that
   `v8DetectionLoss` already honors (bce_loss *= class_weights). Choose
   'invsqrt' or an effective-number beta (Cui et al., CVPR 2019).
   Default beta is 0.99, NOT the customary 0.999: measured on this dataset
   (34,320:1 imbalance) the tail weight saturates near 2.0 past 0.99 while the
   head weight collapses to 0.009 and then 0.001, so a larger beta helps the
   tail not at all and merely erases the head. See ABLATION_PLAN.md.

2. Boundary-aware mask loss (`--boundary-weight`): adds a soft-boundary Dice
   term to the per-instance mask BCE in `single_mask_loss`. Boundary bands
   are extracted differentiably with a max-pool morphological gradient
   (dilate(x) - x). Motivation: mask mAP on thin/elongated dental structures
   (Mandibular Canal, Bone Loss) is dominated by boundary error, which plain
   BCE under-penalizes.

Verified against ultralytics 8.4.104 internals (single_mask_loss signature,
class_weights hook, SegmentationTrainer.get_model). Pin: ultralytics>=8.4,<8.5.

Usage:
    python yolov8_seg_longtail/train_seg.py --data data/data.yaml \
        --model yolov8x-seg.pt --epochs 50 --imgsz 640 --seed 42 \
        --weights 0.99 --boundary-weight 0.5 --name lt_boundary
"""
import argparse
import glob
import os
from typing import List, Optional

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel

from yolov8_seg_longtail import comparator_losses
from ultralytics.utils import RANK
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.ops import crop_mask
from ultralytics.nn.modules import Conv


def class_counts_from_yolo_labels(labels_dir: str, nc: int) -> torch.Tensor:
    counts = torch.zeros(nc)
    for path in glob.glob(os.path.join(labels_dir, "*.txt")):
        with open(path) as f:
            for ln in f:
                parts = ln.split()
                if parts:
                    cid = int(parts[0])
                    if 0 <= cid < nc:
                        counts[cid] += 1
    return counts


def effective_number_weights(counts: torch.Tensor, beta: float = 0.999) -> torch.Tensor:
    """w_c = (1 - beta) / (1 - beta^n_c), normalized to mean 1.

    Classes with zero training instances get the max weight among seen
    classes (they can only appear via later augmentation).

    NOTE on choosing beta: the customary 0.999 comes from CIFAR-LT, whose
    imbalance is ~100:1. Measured on this dataset (34,320:1), the tail weight
    saturates near 2.0 beyond beta=0.99 while the head weight keeps collapsing
    (0.083 -> 0.009 -> 0.001). Raising beta past 0.99 therefore buys the tail
    nothing and merely deletes the head classes from the loss. See
    ABLATION_PLAN.md for the measured table.
    """
    seen = counts > 0
    eff = torch.ones_like(counts)
    eff[seen] = (1.0 - beta) / (1.0 - beta ** counts[seen])
    if seen.any():
        eff[~seen] = eff[seen].max()
    return eff / eff.mean()


def inverse_sqrt_weights(counts: torch.Tensor) -> torch.Tensor:
    """w_c proportional to 1/sqrt(n_c), normalized to mean 1.

    Included as a distinct arm because it is far gentler on the MID-frequency
    classes than effective-number weighting at comparable tail strength
    (mid 0.220 vs 0.092 for beta=0.99, with tail ~1.9 in both). Most of the
    recoverable headroom on this dataset sits in the mid group, so the softer
    treatment may win outright.
    """
    inv = counts.clamp(min=1).pow(-0.5)
    return inv / inv.mean()


def build_class_weights(counts: torch.Tensor, scheme: str) -> Optional[torch.Tensor]:
    """scheme: 'none' | 'invsqrt' | a float beta for effective-number."""
    if scheme in ("none", "off", ""):
        return None
    if scheme == "invsqrt":
        return inverse_sqrt_weights(counts)
    return effective_number_weights(counts, beta=float(scheme))


def soft_boundary(x: torch.Tensor, band: int = 3) -> torch.Tensor:
    """Differentiable boundary band: morphological gradient via max-pool.

    x: (N, H, W) in [0, 1]. Returns (N, H, W) highlighting a ~band-px edge.
    """
    x4 = x.unsqueeze(1)
    dilated = F.max_pool2d(x4, kernel_size=band, stride=1, padding=band // 2)
    eroded = -F.max_pool2d(-x4, kernel_size=band, stride=1, padding=band // 2)
    return (dilated - eroded).squeeze(1)


class _GatedBCE(torch.nn.Module):
    """BCE that refuses to punish confident detections in unannotated regions.

    THE PROBLEM IT TREATS. Consensus mining found 911 validation annotations
    (43 % of periapical lesion, 33 % of bone loss, 26 % of caries) that no
    model in a fifteen-model zoo can detect, and the model that raised caries
    by 35 % recovers 2.7 % of them: the pathology labels are unreliable. An
    unreliable label set is unreliable in BOTH directions, and the unannotated
    real finding is the direction that poisons training: every anchor on it is
    supervised toward background, so the model is actively taught to suppress
    true lesions.

    THE MECHANISM, AND ITS SOURCES. For anchors that (a) received no
    assignment, (b) do NOT lie inside any annotated ground-truth box, and
    (c) predict some class above a confidence threshold, the classification
    loss is zeroed instead of pushing the score to zero. This is the shared
    core of three published treatments of missing-annotation detection:

      * Background Recalibration Loss (Zhang et al., ICASSP 2020,
        arXiv:2002.05274) restricts recalibration to "confusion" anchors with
        IoU < 0.1 to any ground truth and switches the loss branch at an
        activation threshold t = 0.5, which is where our tau default comes
        from. BRL goes further than we do, mirroring the positive branch
        (actively encouraging the prediction); we only STOP suppressing,
        which is the conservative half of the same move.
      * Soft Sampling (Wu et al., arXiv:1806.06986) down-weights the gradient
        of negatives as a function of overlap with annotated positives, on
        the observation that negatives NEAR annotated boxes are reliable
        negatives while far-from-annotation background is uncertain. That is
        why condition (b) exempts anchors inside annotated boxes: those
        negatives stay fully supervised.
      * SparseDet (Suri et al., arXiv:2201.04620) separates proposals into
        labeled and unlabeled regions and mines pseudo-positives from the
        unlabeled ones rather than treating them as background.

    Deviation from BRL, stated: BRL is formulated on focal loss over
    per-anchor-per-class terms; YOLOv8 supervises BCE over task-aligned
    assignments. We gate per ANCHOR (an anchor whose best class exceeds tau is
    ignored for all classes) rather than per class, and we ignore rather than
    flip the branch. Both choices are the cautious end of the published range.
    """

    def __init__(self, owner):
        super().__init__()
        self._owner = owner
        self._inner = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, target):
        loss = self._inner(pred, target)
        m = getattr(self._owner, "_bg_gate_mask", None)
        if m is not None and m.shape == pred.shape[:2]:
            loss = loss * (~m).unsqueeze(-1).to(loss.dtype)
            self._owner._bg_gate_mask = None
        return loss


def _install_bg_gate(crit, tau):
    """Wrap the assigner so every step records which negative anchors are
    confident detections OUTSIDE all annotated boxes, then swap the BCE for
    the gated one. The assigner is called with pixel-space anchor points and
    ground-truth boxes and with already-sigmoided scores (see
    v8DetectionLoss.__call__), so everything the gate needs passes through it.
    """
    assigner = crit.assigner
    orig_forward = assigner.forward

    def recording_forward(pd_scores, pd_bboxes, anc_points, gt_labels,
                          gt_bboxes, mask_gt, *a, **k):
        out = orig_forward(pd_scores, pd_bboxes, anc_points, gt_labels,
                           gt_bboxes, mask_gt, *a, **k)
        try:
            fg_mask = out[3].bool()                       # (b, A)
            ax = anc_points[:, 0].view(1, -1, 1)          # (1, A, 1)
            ay = anc_points[:, 1].view(1, -1, 1)
            x1 = gt_bboxes[..., 0].unsqueeze(1)           # (b, 1, n)
            y1 = gt_bboxes[..., 1].unsqueeze(1)
            x2 = gt_bboxes[..., 2].unsqueeze(1)
            y2 = gt_bboxes[..., 3].unsqueeze(1)
            inside = ((ax >= x1) & (ax <= x2) & (ay >= y1) & (ay <= y2))
            valid = mask_gt.view(mask_gt.shape[0], 1, -1).bool()
            inside_any = (inside & valid).any(-1)         # (b, A)
            confident = pd_scores.amax(-1) > tau          # scores arrive sigmoided
            crit._bg_gate_mask = (~fg_mask) & (~inside_any) & confident
        except Exception:
            crit._bg_gate_mask = None                     # never break training
        return out

    assigner.forward = recording_forward
    crit.bce = _GatedBCE(crit)


class BoundaryAwareSegLoss(v8SegmentationLoss):
    """v8SegmentationLoss with a selectable auxiliary term on the mask loss.

    Reads `model.boundary_weight` (float, 0 disables) and `model.mask_aux`
    (which auxiliary loss to add; default 'band', our objective). The parent
    class reads `model.class_weights` for the cls loss on its own.

    The auxiliary slot exists so the proposed objective can be measured against
    the losses people already publish -- soft Dice, Tversky, Focal Tversky, and
    Kervadec's boundary loss -- with the BCE base, the crop, the normalisation
    and the schedule held identical, so the auxiliary term is the only thing
    that differs between arms. See `comparator_losses.py`.
    """

    def __init__(self, model, tal_topk: int = 10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        self.boundary_weight = float(getattr(model, "boundary_weight", 0.0))
        self.mask_aux = str(getattr(model, "mask_aux", "band") or "band")
        self.coeff_weight = float(getattr(model, "coeff_weight", 0.0))
        # ridge is now RELATIVE to trace(A)/nm, so 1e-2 means "the smallest
        # eigenvalue is at least a hundredth of the mean one"
        self.coeff_ridge = float(getattr(model, "coeff_ridge", 1e-2))
        self.coeff_max_norm = float(getattr(model, "coeff_max_norm", 50.0))
        self.coeff_clip = float(getattr(model, "coeff_clip", 10.0))
        self.bg_gate = float(getattr(model, "bg_gate", 0.0))
        if self.bg_gate > 0:
            _install_bg_gate(self, self.bg_gate)

    def oracle_coefficients(self, gt_mask, proto, boxes):
        """Closed-form best coefficients for these masks on these prototypes.

        The mask deficit on this corpus was attributed by
        `tools/oracle_coefficients.py`: of the 0.1994 Dice gap between what the
        model achieves and what its prototype grid allows, 0.0304 is basis
        expressiveness and **0.1690 -- 85 percent -- is the coefficient head
        failing to locate the right point in a basis that already spans the
        shapes**.

        Every loss tried before this one supervises in PIXEL space, so the
        gradient reaching the coefficients has to travel back through the
        prototype product. This supervises the coefficients directly, against a
        teacher that is exact and free:

            c* = (P P^T + lam I)^-1 P y

        with y the ground truth mapped to {-1, +1}. The prototypes are detached,
        so this term trains the coefficient head only and cannot degenerate by
        moving the basis to meet the prediction.

        One 32x32 solve is shared across every instance in the image, because
        they share the prototype stack; only the right-hand side differs.
        """
        with torch.no_grad():
            nm, h, w = proto.shape
            P = proto.detach().float().reshape(nm, -1)                # (32, HW)
            y = 2.0 * gt_mask.float().reshape(gt_mask.shape[0], -1) - 1.0
            n = y.shape[0]

            # Box restriction, and it is NOT an optimisation. Solved over the
            # whole image the objective is dominated by background: an instance
            # covers a percent or two of the pixels, so "predict -1 everywhere"
            # beats any reconstruction and c* collapses. Measured on the trained
            # model, the unrestricted solve reaches Dice 0.0008 against 0.96 for
            # the box-restricted one. The box is also what the mask loss itself
            # scores, since ultralytics crops the predicted mask to the same box.
            ys = torch.arange(h, device=P.device, dtype=torch.float32)
            xs = torch.arange(w, device=P.device, dtype=torch.float32)
            bx = boxes.float()
            # same convention as ultralytics crop_mask: inclusive low, exclusive
            # high, so the solve covers exactly the pixels the BCE is scored on
            inx = (xs[None, :] >= bx[:, 0:1]) & (xs[None, :] < bx[:, 2:3])
            iny = (ys[None, :] >= bx[:, 1:2]) & (ys[None, :] < bx[:, 3:4])
            B = (iny[:, :, None] & inx[:, None, :]).reshape(n, -1).float()

            eye = torch.eye(nm, device=P.device)
            out = []
            for i in range(0, n, 8):          # chunked: (k, 32, HW) is the peak
                b = B[i:i + 8]
                PB = P[None] * b[:, None]                             # (k,32,HW)
                A = PB @ P.T                                          # (k,32,32)
                r = (PB * y[i:i + 8, None]).sum(-1)                   # (k,32)

                # RELATIVE ridge, and this is the difference between a target
                # and a detonation. A = P B P^T is formed from learned features
                # restricted to one box, so it is severely ill-conditioned:
                # measured on COCO-pretrained prototypes over real dental
                # instances its eigenvalues span 1e-6 to 1e3, and rounding puts
                # the smallest slightly NEGATIVE. A fixed ridge of 1e-3 against
                # that is not regularisation, it is a coin flip on whether the
                # instance is singular, and a single singular instance out of
                # 95,745 annotations sends c* to infinity. The first version of
                # this used a fixed ridge and drove seg_loss to 7e7 with NaN by
                # epoch 10. Scaling the ridge by the matrix's own trace bounds
                # the condition number by construction, whatever the prototype
                # scale or box size.
                tr = A.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12)
                lam = self.coeff_ridge * (tr / nm)                    # (k,)
                A = A + lam[:, None, None] * eye
                try:
                    c = torch.linalg.solve(A, r[..., None])[..., 0]
                except Exception:
                    c = torch.linalg.lstsq(A, r[..., None]).solution[..., 0]

                # Belt and braces: a target that is not finite, or absurdly
                # large, teaches nothing and destroys the run. Drop it to zero
                # weight rather than propagate it.
                c = torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
                nrm = c.norm(dim=1, keepdim=True).clamp_min(1e-12)
                c = c * (nrm.clamp_max(self.coeff_max_norm) / nrm)
                out.append(c)
            return torch.cat(out).to(gt_mask.dtype)

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
        if self.coeff_weight > 0 and gt_mask.numel():
            c_star = self.oracle_coefficients(gt_mask, proto, xyxy)
            # RELATIVE error, not raw MSE. The magnitude of c* depends on the
            # prototype scale and the box area, neither of which is fixed during
            # training, so a raw MSE makes the weight mean something different
            # at every step. Dividing by the target's own energy makes the term
            # dimensionless and O(1): 1.0 means "as wrong as predicting zero".
            num = (pred.float() - c_star.float()).pow(2).mean(dim=1)
            den = c_star.float().pow(2).mean(dim=1) + 1e-6
            coeff_loss = (num / den).clamp_max(self.coeff_clip).sum()
        bce = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        loss = (crop_mask(bce, xyxy).mean(dim=(1, 2)) / area).sum()
        if self.boundary_weight > 0 and self.mask_aux != "none":
            p = pred_mask.sigmoid()
            if self.mask_aux == "band":
                # Left exactly as originally written -- morphology on the FULL
                # map, then crop -- so every arm trained before this refactor
                # still reproduces bit-for-bit. Cropping first would let the
                # crop edge register as contour and change the numbers.
                pb = crop_mask(soft_boundary(p), xyxy)
                gb = crop_mask(soft_boundary(gt_mask), xyxy)
                inter = (pb * gb).sum(dim=(1, 2))
                denom = pb.sum(dim=(1, 2)) + gb.sum(dim=(1, 2))
                aux = 1.0 - (2 * inter + 1.0) / (denom + 1.0)
            elif self.mask_aux == "kervadec":
                # the distance map is a property of the ground truth, so it is
                # built on the full map for the same reason, then cropped
                phi = crop_mask(comparator_losses.signed_distance(gt_mask), xyxy)
                aux = (phi * crop_mask(p, xyxy)).mean(dim=(1, 2))
            else:
                fn = comparator_losses.get_aux_loss(self.mask_aux)
                aux = fn(crop_mask(p, xyxy), crop_mask(gt_mask, xyxy))
            loss = loss + self.boundary_weight * aux.sum()
        if self.coeff_weight > 0 and gt_mask.numel():
            loss = loss + self.coeff_weight * coeff_loss
        return loss


class HighResProto(torch.nn.Module):
    """Prototype head at input/2 instead of the stock input/4.

    Motivation is measured, not assumed. YOLOv8-seg reconstructs every instance
    mask as a linear combination of 32 prototype maps produced at input/4 -- a
    160x160 grid at imgsz 640. On this dataset the median annotated instance is
    ~24 px across, which is 6 px on that grid, and 68 % of instances are under
    8 px. Round-tripping the ground truth through the grid
    (`tools/mask_resolution_ceiling.py`) shows 17.7 % of instances cannot reach
    IoU 0.75 no matter how good the model is, with root canal treatment capped
    at 0.622 and caries at 0.721 -- both below the threshold. AP75 is bounded by
    the representation before training begins.

    Doubling the grid cuts that structurally impossible share to 5.4 % and
    raises the achievable Dice ceiling from 0.896 to 0.949.

    The extra stage is sub-pixel convolution (PixelShuffle, Shi et al. CVPR
    2016) rather than another transposed convolution: it rearranges the 256
    channels of the existing stage into 64 channels at twice the resolution, so
    the wide tensor never exists at high resolution and the memory cost stays
    near 210 MB at batch 8. A transposed convolution holding 256 channels at
    320x320 would cost roughly four times that.

    cv1, upsample and cv2 are carried over from the pretrained head unchanged.
    Only the final 1x1 projection is rebuilt, because its input width changes
    from 256 to 64; it is 8k parameters and retrains quickly.

    NOTE: must be paired with mask_ratio=2. This ultralytics build resizes the
    *proto* to the ground-truth mask resolution, so leaving the ground truth at
    input/4 would immediately discard the extra resolution.
    """

    def __init__(self, proto: torch.nn.Module):
        super().__init__()
        self.cv1 = proto.cv1
        self.upsample = proto.upsample
        self.cv2 = proto.cv2
        c_ = proto.cv2.conv.out_channels          # 256
        nm = proto.cv3.conv.out_channels          # 32 prototypes
        self.shuffle = torch.nn.PixelShuffle(2)   # 256 @ HxW -> 64 @ 2H x 2W
        self.cv3 = torch.nn.Conv2d(c_ // 4, nm, 1)

    def forward(self, x):
        return self.cv3(self.shuffle(self.cv2(self.upsample(self.cv1(x)))))


class P2Proto(torch.nn.Module):
    """Prototype head fed from P2 (stride 4) instead of P3 (stride 8).

    This supersedes HighResProto, which raised the prototype grid by upsampling
    P3-derived features. That added resolution without adding information: the
    features were already band-limited at stride 8. Feeding the head from P2
    supplies genuine stride-4 detail, which is the established fix for small
    objects in this architecture family -- the YOLO P2-head variants exist for
    exactly this reason, and PointRend, RefineMask and Mask Transfiner all
    address the same coarse-mask failure mode in two-stage detectors.

    Why it matters here specifically. `tools/mask_resolution_ceiling.py` shows
    the stock input/4 grid caps mean Dice at 0.8963 and leaves 17.7 % of
    instances unable to reach IoU 0.75 at all, with root canal treatment at
    0.622 and caries at 0.721 -- both below the threshold. The median instance
    is 6 px on that grid.

    Stock path : cv1(P3 @80) -> 256@80, upsample -> 256@160, cv2 -> 256@160,
                 cv3 -> 32@160
    This path  : cv1(P2 @160) -> 256@160, cv2 -> 256@160, PixelShuffle ->
                 64@320, cv3 -> 32@320

    `cv2` is reused unchanged and still runs at 160x160, the same role it had
    after the stock upsample, so its pretrained weights stay meaningful. `cv1`
    and `cv3` are rebuilt because their channel widths change. The transposed
    upsample is dropped in favour of sub-pixel convolution (Shi et al., CVPR
    2016), which keeps the 256-channel tensor off the 320x320 grid -- a
    ConvTranspose there would cost roughly four times the activation memory.

    Net effect on capacity is NEGATIVE: cv1 shrinks (160 vs 320 input
    channels), cv3 shrinks (64 vs 256 input channels), and the transposed
    convolution disappears. Any gain therefore cannot be attributed to a bigger
    model.
    """

    def __init__(self, proto: torch.nn.Module, p2_ch: int):
        super().__init__()
        c_ = proto.cv2.conv.out_channels      # 256
        nm = proto.cv3.conv.out_channels      # 32 prototypes
        self.cv1 = Conv(p2_ch, c_, k=3)       # rebuilt for P2's channel count
        self.cv2 = proto.cv2                  # pretrained, unchanged role
        self.shuffle = torch.nn.PixelShuffle(2)
        self.cv3 = torch.nn.Conv2d(c_ // 4, nm, 1)
        self._src = None

    def set_source(self, t: torch.Tensor) -> None:
        """Hand the head its P2 feature map for this forward pass."""
        self._src = t

    def forward(self, x):
        src = self._src if self._src is not None else x
        self._src = None                      # never reuse a stale map
        return self.cv3(self.shuffle(self.cv2(self.cv1(src))))


class LongTailSegModel(SegmentationModel):
    def init_criterion(self):
        return BoundaryAwareSegLoss(self)


class P2ProtoSegModel(LongTailSegModel):
    """SegmentationModel that hands P2 to the prototype head.

    The stock forward keeps only the layers listed in `self.save`, and P2
    (layer 2) is not among them, so its output is discarded before the head
    runs. This subclass adds it to `save` and passes it to the head each
    forward. Implementing it here rather than with a forward hook matters:
    the model is pickled into the checkpoint, and hooks do not survive that,
    so a hook-based version would silently fall back to P3 at inference.
    """

    P2_INDEX = 2

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            proto = getattr(m, "proto", None)
            if proto is not None and hasattr(proto, "set_source"):
                proto.set_source(y[self.P2_INDEX])
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x


def widen_coefficient_head(model, width: int) -> None:
    """Widen the mask-coefficient branch `cv4` of the Segment head.

    `cv4` is the branch that turns features into the 32 prototype coefficients,
    and the oracle fit attributes 85 percent of the mask deficit to it. Stock
    ultralytics sizes its hidden layer as `max(ch[0] // 4, nm)`, which on this
    model is 80 channels -- a default chosen for the COCO configuration, never
    tuned for anything. The whole branch holds 1.33 M parameters against 2.26 M
    in the prototype head and 7.41 M in the classifier, so the smallest head in
    the model carries the largest share of the error.

    This is the capacity control for the distillation arm. If widening alone
    closes much of the gap the problem is capacity; if only distillation moves
    it, the problem is supervision. Run together they read as a 2x2.

    The branch is rebuilt, so it starts from scratch while the rest of the head
    keeps whatever it was loaded with.
    """
    head = model.model[-1]
    in_ch = [seq[0].conv.in_channels for seq in head.cv4]
    before = sum(p.numel() for p in head.cv4.parameters())
    dev = next(head.cv4.parameters()).device
    dtype = next(head.cv4.parameters()).dtype
    head.cv4 = torch.nn.ModuleList(
        torch.nn.Sequential(Conv(c, width, 3), Conv(width, width, 3),
                            torch.nn.Conv2d(width, head.nm, 1))
        for c in in_ch).to(device=dev, dtype=dtype)
    after = sum(p.numel() for p in head.cv4.parameters())
    print("coefficient head cv4 widened: hidden %d -> %d channels, "
          "%.2f M -> %.2f M parameters (rebuilt, not loaded)"
          % (max(in_ch[0] // 4, head.nm), width, before / 1e6, after / 1e6))


class LongTailSegTrainer(SegmentationTrainer):
    """SegmentationTrainer that builds LongTailSegModel and attaches the
    class-balance / boundary knobs before the criterion is created."""

    def __init__(self, overrides=None, _callbacks=None,
                 class_weights: Optional[torch.Tensor] = None,
                 boundary_weight: float = 0.0,
                 mask_aux: str = "band",
                 proto_scale: int = 4,
                 proto_src: str = "p3",
                 coeff_weight: float = 0.0,
                 coeff_ridge: float = 1e-2,
                 coeff_width: int = 0,
                 bg_gate: float = 0.0):
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        self._lt_class_weights = class_weights
        self._lt_boundary_weight = boundary_weight
        self._lt_mask_aux = mask_aux
        self._lt_proto_scale = proto_scale
        self._lt_proto_src = proto_src
        self._lt_coeff_weight = coeff_weight
        self._lt_coeff_ridge = coeff_ridge
        self._lt_coeff_width = coeff_width
        self._lt_bg_gate = bg_gate

    def get_model(self, cfg=None, weights=None, verbose=True):
        cls = (P2ProtoSegModel if getattr(self, "_lt_proto_src", "p3") == "p2"
               else LongTailSegModel)
        model = self.set_model_names_for_load(
            cls(cfg, nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1))
        if weights:
            model.load(weights)
        if self._lt_class_weights is not None:
            model.class_weights = self._lt_class_weights
        model.boundary_weight = self._lt_boundary_weight
        model.mask_aux = self._lt_mask_aux
        model.coeff_weight = getattr(self, "_lt_coeff_weight", 0.0)
        model.coeff_ridge = getattr(self, "_lt_coeff_ridge", 1e-2)
        model.bg_gate = getattr(self, "_lt_bg_gate", 0.0)
        if getattr(self, "_lt_coeff_width", 0):
            widen_coefficient_head(model, self._lt_coeff_width)
        if getattr(self, "_lt_proto_src", "p3") == "p2":
            head = model.model[-1]
            p2_ch = model.model[P2ProtoSegModel.P2_INDEX].cv2.conv.out_channels
            head.proto = P2Proto(head.proto, p2_ch)
            # P2 is not in the stock save list, so its output would be dropped
            # before the head runs
            if P2ProtoSegModel.P2_INDEX not in model.save:
                model.save = sorted(set(list(model.save) + [P2ProtoSegModel.P2_INDEX]))
            print("prototype head fed from P2 (stride 4, %d ch) -> protos at "
                  "input/2; cv2 reused, cv1 and cv3 rebuilt" % p2_ch)
        elif getattr(self, "_lt_proto_scale", 4) == 2:
            # swap AFTER load() so the reused stages keep their pretrained weights
            head = model.model[-1]
            head.proto = HighResProto(head.proto)
            print("prototype head: input/2 (%dx%d at imgsz 640) via sub-pixel "
                  "convolution; only the final 1x1 projection is new"
                  % (640 // 2, 640 // 2))
        return model


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data.yaml")
    ap.add_argument("--model", default="yolov8s-seg.pt")
    ap.add_argument("--train-labels", default=None,
                    help="train labels dir for class counts "
                         "(default: <data.yaml dir>/train/labels)")
    ap.add_argument("--nc", type=int, default=31)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", default="0.99",
                    help="class-weighting scheme: 'none', 'invsqrt', or a beta "
                         "value for effective-number weighting (e.g. 0.99). "
                         "Default 0.99 -- see ABLATION_PLAN.md for why not 0.999")
    ap.add_argument("--boundary-weight", type=float, default=0.5,
                    help="0 disables the boundary Dice term")
    ap.add_argument("--name", default="longtail_seg")
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", default=None, metavar="LAST_PT",
                    help="resume an interrupted run from its last.pt. Restores "
                         "optimizer state and epoch counter; continues in the "
                         "original run directory to the original epoch target. "
                         "Class weights / boundary term are re-attached by the "
                         "trainer, so the criterion is identical after resume.")
    ap.add_argument("--proto-src", default="p3", choices=["p3", "p2"],
                    help="which feature level feeds the prototype head. p3 is "
                         "stock (stride 8, protos at input/4). p2 uses the "
                         "stride-4 level, giving protos at input/2 from genuine "
                         "high-resolution features rather than upsampled coarse "
                         "ones. Implies --no-val, because ultralytics' own "
                         "validator assumes the stock grid.")
    ap.add_argument("--no-val", action="store_true",
                    help="skip ultralytics' internal per-epoch validation. Its "
                         "mask_iou compares ground truth and predictions at "
                         "resolutions it derives itself, which breaks when the "
                         "prototype grid changes. Scoring is done afterwards by "
                         "predict_to_coco.py, which does not use that path.")
    ap.add_argument("--proto-scale", type=int, default=4, choices=[2, 4],
                    help="prototype grid as input/N. 4 is stock (160x160 at "
                         "imgsz 640); 2 doubles it to 320x320, which cuts the "
                         "share of instances that cannot reach IoU 0.75 from "
                         "17.7%% to 5.4%% (see tools/mask_resolution_ceiling.py). "
                         "Automatically sets mask_ratio to match.")
    ap.add_argument("--coeff-weight", type=float, default=0.0,
                    help="weight on the coefficient-distillation term. The mask "
                         "deficit is 85 %% coefficient head and 15 %% basis, so "
                         "this supervises the coefficients directly against the "
                         "closed-form optimum on the model's own prototypes "
                         "instead of relying on gradient reaching them through "
                         "the prototype product. 0 disables.")
    ap.add_argument("--coeff-ridge", type=float, default=1e-2,
                    help="ridge in the closed-form solve, RELATIVE to trace(A)/nm. "
                         "A = P B P^T is severely ill-conditioned (eigenvalues "
                         "measured spanning 1e-6 to 1e3), so an absolute ridge "
                         "regularises some instances and not others; a relative "
                         "one bounds the condition number by construction.")
    ap.add_argument("--bg-gate", type=float, default=0.0,
                    help="missing-annotation guard: zero the classification "
                         "loss for unassigned anchors OUTSIDE every annotated "
                         "box whose best class score exceeds this threshold, "
                         "instead of training them toward background. 0.5 "
                         "follows the switch threshold of Background "
                         "Recalibration Loss (arXiv:2002.05274); anchors "
                         "inside annotated boxes stay fully supervised per "
                         "Soft Sampling (arXiv:1806.06986). 0 disables.")
    ap.add_argument("--coeff-width", type=int, default=0,
                    help="rebuild the cv4 coefficient branch with this hidden "
                         "width (stock is 80). Capacity control for "
                         "--coeff-weight. 0 leaves the head alone.")
    ap.add_argument("--mask-aux", default="band",
                    choices=["none", "band", "dice", "tversky",
                             "focal_tversky", "kervadec"],
                    help="which auxiliary term the mask loss adds on top of "
                         "BCE, scaled by --boundary-weight. 'band' is our "
                         "objective; the rest are published comparators so the "
                         "proposed loss is measured against the losses people "
                         "already use, not only against stock BCE.")
    # Accuracy-neutral throughput settings, verified by tools/bench_train_speed.py.
    # They must be identical across every compared run, so the ablation queue
    # passes the same set to all arms.
    ap.add_argument("--cache", default=None,
                    help="'ram' or 'disk' to decode images once instead of every epoch")
    ap.add_argument("--channels-last", action="store_true",
                    help="NHWC memory format (numerically identical, tensor-core friendly)")
    ap.add_argument("--compile", default=None,
                    help="torch.compile mode; safe at fixed imgsz")
    ap.add_argument("--workers", type=int, default=None,
                    help="dataloader workers (default 8; this box has 24 cores)")
    args = ap.parse_args(argv)

    weights = None
    labels_dir = args.train_labels or os.path.join(
        os.path.dirname(os.path.abspath(args.data)), "train", "labels")
    counts = class_counts_from_yolo_labels(labels_dir, args.nc)
    weights = build_class_weights(counts, args.weights)
    print("class-weighting scheme:", args.weights)
    if weights is not None:
        print("class counts:", counts.tolist())
        print("class weights:", [round(w, 3) for w in weights.tolist()])
        print("weight ratio max/min: %.1fx" % float(weights.max() / weights.min()))

    overrides = dict(model=args.model, data=args.data, epochs=args.epochs,
                     imgsz=args.imgsz, batch=args.batch, seed=args.seed,
                     deterministic=True, name=args.name)
    if args.resume:
        # ultralytics restores the interrupted run's own args from the
        # checkpoint; model must point at the last.pt being resumed.
        overrides["model"] = args.resume
        overrides["resume"] = True
    if args.device is not None:
        overrides["device"] = args.device
    if args.cache:
        overrides["cache"] = args.cache
    if args.channels_last:
        overrides["channels_last"] = True
    if args.compile:
        overrides["compile"] = (True if args.compile in ("1", "true", "True")
                                else args.compile)
    if args.workers is not None:
        overrides["workers"] = args.workers
    print("throughput overrides:", {k: overrides[k] for k in
          ("cache", "channels_last", "compile", "workers") if k in overrides})
    print("mask auxiliary term:", comparator_losses.describe(args.mask_aux),
          "(weight %.3f)" % args.boundary_weight)
    if args.proto_scale == 2 or args.proto_src == "p2":
        # This build resizes the PROTO to the ground-truth mask resolution, so
        # leaving mask_ratio at 4 would downsample the new prototypes straight
        # back to 160 and discard the change entirely.
        overrides["mask_ratio"] = 2
        print("mask_ratio forced to 2 to match the input/2 prototype grid")
    if args.proto_src == "p2" or args.no_val:
        overrides["val"] = False
    trainer = LongTailSegTrainer(overrides=overrides,
                                 class_weights=weights,
                                 boundary_weight=args.boundary_weight,
                                 mask_aux=args.mask_aux,
                                 proto_scale=args.proto_scale,
                                 proto_src=args.proto_src,
                                 coeff_weight=args.coeff_weight,
                                 coeff_ridge=args.coeff_ridge,
                                 coeff_width=args.coeff_width,
                                 bg_gate=args.bg_gate)
    trainer.train()


if __name__ == "__main__":
    main()
