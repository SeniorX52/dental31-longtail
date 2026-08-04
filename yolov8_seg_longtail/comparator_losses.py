"""Published segmentation losses, as fair comparators for the band-Dice term.

Beating stock BCE is a necessary control, not a sufficient claim. A reviewer
asks whether the proposed objective beats the losses people already use, so
this module implements the standard region-based and boundary-aware families
behind one interface, to be swapped into the SAME per-instance mask loss slot
that the band term occupies.

Every loss here takes `(pred_prob, gt_mask)` already cropped to the instance
box and returns a per-instance vector, so the only thing that differs between
arms is the auxiliary term. The BCE base, the crop, the normalisation and the
schedule are identical, which is what makes the comparison attributable.

  dice            soft Dice (Milletari et al., 3DV 2016) -- the canonical
                  region-based loss any boundary method must beat
  tversky         Tversky index (Salehi et al., MLMI 2017), alpha/beta trade
                  false negatives against false positives; alpha=beta=0.5
                  recovers Dice
  focal_tversky   Focal Tversky (Abraham & Khan, ISBI 2019), (1-TI)^gamma,
                  focuses on hard, usually small, structures
  kervadec        Boundary loss (Kervadec et al., MIDL 2018): integral of the
                  prediction against the GT signed distance map
  band            our band-Dice term, for reference (see BOUNDARY_OBJECTIVE.md)

DISTANCE MAP FOR THE KERVADEC LOSS
The published formulation precomputes an exact Euclidean signed distance
transform per mask offline. That is not available here: masks arrive at
prototype resolution inside the training loop, augmented differently every
epoch, so an exact transform would mean a CPU round-trip per instance per step.

Instead the signed distance is approximated on GPU by iterated morphological
erosion and dilation, each a stride-1 max-pool:

    d_out(x) = #{k in 1..K : x not covered by dilate^k(G)}
    d_in(x)  = #{k in 1..K : x still inside erode^k(G)}
    phi      = d_out - d_in          (positive outside G, negative inside)

This is the standard chamfer-style staircase approximation, exact to +-1 pixel
for the 3x3 structuring element, truncated at K pixels. Truncation is the
honest limitation: gradients saturate beyond K px from the contour, so K is
reported alongside any result produced with this loss. The GT is constant with
respect to the prediction, so phi carries no gradient and the loss stays linear
in the prediction exactly as published.
"""
from typing import Optional

import torch
import torch.nn.functional as F

_EPS = 1.0


def _dilate(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)


def _erode(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    return -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k // 2)


def soft_dice(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """1 - Dice, per instance. pred in [0,1], both (N, H, W)."""
    inter = (pred * gt).sum(dim=(1, 2))
    denom = pred.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))
    return 1.0 - (2 * inter + _EPS) / (denom + _EPS)


def tversky(pred: torch.Tensor, gt: torch.Tensor,
            alpha: float = 0.7, beta: float = 0.3) -> torch.Tensor:
    """1 - Tversky index. alpha weights false negatives, beta false positives.

    alpha > beta penalises misses harder than false alarms, which is the
    setting normally used for small or under-segmented structures. alpha =
    beta = 0.5 reduces exactly to Dice.
    """
    tp = (pred * gt).sum(dim=(1, 2))
    fn = ((1 - pred) * gt).sum(dim=(1, 2))
    fp = (pred * (1 - gt)).sum(dim=(1, 2))
    return 1.0 - (tp + _EPS) / (tp + alpha * fn + beta * fp + _EPS)


def focal_tversky(pred: torch.Tensor, gt: torch.Tensor, alpha: float = 0.7,
                  beta: float = 0.3, gamma: float = 1.33) -> torch.Tensor:
    return tversky(pred, gt, alpha, beta) ** gamma


def signed_distance(gt: torch.Tensor, k: int = 21) -> torch.Tensor:
    """Truncated signed distance to the GT contour, in pixels.

    Positive outside the mask, negative inside, zero on the contour, saturating
    at +-k. Computed with iterated 3x3 morphology so it runs on GPU inside the
    training loop; see the module docstring for why the exact transform is not
    used. Carries no gradient -- the ground truth is a constant.
    """
    with torch.no_grad():
        g = gt.unsqueeze(1)
        out = torch.zeros_like(g)
        cur = g
        for _ in range(k):                       # distance outside
            cur = _dilate(cur)
            out += (1.0 - cur)
        inn = torch.zeros_like(g)
        cur = g
        for _ in range(k):                       # distance inside
            cur = _erode(cur)
            inn += cur
        return (out - inn).squeeze(1)


def kervadec_boundary(pred: torch.Tensor, gt: torch.Tensor,
                      k: int = 21) -> torch.Tensor:
    """Boundary loss: mean of phi_G * prediction, per instance.

    Pixels outside the mask carry positive phi, so predicting there is
    penalised in proportion to how far outside they are; pixels inside carry
    negative phi and reward correct filling. Normalised by area so it composes
    with the BCE term the same way the other comparators do.
    """
    phi = signed_distance(gt, k=k)
    return (phi * pred).mean(dim=(1, 2))


def band(pred: torch.Tensor, gt: torch.Tensor, width: int = 3) -> torch.Tensor:
    """Our band-Dice term: Dice between the two morphological-gradient bands."""
    pb = _dilate(pred.unsqueeze(1), width) - _erode(pred.unsqueeze(1), width)
    gb = _dilate(gt.unsqueeze(1), width) - _erode(gt.unsqueeze(1), width)
    pb, gb = pb.squeeze(1), gb.squeeze(1)
    inter = (pb * gb).sum(dim=(1, 2))
    denom = pb.sum(dim=(1, 2)) + gb.sum(dim=(1, 2))
    return 1.0 - (2 * inter + _EPS) / (denom + _EPS)


_REGISTRY = {
    "none": None,
    "band": band,
    "dice": soft_dice,
    "tversky": tversky,
    "focal_tversky": focal_tversky,
    "kervadec": kervadec_boundary,
}


def get_aux_loss(name: str):
    """-> callable(pred, gt) -> (N,) per-instance loss, or None."""
    key = (name or "none").strip().lower()
    if key not in _REGISTRY:
        raise ValueError("unknown mask-aux loss %r; choose from %s"
                         % (name, sorted(_REGISTRY)))
    return _REGISTRY[key]


def describe(name: str) -> str:
    return {
        "none": "stock BCE mask loss only",
        "band": "BCE + band-Dice on morphological-gradient maps (ours)",
        "dice": "BCE + soft Dice (Milletari et al. 2016)",
        "tversky": "BCE + Tversky alpha=0.7 beta=0.3 (Salehi et al. 2017)",
        "focal_tversky": "BCE + Focal Tversky gamma=1.33 (Abraham & Khan 2019)",
        "kervadec": "BCE + Boundary loss, truncated SDF K=21 (Kervadec et al. 2018)",
    }.get((name or "none").strip().lower(), name)
