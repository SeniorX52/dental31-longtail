"""Long-tail classification losses for DINO-DETR, applied consistently in the
Hungarian matching cost AND the training loss.

Rationale: DINO's focal classification loss is frequency-blind — with a
33,000:4 imbalance the tail logits are dominated by negative gradient from
head-class queries. Two remedies, both drop-in:

  * `logit_adjusted_focal_loss` — subtracts tau * log(prior_c) from each
    class logit before the sigmoid focal loss (Menon et al. 2021, "Long-Tail
    Learning via Logit Adjustment", adapted to sigmoid/multi-label form).
    Cheap, one hyperparameter, our default.

  * `SeesawBCE` — per-class-pair gradient rescaling (Wang et al. 2021,
    "Seesaw Loss for Long-Tailed Instance Segmentation") in a sigmoid
    formulation compatible with DETR-style one-vs-all classification.

CRITICAL consistency point: DETR-family models select which query is
supervised for which GT via the matching cost. If the loss is
prior-adjusted but the cost is not, matching keeps assigning tail GTs to
queries whose unadjusted scores look good, and the loss adjustment fights
the matcher. `logit_adjusted_cost_class` mirrors the adjustment inside the
matcher. Ablate them jointly and separately (see INTEGRATION.md).

Class priors come from `class_priors_from_coco`, computed on the TRAIN json
only — never from valid/test.
"""
import json
from typing import Optional

import torch
import torch.nn.functional as F


def class_priors_from_coco(coco_json_path: str, num_classes: int) -> torch.Tensor:
    """Instance-count priors p(c), aligned to 0-based contiguous class ids.

    Assumes category_id 1..num_classes map to classes 0..num_classes-1 (the
    layout our converter emits). Classes absent from train get the minimum
    observed prior rather than 0, so log-priors stay finite.
    """
    with open(coco_json_path) as f:
        coco = json.load(f)
    counts = torch.zeros(num_classes)
    for ann in coco["annotations"]:
        cid = ann["category_id"] - 1
        if 0 <= cid < num_classes:
            counts[cid] += 1
    nonzero_min = counts[counts > 0].min() if (counts > 0).any() else torch.tensor(1.0)
    counts = torch.where(counts > 0, counts, nonzero_min)
    return counts / counts.sum()


def adjust_logits(logits: torch.Tensor, log_priors: torch.Tensor,
                  tau: float = 1.0) -> torch.Tensor:
    """logits_c - tau * log p(c): rare classes get a head start at the sigmoid."""
    return logits - tau * log_priors.to(logits.device, logits.dtype)


def logit_adjusted_focal_loss(logits: torch.Tensor,
                              targets_onehot: torch.Tensor,
                              log_priors: torch.Tensor,
                              tau: float = 1.0,
                              alpha: float = 0.25,
                              gamma: float = 2.0,
                              reduction: str = "none") -> torch.Tensor:
    """Sigmoid focal loss on prior-adjusted logits.

    Same signature/semantics as torchvision's sigmoid_focal_loss, so it can
    replace DINO's `sigmoid_focal_loss` in models/dino/dino.py:loss_labels.
    """
    logits = adjust_logits(logits, log_priors, tau)
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets_onehot, reduction="none")
    p_t = p * targets_onehot + (1 - p) * (1 - targets_onehot)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets_onehot + (1 - alpha) * (1 - targets_onehot)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def logit_adjusted_cost_class(out_prob_logits: torch.Tensor,
                              tgt_ids: torch.Tensor,
                              log_priors: torch.Tensor,
                              tau: float = 1.0,
                              alpha: float = 0.25,
                              gamma: float = 2.0) -> torch.Tensor:
    """Focal-style classification cost on adjusted logits, for HungarianMatcher.

    Mirrors DINO's matcher cost (models/dino/matcher.py): returns a
    [num_queries, num_targets] cost matrix slice given flat query logits
    [num_queries, num_classes] and target class ids [num_targets].
    """
    logits = adjust_logits(out_prob_logits, log_priors, tau)
    out_prob = logits.sigmoid()
    neg_cost = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
    pos_cost = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
    return pos_cost[:, tgt_ids] - neg_cost[:, tgt_ids]


class SeesawBCE(torch.nn.Module):
    """Sigmoid-form Seesaw loss with mitigation + compensation factors.

    Maintains running instance counts per class (as in the paper) so the
    mitigation factor M_ij = min(1, (N_j / N_i)^p) down-weights negative
    gradient flowing from head class i onto tail class j. The compensation
    factor boosts misclassified positives by (p_neg / p_pos)^q.
    """

    def __init__(self, num_classes: int, p: float = 0.8, q: float = 2.0):
        super().__init__()
        self.p = p
        self.q = q
        self.register_buffer("cum_counts", torch.ones(num_classes))

    @torch.no_grad()
    def _update_counts(self, targets_onehot: torch.Tensor) -> None:
        self.cum_counts += targets_onehot.sum(dim=tuple(range(targets_onehot.dim() - 1)))

    def forward(self, logits: torch.Tensor, targets_onehot: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update_counts(targets_onehot)
        n = self.cum_counts
        # mitigation: [C, C] with M[i, j] = min(1, (n_j / n_i)^p)
        ratio = (n.unsqueeze(0) / n.unsqueeze(1)).clamp(max=1.0) ** self.p
        # For each sample, a negative class j is down-weighted by the max
        # mitigation w.r.t. its positive classes i (queries matched to a GT).
        pos_mask = targets_onehot > 0.5
        has_pos = pos_mask.any(dim=-1, keepdim=True)
        probs = logits.sigmoid()

        # mitigation on negatives: for a sample whose positive is class i,
        # negative class j is weighted by M[i, j]; with multiple positives,
        # take the mean over them (matched queries have exactly one positive).
        weights = torch.ones_like(logits)
        if pos_mask.any():
            pos_norm = pos_mask.float() / pos_mask.float().sum(-1, keepdim=True).clamp(min=1)
            sample_mit = pos_norm @ ratio  # [..., C]
            weights = torch.where(has_pos & ~pos_mask, sample_mit, weights)

        # compensation on negatives: if a negative class outscores the
        # sample's positive, scale its penalty by (p_neg / p_pos)^q (>= 1).
        comp = torch.ones_like(logits)
        if pos_mask.any():
            pos_prob = (probs * pos_mask.float()).sum(-1, keepdim=True) / \
                pos_mask.float().sum(-1, keepdim=True).clamp(min=1)
            rel = (probs / pos_prob.clamp(min=1e-8)).clamp(min=1.0) ** self.q
            comp = torch.where(has_pos & ~pos_mask, rel, comp)

        ce = F.binary_cross_entropy_with_logits(logits, targets_onehot, reduction="none")
        return ce * weights * comp
