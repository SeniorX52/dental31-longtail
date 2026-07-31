#!/usr/bin/env python3
"""Unit tests for the long-tail losses (CPU torch, no GPU needed).

Run:  python tests/test_losses.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dino_longtail.losses import (  # noqa: E402
    adjust_logits, logit_adjusted_focal_loss, logit_adjusted_cost_class,
    SeesawBCE)

torch.manual_seed(0)
C = 4                                       # classes: 0 head ... 3 tail
priors = torch.tensor([0.70, 0.20, 0.09, 0.01])
log_priors = priors.log()


def test_adjust_logits_direction():
    logits = torch.zeros(2, C)
    adj = adjust_logits(logits, log_priors, tau=1.0)
    # -log p is larger for rarer classes -> tail gets the biggest boost
    assert torch.argmax(adj[0]) == 3
    assert adj[0, 3] > adj[0, 0]


def test_focal_favors_tail_positive():
    """Same raw logit, positive label: tail class must incur LESS loss than
    head class (the adjustment has already 'paid' part of its margin)."""
    logits = torch.full((1, C), 0.5)
    tail_t = torch.zeros(1, C); tail_t[0, 3] = 1
    head_t = torch.zeros(1, C); head_t[0, 0] = 1
    l_tail = logit_adjusted_focal_loss(logits, tail_t, log_priors)[0, 3]
    l_head = logit_adjusted_focal_loss(logits, head_t, log_priors)[0, 0]
    assert l_tail < l_head, (l_tail.item(), l_head.item())


def test_focal_gradient_sane():
    logits = torch.zeros(1, C, requires_grad=True)
    t = torch.zeros(1, C); t[0, 2] = 1
    loss = logit_adjusted_focal_loss(logits, t, log_priors, reduction="sum")
    loss.backward()
    # positive class pushed up, negatives pushed down
    assert logits.grad[0, 2] < 0
    assert (logits.grad[0, [0, 1, 3]] > 0).all()


def test_cost_matrix_consistency():
    """Matcher cost must prefer assigning a tail GT to the same query that the
    adjusted loss would supervise — i.e. adjusted cost for the tail class is
    lower than the unadjusted one."""
    q_logits = torch.randn(5, C)
    tgt = torch.tensor([3, 0])
    cost_adj = logit_adjusted_cost_class(q_logits, tgt, log_priors, tau=1.0)
    cost_raw = logit_adjusted_cost_class(q_logits, tgt, log_priors * 0, tau=1.0)
    assert cost_adj.shape == (5, 2)
    assert (cost_adj[:, 0] < cost_raw[:, 0]).all(), \
        "adjusted cost must make tail assignments cheaper"


def test_seesaw_mitigation_and_compensation():
    seesaw = SeesawBCE(C, p=0.8, q=2.0)
    seesaw.train()
    # burn in imbalanced counts: 1000 head-0 positives, 10 tail-3 positives
    head_batch = torch.zeros(1000, C); head_batch[:, 0] = 1
    tail_batch = torch.zeros(10, C); tail_batch[:, 3] = 1
    seesaw._update_counts(head_batch)
    seesaw._update_counts(tail_batch)
    seesaw.eval()  # freeze counts for the assertion phase

    # sample whose positive is HEAD class 0: the tail negative (class 3)
    # must be down-weighted relative to plain BCE (mitigation < 1)...
    logits = torch.zeros(1, C)
    t_head = torch.zeros(1, C); t_head[0, 0] = 1
    loss = seesaw(logits, t_head)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, t_head, reduction="none")
    assert loss[0, 3] < bce[0, 3], "tail negative must be mitigated"

    # ...while for a sample whose positive is TAIL class 3, the head negative
    # keeps full weight (n_head > n_tail -> ratio clamped to 1) and gains
    # compensation if it outscores the positive.
    t_tail = torch.zeros(1, C); t_tail[0, 3] = 1
    hot_logits = torch.tensor([[3.0, 0.0, 0.0, -1.0]])  # head outscores tail
    loss2 = seesaw(hot_logits, t_tail)
    bce2 = torch.nn.functional.binary_cross_entropy_with_logits(
        hot_logits, t_tail, reduction="none")
    assert loss2[0, 0] > bce2[0, 0], "outscoring head negative must be compensated up"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS %s" % fn.__name__)
    print("\nALL CHECKS PASSED: long-tail losses behave as designed")
