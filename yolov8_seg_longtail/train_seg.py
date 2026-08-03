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
from ultralytics.utils import RANK
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.ops import crop_mask


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


class BoundaryAwareSegLoss(v8SegmentationLoss):
    """v8SegmentationLoss whose per-instance mask loss adds a boundary Dice term.

    Reads `model.boundary_weight` (float, 0 disables). The parent class reads
    `model.class_weights` for the cls loss on its own.
    """

    def __init__(self, model, tal_topk: int = 10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        self.boundary_weight = float(getattr(model, "boundary_weight", 0.0))

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
        bce = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        loss = (crop_mask(bce, xyxy).mean(dim=(1, 2)) / area).sum()
        if self.boundary_weight > 0:
            pb = soft_boundary(pred_mask.sigmoid())
            gb = soft_boundary(gt_mask)
            pb = crop_mask(pb, xyxy)
            gb = crop_mask(gb, xyxy)
            inter = (pb * gb).sum(dim=(1, 2))
            denom = pb.sum(dim=(1, 2)) + gb.sum(dim=(1, 2))
            bdice = 1.0 - (2 * inter + 1.0) / (denom + 1.0)
            loss = loss + self.boundary_weight * bdice.sum()
        return loss


class LongTailSegModel(SegmentationModel):
    def init_criterion(self):
        return BoundaryAwareSegLoss(self)


class LongTailSegTrainer(SegmentationTrainer):
    """SegmentationTrainer that builds LongTailSegModel and attaches the
    class-balance / boundary knobs before the criterion is created."""

    def __init__(self, overrides=None, _callbacks=None,
                 class_weights: Optional[torch.Tensor] = None,
                 boundary_weight: float = 0.0):
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        self._lt_class_weights = class_weights
        self._lt_boundary_weight = boundary_weight

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(
            LongTailSegModel(cfg, nc=self.data["nc"],
                             ch=self.data["channels"],
                             verbose=verbose and RANK == -1))
        if weights:
            model.load(weights)
        if self._lt_class_weights is not None:
            model.class_weights = self._lt_class_weights
        model.boundary_weight = self._lt_boundary_weight
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
    trainer = LongTailSegTrainer(overrides=overrides,
                                 class_weights=weights,
                                 boundary_weight=args.boundary_weight)
    trainer.train()


if __name__ == "__main__":
    main()
