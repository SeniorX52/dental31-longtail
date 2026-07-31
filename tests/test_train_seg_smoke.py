#!/usr/bin/env python3
"""CPU smoke test: one epoch of LongTailSegTrainer on a tiny synthetic dataset.

Proves the custom criterion (class-balanced BCE + boundary Dice) actually
builds, runs forward/backward inside the real ultralytics train loop, and
that the boundary term is active (loss changes when enabled).

Run:  python tests/test_train_seg_smoke.py     (~1-2 min on CPU)
"""
import os
import sys
import tempfile

import cv2
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from yolov8_seg_longtail.train_seg import (  # noqa: E402
    LongTailSegTrainer, effective_number_weights, class_counts_from_yolo_labels)

W = H = 160


def build_dataset(root):
    rng = np.random.RandomState(0)
    for split in ("train", "val"):
        os.makedirs(os.path.join(root, split, "images"))
        os.makedirs(os.path.join(root, split, "labels"))
        for i in range(4):
            img = rng.randint(40, 200, (H, W, 3), dtype=np.uint8)
            # draw the object so there is actually signal to fit
            cv2.rectangle(img, (16, 16), (80, 80), (255, 255, 255), -1)
            cv2.circle(img, (110, 110), 24, (0, 0, 0), -1)
            cv2.imwrite(os.path.join(root, split, "images", "im%d.jpg" % i), img)
            with open(os.path.join(root, split, "labels", "im%d.txt" % i), "w") as f:
                f.write("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")   # square, class 0
                f.write("1 0.54 0.69 0.84 0.69 0.69 0.84\n")      # triangle, class 1
    yaml_path = os.path.join(root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write("path: %s\ntrain: train/images\nval: val/images\n"
                "names:\n  0: square\n  1: triangle\n" % root)
    return yaml_path


def main():
    tmp = tempfile.mkdtemp(prefix="seg_smoke_")
    yaml_path = build_dataset(tmp)

    counts = class_counts_from_yolo_labels(os.path.join(tmp, "train", "labels"), 2)
    assert counts.tolist() == [4.0, 4.0]
    weights = effective_number_weights(torch.tensor([33000.0, 4.0]), beta=0.999)
    assert weights[1] > weights[0], "tail class must get the larger weight"

    trainer = LongTailSegTrainer(
        overrides=dict(model="yolov8n-seg.yaml", data=yaml_path, epochs=1,
                       imgsz=64, batch=2, device="cpu", workers=0, seed=42,
                       deterministic=True, val=False, plots=False, save=False,
                       project=os.path.join(tmp, "runs"), name="smoke",
                       exist_ok=True),
        class_weights=effective_number_weights(counts, beta=0.999),
        boundary_weight=0.5)
    trainer.train()

    crit = trainer.model.init_criterion()
    assert type(crit).__name__ == "BoundaryAwareSegLoss"
    assert crit.boundary_weight == 0.5
    assert trainer.model.class_weights is not None

    # boundary term must actually change the mask loss
    gt = torch.zeros(2, 32, 32); gt[:, 8:24, 8:24] = 1
    proto = torch.randn(16, 32, 32)
    pred = torch.randn(2, 16, requires_grad=True)
    xyxy = torch.tensor([[4.0, 4.0, 28.0, 28.0]] * 2)
    area = torch.tensor([0.5, 0.5])
    crit.boundary_weight = 0.0
    base = crit.single_mask_loss(gt, pred, proto, xyxy, area)
    crit.boundary_weight = 0.5
    with_b = crit.single_mask_loss(gt, pred, proto, xyxy, area)
    assert with_b > base, "boundary term should add positive loss here"
    with_b.backward()  # gradient flows through the boundary term
    assert pred.grad is not None and torch.isfinite(pred.grad).all()

    print("\nALL CHECKS PASSED: custom seg criterion trains end-to-end "
          "(class weights + active, differentiable boundary term)")


if __name__ == "__main__":
    main()
