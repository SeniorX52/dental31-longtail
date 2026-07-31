#!/usr/bin/env python3
"""Repeat-factor sampling (LVIS-style) computed from a COCO annotations JSON.

For each category c with image-frequency f(c) (fraction of training images
containing at least one instance of c), the category repeat factor is

    r(c) = max(1, sqrt(t / f(c)))

and each image's repeat factor is the max over the categories it contains
(Gupta et al., "LVIS: A Dataset for Large Vocabulary Instance Segmentation").
With t = 0.001, a class appearing in 0.01% of images is oversampled ~3.2x,
while all classes above the threshold are left untouched — this targets
exactly the single-digit tail (TAD = 4, Fracture teeth = 9) without
distorting the head.

Pure-python computation (testable without torch); `RepeatFactorSampler` at
the bottom wraps it for a PyTorch DataLoader.
"""
import json
import math
import random
from collections import defaultdict
from typing import Dict, List


def category_frequencies(coco_json_path: str) -> Dict[int, float]:
    """f(c): fraction of images containing >=1 instance of category c."""
    with open(coco_json_path) as f:
        coco = json.load(f)
    n_images = len(coco["images"])
    imgs_per_cat = defaultdict(set)
    for ann in coco["annotations"]:
        imgs_per_cat[ann["category_id"]].add(ann["image_id"])
    return {cat["id"]: len(imgs_per_cat[cat["id"]]) / max(n_images, 1)
            for cat in coco["categories"]}


def image_repeat_factors(coco_json_path: str, t: float = 0.001) -> Dict[int, float]:
    """r(img) = max over contained categories of max(1, sqrt(t / f(c))).

    Images with no annotations get r = 1.
    """
    freq = category_frequencies(coco_json_path)
    cat_rf = {c: max(1.0, math.sqrt(t / f)) if f > 0 else 1.0
              for c, f in freq.items()}
    with open(coco_json_path) as f:
        coco = json.load(f)
    cats_per_img = defaultdict(set)
    for ann in coco["annotations"]:
        cats_per_img[ann["image_id"]].add(ann["category_id"])
    return {img["id"]: max((cat_rf[c] for c in cats_per_img[img["id"]]), default=1.0)
            for img in coco["images"]}


def expand_indices(repeat_factors: List[float], seed: int) -> List[int]:
    """Stochastic rounding of per-index repeat factors into an epoch index list.

    An index with r = 2.3 appears 2 times always and a 3rd time with p = 0.3,
    using the epoch-seeded RNG so runs are reproducible.
    """
    rng = random.Random(seed)
    out: List[int] = []
    for idx, r in enumerate(repeat_factors):
        whole = int(r)
        frac = r - whole
        reps = whole + (1 if rng.random() < frac else 0)
        out.extend([idx] * reps)
    rng.shuffle(out)
    return out


try:
    import torch
    from torch.utils.data import Sampler

    class RepeatFactorSampler(Sampler):
        """Drop-in replacement for RandomSampler in the DINO train loader.

        repeat_factors: per-dataset-index repeat factor, aligned with the
        dataset's own index order (use `image_repeat_factors` + the dataset's
        image-id order to build it). Call `set_epoch(e)` before each epoch,
        as with DistributedSampler.
        """

        def __init__(self, repeat_factors: List[float], seed: int = 42):
            self.repeat_factors = repeat_factors
            self.seed = seed
            self.epoch = 0

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

        def __iter__(self):
            return iter(expand_indices(self.repeat_factors, self.seed + self.epoch))

        def __len__(self):
            # deterministic upper bound (ceil of fractional parts)
            return sum(int(math.ceil(r)) for r in self.repeat_factors)

except ImportError:  # torch-free environments can still use the math above
    pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("coco_json")
    ap.add_argument("--t", type=float, default=0.001)
    args = ap.parse_args()
    freq = category_frequencies(args.coco_json)
    print("cat_id  freq      repeat_factor")
    for c, f in sorted(freq.items(), key=lambda kv: kv[1]):
        rf = max(1.0, math.sqrt(args.t / f)) if f > 0 else 1.0
        print("%6d  %.6f  %.2f" % (c, f, rf))
