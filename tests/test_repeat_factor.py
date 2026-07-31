#!/usr/bin/env python3
"""Unit test for LVIS-style repeat-factor computation and epoch expansion.

Run:  python tests/test_repeat_factor.py
"""
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dino_longtail.repeat_factor import (  # noqa: E402
    category_frequencies, image_repeat_factors, expand_indices)


def make_coco(tmpdir):
    """100 images: class 1 in all 100, class 2 in 1 image, image 99 empty."""
    images = [{"id": i, "file_name": "%d.jpg" % i, "width": 10, "height": 10}
              for i in range(100)]
    anns = []
    aid = 1
    for i in range(99):
        anns.append({"id": aid, "image_id": i, "category_id": 1,
                     "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0,
                     "segmentation": [[0, 0, 5, 0, 5, 5]]})
        aid += 1
    anns.append({"id": aid, "image_id": 50, "category_id": 2,
                 "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0,
                 "segmentation": [[0, 0, 5, 0, 5, 5]]})
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "head"}, {"id": 2, "name": "tail"}]}
    path = os.path.join(tmpdir, "coco.json")
    with open(path, "w") as f:
        json.dump(coco, f)
    return path


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = make_coco(tmp)
        t = 0.001

        freq = category_frequencies(path)
        assert abs(freq[1] - 0.99) < 1e-9
        assert abs(freq[2] - 0.01) < 1e-9

        rf = image_repeat_factors(path, t=t)
        # head-only images: f=0.99 > t -> factor 1
        assert rf[0] == 1.0
        # image 50 contains the tail class: sqrt(0.001/0.01) < 1 -> still 1.0?
        # No: sqrt(t/f) = sqrt(0.1) = 0.316 -> max(1, .) = 1. Use rarer t.
        rf2 = image_repeat_factors(path, t=0.04)
        expect = math.sqrt(0.04 / 0.01)  # = 2.0
        assert abs(rf2[50] - expect) < 1e-9, rf2[50]
        # empty image gets 1.0
        assert rf2[99] == 1.0

        # expansion: deterministic given seed, correct expectation
        factors = [rf2[i] for i in range(100)]
        idx_a = expand_indices(factors, seed=7)
        idx_b = expand_indices(factors, seed=7)
        assert idx_a == idx_b, "expansion must be reproducible for a fixed seed"
        # image 50 (r=2.0 exactly) must appear exactly twice
        assert idx_a.count(50) == 2
        # total length: 99 singles + 1 double
        assert len(idx_a) == 101

    print("ALL CHECKS PASSED: repeat-factor math and seeded expansion correct")


if __name__ == "__main__":
    main()
