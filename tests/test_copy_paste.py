#!/usr/bin/env python3
"""End-to-end smoke test for rare-class copy-paste augmentation.

Builds a synthetic split (head class everywhere, one rare-class instance),
runs the script via its CLI, then checks determinism and label validity.

Run:  python tests/test_copy_paste.py
"""
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(ROOT, "yolov8_seg_longtail", "copy_paste_rare.py")
W, H = 320, 240


def build(root):
    img_dir = os.path.join(root, "images"); os.makedirs(img_dir)
    lbl_dir = os.path.join(root, "labels"); os.makedirs(lbl_dir)
    rng = np.random.RandomState(0)
    for i in range(6):
        img = rng.randint(0, 255, (H, W, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(img_dir, "im%d.jpg" % i), img)
        with open(os.path.join(lbl_dir, "im%d.txt" % i), "w") as f:
            f.write("0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n")   # head class 0
            if i == 0:  # single rare instance of class 1
                f.write("1 0.6 0.6 0.9 0.6 0.75 0.9\n")
    return img_dir, lbl_dir


def run(img_dir, lbl_dir, out_root, seed=42):
    oi = os.path.join(out_root, "images"); ol = os.path.join(out_root, "labels")
    mf = os.path.join(out_root, "manifest.json")
    subprocess.run([sys.executable, SCRIPT,
                    "--images", img_dir, "--labels", lbl_dir,
                    "--out-images", oi, "--out-labels", ol,
                    "--max-count", "3", "--copies", "5",
                    "--seed", str(seed), "--manifest", mf],
                   check=True, cwd=out_root,
                   stdout=subprocess.PIPE)
    with open(mf) as f:
        return json.load(f), oi, ol


def main():
    with tempfile.TemporaryDirectory() as tmp:
        img_dir, lbl_dir = build(tmp)

        out_a = os.path.join(tmp, "a"); os.makedirs(out_a)
        man_a, oi, ol = run(img_dir, lbl_dir, out_a)
        assert man_a["pastes"], "no pastes produced"
        # only the rare class (1) is pasted, never the head class
        assert {p["class"] for p in man_a["pastes"]} == {1}

        for p in man_a["pastes"]:
            ipath = os.path.join(oi, p["out"])
            lpath = os.path.join(ol, os.path.splitext(p["out"])[0] + ".txt")
            assert os.path.exists(ipath) and os.path.exists(lpath)
            with open(lpath) as f:
                lines = [ln.split() for ln in f if ln.strip()]
            # host's original head instance kept + pasted rare instance added
            classes = [int(l[0]) for l in lines]
            assert classes.count(1) >= 1 and classes.count(0) >= 1
            coords = np.array([float(v) for l in lines for v in l[1:]])
            assert (coords >= -1e-6).all() and (coords <= 1 + 1e-6).all(), \
                "pasted polygon out of normalized bounds"

        # determinism: same seed -> identical manifest
        out_b = os.path.join(tmp, "b"); os.makedirs(out_b)
        man_b, _, _ = run(img_dir, lbl_dir, out_b)
        assert man_a["pastes"] == man_b["pastes"], "same seed must reproduce exactly"

    print("ALL CHECKS PASSED: copy-paste is rare-only, in-bounds, deterministic")


if __name__ == "__main__":
    main()
