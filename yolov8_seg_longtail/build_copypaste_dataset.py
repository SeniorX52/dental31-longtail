#!/usr/bin/env python3
"""Build a copy-paste-augmented TRAINING set, leaving the frozen split intact.

Produces a dataset whose valid/test splits are the originals (symlinked, so
evaluation is byte-identical to every other arm) and whose train split is the
originals PLUS synthetic images carrying extra instances of rare classes.

Augmenting only train is the whole point: if pasted instances reached valid or
test, the rare-class numbers would be measuring synthetic pixels rather than
real findings.

Usage:
    python yolov8_seg_longtail/build_copypaste_dataset.py \\
        --clean-root data_clean --out data_clean_cp \\
        --max-count 100 --copies 10 --seed 42
"""
import argparse
import os
import subprocess
import sys
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def link_tree(src: str, dst: str) -> int:
    os.makedirs(dst, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(src)):
        s = os.path.realpath(os.path.join(src, name))
        d = os.path.join(dst, name)
        if os.path.islink(d) or os.path.exists(d):
            os.remove(d)
        os.symlink(s, d)
        n += 1
    return n


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-count", type=int, default=100)
    ap.add_argument("--copies", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iou", type=float, default=0.10)
    args = ap.parse_args(argv)

    # valid / test: plain symlinks to the frozen originals
    for split in ("valid", "test"):
        for kind in ("images", "labels"):
            n = link_tree(os.path.join(args.clean_root, split, kind),
                          os.path.join(args.out, split, kind))
        print("%-6s linked (untouched)" % split)

    # train: originals first, then the synthetic additions on top
    ti = os.path.join(args.out, "train", "images")
    tl = os.path.join(args.out, "train", "labels")
    n_img = link_tree(os.path.join(args.clean_root, "train", "images"), ti)
    n_lbl = link_tree(os.path.join(args.clean_root, "train", "labels"), tl)
    print("train  linked %d originals" % n_img)

    cmd = [sys.executable, os.path.join(HERE, "copy_paste_rare.py"),
           "--images", os.path.join(args.clean_root, "train", "images"),
           "--labels", os.path.join(args.clean_root, "train", "labels"),
           "--out-images", ti, "--out-labels", tl,
           "--max-count", str(args.max_count), "--copies", str(args.copies),
           "--max-iou", str(args.max_iou), "--seed", str(args.seed),
           "--manifest", os.path.join(args.out, "copypaste_manifest.json")]
    print("\n" + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=True)

    total = len(os.listdir(ti))
    print("\ntrain now %d images (%d original + %d synthetic)"
          % (total, n_img, total - n_img))

    # data.yaml pointing at the augmented train, original valid/test
    import yaml
    names = yaml.safe_load(open(os.path.join(args.clean_root, "data.yaml")))["names"]
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write("# copy-paste augmented TRAIN; valid/test are the frozen originals\n")
        f.write("path: %s\n" % os.path.abspath(args.out))
        f.write("train: train/images\nval: valid/images\ntest: test/images\n")
        f.write("names:\n")
        for i in sorted(names):
            f.write("  %d: %s\n" % (i, names[i]))
    print("wrote %s/data.yaml" % args.out)


if __name__ == "__main__":
    main()
