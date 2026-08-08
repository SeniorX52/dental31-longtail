#!/usr/bin/env python3
"""Build a training subset of a YOLO corpus, holding validation and test fixed.

WHY. Two unrelated architectures converge to within 0.7 pp on this corpus
(DINO-DETR 0.1625, YOLOv8x box 0.1551) where on COCO the same families are
several AP apart. Convergence across families is the signature of a data-limited
regime rather than a method-limited one, but it is circumstantial. A learning
curve measures it directly: train the identical configuration on 25, 50, 75 and
100 percent of the training images and read the slope at 100 percent. If the
curve has flattened, more data will not help and the ceiling is the labels or
the task; if it is still climbing, the honest answer to "what would beat the
baseline" is more data, with a measured slope attached.

The sampling is at the IMAGE level and stratified by the rarest class each image
contains, so that a 25 percent subset still carries roughly a quarter of the
instances of every class rather than dropping the tail entirely, which would
confound the data-quantity question with a class-coverage change. Images are
symlinked, so a subset costs nothing on disk.

Validation and test point at the ORIGINAL directories, unchanged across every
fraction, so the curve is read on one fixed measuring stick.

Usage:
    python tools/make_subset.py --src data_clean --out data_frac25 --frac 0.25
"""
import argparse
import json
import os
import random
from collections import defaultdict

import yaml


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not 0 < args.frac <= 1:
        raise SystemExit("--frac must be in (0, 1]")

    src = os.path.abspath(args.src)
    with open(os.path.join(src, "data.yaml")) as f:
        cfg = yaml.safe_load(f)

    lbl_dir = os.path.join(src, "train", "labels")
    img_dir = os.path.join(src, "train", "images")
    stems = [os.path.splitext(f)[0] for f in sorted(os.listdir(lbl_dir))
             if f.endswith(".txt")]

    # class content of each image, and global class frequency
    per_image, freq = {}, defaultdict(int)
    for s in stems:
        classes = set()
        with open(os.path.join(lbl_dir, s + ".txt")) as f:
            for ln in f:
                p = ln.split()
                if p:
                    classes.add(int(p[0]))
        per_image[s] = classes
        for c in classes:
            freq[c] += 1

    # Stratify on the RAREST class an image contains. Sampling a fixed fraction
    # within each stratum keeps tail classes represented at the target rate; a
    # flat random sample would thin them by chance and confound the experiment.
    strata = defaultdict(list)
    for s in stems:
        cs = per_image[s]
        key = min(cs, key=lambda c: freq[c]) if cs else -1   # -1 = background
        strata[key].append(s)

    rng = random.Random(args.seed)
    keep = []
    for key in sorted(strata):
        grp = sorted(strata[key])
        rng.shuffle(grp)
        n = int(round(len(grp) * args.frac))
        if grp and n == 0:
            n = 1                      # never drop a stratum entirely
        keep.extend(grp[:n])
    keep.sort()

    out_img = os.path.join(os.path.abspath(args.out), "train", "images")
    out_lbl = os.path.join(os.path.abspath(args.out), "train", "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    n_ann = 0
    kept_classes = defaultdict(int)
    for s in keep:
        lp = os.path.join(lbl_dir, s + ".txt")
        dst = os.path.join(out_lbl, s + ".txt")
        if not os.path.exists(dst):
            os.symlink(lp, dst)
        with open(lp) as f:
            for ln in f:
                p = ln.split()
                if p:
                    n_ann += 1
                    kept_classes[int(p[0])] += 1
        for ext in (".jpg", ".jpeg", ".png"):
            ip = os.path.join(img_dir, s + ext)
            if os.path.exists(ip):
                d = os.path.join(out_img, s + ext)
                if not os.path.exists(d):
                    os.symlink(ip, d)
                break

    # valid/test point back at the ORIGINAL split, identical for every fraction
    out_cfg = {
        "path": os.path.abspath(args.out),
        "train": "train/images",
        "val": os.path.join(src, "valid", "images"),
        "test": os.path.join(src, "test", "images"),
        "names": cfg["names"],
    }
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        yaml.safe_dump(out_cfg, f, sort_keys=False, default_flow_style=False)

    meta = {"src": src, "frac": args.frac, "seed": args.seed,
            "images_kept": len(keep), "images_total": len(stems),
            "annotations_kept": n_ann,
            "classes_present": len(kept_classes),
            "per_class": {str(k): v for k, v in sorted(kept_classes.items())}}
    with open(os.path.join(args.out, "subset.json"), "w") as f:
        json.dump(meta, f, indent=1)

    print("  %s: %d/%d images (%.1f %%), %d annotations, %d/%d classes present"
          % (args.out, len(keep), len(stems), 100 * len(keep) / len(stems),
             n_ann, len(kept_classes), len(cfg["names"])))


if __name__ == "__main__":
    main()
