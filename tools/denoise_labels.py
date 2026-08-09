#!/usr/bin/env python3
"""Build a training corpus with consensus-flagged suspect annotations removed.

THE GT-SIDE HALF OF LABEL-NOISE-ROBUST TRAINING. The loss-side half (the
--bg-gate flag in train_seg.py, after Background Recalibration Loss,
arXiv:2002.05274, and Soft Sampling, arXiv:1806.06986) stops the model being
punished for detecting real findings that were never annotated. This tool
treats the opposite error: annotations of findings that are not usably there.

The suspect list comes from tools/universal_misses.py run over the TRAIN split
with models that trained ON that split. An annotation that a model cannot fit
after being explicitly optimised to fit it is a stronger label-error signal
than a validation miss: the model had every opportunity to memorise it and
still could not. This is the training-side analogue of the consensus-outlier
attribution used on validation, where 911 pathology annotations were invisible
to fifteen models and the model that improved caries 35 percent recovered 2.7
percent of them.

Removing them follows SparseDet's framing (arXiv:2201.04620): regions whose
labels cannot be trusted should not supervise, in either direction. Dropping
the annotation stops it teaching the model that an invisible pattern is a
lesion; the image itself is kept, because its OTHER annotations are fine.

Matching COCO suspects to YOLO label lines is by class + IoU >= 0.5 between
the suspect bbox and the label line's polygon bbox, both in pixels.

Usage:
    python tools/denoise_labels.py --src data_clean \\
        --suspects reports/universal_misses_train.json \\
        --gt data_clean/annotations/instances_train.json \\
        --out data_clean_dn
"""
import argparse
import json
import os

import yaml


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--suspects", required=True,
                    help="universal_misses output over the TRAIN split")
    ap.add_argument("--gt", required=True,
                    help="the train COCO file, for image sizes and class ids")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sus = json.load(open(args.suspects))["instances"]
    gt = json.load(open(args.gt))
    name_to_yolo = {}
    with open(os.path.join(args.src, "data.yaml")) as f:
        names = yaml.safe_load(f)["names"]
    for k, v in (names.items() if isinstance(names, dict) else enumerate(names)):
        name_to_yolo[v] = int(k)
    wh = {os.path.splitext(im["file_name"])[0]: (im["width"], im["height"])
          for im in gt["images"]}

    by_stem = {}
    for r in sus:
        stem = os.path.splitext(r["image"])[0]
        by_stem.setdefault(stem, []).append((name_to_yolo[r["class"]], r["bbox"]))

    src_lbl = os.path.join(args.src, "train", "labels")
    src_img = os.path.join(args.src, "train", "images")
    out_lbl = os.path.join(args.out, "train", "labels")
    out_img = os.path.join(args.out, "train", "images")
    os.makedirs(out_lbl, exist_ok=True)
    os.makedirs(out_img, exist_ok=True)

    dropped = kept = 0
    touched = 0
    for fn in sorted(os.listdir(src_lbl)):
        if not fn.endswith(".txt"):
            continue
        stem = os.path.splitext(fn)[0]
        suspects = by_stem.get(stem, [])
        W, H = wh.get(stem, (None, None))
        out_lines = []
        with open(os.path.join(src_lbl, fn)) as f:
            for ln in f:
                p = ln.split()
                if not p:
                    continue
                drop = False
                if suspects and W:
                    cid = int(p[0])
                    xs = [float(v) * W for v in p[1::2]]
                    ys = [float(v) * H for v in p[2::2]]
                    line_box = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                    for scid, sbox in suspects:
                        if scid == cid and iou(line_box, sbox) >= args.iou:
                            drop = True
                            break
                if drop:
                    dropped += 1
                else:
                    kept += 1
                    out_lines.append(ln.rstrip("\n"))
        if suspects:
            touched += 1
        with open(os.path.join(out_lbl, fn), "w") as f:
            f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
        for ext in (".jpg", ".jpeg", ".png"):
            sp = os.path.join(src_img, stem + ext)
            if os.path.exists(sp):
                dp = os.path.join(out_img, stem + ext)
                if not os.path.exists(dp):
                    os.symlink(os.path.abspath(sp), dp)
                break

    src_abs = os.path.abspath(args.src)
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        yaml.safe_dump({
            "path": os.path.abspath(args.out),
            "train": "train/images",
            "val": os.path.join(src_abs, "valid", "images"),
            "test": os.path.join(src_abs, "test", "images"),
            "names": names,
        }, f, sort_keys=False, default_flow_style=False)

    n_sus = len(sus)
    print("  suspects listed: %d  |  label lines dropped: %d  kept: %d"
          % (n_sus, dropped, kept))
    print("  images with at least one suspect: %d" % touched)
    print("  match rate: %.0f %% of suspects found a label line"
          % (100 * dropped / max(n_sus, 1)))
    print("  wrote %s (val/test point at the untouched clean split)" % args.out)


if __name__ == "__main__":
    main()
