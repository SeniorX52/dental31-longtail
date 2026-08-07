#!/usr/bin/env python3
"""Build a corpus restricted to the classes that can support a claim.

Sixteen of the 31 classes appear in fewer than 10 images in valid or test. Every
apparent tail improvement in this project decomposed to two or three detections
on classes with 2-8 instances, so those classes cannot carry a result -- but they
are still in the objective, contributing gradient with no usable signal and
occupying capacity in a shared classification head.

This builds a parallel corpus keeping only classes with at least 10 images in
BOTH valid and test, remapping class ids to a contiguous range. Images are
symlinked, so the corpus costs nothing on disk; only the label files are
rewritten.

Two reasons to run it. It tests whether removing unmeasurable classes helps the
measurable ones, which is a real question and not a foregone conclusion. And it
is the corpus a defensible paper would use, since it is exactly the subset on
which per-class claims are supportable.

Usage:
    python tools/restrict_classes.py --src data_clean --out data_clean_15 \\
        --spec reports/measurable_classes.json
"""
import argparse
import json
import os
import shutil

import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", required=True,
                    help="json with a 'keep' list of class names")
    ap.add_argument("--splits", default="train,valid,test")
    args = ap.parse_args()

    with open(os.path.join(args.src, "data.yaml")) as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]                      # {yolo_id: name}
    keep_names = set(json.load(open(args.spec))["keep"])

    # old yolo id -> new contiguous yolo id, ordered as in the source vocabulary
    old_ids = [i for i in sorted(names) if names[i] in keep_names]
    remap = {old: new for new, old in enumerate(old_ids)}
    new_names = {new: names[old] for old, new in remap.items()}
    if len(old_ids) != len(keep_names):
        missing = keep_names - {names[i] for i in old_ids}
        raise SystemExit("names in spec not found in data.yaml: %s" % missing)

    splits = [s.strip() for s in args.splits.split(",")]
    stats = {}
    for sp in splits:
        src_img = os.path.join(args.src, sp, "images")
        src_lbl = os.path.join(args.src, sp, "labels")
        out_img = os.path.join(args.out, sp, "images")
        out_lbl = os.path.join(args.out, sp, "labels")
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)

        kept = dropped = empties = 0
        for fn in sorted(os.listdir(src_lbl)):
            if not fn.endswith(".txt"):
                continue
            lines_out = []
            with open(os.path.join(src_lbl, fn)) as f:
                for ln in f:
                    parts = ln.split()
                    if not parts:
                        continue
                    cid = int(parts[0])
                    if cid in remap:
                        parts[0] = str(remap[cid])
                        lines_out.append(" ".join(parts))
                        kept += 1
                    else:
                        dropped += 1
            # An image whose every annotation was dropped becomes a background
            # image. Keep it: it is still a valid negative and removing it would
            # change the image distribution as well as the label set, which
            # would confound the comparison.
            if not lines_out:
                empties += 1
            with open(os.path.join(out_lbl, fn), "w") as f:
                f.write("\n".join(lines_out) + ("\n" if lines_out else ""))

            stem = os.path.splitext(fn)[0]
            for ext in (".jpg", ".jpeg", ".png"):
                s = os.path.join(src_img, stem + ext)
                if os.path.exists(s):
                    d = os.path.join(out_img, stem + ext)
                    if not os.path.exists(d):
                        os.symlink(os.path.abspath(s), d)
                    break
        stats[sp] = {"annotations_kept": kept, "annotations_dropped": dropped,
                     "images_now_background": empties}
        print("  %-6s kept %6d, dropped %6d annotations; %d images became background"
              % (sp, kept, dropped, empties))

    out_cfg = {
        "path": os.path.abspath(args.out),
        "train": "train/images", "val": "valid/images", "test": "test/images",
        "names": new_names,
    }
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        yaml.safe_dump(out_cfg, f, sort_keys=False, default_flow_style=False)

    with open(os.path.join(args.out, "class_map.json"), "w") as f:
        json.dump({"kept_classes": new_names,
                   "old_to_new_yolo_id": {str(k): v for k, v in remap.items()},
                   "stats": stats}, f, indent=1)
    print("\n  %d classes kept, wrote %s/data.yaml" % (len(new_names), args.out))


if __name__ == "__main__":
    main()
