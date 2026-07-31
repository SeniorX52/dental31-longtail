#!/usr/bin/env python3
"""Materialize a frozen split into a training-ready dataset.

Takes the file lists produced by `make_clean_split.py` and builds:

    <out>/train|valid|test/images/*        symlinks to the originals
    <out>/train|valid|test/labels/*.txt    symlinks to the originals
    <out>/data.yaml                        relative paths, correct test: entry
    <out>/annotations/instances_*.json     COCO, rebuilt from the YOLO polygons

Symlinks are used so the 2.8 GB of pixels is never duplicated, while
ultralytics and DINO both see a conventional layout.

The COCO files are regenerated from the polygon labels rather than copied from
the shipped export, which fixes three defects in one step: the shipped test
JSON contains a bogus `croen` category that shifts every class id above 11,
its `segmentation` fields are all empty so masks cannot be supervised or
scored, and its category set (14) disagrees with train/valid (31). Rebuilding
guarantees one identical 31-category vocabulary across all three splits.

Usage:
    python tools/build_clean_dataset.py \\
        --split-dir splits/clean_v1 \\
        --names data_raw/dental31/YOLO/YOLO/data.yaml \\
        --out data_clean
"""
import argparse
import json
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.yolo_polygons_to_coco import convert, load_class_names  # noqa: E402

SPLITS = ("train", "valid", "test")


def link(src: str, dst: str) -> None:
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", required=True, help="dir with train/valid/test.txt")
    ap.add_argument("--names", required=True, help="source data.yaml (class names)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    class_names = load_class_names(args.names)
    print("classes: %d" % len(class_names))

    totals = {}
    for split in SPLITS:
        list_path = os.path.join(args.split_dir, split + ".txt")
        with open(list_path) as f:
            img_paths = [ln.strip() for ln in f if ln.strip()]

        img_dir = os.path.join(args.out, split, "images")
        lbl_dir = os.path.join(args.out, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        n_lbl_missing = 0
        for ip in img_paths:
            stem = os.path.splitext(os.path.basename(ip))[0]
            link(ip, os.path.join(img_dir, os.path.basename(ip)))
            # labels live in ../labels/<stem>.txt relative to the source image
            lp = os.path.join(os.path.dirname(os.path.dirname(ip)), "labels", stem + ".txt")
            if os.path.exists(lp):
                link(lp, os.path.join(lbl_dir, stem + ".txt"))
            else:
                n_lbl_missing += 1
        print("%-6s linked %5d images (%d without a label file)"
              % (split, len(img_paths), n_lbl_missing))
        totals[split] = len(img_paths)

    # ---- COCO, rebuilt from polygons, one shared 31-category vocabulary -----
    ann_dir = os.path.join(args.out, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    for split in SPLITS:
        res = convert(os.path.join(args.out, split, "images"),
                      os.path.join(args.out, split, "labels"),
                      class_names)
        out_json = os.path.join(ann_dir, "instances_%s.json" % split)
        with open(out_json, "w") as f:
            json.dump(res["coco"], f)
        coco = res["coco"]
        n_seg = sum(1 for a in coco["annotations"] if a.get("segmentation"))
        print("%-6s COCO: %5d images, %6d anns, %d categories, %d with segmentation | dropped %s"
              % (split, len(coco["images"]), len(coco["annotations"]),
                 len(coco["categories"]), n_seg, res["dropped"]))

    # ---- data.yaml with a CORRECT test: path and relative paths ------------
    yaml_path = os.path.join(args.out, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write("# rebuilt by tools/build_clean_dataset.py -- patient/source-grouped split\n")
        f.write("path: %s\n" % os.path.abspath(args.out))
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("test: test/images\n")   # the shipped file pointed this at valid
        f.write("names:\n")
        for i, n in enumerate(class_names):
            f.write("  %d: %s\n" % (i, n))
    print("\nwrote %s" % yaml_path)

    # sanity: identical category vocabulary across the three files
    vocabs = []
    for split in SPLITS:
        d = json.load(open(os.path.join(ann_dir, "instances_%s.json" % split)))
        vocabs.append(tuple((c["id"], c["name"]) for c in d["categories"]))
    assert len(set(vocabs)) == 1, "category vocabularies differ between splits"
    print("VERIFIED: all three COCO files share one identical 31-category vocabulary")
    print("dataset ready at %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
