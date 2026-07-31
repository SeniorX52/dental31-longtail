#!/usr/bin/env python3
"""Lay out the clean split in the directory shape DINO's COCO loader expects.

DINO (IDEA-Research) hardcodes the COCO 2017 convention in
`datasets/coco.py`:

    <coco_path>/train2017/*.jpg
    <coco_path>/val2017/*.jpg
    <coco_path>/annotations/instances_train2017.json
    <coco_path>/annotations/instances_val2017.json

Rather than patch the loader (one more thing to keep in sync with upstream),
this script presents our split under those names using symlinks, so no pixels
are copied and the upstream repo runs unmodified.

`test` is exposed as an extra `test2017` + `instances_test2017.json` pair; the
final evaluation points DINO's val path at it once, after training, so the
test split is never seen during model selection.

Usage:
    python dino_longtail/prepare_dino_data.py \\
        --clean-root data_clean --out data_coco
"""
import argparse
import json
import os
from typing import List, Optional

# our split name -> the COCO-2017 name DINO expects
SPLIT_MAP = {"train": "train2017", "valid": "val2017", "test": "test2017"}


def link_dir(src_dir: str, dst_dir: str) -> int:
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for fname in sorted(os.listdir(src_dir)):
        dst = os.path.join(dst_dir, fname)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(os.path.join(src_dir, fname)), dst)
        n += 1
    return n


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-root", required=True,
                    help="output of tools/build_clean_dataset.py")
    ap.add_argument("--out", required=True, help="coco_path to hand DINO")
    args = ap.parse_args(argv)

    ann_out = os.path.join(args.out, "annotations")
    os.makedirs(ann_out, exist_ok=True)

    for split, coco_name in SPLIT_MAP.items():
        img_src = os.path.join(args.clean_root, split, "images")
        if not os.path.isdir(img_src):
            print("skip %s (missing %s)" % (split, img_src))
            continue
        n = link_dir(img_src, os.path.join(args.out, coco_name))

        src_json = os.path.join(args.clean_root, "annotations",
                                "instances_%s.json" % split)
        dst_json = os.path.join(ann_out, "instances_%s.json" % coco_name)
        with open(src_json) as f:
            d = json.load(f)
        with open(dst_json, "w") as f:
            json.dump(d, f)
        cats = d["categories"]
        print("%-6s -> %-10s %5d images | %6d anns | %d categories (ids %d..%d)"
              % (split, coco_name, n, len(d["annotations"]), len(cats),
                 min(c["id"] for c in cats), max(c["id"] for c in cats)))

    # DINO builds class_embed with num_classes outputs and indexes it directly
    # by category_id, so the highest id must be < num_classes. Our ids are
    # 1..31, hence num_classes=32 (index 0 unused) rather than 31.
    with open(os.path.join(ann_out, "instances_train2017.json")) as f:
        cats = json.load(f)["categories"]
    max_id = max(c["id"] for c in cats)
    print()
    print("category ids run 1..%d  ->  use  num_classes=%d" % (max_id, max_id + 1))
    print("(DINO indexes the classification head by category_id directly, so the")
    print(" head must have max_id+1 outputs; index 0 is simply never used.)")
    print()
    print("coco_path for DINO:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
