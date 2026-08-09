#!/usr/bin/env python3
"""Build a single-class tooth-detection corpus from the DENTEX enumeration sets.

STAGE ONE OF THE TOOTH-CONDITIONED PIPELINE. The domain-correct inductive bias
for dental diagnosis is that a finding is a property of a TOOTH: DENTEX's own
label hierarchy is quadrant -> enumeration -> diagnosis (arXiv:2305.19112), and
the organisers' follow-up detector is built around exactly that hierarchy
(HierarchicalDet, arXiv:2303.06500). Our corpus annotates lesions but carries
no per-tooth ground truth, so the tooth detector has to come from DENTEX:

  training_data/quadrant_enumeration      634 x-rays, every tooth boxed
  training_data/quadrant-enumeration-disease  705 x-rays, abnormal teeth boxed

Both collapse to one class, "tooth". The disease set is included even though it
boxes only abnormal teeth, because a missing annotation in tooth detection is
exactly the sparse-label situation the rest of this project deals with; its
images are marked so the trainer's --bg-gate can be used if needed.

Boxes become rectangle polygons so the corpus trains a -seg model directly and
stage two can reuse the same tooling end to end. A 90/10 image-level split
gives an honest validation number for the tooth detector itself.

Usage:
    python tools/build_tooth_corpus.py \\
        --dentex /media/mostafa/EGYPT_SSD/dental31/dentex/extracted \\
        --out data_tooth
"""
import argparse
import glob
import json
import os
import random

import yaml


def load_set(root, sub):
    d = os.path.join(root, "training_data", sub)
    js = glob.glob(os.path.join(d, "*.json"))
    if not js:
        return None, None, None
    data = json.load(open(js[0]))
    return data, os.path.join(d, "xrays"), js[0]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dentex", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sets = []
    for sub in ("quadrant_enumeration", "quadrant-enumeration-disease"):
        data, imgdir, js = load_set(args.dentex, sub)
        if data:
            sets.append((sub, data, imgdir))
            print("  using %s (%d images, %d boxes)"
                  % (js.split("/")[-1], len(data["images"]), len(data["annotations"])))
    if not sets:
        raise SystemExit("no enumeration jsons found under %s" % args.dentex)

    records = []   # (unique_stem, src_path, [(x,y,w,h)...], W, H)
    for sub, data, imgdir in sets:
        anns = {}
        for a in data["annotations"]:
            anns.setdefault(a["image_id"], []).append(a["bbox"])
        for im in data["images"]:
            src = os.path.join(imgdir, im["file_name"])
            if not os.path.exists(src):
                continue
            stem = sub.replace("-", "_") + "__" + os.path.splitext(im["file_name"])[0]
            records.append((stem, src, anns.get(im["id"], []),
                            im["width"], im["height"]))

    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * args.val_frac))
    splits = {"valid": records[:n_val], "train": records[n_val:]}

    total_boxes = 0
    for split, recs in splits.items():
        oi = os.path.join(args.out, split, "images")
        ol = os.path.join(args.out, split, "labels")
        os.makedirs(oi, exist_ok=True)
        os.makedirs(ol, exist_ok=True)
        for stem, src, boxes, W, H in recs:
            ext = os.path.splitext(src)[1]
            dst = os.path.join(oi, stem + ext)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)
            lines = []
            for (x, y, w, h) in boxes:
                x1, y1 = max(x, 0) / W, max(y, 0) / H
                x2, y2 = min(x + w, W) / W, min(y + h, H) / H
                if x2 <= x1 or y2 <= y1:
                    continue
                lines.append("0 %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f"
                             % (x1, y1, x2, y1, x2, y2, x1, y2))
                total_boxes += 1
            with open(os.path.join(ol, stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        yaml.safe_dump({"path": os.path.abspath(args.out),
                        "train": "train/images", "val": "valid/images",
                        "names": {0: "tooth"}},
                       f, sort_keys=False, default_flow_style=False)
    print("  wrote %s: %d train / %d valid images, %d tooth boxes"
          % (args.out, len(splits["train"]), len(splits["valid"]), total_boxes))


if __name__ == "__main__":
    main()
