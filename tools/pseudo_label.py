#!/usr/bin/env python3
"""Self-training data plumbing: skeleton COCO for unlabeled images, and
COCO predictions into YOLO-seg pseudo-labels.

RECIPE AND SOURCE. This implements the data side of STAC (Sohn et al.,
arXiv:2005.04757): a trained teacher predicts on unlabeled domain images, the
predictions above a high confidence threshold become hard pseudo-labels, and a
student trains on labeled + pseudo-labeled data together. STAC's reported
setting is tau = 0.9, which is the default here, with unsupervised loss weight
lambda_u in [1, 2]; our pipeline mixes pseudo-labeled images into the ordinary
dataloader, which corresponds to lambda_u = 1, the conservative end. STAC
gains +4.8 to +5.9 mAP on the COCO low-label protocols; dental gains will be
smaller since our labeled set is larger, but the unlabeled pool (about 2,900
DENTEX panoramics) is also the exact external domain we want to transfer to,
so this doubles as domain adaptation.

Two subcommands:

  index   scan an image directory into a minimal COCO json (images + our 31
          categories, no annotations) so predict_to_coco can address the files.

  labels  turn a predictions json into YOLO-seg label files: score >= tau,
          polygon from the predicted mask (largest contour), bbox rectangle as
          the fallback when no usable mask is present.

Usage:
    python tools/pseudo_label.py index --images <dir> \\
        --like data_clean/annotations/instances_valid.json --out skel.json
    python yolov8_seg_longtail/predict_to_coco.py --weights <teacher> \\
        --gt skel.json --images <dir> --out preds.json --imgsz 2176 ...
    python tools/pseudo_label.py labels --preds preds.json --skel skel.json \\
        --tau 0.9 --out-labels <labels_dir>
"""
import argparse
import json
import os

import cv2
import numpy as np


def cmd_index(args):
    ref = json.load(open(args.like))
    images = []
    iid = 1
    for fn in sorted(os.listdir(args.images)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        im = cv2.imread(os.path.join(args.images, fn))
        if im is None:
            continue
        h, w = im.shape[:2]
        images.append({"id": iid, "file_name": fn, "width": w, "height": h})
        iid += 1
    out = {"images": images, "annotations": [], "categories": ref["categories"]}
    json.dump(out, open(args.out, "w"))
    print("  indexed %d images -> %s" % (len(images), args.out))


def poly_from_segmentation(seg, w, h):
    """Largest-contour polygon from a COCO segmentation (RLE or polygon)."""
    if isinstance(seg, list) and seg and isinstance(seg[0], list):
        best = max(seg, key=len)
        pts = np.array(best, dtype=np.float64).reshape(-1, 2)
        return pts
    if isinstance(seg, dict):
        from pycocotools import mask as maskutil
        m = maskutil.decode(seg)
        if m.ndim == 3:
            m = m[:, :, 0]
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return None
        c = max(cs, key=cv2.contourArea)
        if cv2.contourArea(c) < 16:
            return None
        eps = 0.01 * cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, eps, True)
        return c.reshape(-1, 2).astype(np.float64)
    return None


def cmd_labels(args):
    skel = json.load(open(args.skel))
    info = {im["id"]: im for im in skel["images"]}
    # predictions carry ORIGINAL category ids; YOLO wants the contiguous index
    cats = sorted(c["id"] for c in skel["categories"])
    coco_to_yolo = {cid: i for i, cid in enumerate(cats)}
    preds = json.load(open(args.preds))

    os.makedirs(args.out_labels, exist_ok=True)
    per_image = {}
    n_kept = n_bbox_fallback = 0
    for d in preds:
        if d.get("score", 0.0) < args.tau:
            continue
        im = info.get(d["image_id"])
        if im is None:
            continue
        w, h = im["width"], im["height"]
        pts = poly_from_segmentation(d.get("segmentation"), w, h)
        if pts is None or len(pts) < 3:
            x, y, bw, bh = d["bbox"]
            pts = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]],
                           dtype=np.float64)
            n_bbox_fallback += 1
        pts[:, 0] = np.clip(pts[:, 0] / w, 0, 1)
        pts[:, 1] = np.clip(pts[:, 1] / h, 0, 1)
        line = str(coco_to_yolo[d["category_id"]]) + " " + \
            " ".join("%.6f" % v for v in pts.reshape(-1))
        per_image.setdefault(im["file_name"], []).append(line)
        n_kept += 1

    for im in skel["images"]:
        stem = os.path.splitext(im["file_name"])[0]
        lines = per_image.get(im["file_name"], [])[: args.max_per_image]
        with open(os.path.join(args.out_labels, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    n_img = sum(1 for v in per_image.values() if v)
    print("  pseudo-labels: %d detections >= tau %.2f on %d/%d images "
          "(%d bbox-rectangle fallbacks)"
          % (n_kept, args.tau, n_img, len(skel["images"]), n_bbox_fallback))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index")
    a.add_argument("--images", required=True)
    a.add_argument("--like", required=True,
                   help="COCO file whose categories to copy")
    a.add_argument("--out", required=True)
    b = sub.add_parser("labels")
    b.add_argument("--preds", required=True)
    b.add_argument("--skel", required=True)
    b.add_argument("--tau", type=float, default=0.9,
                   help="STAC's confidence threshold (arXiv:2005.04757)")
    b.add_argument("--max-per-image", type=int, default=100)
    b.add_argument("--out-labels", required=True)
    args = ap.parse_args()
    if args.cmd == "index":
        cmd_index(args)
    else:
        cmd_labels(args)


if __name__ == "__main__":
    main()
