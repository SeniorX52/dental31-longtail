#!/usr/bin/env python3
"""Offline rare-class copy-paste augmentation for YOLO polygon datasets.

Ghiasi et al. ("Simple Copy-Paste is a Strong Data Augmentation Method for
Instance Segmentation", CVPR 2021) — restricted to rare classes: instances of
classes below `--max-count` are cropped by their polygon mask and pasted into
other training images at plausible positions, writing new image + label files
alongside the originals. Offline (rather than in the dataloader) so the exact
augmented dataset is versioned and reproducible — an auditor can diff it.

Guard rails:
  * pastes never overlap an existing instance by more than `--max-iou` (IoU
    of bounding boxes) so real supervision is not occluded;
  * paste scale is jittered mildly (0.8-1.2) and position drawn uniformly
    from valid locations; a fixed `--seed` makes the whole run deterministic;
  * output images get the suffix `_cp{k}` and a manifest JSON records the
    provenance of every paste (source image, class, transform).

Usage:
    python yolov8_seg_longtail/copy_paste_rare.py \
        --images data/train/images --labels data/train/labels \
        --out-images data/train/images --out-labels data/train/labels \
        --max-count 100 --copies 20 --seed 42 --manifest cp_manifest.json
"""
import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def read_labels(path: str) -> List[Tuple[int, np.ndarray]]:
    """[(cls, Nx2 normalized polygon)] for one label file."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 7:
                continue
            cls = int(parts[0])
            pts = np.array([float(v) for v in parts[1:]], dtype=np.float64)
            out.append((cls, pts.reshape(-1, 2)))
    return out


def write_labels(path: str, insts: List[Tuple[int, np.ndarray]]) -> None:
    with open(path, "w") as f:
        for cls, poly in insts:
            flat = " ".join("%.6f" % v for v in poly.reshape(-1))
            f.write("%d %s\n" % (cls, flat))


def poly_bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0 = a.min(0); ax1, ay1 = a.max(0)
    bx0, by0 = b.min(0); bx1, by1 = b.max(0)
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def collect_instances(images_dir: str, labels_dir: str) -> Dict[int, List[dict]]:
    """class -> [{image, poly}] over the whole train split."""
    by_class = defaultdict(list)
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(IMG_EXTS):
            continue
        lp = os.path.join(labels_dir, os.path.splitext(fname)[0] + ".txt")
        for cls, poly in read_labels(lp):
            by_class[cls].append({"image": fname, "poly": poly})
    return by_class


def paste_instance(dst_img: np.ndarray, dst_insts: List[Tuple[int, np.ndarray]],
                   src_img: np.ndarray, poly_norm: np.ndarray, cls: int,
                   rng: random.Random, max_iou: float,
                   feather_px: int = 3) -> Tuple[bool, np.ndarray]:
    """Try to paste one masked instance into dst. Returns (ok, new_poly_norm)."""
    sh, sw = src_img.shape[:2]
    dh, dw = dst_img.shape[:2]
    poly_px = poly_norm * [sw, sh]
    x0, y0 = poly_px.min(0).astype(int)
    x1, y1 = np.ceil(poly_px.max(0)).astype(int)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return False, poly_norm

    patch = src_img[y0:y1, x0:x1]
    local = (poly_px - [x0, y0]).astype(np.int32)
    mask = np.zeros(patch.shape[:2], np.uint8)
    cv2.fillPoly(mask, [local], 255)

    scale = rng.uniform(0.8, 1.2)
    pw, ph = max(4, int(patch.shape[1] * scale)), max(4, int(patch.shape[0] * scale))
    if pw >= dw or ph >= dh:
        return False, poly_norm
    patch = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST)

    for _ in range(20):  # rejection-sample a position
        px = rng.randint(0, dw - pw)
        py = rng.randint(0, dh - ph)
        new_poly_px = (poly_px - [x0, y0]) * scale + [px, py]
        new_poly_norm = new_poly_px / [dw, dh]
        if all(poly_bbox_iou(new_poly_norm, ex) <= max_iou
               for _, ex in dst_insts):
            soft = cv2.GaussianBlur(mask, (0, 0), feather_px).astype(np.float32) / 255.0
            soft = soft[..., None] if dst_img.ndim == 3 else soft
            roi = dst_img[py:py + ph, px:px + pw].astype(np.float32)
            dst_img[py:py + ph, px:px + pw] = \
                (roi * (1 - soft) + patch.astype(np.float32) * soft).astype(np.uint8)
            return True, new_poly_norm
    return False, poly_norm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-images", required=True)
    ap.add_argument("--out-labels", required=True)
    ap.add_argument("--max-count", type=int, default=100,
                    help="classes with <= this many instances are augmented")
    ap.add_argument("--copies", type=int, default=20,
                    help="target extra copies per rare instance")
    ap.add_argument("--max-iou", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", default="cp_manifest.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_images, exist_ok=True)
    os.makedirs(args.out_labels, exist_ok=True)

    by_class = collect_instances(args.images, args.labels)
    counts = {c: len(v) for c, v in by_class.items()}
    rare = {c for c, n in counts.items() if n <= args.max_count}
    print("instance counts:", dict(sorted(counts.items())))
    print("rare classes (<=%d):" % args.max_count, sorted(rare))

    host_pool = [f for f in sorted(os.listdir(args.images))
                 if f.lower().endswith(IMG_EXTS)]
    manifest = []
    made = 0
    for cls in sorted(rare):
        for inst_idx, inst in enumerate(by_class[cls]):
            src = cv2.imread(os.path.join(args.images, inst["image"]))
            if src is None:
                continue
            for k in range(args.copies):
                host_name = rng.choice(host_pool)
                stem, ext = os.path.splitext(host_name)
                out_name = "%s_cp%d_%d_%d%s" % (stem, cls, inst_idx, k, ext)
                host = cv2.imread(os.path.join(args.images, host_name))
                if host is None:
                    continue
                insts = read_labels(
                    os.path.join(args.labels, stem + ".txt"))
                ok, new_poly = paste_instance(
                    host, insts, src, inst["poly"], cls, rng, args.max_iou)
                if not ok:
                    continue
                insts.append((cls, new_poly))
                cv2.imwrite(os.path.join(args.out_images, out_name), host)
                write_labels(os.path.join(args.out_labels,
                                          os.path.splitext(out_name)[0] + ".txt"),
                             insts)
                manifest.append({"out": out_name, "host": host_name,
                                 "source": inst["image"], "class": cls,
                                 "copy": k})
                made += 1

    with open(args.manifest, "w") as f:
        json.dump({"seed": args.seed, "max_count": args.max_count,
                   "copies": args.copies, "pastes": manifest}, f, indent=1)
    print("wrote %d augmented images; manifest -> %s" % (made, args.manifest))


if __name__ == "__main__":
    main()
