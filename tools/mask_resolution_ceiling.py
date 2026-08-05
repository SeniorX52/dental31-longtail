#!/usr/bin/env python3
"""Measure the mask quality ceiling imposed by the prototype resolution.

YOLOv8-seg does not predict a mask per pixel. It predicts 32 coefficients per
instance and reconstructs the mask as a linear combination of prototype maps
produced at input/4 resolution -- 160x160 for a 640 input. Whatever the loss
function does, the reconstructed mask is band-limited to that grid before it is
upsampled back to image size.

On this dataset the median annotated instance is about 24 px across at input
resolution, which is **6 px** on the prototype grid, and 68 % of instances are
under 8 px there. A 6 px blob has almost no contour to refine, which is a
plausible explanation for why a boundary-shaping objective changes the AP-family
numbers slightly and the direct contour metrics not at all.

This script measures that ceiling directly and without a model. For every
ground-truth instance it:

  1. rasterises the mask at image resolution,
  2. downsamples it to the prototype grid, exactly as the representation would,
  3. thresholds and upsamples it back,
  4. scores the round trip against the original.

The result is the **best Dice and IoU any model could achieve** with perfect
coefficients and perfect prototypes. No loss function can exceed it. Comparing
the ceiling at input/4 against input/2 also quantifies how much is recoverable
by raising the prototype resolution, which is an architectural change rather
than an objective change.

Reported per frequency group and per class, because the ceiling depends on
object size and the tail classes are not the small ones here.

Usage:
    python tools/mask_resolution_ceiling.py \
        --gt data_clean/annotations/instances_valid.json \
        --train-json data_clean/annotations/instances_train.json \
        --imgsz 640 --out reports/mask_resolution_ceiling
"""
import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
from pycocotools.coco import COCO

cv2.setNumThreads(1)

HEAD_MIN = 5000
TAIL_MAX = 100


def round_trip(mask: np.ndarray, grid: int, imgsz: int) -> np.ndarray:
    """Downsample to the prototype grid and back, as the representation does.

    The mask is first placed on the square model canvas (the trainer letterboxes
    to imgsz), reduced to the prototype grid with area averaging, thresholded at
    0.5 as the sigmoid output would be, then restored to the original shape.
    """
    h, w = mask.shape
    canvas = cv2.resize(mask, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    small = cv2.resize(canvas, (grid, grid), interpolation=cv2.INTER_AREA)
    back = cv2.resize((small >= 0.5).astype(np.float32), (imgsz, imgsz),
                      interpolation=cv2.INTER_LINEAR)
    out = cv2.resize(back, (w, h), interpolation=cv2.INTER_LINEAR)
    return (out >= 0.5).astype(np.uint8)


def dice_iou(a: np.ndarray, b: np.ndarray):
    inter = float(np.logical_and(a, b).sum())
    sa, sb = float(a.sum()), float(b.sum())
    union = sa + sb - inter
    if sa + sb == 0:
        return None, None
    return (2 * inter / (sa + sb)), (inter / union if union else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--grids", default="160,320,640",
                    help="prototype grid sizes to test; 160 is what the stock "
                         "head produces at imgsz 640 (input/4)")
    ap.add_argument("--max-instances", type=int, default=0,
                    help="0 = all; set a cap for a quick pass")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    grids = [int(g) for g in args.grids.split(",")]
    coco = COCO(args.gt)
    cat_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    with open(args.train_json) as f:
        tr = json.load(f)
    train_counts = defaultdict(int)
    for a in tr["annotations"]:
        train_counts[a["category_id"]] += 1

    def group_of(c):
        n = train_counts.get(c, 0)
        return "head" if n > HEAD_MIN else ("tail" if n < TAIL_MAX else "mid")

    per_grid = {g: defaultdict(list) for g in grids}   # grid -> cat -> [(dice,iou,side)]
    n_done = 0
    img_ids = sorted(coco.getImgIds())
    for k, img_id in enumerate(img_ids):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=None))
        if not anns:
            continue
        for a in anns:
            m = coco.annToMask(a).astype(np.uint8)
            if m.sum() == 0:
                continue
            side = float(np.sqrt(a.get("area", m.sum())))
            for g in grids:
                rt = round_trip(m.astype(np.float32), g, args.imgsz)
                d, i = dice_iou(m, rt)
                if d is not None:
                    per_grid[g][a["category_id"]].append((d, i, side))
            n_done += 1
            if args.max_instances and n_done >= args.max_instances:
                break
        if args.max_instances and n_done >= args.max_instances:
            break
        if (k + 1) % 250 == 0:
            print("  %d/%d images, %d instances" % (k + 1, len(img_ids), n_done),
                  flush=True)

    result = {"gt": args.gt, "imgsz": args.imgsz, "grids": grids,
              "instances_scored": n_done,
              "note": "ceiling is the best achievable with perfect coefficients; "
                      "no loss function can exceed it",
              "by_grid": {}}
    for g in grids:
        allv = [v for vs in per_grid[g].values() for v in vs]
        if not allv:
            continue
        gr = defaultdict(list)
        for cid, vs in per_grid[g].items():
            for v in vs:
                gr[group_of(cid)].append(v)
        result["by_grid"][str(g)] = {
            "scale_of_input": "input/%d" % (args.imgsz // g) if g <= args.imgsz else "input",
            "mean_dice": float(np.mean([v[0] for v in allv])),
            "mean_iou": float(np.mean([v[1] for v in allv])),
            "median_dice": float(np.median([v[0] for v in allv])),
            "frac_below_iou_075": float(np.mean([v[1] < 0.75 for v in allv])),
            "frac_below_iou_050": float(np.mean([v[1] < 0.50 for v in allv])),
            "by_group": {k: {"mean_dice": float(np.mean([v[0] for v in vs])),
                             "mean_iou": float(np.mean([v[1] for v in vs])),
                             "n": len(vs)}
                         for k, vs in sorted(gr.items())},
            "by_class": {cat_name[cid]: {"mean_dice": float(np.mean([v[0] for v in vs])),
                                         "mean_iou": float(np.mean([v[1] for v in vs])),
                                         "median_side_px": float(np.median([v[2] for v in vs])),
                                         "n": len(vs)}
                         for cid, vs in sorted(per_grid[g].items())},
        }

    with open(args.out + ".json", "w") as f:
        json.dump(result, f, indent=1)

    L = ["# Mask resolution ceiling", "",
         "Best Dice and IoU achievable by *any* model whose masks are reconstructed "
         "on a prototype grid, measured by round-tripping the ground truth through "
         "that grid. No loss function can exceed these numbers.", "",
         "Input size %d. The stock YOLOv8-seg head produces prototypes at input/4."
         % args.imgsz, "",
         "| prototype grid | scale | mean Dice | mean IoU | share below IoU 0.75 | share below IoU 0.50 |",
         "|---|---|---|---|---|---|"]
    for g in grids:
        r = result["by_grid"].get(str(g))
        if not r:
            continue
        L.append("| %dx%d | %s | %.4f | %.4f | %.1f %% | %.1f %% |"
                 % (g, g, r["scale_of_input"], r["mean_dice"], r["mean_iou"],
                    100 * r["frac_below_iou_075"], 100 * r["frac_below_iou_050"]))
    L += ["", "## By frequency group", "",
          "| grid | group | mean Dice | mean IoU | instances |", "|---|---|---|---|---|"]
    for g in grids:
        r = result["by_grid"].get(str(g))
        if not r:
            continue
        for grp in ("head", "mid", "tail"):
            if grp in r["by_group"]:
                v = r["by_group"][grp]
                L.append("| %d | %s | %.4f | %.4f | %d |"
                         % (g, grp, v["mean_dice"], v["mean_iou"], v["n"]))
    base = result["by_grid"].get(str(grids[0]))
    if base:
        L += ["", "## Smallest-object classes at the stock grid", "",
              "| class | median side (px) | mean Dice | mean IoU | n |",
              "|---|---|---|---|---|"]
        rows = sorted(base["by_class"].items(), key=lambda kv: kv[1]["median_side_px"])
        for name, v in rows[:12]:
            L.append("| %s | %.1f | %.4f | %.4f | %d |"
                     % (name, v["median_side_px"], v["mean_dice"], v["mean_iou"], v["n"]))
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(L) + "\n")

    print("\ninstances scored: %d" % n_done)
    for g in grids:
        r = result["by_grid"].get(str(g))
        if r:
            print("  grid %-4d (%s): mean Dice %.4f  mean IoU %.4f  | %.1f%% below IoU 0.75"
                  % (g, r["scale_of_input"], r["mean_dice"], r["mean_iou"],
                     100 * r["frac_below_iou_075"]))
    print("wrote %s.md and %s.json" % (args.out, args.out))


if __name__ == "__main__":
    main()
