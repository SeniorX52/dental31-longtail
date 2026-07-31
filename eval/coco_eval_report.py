#!/usr/bin/env python3
"""Standardized evaluation report: COCO mAP + per-class AP + head/mid/tail groups.

One eval script for every run (baseline, each ablation arm, final model, both
projects) so numbers are always comparable. Emits a markdown table ready to
paste into the ablation write-up, and a machine-readable JSON next to it.

Group definition (by TRAIN-split instance count, supplied via --train-json so
the grouping never peeks at test statistics):
    head  > 5000    mid  100-5000    tail  < 100

Classes with fewer than 10 instances in the eval split are additionally
flagged `unstable` in the output: their single-class AP is reported for
transparency but group-level AP is the claimable number at those counts.

Usage:
    python eval/coco_eval_report.py \
        --gt annotations/instances_test.json \
        --dt runs/final/test_predictions.json \
        --train-json annotations/instances_train.json \
        --iou-type bbox --out report_test_bbox
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HEAD_MIN = 5000
TAIL_MAX = 100
UNSTABLE_EVAL_COUNT = 10


def per_class_ap(coco_eval: COCOeval, cat_ids):
    """AP@[.5:.95] and AP@.5 per category from the accumulated precision array."""
    p = coco_eval.eval["precision"]  # [T_iou, R_recall, K_cat, A_area, M_maxdet]
    out = {}
    for k, cat_id in enumerate(cat_ids):
        pr_all = p[:, :, k, 0, -1]
        pr_50 = p[0, :, k, 0, -1]
        out[cat_id] = (
            float(np.mean(pr_all[pr_all > -1])) if (pr_all > -1).any() else float("nan"),
            float(np.mean(pr_50[pr_50 > -1])) if (pr_50 > -1).any() else float("nan"),
        )
    return out


def instance_counts(json_path):
    with open(json_path) as f:
        data = json.load(f)
    counts = defaultdict(int)
    for ann in data["annotations"]:
        counts[ann["category_id"]] += 1
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="ground-truth COCO json (eval split)")
    ap.add_argument("--dt", required=True, help="COCO-format detections json")
    ap.add_argument("--train-json", required=True,
                    help="TRAIN split json; defines head/mid/tail grouping")
    ap.add_argument("--iou-type", choices=["bbox", "segm"], default="bbox")
    ap.add_argument("--out", required=True, help="basename for .md and .json output")
    args = ap.parse_args()

    coco_gt = COCO(args.gt)
    coco_dt = coco_gt.loadRes(args.dt)
    ev = COCOeval(coco_gt, coco_dt, args.iou_type)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    cat_ids = sorted(coco_gt.getCatIds())
    names = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}
    ap_by_cat = per_class_ap(ev, cat_ids)
    train_counts = instance_counts(args.train_json)
    eval_counts = instance_counts(args.gt)

    def group_of(cid):
        n = train_counts.get(cid, 0)
        return "head" if n > HEAD_MIN else ("tail" if n < TAIL_MAX else "mid")

    rows, groups = [], defaultdict(list)
    for cid in cat_ids:
        ap_all, ap50 = ap_by_cat[cid]
        g = group_of(cid)
        unstable = eval_counts.get(cid, 0) < UNSTABLE_EVAL_COUNT
        rows.append({"id": cid, "class": names[cid], "group": g,
                     "train_instances": train_counts.get(cid, 0),
                     "eval_instances": eval_counts.get(cid, 0),
                     "AP": ap_all, "AP50": ap50, "unstable": unstable})
        if not np.isnan(ap_all):
            groups[g].append(ap_all)

    result = {
        "iou_type": args.iou_type,
        "coco_stats": {  # the 12 standard numbers, keyed for the write-up
            "mAP": ev.stats[0], "AP50": ev.stats[1], "AP75": ev.stats[2],
            "AP_small": ev.stats[3], "AP_medium": ev.stats[4], "AP_large": ev.stats[5],
            "AR1": ev.stats[6], "AR10": ev.stats[7], "AR100": ev.stats[8],
            "AR_small": ev.stats[9], "AR_medium": ev.stats[10], "AR_large": ev.stats[11],
        },
        "group_AP": {g: float(np.mean(v)) for g, v in sorted(groups.items())},
        "per_class": rows,
        "grouping": {"head_min": HEAD_MIN, "tail_max": TAIL_MAX,
                     "train_json": args.train_json},
    }
    with open(args.out + ".json", "w") as f:
        json.dump(result, f, indent=1)

    lines = ["# %s eval — %s" % (args.iou_type, args.dt), "",
             "| metric | value |", "|---|---|"]
    for k in ("mAP", "AP50", "AP75"):
        lines.append("| %s | %.4f |" % (k, result["coco_stats"][k]))
    for g in ("head", "mid", "tail"):
        if g in result["group_AP"]:
            lines.append("| %s AP | %.4f |" % (g, result["group_AP"][g]))
    lines += ["", "| class | group | train n | eval n | AP | AP50 | |",
              "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["train_instances"]):
        lines.append("| %s | %s | %d | %d | %.4f | %.4f | %s |" % (
            r["class"], r["group"], r["train_instances"], r["eval_instances"],
            r["AP"], r["AP50"], "unstable" if r["unstable"] else ""))
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\nwrote %s.md and %s.json" % (args.out, args.out))
    print("group AP:", {g: round(v, 4) for g, v in result["group_AP"].items()})


if __name__ == "__main__":
    main()
