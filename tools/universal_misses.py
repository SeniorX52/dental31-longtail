#!/usr/bin/env python3
"""Find ground-truth instances that EVERY model in the zoo fails to detect.

WHY. A single model missing an annotation says something about that model. A
dozen independently-trained models -- different architectures, different loss
configurations, different seeds -- all missing the same annotation says
something about the annotation. This is the consensus-outlier argument used to
attribute a large share of worst-case episodes to recorded-vehicle failures
rather than predictor error in the blimp thesis (Noaman 2026, sec. 6.8), moved
across to labels.

On this corpus the finding is severe: with fifteen prediction files, a lenient
0.15 confidence cut and a 0.10 box-IoU hit criterion, 43 % of Periapical
lesion, 33 % of Bone Loss and 26 % of Caries ground truth is invisible to every
model. Those three classes are also the ones sitting near the metric floor. No
architecture can beat labels it cannot see, so this list bounds what any model
on this corpus can achieve, and it is the sampling frame a label audit should
start from.

The criterion is deliberately generous. A 0.10 IoU at a 0.15 confidence cut
asks only "did any model put a box of the right class roughly here", not "did
it segment it well". An instance that fails even this is not a hard example, it
is an absent one.

TWO USES.

  1. Bare, over the whole zoo: the audit list.
  2. With --probe MODEL: hold one model out of the consensus, compute the
     universal misses over the remaining models, and report how many of them
     the held-out model recovers. This splits "too small to resolve at the
     training resolution" from "not visible in the image at all". A
     higher-resolution model that recovers a large share argues resolution; one
     that recovers almost none argues the labels.

Usage:
    python tools/universal_misses.py \\
        --gt data_clean/annotations/instances_valid.json \\
        --preds 'preds/ablation_*_valid.json' \\
        --classes 'Caries,Bone Loss,Periapical lesion' \\
        --out reports/universal_misses_pathology
    # with a held-out probe
    python tools/universal_misses.py ... \\
        --probe preds/ablation_abl_HR1280ft_valid.json
"""
import argparse
import glob
import json
import os
from collections import defaultdict


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def missed_set(pred_file, gt_by, conf, thr):
    """Ann ids in gt_by that this one prediction file fails to hit."""
    try:
        dets = json.load(open(pred_file))
    except Exception:
        return None
    by_key = defaultdict(list)
    for d in dets:
        if d.get("score", 0.0) >= conf:
            by_key[(d["image_id"], d["category_id"])].append(d["bbox"])
    missed = set()
    for key, anns in gt_by.items():
        cand = by_key.get(key, [])
        for a in anns:
            if not any(iou_xywh(a["bbox"], b) >= thr for b in cand):
                missed.add(a["id"])
    return missed


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--preds", required=True,
                    help="glob over prediction json files forming the consensus")
    ap.add_argument("--classes", default=None,
                    help="comma-separated class names; default is every class")
    ap.add_argument("--conf", type=float, default=0.15,
                    help="confidence cut, matching the frozen operating point")
    ap.add_argument("--iou", type=float, default=0.10,
                    help="box IoU counted as a hit; deliberately lenient")
    ap.add_argument("--probe", default=None,
                    help="hold this prediction file OUT of the consensus and "
                         "report how much of the missed set it recovers")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt = json.load(open(args.gt))
    names = {c["id"]: c["name"] for c in gt["categories"]}
    imgname = {im["id"]: im["file_name"] for im in gt["images"]}
    if args.classes:
        want = {s.strip() for s in args.classes.split(",")}
        focus = {cid for cid, n in names.items() if n in want}
        missing = want - {names[c] for c in focus}
        if missing:
            raise SystemExit("class names not in the ground truth: %s" % sorted(missing))
    else:
        focus = set(names)

    gt_by = defaultdict(list)
    for a in gt["annotations"]:
        if a["category_id"] in focus and not a.get("iscrowd", 0):
            gt_by[(a["image_id"], a["category_id"])].append(a)
    ann_of = {a["id"]: a for anns in gt_by.values() for a in anns}

    files = sorted(glob.glob(args.preds))
    probe_abs = os.path.abspath(args.probe) if args.probe else None
    # The probe must not vote in the consensus it is being tested against,
    # otherwise the question is circular.
    files = [f for f in files if os.path.abspath(f) != probe_abs]
    if not files:
        raise SystemExit("no prediction files matched %s" % args.preds)

    per_file = {}
    for f in files:
        m = missed_set(f, gt_by, args.conf, args.iou)
        if m is not None:
            per_file[f] = m
    if not per_file:
        raise SystemExit("no prediction file could be read")

    universal = set.intersection(*per_file.values())

    print("consensus models : %d" % len(per_file))
    print("criterion        : score >= %.2f, box IoU >= %.2f" % (args.conf, args.iou))
    print("\nground truth missed by EVERY model:")
    print("  %-22s %10s %8s %8s" % ("class", "universal", "total", "share"))
    per_class = {}
    for cid in sorted(focus, key=lambda c: names[c]):
        tot = sum(1 for a in ann_of.values() if a["category_id"] == cid)
        uni = sum(1 for i in universal if ann_of[i]["category_id"] == cid)
        if tot == 0:
            continue
        per_class[names[cid]] = {"universal": uni, "total": tot,
                                 "share": uni / tot}
        print("  %-22s %10d %8d %7.1f%%" % (names[cid], uni, tot, 100 * uni / tot))

    result = {"consensus_files": files, "conf": args.conf, "iou": args.iou,
              "per_class": per_class, "n_universal": len(universal)}

    if probe_abs:
        pm = missed_set(args.probe, gt_by, args.conf, args.iou)
        if pm is None:
            print("\nprobe unreadable: %s" % args.probe)
        else:
            recovered = universal - pm
            print("\nPROBE %s" % os.path.basename(args.probe))
            print("  held out of the consensus above")
            print("  recovers %d of the %d universally-missed instances (%.1f %%)"
                  % (len(recovered), len(universal),
                     100 * len(recovered) / max(len(universal), 1)))
            rec_by = defaultdict(int)
            for i in recovered:
                rec_by[names[ann_of[i]["category_id"]]] += 1
            for k in sorted(rec_by):
                print("    %-22s %d" % (k, rec_by[k]))
            print("  Reading: a large share argues these instances are real but "
                  "unresolved at the\n  baseline resolution; a near-zero share "
                  "argues the labels themselves.")
            result["probe"] = {
                "file": args.probe,
                "recovered": len(recovered),
                "of_universal": len(universal),
                "recovered_per_class": dict(rec_by),
            }

    rows = [{"ann_id": i, "image": imgname[ann_of[i]["image_id"]],
             "class": names[ann_of[i]["category_id"]],
             "bbox": ann_of[i]["bbox"], "area": ann_of[i].get("area")}
            for i in sorted(universal)]
    rows.sort(key=lambda r: (r["class"], -(r["area"] or 0)))
    result["instances"] = rows
    with open(args.out + ".json", "w") as f:
        json.dump(result, f, indent=1)
    print("\nwrote %s.json  (%d instances listed, largest first per class)"
          % (args.out, len(rows)))


if __name__ == "__main__":
    main()
