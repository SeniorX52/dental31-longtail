#!/usr/bin/env python3
"""Region and contour metrics with image-level bootstrap confidence intervals.

COCO mAP answers "did the model find the thing, and roughly where". It does not
answer "is the outline right", which is the claim the boundary-aware objective
actually makes, and it is not the metric family a medical-imaging reviewer will
look for. This script reports, per class and pooled:

    Dice, IoU                region overlap
    boundary F-score         contour agreement within a tolerance
    HD95, ASSD               contour distance, in pixels

EVALUATION PROTOCOL (semantic, per class per image)
For each image and each class, all ground-truth instances of that class are
unioned into one binary mask and all predicted instances of that class above
--conf are unioned into another. Metrics are computed between those two masks.

Why union rather than per-instance matching: the clinical quantity these masks
feed is area- or distance-based over a region (how much bone is lost, how far
the crestal bone sits from the CEJ), not instance identity. Pooling to a
per-class region measures the quantity the downstream endpoint actually
consumes. A per-instance variant is available with --protocol matched, which
greedily pairs predictions to ground truth at --match-iou and scores the pairs;
it is reported as a secondary table because unmatched instances make the
distance metrics conditional on the matcher.

THRESHOLD. Union masks require a confidence cut, unlike mAP. --conf must be
chosen on the validation split and then frozen; the value used is recorded in
the output so a test-split number can never be quietly re-tuned.

EMPTY-MASK HANDLING -- the part that silently corrupts contour metrics.
Distance metrics are undefined when either mask is empty, so a model that
predicts nothing for a class would otherwise score a perfect HD95 on that
class by having no cases left to average. This script never does that:

    both non-empty   -> every metric contributes
    GT empty, pred empty     -> case dropped (the class is simply absent)
    GT non-empty, pred empty -> counted as a MISS:     Dice/IoU/BF = 0
    GT empty, pred non-empty -> counted as FALSE ALARM: Dice/IoU/BF = 0
    HD95 and ASSD are averaged over BOTH-non-empty cases only, and the counts
    n_both / n_miss / n_false_alarm are reported beside them so the denominator
    is always visible.

CONFIDENCE INTERVALS are bootstrapped over IMAGES, not over per-class records:
records from the same radiograph are not independent, so resampling records
would understate the interval. Percentile intervals, --boot resamples,
--boot-alpha two-sided level.

Usage:
    python eval/contour_metrics.py \
        --gt data_clean/annotations/instances_test.json \
        --dt preds/final_S2_test.json \
        --train-json data_clean/annotations/instances_train.json \
        --conf 0.25 --out reports/final_S2_test_contour
"""
import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
from pycocotools import mask as maskutil
from pycocotools.coco import COCO

cv2.setNumThreads(1)

HEAD_MIN = 5000
TAIL_MAX = 100
UNSTABLE_EVAL_COUNT = 10
# DAVIS convention: boundary tolerance is 0.75% of the image diagonal, so the
# F-score does not silently get easier on larger radiographs.
BOUNDARY_TOL_FRAC = 0.0075
_K3 = np.ones((3, 3), np.uint8)


def boundary_of(mask: np.ndarray) -> np.ndarray:
    """1-pixel inner boundary: mask minus its 3x3 erosion."""
    return cv2.subtract(mask, cv2.erode(mask, _K3, iterations=1))


def dist_to(boundary: np.ndarray) -> np.ndarray:
    """Euclidean distance from every pixel to the nearest boundary pixel."""
    # distanceTransform measures distance to the nearest ZERO pixel, so the
    # boundary must be the zero set.
    return cv2.distanceTransform((boundary == 0).astype(np.uint8),
                                 cv2.DIST_L2, 3)


def pair_metrics(gt: np.ndarray, dt: np.ndarray, tol: float) -> dict:
    """All five metrics for one pair of binary masks (uint8, 0/1)."""
    g, d = gt.sum(), dt.sum()
    inter = int(np.logical_and(gt, dt).sum())
    union = int(np.logical_or(gt, dt).sum())
    out = {"dice": (2.0 * inter / (g + d)) if (g + d) else None,
           "iou": (inter / union) if union else None,
           "bf": None, "hd95": None, "assd": None,
           "gt_px": int(g), "dt_px": int(d)}
    if g == 0 or d == 0:
        # one side empty: overlap metrics are legitimately 0, distances are
        # undefined and are left as None rather than imputed
        if g == 0 and d == 0:
            return out
        out["dice"] = 0.0
        out["iou"] = 0.0
        out["bf"] = 0.0
        return out

    bg, bd = boundary_of(gt), boundary_of(dt)
    if bg.sum() == 0 or bd.sum() == 0:
        return out
    dist_to_g, dist_to_d = dist_to(bg), dist_to(bd)
    d_dt_to_gt = dist_to_g[bd > 0]     # each predicted boundary px -> nearest GT boundary
    d_gt_to_dt = dist_to_d[bg > 0]     # each GT boundary px -> nearest predicted boundary

    precision = float((d_dt_to_gt <= tol).mean())
    recall = float((d_gt_to_dt <= tol).mean())
    out["bf"] = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    # symmetric HD95 as the max of the two directed 95th percentiles (the
    # conservative reading; pooling both directions into one percentile lets a
    # large one-sided error hide behind the other side's mass)
    out["hd95"] = float(max(np.percentile(d_dt_to_gt, 95),
                            np.percentile(d_gt_to_dt, 95)))
    out["assd"] = float((d_dt_to_gt.sum() + d_gt_to_dt.sum()) /
                        (len(d_dt_to_gt) + len(d_gt_to_dt)))
    return out


def union_mask(coco: COCO, anns, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), np.uint8)
    for a in anns:
        out |= coco.annToMask(a).astype(np.uint8)
    return out


def dt_mask(ann, h: int, w: int) -> np.ndarray:
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = maskutil.frPyObjects(seg, h, w)
        rle = maskutil.merge(rles)
    elif isinstance(seg["counts"], list):
        rle = maskutil.frPyObjects(seg, h, w)
    else:
        rle = seg
    return maskutil.decode(rle).astype(np.uint8)


def collect_records(gt_json, dt_json, conf, protocol, match_iou):
    """-> list of per-(image, class) records; each carries its image id."""
    coco = COCO(gt_json)
    with open(dt_json) as f:
        dets = json.load(f)
    if isinstance(dets, dict):
        dets = dets["annotations"]
    dets = [d for d in dets if d.get("score", 1.0) >= conf and "segmentation" in d]
    by_img_cat = defaultdict(list)
    for d in dets:
        by_img_cat[(d["image_id"], d["category_id"])].append(d)

    records = []
    img_ids = sorted(coco.getImgIds())
    for n, img_id in enumerate(img_ids):
        info = coco.loadImgs(img_id)[0]
        h, w = info["height"], info["width"]
        tol = BOUNDARY_TOL_FRAC * float(np.hypot(h, w))
        gt_by_cat = defaultdict(list)
        for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=None)):
            gt_by_cat[a["category_id"]].append(a)
        cats = set(gt_by_cat) | {c for (i, c) in by_img_cat if i == img_id}
        for cat in sorted(cats):
            g_anns, d_anns = gt_by_cat.get(cat, []), by_img_cat.get((img_id, cat), [])
            if protocol == "semantic":
                gm = union_mask(coco, g_anns, h, w) if g_anns else np.zeros((h, w), np.uint8)
                dm = np.zeros((h, w), np.uint8)
                for d in d_anns:
                    dm |= dt_mask(d, h, w)
                if gm.sum() == 0 and dm.sum() == 0:
                    continue
                m = pair_metrics(gm, dm, tol)
                m.update(image_id=img_id, category_id=cat)
                records.append(m)
            else:                                    # matched-instance protocol
                gms = [coco.annToMask(a).astype(np.uint8) for a in g_anns]
                dms = [(d.get("score", 1.0), dt_mask(d, h, w)) for d in d_anns]
                dms.sort(key=lambda t: -t[0])
                used = set()
                for _, dm in dms:
                    best, best_j = 0.0, -1
                    for j, gm in enumerate(gms):
                        if j in used:
                            continue
                        u = np.logical_or(gm, dm).sum()
                        iou = (np.logical_and(gm, dm).sum() / u) if u else 0.0
                        if iou > best:
                            best, best_j = iou, j
                    if best >= match_iou and best_j >= 0:
                        used.add(best_j)
                        m = pair_metrics(gms[best_j], dm, tol)
                        m.update(image_id=img_id, category_id=cat)
                        records.append(m)
        if (n + 1) % 250 == 0:
            print("  %d/%d images" % (n + 1, len(img_ids)), flush=True)
    return coco, records


def aggregate(records, cat_ids):
    """Per-class and macro aggregates from a record list."""
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category_id"]].append(r)

    def agg(rs):
        both = [r for r in rs if r["hd95"] is not None]
        overlap = [r for r in rs if r["dice"] is not None]
        miss = sum(1 for r in rs if r["gt_px"] > 0 and r["dt_px"] == 0)
        fa = sum(1 for r in rs if r["gt_px"] == 0 and r["dt_px"] > 0)
        return {
            "dice": float(np.mean([r["dice"] for r in overlap])) if overlap else None,
            "iou": float(np.mean([r["iou"] for r in overlap])) if overlap else None,
            "bf": float(np.mean([r["bf"] for r in overlap if r["bf"] is not None]))
                  if any(r["bf"] is not None for r in overlap) else None,
            "hd95": float(np.mean([r["hd95"] for r in both])) if both else None,
            "assd": float(np.mean([r["assd"] for r in both])) if both else None,
            "n_cases": len(rs), "n_both": len(both),
            "n_miss": miss, "n_false_alarm": fa,
        }

    per_class = {c: agg(by_cat[c]) for c in cat_ids if by_cat.get(c)}
    # macro = unweighted mean over classes that have at least one case, so a
    # single high-frequency class cannot carry the headline number
    macro = {}
    for k in ("dice", "iou", "bf", "hd95", "assd"):
        vals = [v[k] for v in per_class.values() if v[k] is not None]
        macro[k] = float(np.mean(vals)) if vals else None
    macro["n_classes"] = len(per_class)
    micro = agg(records)
    return per_class, macro, micro


def bootstrap(records, cat_ids, n_boot, alpha, seed):
    """Percentile CIs by resampling IMAGES with replacement."""
    by_img = defaultdict(list)
    for r in records:
        by_img[r["image_id"]].append(r)
    img_ids = sorted(by_img)
    rng = np.random.default_rng(seed)
    keys = ("dice", "iou", "bf", "hd95", "assd")
    macro_draws = {k: [] for k in keys}
    class_draws = defaultdict(lambda: {k: [] for k in keys})
    for b in range(n_boot):
        pick = rng.integers(0, len(img_ids), len(img_ids))
        res = []
        for i in pick:
            res.extend(by_img[img_ids[i]])
        pc, mac, _ = aggregate(res, cat_ids)
        for k in keys:
            if mac[k] is not None:
                macro_draws[k].append(mac[k])
            for c, v in pc.items():
                if v[k] is not None:
                    class_draws[c][k].append(v[k])
        if (b + 1) % 50 == 0:
            print("  bootstrap %d/%d" % (b + 1, n_boot), flush=True)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def ci(d):
        return {k: ([float(np.percentile(v, lo)), float(np.percentile(v, hi))]
                    if len(v) > 1 else None) for k, v in d.items()}
    return ci(macro_draws), {c: ci(v) for c, v in class_draws.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dt", required=True)
    ap.add_argument("--train-json", required=True, help="defines head/mid/tail grouping")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence cut; MUST be chosen on validation and frozen")
    ap.add_argument("--protocol", choices=["semantic", "matched"], default="semantic")
    ap.add_argument("--match-iou", type=float, default=0.5,
                    help="pairing threshold for --protocol matched")
    ap.add_argument("--boot", type=int, default=200, help="bootstrap resamples; 0 = off")
    ap.add_argument("--boot-alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("computing per-image contour metrics (protocol=%s, conf=%.3f) ..."
          % (args.protocol, args.conf), flush=True)
    coco, records = collect_records(args.gt, args.dt, args.conf,
                                    args.protocol, args.match_iou)
    cat_ids = sorted(coco.getCatIds())
    names = {c["id"]: c["name"] for c in coco.loadCats(cat_ids)}
    per_class, macro, micro = aggregate(records, cat_ids)

    with open(args.train_json) as f:
        tr = json.load(f)
    train_counts = defaultdict(int)
    for a in tr["annotations"]:
        train_counts[a["category_id"]] += 1

    def group_of(c):
        n = train_counts.get(c, 0)
        return "head" if n > HEAD_MIN else ("tail" if n < TAIL_MAX else "mid")

    macro_ci, class_ci = ({}, {})
    if args.boot > 0:
        print("bootstrapping over images (%d resamples) ..." % args.boot, flush=True)
        macro_ci, class_ci = bootstrap(records, cat_ids, args.boot,
                                       args.boot_alpha, args.seed)

    groups = defaultdict(lambda: defaultdict(list))
    for c, v in per_class.items():
        for k in ("dice", "iou", "bf", "hd95", "assd"):
            if v[k] is not None:
                groups[group_of(c)][k].append(v[k])
    group_agg = {g: {k: float(np.mean(v)) for k, v in d.items()}
                 for g, d in groups.items()}

    result = {
        "protocol": args.protocol, "conf": args.conf,
        "match_iou": args.match_iou if args.protocol == "matched" else None,
        "boundary_tolerance": "%.4f x image diagonal" % BOUNDARY_TOL_FRAC,
        "distance_units": "pixels at native image resolution",
        "gt": args.gt, "dt": args.dt,
        "bootstrap": {"resamples": args.boot, "alpha": args.boot_alpha,
                      "resampled_unit": "image", "seed": args.seed},
        "macro": macro, "macro_ci": macro_ci,
        "micro_pooled": micro,
        "group": group_agg,
        "per_class": {
            names[c]: dict(v, group=group_of(c),
                           train_instances=train_counts.get(c, 0),
                           unstable=v["n_cases"] < UNSTABLE_EVAL_COUNT,
                           ci=class_ci.get(c, {}))
            for c, v in sorted(per_class.items())},
    }
    with open(args.out + ".json", "w") as f:
        json.dump(result, f, indent=1)

    def fmt(v, ci=None, p=4):
        if v is None:
            return "n/a"
        s = "%.*f" % (p, v)
        if ci:
            s += " [%.*f, %.*f]" % (p, ci[0], p, ci[1])
        return s

    L = ["# Region and contour metrics — %s" % args.dt, "",
         "Protocol **%s**, confidence cut **%.3f** (chosen on validation, frozen)."
         % (args.protocol, args.conf),
         "Boundary tolerance %.2f%% of the image diagonal. Distances in pixels."
         % (100 * BOUNDARY_TOL_FRAC),
         "Intervals are %d%% percentile bootstrap over **images** (%d resamples)."
         % (round(100 * (1 - args.boot_alpha)), args.boot), "",
         "## Headline (macro-average over classes)", "",
         "| metric | value |", "|---|---|"]
    for k, lab in (("dice", "Dice"), ("iou", "IoU"), ("bf", "boundary F-score"),
                   ("hd95", "HD95 (px)"), ("assd", "ASSD (px)")):
        L.append("| %s | %s |" % (lab, fmt(macro[k], macro_ci.get(k))))
    L += ["", "Cases: %d scored, %d with both masks non-empty, %d misses, "
              "%d false alarms. HD95 and ASSD average over the both-non-empty "
              "cases only; Dice, IoU and boundary F count misses and false "
              "alarms as 0."
          % (micro["n_cases"], micro["n_both"], micro["n_miss"],
             micro["n_false_alarm"]), "",
          "## By frequency group", "",
          "| group | Dice | IoU | bF | HD95 | ASSD |", "|---|---|---|---|---|---|"]
    for g in ("head", "mid", "tail"):
        if g in group_agg:
            d = group_agg[g]
            L.append("| %s | %s | %s | %s | %s | %s |"
                     % (g, fmt(d.get("dice")), fmt(d.get("iou")), fmt(d.get("bf")),
                        fmt(d.get("hd95"), p=2), fmt(d.get("assd"), p=2)))
    L += ["", "## Per class", "",
          "| class | group | train n | cases | both | miss | FA | Dice | IoU | bF | HD95 | ASSD | |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in sorted(result["per_class"].items(),
                          key=lambda kv: -kv[1]["train_instances"]):
        L.append("| %s | %s | %d | %d | %d | %d | %d | %s | %s | %s | %s | %s | %s |"
                 % (name, v["group"], v["train_instances"], v["n_cases"],
                    v["n_both"], v["n_miss"], v["n_false_alarm"],
                    fmt(v["dice"]), fmt(v["iou"]), fmt(v["bf"]),
                    fmt(v["hd95"], p=2), fmt(v["assd"], p=2),
                    "unstable" if v["unstable"] else ""))
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(L) + "\n")

    print("\nmacro  Dice %s  IoU %s  bF %s  HD95 %s  ASSD %s"
          % (fmt(macro["dice"]), fmt(macro["iou"]), fmt(macro["bf"]),
             fmt(macro["hd95"], p=2), fmt(macro["assd"], p=2)))
    print("cases %d | both %d | miss %d | false alarm %d"
          % (micro["n_cases"], micro["n_both"], micro["n_miss"], micro["n_false_alarm"]))
    print("wrote %s.md and %s.json" % (args.out, args.out))


if __name__ == "__main__":
    main()
