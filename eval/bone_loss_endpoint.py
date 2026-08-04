#!/usr/bin/env python3
"""Downstream clinical endpoint: bone-loss quantification error.

A better contour metric is not by itself evidence of clinical utility. This
script computes the quantity a clinician would actually read off the
segmentation and asks whether ITS error goes down.

WHICH ENDPOINT, AND WHY THIS ONE
Three endpoint families are used for periodontal bone loss:

  area-based               total area of the bone-loss region
  distance-based           CEJ to crestal-bone distance, in mm
  normalized bone level    that distance as a fraction of root length

Only the **area-based** endpoint is computable from this dataset. The
distance-based and normalized endpoints both require cemento-enamel-junction
and root-apex landmarks, and the annotation set contains neither -- it provides
a `Bone Loss` region polygon and nothing else. Nor do these panoramic images
carry pixel-spacing metadata, so no measurement can be expressed in millimetres.
Reporting a distance endpoint here would mean inventing landmarks, so the area
endpoint is what is reported and the limitation is stated rather than hidden.

Areas are normalised by image area and reported in percent, because the
radiographs vary in resolution and a raw pixel count would confound endpoint
error with image size.

WHAT IS REPORTED
Over images that contain ground-truth bone loss:
  MAE           mean absolute error of the area estimate (percentage points)
  bias          mean signed error; positive means over-segmentation
  MAPE          median absolute error relative to the true area
  r, rho        Pearson and Spearman correlation of predicted vs true area
  LoA           Bland-Altman 95% limits of agreement (bias +- 1.96 SD)
Over images with NO ground-truth bone loss:
  false-positive area, i.e. bone loss reported where there is none. Omitting
  these images would let a model that hallucinates bone loss on healthy
  radiographs post a better endpoint error than one that does not.

Confidence intervals bootstrap over images. Two prediction files can be
compared directly with --dt-b, in which case the PAIRED per-image difference is
bootstrapped, which is the correct test for "did the error decrease".

Usage:
    python eval/bone_loss_endpoint.py \
        --gt data_clean/annotations/instances_test.json \
        --dt preds/final_S1c_test.json --dt-b preds/final_S2_test.json \
        --conf 0.15 --out reports/bone_loss_endpoint_test
"""
import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
from pycocotools import mask as maskutil
from pycocotools.coco import COCO

cv2.setNumThreads(1)
DEFAULT_CLASS = "Bone Loss"


def decode(seg, h, w):
    if isinstance(seg, list):
        return maskutil.decode(maskutil.merge(maskutil.frPyObjects(seg, h, w)))
    if isinstance(seg["counts"], list):
        return maskutil.decode(maskutil.frPyObjects(seg, h, w))
    return maskutil.decode(seg)


def areas_for(gt_json, dt_json, class_name, conf):
    """-> dict image_id -> (gt_area_pct, dt_area_pct)."""
    coco = COCO(gt_json)
    cid = [c["id"] for c in coco.loadCats(coco.getCatIds())
           if c["name"].strip().lower() == class_name.strip().lower()]
    if not cid:
        raise SystemExit("class %r not found in %s" % (class_name, gt_json))
    cid = cid[0]

    with open(dt_json) as f:
        dets = json.load(f)
    if isinstance(dets, dict):
        dets = dets["annotations"]
    by_img = defaultdict(list)
    for d in dets:
        if d.get("category_id") == cid and d.get("score", 1.0) >= conf and "segmentation" in d:
            by_img[d["image_id"]].append(d)

    out = {}
    for img_id in sorted(coco.getImgIds()):
        info = coco.loadImgs(img_id)[0]
        h, w = info["height"], info["width"]
        px = float(h * w)
        g = np.zeros((h, w), np.uint8)
        for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[cid], iscrowd=None)):
            g |= coco.annToMask(a).astype(np.uint8)
        d = np.zeros((h, w), np.uint8)
        for a in by_img.get(img_id, []):
            d |= decode(a["segmentation"], h, w).astype(np.uint8)
        out[img_id] = (100.0 * g.sum() / px, 100.0 * d.sum() / px)
    return out


def stats(pairs):
    """pairs: list of (gt_pct, dt_pct) for images WITH ground-truth bone loss."""
    if not pairs:
        return None
    g = np.array([p[0] for p in pairs])
    d = np.array([p[1] for p in pairs])
    err = d - g
    out = {
        "n_images": len(pairs),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
        "mape_median": float(np.median(np.abs(err) / np.maximum(g, 1e-9)) * 100),
        "loa_low": float(err.mean() - 1.96 * err.std(ddof=1)) if len(err) > 1 else None,
        "loa_high": float(err.mean() + 1.96 * err.std(ddof=1)) if len(err) > 1 else None,
        "mean_gt_area_pct": float(g.mean()),
        "mean_pred_area_pct": float(d.mean()),
    }
    if len(g) > 2 and g.std() > 0 and d.std() > 0:
        out["pearson_r"] = float(np.corrcoef(g, d)[0, 1])
        rg = np.argsort(np.argsort(g)).astype(float)
        rd = np.argsort(np.argsort(d)).astype(float)
        out["spearman_rho"] = float(np.corrcoef(rg, rd)[0, 1])
    else:
        out["pearson_r"] = out["spearman_rho"] = None
    return out


def boot_ci(values, n_boot, alpha, seed, fn=np.mean):
    if len(values) < 2 or n_boot <= 0:
        return None
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    draws = [float(fn(v[rng.integers(0, len(v), len(v))])) for _ in range(n_boot)]
    return [float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dt", required=True, help="predictions for model A")
    ap.add_argument("--dt-b", help="optional second model, paired comparison")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--class-name", default=DEFAULT_CLASS)
    ap.add_argument("--conf", type=float, default=0.15,
                    help="confidence cut; must be the one frozen on validation")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--boot-alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("computing %s areas for %s ..." % (args.class_name, args.label_a), flush=True)
    A = areas_for(args.gt, args.dt, args.class_name, args.conf)
    B = areas_for(args.gt, args.dt_b, args.class_name, args.conf) if args.dt_b else None
    if B:
        print("computing %s areas for %s ..." % (args.class_name, args.label_b), flush=True)

    pos = [i for i, (g, _) in A.items() if g > 0]
    neg = [i for i, (g, _) in A.items() if g == 0]

    def model_block(M):
        s = stats([M[i] for i in pos])
        fp = np.array([M[i][1] for i in neg])
        s["n_images_without_bone_loss"] = len(neg)
        s["false_positive_area_pct_mean"] = float(fp.mean()) if len(fp) else None
        s["false_positive_rate"] = float((fp > 0).mean()) if len(fp) else None
        errs = [abs(M[i][1] - M[i][0]) for i in pos]
        s["mae_ci"] = boot_ci(errs, args.boot, args.boot_alpha, args.seed)
        return s

    result = {
        "endpoint": "area of the %s region, percent of image area" % args.class_name,
        "not_computable": ["CEJ-to-crestal-bone distance (no landmark annotations)",
                           "normalized bone level (no landmark annotations)",
                           "any millimetre measurement (no pixel-spacing metadata)"],
        "conf": args.conf, "gt": args.gt,
        "images_with_bone_loss": len(pos), "images_without": len(neg),
        "bootstrap": {"resamples": args.boot, "alpha": args.boot_alpha,
                      "resampled_unit": "image", "seed": args.seed},
        args.label_a: model_block(A),
    }
    if B:
        result[args.label_b] = model_block(B)
        pa = np.array([abs(A[i][1] - A[i][0]) for i in pos])
        pb = np.array([abs(B[i][1] - B[i][0]) for i in pos])
        diff = pb - pa                       # negative = B has smaller error
        result["paired_comparison"] = {
            "definition": "per-image |error| of %s minus |error| of %s; "
                          "negative means %s is closer to the truth"
                          % (args.label_b, args.label_a, args.label_b),
            "mean_difference": float(diff.mean()),
            "ci": boot_ci(diff, args.boot, args.boot_alpha, args.seed),
            "fraction_images_improved": float((diff < 0).mean()),
        }
    with open(args.out + ".json", "w") as f:
        json.dump(result, f, indent=1)

    L = ["# Downstream clinical endpoint — %s area" % args.class_name, "",
         "Endpoint: **area of the %s region as a percentage of image area**."
         % args.class_name, "",
         "Not computable from this annotation set, and therefore not reported:",
         "", "- CEJ-to-crestal-bone distance — no landmark annotations",
         "- normalized bone level — no landmark annotations",
         "- any measurement in millimetres — no pixel-spacing metadata", "",
         "%d of %d evaluation images contain ground-truth %s; the remaining %d "
         "are used to measure bone loss reported where there is none."
         % (len(pos), len(pos) + len(neg), args.class_name, len(neg)), "",
         "Confidence cut %.3f (frozen on validation). Intervals are %d%% "
         "percentile bootstrap over images (%d resamples)."
         % (args.conf, round(100 * (1 - args.boot_alpha)), args.boot), "",
         "| metric | %s |" % args.label_a + (" %s |" % args.label_b if B else ""),
         "|---|---|" + ("---|" if B else "")]

    def cell(s, k, p=3):
        v = s.get(k)
        return "n/a" if v is None else "%.*f" % (p, v)

    for k, lab in (("mae", "MAE (pp of image area)"), ("bias", "bias (signed)"),
                   ("mape_median", "median APE (%)"), ("pearson_r", "Pearson r"),
                   ("spearman_rho", "Spearman rho"),
                   ("false_positive_area_pct_mean", "FP area on healthy images"),
                   ("false_positive_rate", "fraction of healthy images with FP")):
        row = "| %s | %s |" % (lab, cell(result[args.label_a], k))
        if B:
            row += " %s |" % cell(result[args.label_b], k)
        L.append(row)
    sa = result[args.label_a]
    L += ["", "%s Bland-Altman limits of agreement: [%s, %s] pp."
          % (args.label_a, cell(sa, "loa_low"), cell(sa, "loa_high"))]
    if B:
        sb = result[args.label_b]
        L.append("%s Bland-Altman limits of agreement: [%s, %s] pp."
                 % (args.label_b, cell(sb, "loa_low"), cell(sb, "loa_high")))
        pc = result["paired_comparison"]
        ci = pc["ci"]
        verdict = ("no detectable difference" if ci is None or (ci[0] < 0 < ci[1])
                   else ("%s reduces the endpoint error" % args.label_b if ci[1] < 0
                         else "%s increases the endpoint error" % args.label_b))
        L += ["", "## Paired comparison", "",
              "Per-image absolute endpoint error, %s minus %s "
              "(negative favours %s):" % (args.label_b, args.label_a, args.label_b), "",
              "- mean difference **%.4f pp**" % pc["mean_difference"],
              "- 95%% CI %s" % ("n/a" if ci is None else "[%.4f, %.4f]" % (ci[0], ci[1])),
              "- improved on %.1f%% of images" % (100 * pc["fraction_images_improved"]),
              "", "**Verdict: %s.**" % verdict]
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(L) + "\n")

    print("\n%s: MAE %.4f pp, bias %+.4f, r %s"
          % (args.label_a, sa["mae"], sa["bias"], cell(sa, "pearson_r")))
    if B:
        print("%s: MAE %.4f pp, bias %+.4f, r %s"
              % (args.label_b, sb["mae"], sb["bias"], cell(sb, "pearson_r")))
        print("paired mean difference %.4f pp, CI %s"
              % (result["paired_comparison"]["mean_difference"],
                 result["paired_comparison"]["ci"]))
    print("wrote %s.md and %s.json" % (args.out, args.out))


if __name__ == "__main__":
    main()
