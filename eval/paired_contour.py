#!/usr/bin/env python3
"""Paired contour comparison of two models on the SAME cases.

Why this exists. HD95 and ASSD are undefined when either mask is empty, so
`contour_metrics.py` averages them over the cases where both masks are
non-empty -- correctly, but that denominator differs per model. A model that
misses a hard structure entirely drops that case from its own average and can
post better distances by predicting less. The two arms being compared here
differ by 39 such cases, which is small but is exactly the objection a reviewer
raises first, and it has to be answered rather than argued away.

This script recomputes both models, intersects their case sets down to the
(image, class) pairs where BOTH produced a non-empty mask against non-empty
ground truth, and compares them on that common set only. Every number is then
computed over identical cases, so a difference cannot come from a difference in
which cases were scored.

The comparison is paired: the per-case difference is bootstrapped over images,
which is both the correct construction for "did this model do better on the
same case" and far tighter than comparing two independent intervals.

Coverage is reported alongside, because equal-footing distances say nothing
about how often each model found the structure at all. A model that wins on the
common set while missing more cases has traded recall for contour quality, and
the write-up has to say so.

Usage:
    python eval/paired_contour.py \
        --gt data_clean/annotations/instances_valid.json \
        --dt-a preds/ablation_S1c_valid.json --label-a S1c \
        --dt-b preds/ablation_S2_valid.json  --label-b S2 \
        --conf 0.15 --out reports/paired_contour_S1c_S2_valid
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from contour_metrics import collect_records  # noqa: E402

METRICS = [("dice", "Dice", +1), ("iou", "IoU", +1), ("bf", "boundary F", +1),
           ("hd95", "HD95 (px)", -1), ("assd", "ASSD (px)", -1)]


def index(records):
    return {(r["image_id"], r["category_id"]): r for r in records}


def boot_paired(diffs_by_image, n_boot, alpha, seed):
    """Bootstrap the mean paired difference, resampling IMAGES."""
    imgs = sorted(diffs_by_image)
    if len(imgs) < 2 or n_boot <= 0:
        return None
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(imgs), len(imgs))
        vals = [v for i in pick for v in diffs_by_image[imgs[i]]]
        if vals:
            draws.append(float(np.mean(vals)))
    if not draws:
        return None
    return [float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dt-a", required=True)
    ap.add_argument("--dt-b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--boot-alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("scoring %s ..." % args.label_a, flush=True)
    _, ra = collect_records(args.gt, args.dt_a, args.conf, "semantic", 0.5)
    print("scoring %s ..." % args.label_b, flush=True)
    _, rb = collect_records(args.gt, args.dt_b, args.conf, "semantic", 0.5)
    A, B = index(ra), index(rb)

    keys_all = set(A) | set(B)
    # common set: both models produced a non-empty prediction against non-empty
    # ground truth, so every distance is defined for both
    common = [k for k in keys_all
              if k in A and k in B
              and A[k]["hd95"] is not None and B[k]["hd95"] is not None]
    common.sort()

    result = {
        "gt": args.gt, "conf": args.conf,
        "labels": {"a": args.label_a, "b": args.label_b},
        "bootstrap": {"resamples": args.boot, "alpha": args.boot_alpha,
                      "resampled_unit": "image", "paired": True, "seed": args.seed},
        "coverage": {
            args.label_a: {"cases": len(A),
                           "both_non_empty": sum(1 for r in ra if r["hd95"] is not None),
                           "misses": sum(1 for r in ra if r["gt_px"] > 0 and r["dt_px"] == 0),
                           "false_alarms": sum(1 for r in ra if r["gt_px"] == 0 and r["dt_px"] > 0)},
            args.label_b: {"cases": len(B),
                           "both_non_empty": sum(1 for r in rb if r["hd95"] is not None),
                           "misses": sum(1 for r in rb if r["gt_px"] > 0 and r["dt_px"] == 0),
                           "false_alarms": sum(1 for r in rb if r["gt_px"] == 0 and r["dt_px"] > 0)},
        },
        "common_cases": len(common),
        "paired": {},
    }

    for key, lab, sign in METRICS:
        va = [A[k][key] for k in common if A[k][key] is not None and B[k][key] is not None]
        vb = [B[k][key] for k in common if A[k][key] is not None and B[k][key] is not None]
        if not va:
            continue
        by_img = defaultdict(list)
        for k in common:
            if A[k][key] is None or B[k][key] is None:
                continue
            by_img[k[0]].append(B[k][key] - A[k][key])
        diffs = [d for v in by_img.values() for d in v]
        ci = boot_paired(by_img, args.boot, args.boot_alpha, args.seed)
        mean_d = float(np.mean(diffs))
        # "wins" counts cases where B is better in this metric's own direction
        wins = float(np.mean([(d * sign) < 0 if sign < 0 else d > 0 for d in diffs]))
        significant = ci is not None and (ci[0] > 0 or ci[1] < 0)
        result["paired"][key] = {
            "label": lab, "n": len(diffs),
            "mean_a": float(np.mean(va)), "mean_b": float(np.mean(vb)),
            "mean_difference": mean_d, "ci": ci,
            "fraction_cases_b_better": wins,
            "b_better": (mean_d * sign) < 0 if sign < 0 else mean_d > 0,
            "separable_from_zero": significant,
        }

    with open(args.out + ".json", "w") as f:
        json.dump(result, f, indent=1)

    ca, cb = result["coverage"][args.label_a], result["coverage"][args.label_b]
    L = ["# Paired contour comparison — %s vs %s" % (args.label_b, args.label_a), "",
         "Both models scored on the **same %d cases**: every (image, class) pair where "
         "both produced a non-empty mask against non-empty ground truth, so no "
         "difference below can come from a difference in which cases were "
         "averaged." % len(common), "",
         "Confidence cut %.3f (frozen on validation using the baseline arm). "
         "Intervals are %d%% percentile bootstrap of the PAIRED per-case "
         "difference, resampling images (%d resamples)."
         % (args.conf, round(100 * (1 - args.boot_alpha)), args.boot), "",
         "## Coverage — read this before the distances", "",
         "| | cases | both non-empty | misses | false alarms |", "|---|---|---|---|---|",
         "| %s | %d | %d | %d | %d |" % (args.label_a, ca["cases"], ca["both_non_empty"],
                                         ca["misses"], ca["false_alarms"]),
         "| %s | %d | %d | %d | %d |" % (args.label_b, cb["cases"], cb["both_non_empty"],
                                         cb["misses"], cb["false_alarms"]), "",
         "## Paired differences on the common set", "",
         "| metric | %s | %s | difference | 95%% CI | %s better on | separable |"
         % (args.label_a, args.label_b, args.label_b),
         "|---|---|---|---|---|---|---|"]
    for key, lab, sign in METRICS:
        r = result["paired"].get(key)
        if not r:
            continue
        ci = r["ci"]
        L.append("| %s | %.4f | %.4f | %+.4f | %s | %.0f%% of cases | %s |"
                 % (lab, r["mean_a"], r["mean_b"], r["mean_difference"],
                    "[%+.4f, %+.4f]" % (ci[0], ci[1]) if ci else "n/a",
                    100 * r["fraction_cases_b_better"],
                    "**yes**" if r["separable_from_zero"] else "no"))
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(L) + "\n")

    print("\ncommon cases: %d" % len(common))
    print("coverage  %s: %d misses | %s: %d misses"
          % (args.label_a, ca["misses"], args.label_b, cb["misses"]))
    for key, lab, sign in METRICS:
        r = result["paired"].get(key)
        if r:
            print("  %-12s %s %+.4f  CI %s  %s"
                  % (lab, args.label_b, r["mean_difference"],
                     ("[%+.4f, %+.4f]" % tuple(r["ci"])) if r["ci"] else "n/a",
                     "SEPARABLE" if r["separable_from_zero"] else "not separable"))
    print("wrote %s.md and %s.json" % (args.out, args.out))


if __name__ == "__main__":
    main()
