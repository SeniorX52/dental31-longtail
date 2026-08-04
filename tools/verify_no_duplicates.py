#!/usr/bin/env python3
"""Verify no duplicate or near-duplicate image is shared between split partitions.

This answers one question and only that question, because it has to run while a
`cache=ram` training job owns most of system memory. `tools/dataset_audit.py`
parses every YOLO polygon in the dataset (~137k coordinate lists) and cannot be
run concurrently; this script keeps three small fields per image and a single
image pair in memory at a time.

Detection is two-stage and identical in substance to dataset_audit.py:

  stage 1  dHash (8x8, 64-bit), Hamming distance <= NEAR_DUP_MAX_BITS  -> CANDIDATE
  stage 2  normalized cross-correlation >= NCC_CONFIRM                 -> CONFIRMED

Stage 2 is not optional. Panoramic radiographs are all the same anatomy in the
same framing, so unrelated studies routinely land within 5 bits of each other:
on this dataset stage 1 alone flags ~1500 test images, of which the great
majority are merely look-alikes. Pixel correlation separates them cleanly --
genuine re-encoded/augmented copies score >= 0.99, distinct studies <= 0.90 --
so the confirmed count is the only one that may be reported as leakage.

Two differences from dataset_audit.py, both deliberate:

  * ALL ordered split pairs are checked (test-train, test-valid, valid-train),
    not just test-vs-rest. The requirement is that no two partitions share an
    image, which is a statement about every pair.
  * The NCC score of every candidate is recorded, confirmed or not, so the
    separation between duplicates and look-alikes is evidence in the report
    rather than an assertion about a threshold.

Also reports, from filename structure alone:
  * exact byte-identical duplicates within a single split (these do not leak
    but do inflate that split's effective size)
  * source-image and patient identifiers appearing in more than one split,
    which is the leak that survives even when no two files are pixel-similar

Usage:
    python tools/verify_no_duplicates.py --root data_clean --out reports/dup_check
"""
import argparse
import json
import os
import resource
import sys
from collections import defaultdict

import cv2
import numpy as np

# This runs alongside a training job that owns the machine's CPU and RAM.
# OpenCV would otherwise spawn a thread per core and contend with the
# dataloader workers for no gain -- the work here is I/O bound.
cv2.setNumThreads(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.dataset_audit import (  # noqa: E402
    NCC_CONFIRM, NEAR_DUP_MAX_BITS, dhash, ncc, sha256_file)
from tools.make_clean_split import patient_key, source_key  # noqa: E402
from tools.yolo_polygons_to_coco import IMG_EXTS  # noqa: E402

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_vec(query: int, arr: np.ndarray) -> np.ndarray:
    """Hamming distance from one 64-bit hash to a whole array of them."""
    x = np.bitwise_xor(arr, np.uint64(query))
    return _POPCOUNT[x.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def scan(root: str, split: str):
    """-> list of dicts, one per image: file, sha, dhash, source, patient."""
    img_dir = os.path.join(root, split, "images")
    out = []
    names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS))
    for i, fname in enumerate(names):
        path = os.path.join(img_dir, fname)
        stem = os.path.splitext(fname)[0]
        out.append({"file": fname,
                    "sha": sha256_file(path),
                    "dhash": dhash(path),
                    "source": source_key(stem),
                    "patient": patient_key(stem)})
        if (i + 1) % 1000 == 0:
            print("  %s: %d/%d" % (split, i + 1, len(names)), flush=True)
    print("  %s: %d images" % (split, len(out)), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument("--out", default="reports/dup_check")
    ap.add_argument("--mem-cap-gb", type=float, default=3.0,
                    help="fail loudly instead of pushing the training job into swap")
    args = ap.parse_args()

    cap = int(args.mem_cap_gb * (1 << 30))
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    print("scanning ...", flush=True)
    data = {s: scan(args.root, s) for s in splits}
    img_dirs = {s: os.path.join(args.root, s, "images") for s in splits}

    # ---- exact duplicates, cross-split and within-split -------------------
    by_sha = defaultdict(list)
    for s in splits:
        for im in data[s]:
            by_sha[im["sha"]].append((s, im["file"]))
    exact_cross = [v for v in by_sha.values() if len({s for s, _ in v}) > 1]
    within = {}
    for s in splits:
        seen = defaultdict(list)
        for im in data[s]:
            seen[im["sha"]].append(im["file"])
        groups = [v for v in seen.values() if len(v) > 1]
        if groups:
            within[s] = groups

    # ---- near-duplicates across every ordered split pair ------------------
    # Exact cross-split hits are excluded: they are already reported above and
    # would otherwise be double-counted as near-duplicates.
    exact_shas = {sha for sha, v in by_sha.items() if len({s for s, _ in v}) > 1}
    hashes = {s: np.array([im["dhash"] for im in data[s] if im["dhash"] is not None],
                          dtype=np.uint64) for s in splits}
    files = {s: [im["file"] for im in data[s] if im["dhash"] is not None]
             for s in splits}

    confirmed, rejected, scores = [], [], []
    pairs = [(a, b) for i, a in enumerate(splits) for b in splits[i + 1:]]
    for a, b in pairs:
        print("comparing %s vs %s ..." % (a, b), flush=True)
        for im in data[a]:
            if im["dhash"] is None or im["sha"] in exact_shas:
                continue
            dists = hamming_vec(im["dhash"], hashes[b])
            for j in np.nonzero(dists <= NEAR_DUP_MAX_BITS)[0]:
                other = files[b][int(j)]
                score = ncc(os.path.join(img_dirs[a], im["file"]),
                            os.path.join(img_dirs[b], other))
                rec = {"image": im["file"], "split": a,
                       "other_image": other, "other_split": b,
                       "hamming": int(dists[j]),
                       "ncc": None if score is None else round(float(score), 5)}
                if score is not None:
                    scores.append(float(score))
                if score is not None and score < NCC_CONFIRM:
                    rejected.append(rec)
                else:
                    confirmed.append(rec)
                    break            # one confirmed partner is enough

    # ---- identifier-level overlap (survives pixel-level dissimilarity) ----
    ident = {}
    for key in ("source", "patient"):
        where = defaultdict(set)
        for s in splits:
            for im in data[s]:
                where[im[key]].add(s)
        shared = {k: sorted(v) for k, v in where.items() if len(v) > 1}
        ident[key] = {"n_shared": len(shared),
                      "examples": dict(list(shared.items())[:20])}

    scores_arr = np.array(scores) if scores else np.array([0.0])
    leak = bool(exact_cross) or bool(confirmed) or ident["source"]["n_shared"] \
        or ident["patient"]["n_shared"]
    report = {
        "root": args.root,
        "totals": {s: len(data[s]) for s in splits},
        "thresholds": {"dhash_max_bits": NEAR_DUP_MAX_BITS, "ncc_confirm": NCC_CONFIRM},
        "exact_cross_split_groups": exact_cross,
        "near_duplicate_confirmed": confirmed,
        "near_duplicate_rejected_as_lookalike": len(rejected),
        "candidates_examined": len(confirmed) + len(rejected),
        "ncc_score_distribution": {
            "min": float(scores_arr.min()), "max": float(scores_arr.max()),
            "mean": float(scores_arr.mean()),
            "p50": float(np.percentile(scores_arr, 50)),
            "p95": float(np.percentile(scores_arr, 95)),
            "p99": float(np.percentile(scores_arr, 99)),
            "n_at_or_above_confirm": int((scores_arr >= NCC_CONFIRM).sum()),
        },
        "within_split_exact_duplicate_groups": {k: len(v) for k, v in within.items()},
        "within_split_examples": {k: v[:5] for k, v in within.items()},
        "identifier_overlap": ident,
        "verdict": {"partitions_disjoint": not leak},
    }
    with open(args.out + ".json", "w") as f:
        json.dump(report, f, indent=1)

    L = ["# Cross-split duplicate verification", "",
         "Root: `%s`" % args.root, "",
         "| split | images |", "|---|---|"]
    for s in splits:
        L.append("| %s | %d |" % (s, len(data[s])))
    L += ["", "## Cross-split leakage", "",
          "| check | result |", "|---|---|",
          "| exact duplicate groups (SHA-256) | **%d** |" % len(exact_cross),
          "| dHash candidates examined (<=%d bits) | %d |"
          % (NEAR_DUP_MAX_BITS, len(confirmed) + len(rejected)),
          "| rejected as look-alikes (NCC < %.2f) | %d |" % (NCC_CONFIRM, len(rejected)),
          "| **confirmed near-duplicates (NCC >= %.2f)** | **%d** |"
          % (NCC_CONFIRM, len(confirmed)),
          "| source images in >1 split | %d |" % ident["source"]["n_shared"],
          "| patients in >1 split | %d |" % ident["patient"]["n_shared"],
          "", "NCC over all candidates: min %.4f, median %.4f, p95 %.4f, p99 %.4f, max %.4f."
          % (scores_arr.min(), np.percentile(scores_arr, 50), np.percentile(scores_arr, 95),
             np.percentile(scores_arr, 99), scores_arr.max()),
          "", "## Within-split exact duplicates", "",
          "Not leakage, but they inflate the effective size of the split.", "",
          "| split | duplicate groups |", "|---|---|"]
    for s in splits:
        L.append("| %s | %d |" % (s, len(within.get(s, []))))
    L += ["", "**VERDICT: %s**" % ("PARTITIONS DISJOINT" if not leak
                                   else "OVERLAP DETECTED"), ""]
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(L) + "\n")

    print("\nexact cross-split groups: %d" % len(exact_cross))
    print("candidates examined: %d | rejected look-alike: %d | CONFIRMED: %d"
          % (len(confirmed) + len(rejected), len(rejected), len(confirmed)))
    print("source overlap: %d | patient overlap: %d"
          % (ident["source"]["n_shared"], ident["patient"]["n_shared"]))
    print("wrote %s.json and %s.md" % (args.out, args.out))


if __name__ == "__main__":
    main()
