#!/usr/bin/env python3
"""Dataset integrity + leakage audit for the 31-class dental splits.

Answers, with evidence, the questions a reviewer (or anyone re-running the
pipeline) should ask before trusting any number reported on this dataset:

  1. LEAKAGE — does any test image also appear in train/valid? Checked two
     ways: exact content hash (SHA-256) catches byte-identical copies, and a
     perceptual dHash catches re-encoded / resized / renamed duplicates, which
     is the realistic failure mode when splits are made by copying files.
     Any cross-split hit is a HARD FAIL: every metric downstream is invalid.
  2. PAIRING — every image has a label file, every label file has an image,
     no unreadable images.
  3. LABEL VALIDITY — class ids in range, polygons well-formed and in [0,1],
     no zero-area instances.
  4. RECONCILIATION — if a COCO JSON is supplied for a split, do its image
     set and per-class instance counts agree with the YOLO labels? (The
     provided COCO export has empty `segmentation`, so YOLO polygons are the
     source of truth; this check quantifies any disagreement rather than
     silently preferring one.)
  5. DISTRIBUTION — per-class instance counts per split with head/mid/tail
     grouping, so the long-tail claim is documented from the actual data.

Usage:
    python tools/dataset_audit.py --root data --names data/data.yaml \\
        --coco train=annotations/instances_train.json \\
        --coco valid=annotations/instances_valid.json \\
        --coco test=annotations/instances_test.json \\
        --out audit_report

Expects <root>/<split>/images and <root>/<split>/labels for splits
train/valid/test (override with --splits).
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.yolo_polygons_to_coco import (  # noqa: E402
    IMG_EXTS, load_class_names, shoelace_area)

HEAD_MIN = 5000
TAIL_MAX = 100
DHASH_SIZE = 8
NEAR_DUP_MAX_BITS = 5   # dHash Hamming distance: CANDIDATE threshold only
NCC_CONFIRM = 0.98      # normalized cross-correlation required to CONFIRM a duplicate

# Two-stage near-duplicate detection. dHash alone is far too permissive on
# panoramic radiographs: every image is the same anatomy in the same framing,
# so unrelated studies routinely land within 5 bits of each other. Measured on
# this dataset, genuine duplicates score NCC 0.9988-1.0000 while merely
# similar-looking radiographs at the same Hamming distance score <= 0.87 --
# a clean separation. dHash is therefore used as a cheap candidate generator
# and every candidate is confirmed by pixel correlation before being reported
# as leakage.


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: str, size: int = DHASH_SIZE) -> Optional[int]:
    """Difference hash: robust to re-encoding, resizing and mild compression."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    out = 0
    for bit in bits.flatten():
        out = (out << 1) | int(bit)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def ncc(path_a: str, path_b: str, size: int = 128) -> Optional[float]:
    """Normalized cross-correlation of two images at a common resolution.

    1.0 means identical content; re-encoded/resized copies of one radiograph
    stay above 0.99, unrelated studies of the same anatomy stay below 0.90.
    """
    ia = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
    ib = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)
    if ia is None or ib is None:
        return None
    ia = cv2.resize(ia, (size, size)).astype(np.float32)
    ib = cv2.resize(ib, (size, size)).astype(np.float32)
    ia -= ia.mean()
    ib -= ib.mean()
    denom = np.linalg.norm(ia) * np.linalg.norm(ib)
    return float((ia * ib).sum() / denom) if denom else None


def scan_split(images_dir: str, labels_dir: str,
               class_names: List[str]) -> Dict:
    """Hash images and validate labels for one split."""
    rec = {
        "images": [],            # [{file, sha, dhash, w, h}]
        "counts": Counter(),     # class name -> instances
        "problems": defaultdict(list),
    }
    if not os.path.isdir(images_dir):
        rec["problems"]["missing_images_dir"].append(images_dir)
        return rec

    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS))
    label_files = set()
    if os.path.isdir(labels_dir):
        label_files = {f for f in os.listdir(labels_dir) if f.endswith(".txt")}
    else:
        rec["problems"]["missing_labels_dir"].append(labels_dir)

    seen_labels = set()
    for fname in files:
        ipath = os.path.join(images_dir, fname)
        img = cv2.imread(ipath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            rec["problems"]["unreadable_image"].append(fname)
            continue
        h, w = img.shape[:2]
        rec["images"].append({"file": fname, "sha": sha256_file(ipath),
                              "dhash": dhash(ipath), "w": w, "h": h})

        stem = os.path.splitext(fname)[0]
        lname = stem + ".txt"
        lpath = os.path.join(labels_dir, lname)
        if lname not in label_files:
            rec["problems"]["image_without_label"].append(fname)
            continue
        seen_labels.add(lname)

        with open(lpath) as f:
            for lineno, ln in enumerate(f, 1):
                parts = ln.split()
                if not parts:
                    continue
                where = "%s:%d" % (lname, lineno)
                try:
                    cid = int(parts[0])
                except ValueError:
                    rec["problems"]["malformed_line"].append(where)
                    continue
                coords = parts[1:]
                if len(coords) < 6 or len(coords) % 2 != 0:
                    rec["problems"]["malformed_polygon"].append(where)
                    continue
                if not 0 <= cid < len(class_names):
                    rec["problems"]["class_id_out_of_range"].append(where)
                    continue
                try:
                    vals = [float(v) for v in coords]
                except ValueError:
                    rec["problems"]["malformed_line"].append(where)
                    continue
                if any(v < -1e-6 or v > 1 + 1e-6 for v in vals):
                    rec["problems"]["coords_out_of_unit_range"].append(where)
                xs = [v * w for v in vals[0::2]]
                ys = [v * h for v in vals[1::2]]
                if shoelace_area(xs, ys) < 1.0:
                    rec["problems"]["zero_area_polygon"].append(where)
                    continue
                rec["counts"][class_names[cid]] += 1

    for lname in sorted(label_files - seen_labels):
        rec["problems"]["label_without_image"].append(lname)
    return rec


def find_cross_split_leaks(splits: Dict[str, Dict],
                           image_dirs: Optional[Dict[str, str]] = None) -> Dict:
    """Exact and near-duplicate images shared between different splits.

    image_dirs maps split -> images directory; when supplied, every dHash
    candidate is confirmed by pixel correlation (see NCC_CONFIRM) so that
    look-alike radiographs are not reported as leakage.
    """
    exact = defaultdict(list)      # sha -> [(split, file)]
    for sname, rec in splits.items():
        for im in rec["images"]:
            exact[im["sha"]].append((sname, im["file"]))
    exact_leaks = [v for v in exact.values()
                   if len({s for s, _ in v}) > 1]

    # near-duplicates: compare test against train/valid (the decisive direction)
    near_leaks, candidates_checked, rejected_by_ncc = [], 0, 0
    if "test" in splits:
        others = [(s, im) for s, rec in splits.items() if s != "test"
                  for im in rec["images"] if im["dhash"] is not None]
        exact_shas = {sha for sha, v in exact.items() if len({s for s, _ in v}) > 1}
        for im in splits["test"]["images"]:
            if im["dhash"] is None or im["sha"] in exact_shas:
                continue
            for sname, other in others:
                d = hamming(im["dhash"], other["dhash"])
                if d > NEAR_DUP_MAX_BITS:
                    continue
                candidates_checked += 1
                score = None
                if image_dirs:
                    score = ncc(os.path.join(image_dirs["test"], im["file"]),
                                os.path.join(image_dirs[sname], other["file"]))
                    if score is not None and score < NCC_CONFIRM:
                        rejected_by_ncc += 1
                        continue          # look-alike, not a duplicate
                near_leaks.append({"test_image": im["file"],
                                   "other_split": sname,
                                   "other_image": other["file"],
                                   "hamming": d,
                                   "ncc": None if score is None else round(score, 5)})
                break

    # duplicates inside a split (not fatal, but inflates counts)
    within = {}
    for sname, rec in splits.items():
        seen = defaultdict(list)
        for im in rec["images"]:
            seen[im["sha"]].append(im["file"])
        dups = {k: v for k, v in seen.items() if len(v) > 1}
        if dups:
            within[sname] = [v for v in dups.values()]
    return {"exact_cross_split": exact_leaks,
            "near_duplicate_test": near_leaks,
            "within_split_duplicates": within,
            "dhash_candidates_checked": candidates_checked,
            "rejected_as_lookalike_by_ncc": rejected_by_ncc}


def reconcile_coco(coco_path: str, rec: Dict, class_names: List[str]) -> Dict:
    """Compare a supplied COCO JSON against the YOLO-derived truth."""
    with open(coco_path) as f:
        coco = json.load(f)
    coco_files = {im["file_name"] for im in coco["images"]}
    yolo_files = {im["file"] for im in rec["images"]}
    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    coco_counts = Counter()
    empty_seg = 0
    for ann in coco["annotations"]:
        coco_counts[id_to_name.get(ann["category_id"], "?")] += 1
        if not ann.get("segmentation"):
            empty_seg += 1
    diffs = {n: coco_counts.get(n, 0) - rec["counts"].get(n, 0)
             for n in set(list(coco_counts) + list(rec["counts"]))}
    return {
        "coco_images": len(coco_files),
        "yolo_images": len(yolo_files),
        "only_in_coco": sorted(coco_files - yolo_files)[:20],
        "only_in_yolo": sorted(yolo_files - coco_files)[:20],
        "coco_annotations": sum(coco_counts.values()),
        "yolo_annotations": sum(rec["counts"].values()),
        "annotations_with_empty_segmentation": empty_seg,
        "per_class_count_diff": {k: v for k, v in sorted(diffs.items()) if v != 0},
        "category_names_match": sorted(id_to_name.values()) == sorted(class_names),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument("--coco", action="append", default=[],
                    metavar="SPLIT=PATH", help="repeatable, e.g. test=inst_test.json")
    ap.add_argument("--out", default="audit_report")
    args = ap.parse_args()

    class_names = load_class_names(args.names)
    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]

    splits = {}
    for s in split_names:
        print("scanning %s ..." % s)
        splits[s] = scan_split(os.path.join(args.root, s, "images"),
                               os.path.join(args.root, s, "labels"),
                               class_names)

    image_dirs = {s: os.path.join(args.root, s, "images") for s in split_names}
    leaks = find_cross_split_leaks(splits, image_dirs)
    recon = {}
    for spec in args.coco:
        sname, _, path = spec.partition("=")
        if sname in splits:
            recon[sname] = reconcile_coco(path, splits[sname], class_names)

    totals = {s: {"images": len(r["images"]),
                  "annotations": sum(r["counts"].values())}
              for s, r in splits.items()}
    train_counts = splits.get("train", {"counts": Counter()})["counts"]

    def group_of(name):
        n = train_counts.get(name, 0)
        return "head" if n > HEAD_MIN else ("tail" if n < TAIL_MAX else "mid")

    problems = {s: {k: len(v) for k, v in r["problems"].items()}
                for s, r in splits.items()}
    n_problems = sum(sum(p.values()) for p in problems.values())
    leak_fail = bool(leaks["exact_cross_split"]) or bool(leaks["near_duplicate_test"])

    report = {"totals": totals, "leakage": leaks, "problems": problems,
              "reconciliation": recon,
              "per_class": {n: {s: splits[s]["counts"].get(n, 0) for s in split_names}
                            for n in class_names},
              "grouping": {n: group_of(n) for n in class_names},
              "verdict": {"leakage_free": not leak_fail,
                          "problem_count": n_problems}}
    with open(args.out + ".json", "w") as f:
        json.dump(report, f, indent=1, default=list)

    lines = ["# Dataset audit", "",
             "## Split totals", "", "| split | images | annotations |", "|---|---|---|"]
    for s in split_names:
        lines.append("| %s | %d | %d |" % (s, totals[s]["images"], totals[s]["annotations"]))
    lines += ["", "## Leakage", "",
              "- exact cross-split duplicate groups: **%d**" % len(leaks["exact_cross_split"]),
              "- test images duplicating train/valid (dHash<=%d AND NCC>=%.2f): **%d**"
              % (NEAR_DUP_MAX_BITS, NCC_CONFIRM, len(leaks["near_duplicate_test"])),
              "- dHash candidates examined: %d, of which rejected as look-alikes "
              "by pixel correlation: %d"
              % (leaks["dhash_candidates_checked"], leaks["rejected_as_lookalike_by_ncc"]),
              "- within-split duplicate groups: %s"
              % {k: len(v) for k, v in leaks["within_split_duplicates"].items()},
              "", "**VERDICT: %s**" % ("LEAKAGE-FREE" if not leak_fail else "LEAKAGE DETECTED — metrics invalid until resolved"),
              "", "## Per-class instance counts", "",
              "| class | group | " + " | ".join(split_names) + " |",
              "|---|---|" + "---|" * len(split_names)]
    for n in sorted(class_names, key=lambda n: -train_counts.get(n, 0)):
        lines.append("| %s | %s | %s |" % (
            n, group_of(n), " | ".join(str(splits[s]["counts"].get(n, 0))
                                       for s in split_names)))
    if recon:
        lines += ["", "## YOLO vs COCO reconciliation", ""]
        for s, r in recon.items():
            lines.append("- **%s**: %d/%d images (coco/yolo), %d/%d annotations, "
                         "%d anns with empty segmentation, %d classes disagree"
                         % (s, r["coco_images"], r["yolo_images"],
                            r["coco_annotations"], r["yolo_annotations"],
                            r["annotations_with_empty_segmentation"],
                            len(r["per_class_count_diff"])))
    if n_problems:
        lines += ["", "## Label problems", ""]
        for s, p in problems.items():
            if p:
                lines.append("- %s: %s" % (s, dict(p)))
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines[-6:]) if not leak_fail else "LEAKAGE DETECTED")
    print("\nwrote %s.md / %s.json" % (args.out, args.out))
    sys.exit(1 if leak_fail else 0)


if __name__ == "__main__":
    main()
