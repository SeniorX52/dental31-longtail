#!/usr/bin/env python3
"""Build a frozen, leakage-free, class-stratified split for the dental dataset.

Why this exists: in the dataset as shipped, augmentation was applied BEFORE
splitting, so augmented copies of one radiograph land in different splits
(1,516 of 1,580 test files have their source image in train/valid). Splitting
must therefore happen at the **patient** level, which also keeps every
augmented copy of every image of that patient inside one split.

Algorithm
---------
1. Group every image file by patient key parsed from the Roboflow filename
   `<uuid8>-<PATIENT>_<date/time>_jpg.rf.<hash>.jpg`. Patient key is the
   name portion with the trailing date/time/age tokens stripped. Files that
   don't match the pattern fall back to their own source-image key, so they
   are still never split apart.
2. Assign whole patient groups to train/valid/test with a greedy
   stratification pass: process classes from rarest to most common, and for
   each class push its patient groups to whichever split is furthest below
   its quota for that class. This is what gets tail classes into test, which
   the original split failed to do (18 of 31 classes had zero test instances).
3. Verify: no patient and no source image crosses splits; report per-class
   counts per split; fail loudly if any class ends up absent from train.
4. Write explicit file lists (train.txt / valid.txt / test.txt) plus a JSON
   manifest with the seed and the resulting statistics. The split is then
   frozen -- these lists are the split, forever, for baseline and method
   alike.

Usage:
    python tools/make_clean_split.py \\
        --yolo-root data_raw/dental31/YOLO/YOLO \\
        --names data_raw/dental31/YOLO/YOLO/data.yaml \\
        --out splits/clean_v1 --ratios 0.70 0.15 0.15 --seed 42
"""
import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
RF_RE = re.compile(r"^(?:[0-9a-f]{8}-)?(?P<body>.*?)_jpg\.rf\.[0-9a-f]+$", re.I)


def load_class_names(path: str) -> List[str]:
    names: Dict[int, str] = {}
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            m = re.match(r"^(\d+)\s*:\s*(.+)$", s)
            if m:
                names[int(m.group(1))] = m.group(2).strip().strip("'\"")
    if not names:
        raise SystemExit("no `id: name` entries found in %s" % path)
    return [names[k] for k in sorted(names)]


def source_key(stem: str) -> str:
    """Identity of the underlying radiograph, ignoring augmentation hash."""
    m = RF_RE.match(stem)
    return m.group("body") if m else stem


def patient_key(stem: str) -> str:
    """Identity of the patient: source key minus trailing date/time/age."""
    s = source_key(stem)
    s = re.sub(r"[_-]?\d{4}[-_]?\d{2}[-_]?\d{2}.*$", "", s)   # 2020-06-13...
    s = re.sub(r"[_-]?\d{1,3}\s*yo.*$", "", s, flags=re.I)     # 39yo
    s = re.sub(r"[_-]?\d{6,}.*$", "", s)                       # 180215 / 07102020
    s = s.strip("_- ").lower()
    return s if s else source_key(stem).lower()


def collect(yolo_root: str, splits=("train", "valid", "test")):
    """-> {patient: {"files": [(split, stem, img_path, lbl_path)], "sources": set}}"""
    groups = defaultdict(lambda: {"files": [], "sources": set()})
    for split in splits:
        img_dir = os.path.join(yolo_root, split, "images")
        lbl_dir = os.path.join(yolo_root, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith(IMG_EXTS):
                continue
            stem = os.path.splitext(fname)[0]
            pk = patient_key(stem)
            groups[pk]["files"].append(
                (split, stem, os.path.join(img_dir, fname),
                 os.path.join(lbl_dir, stem + ".txt")))
            groups[pk]["sources"].add(source_key(stem))
    return groups


def group_class_counts(group, num_classes: int) -> Counter:
    c = Counter()
    for _, _, _, lbl in group["files"]:
        if not os.path.exists(lbl):
            continue
        with open(lbl) as f:
            for ln in f:
                parts = ln.split()
                if parts:
                    cid = int(parts[0])
                    if 0 <= cid < num_classes:
                        c[cid] += 1
    return c


def stratified_assign(groups, num_classes: int, ratios, seed: int):
    """Greedy: rarest class first, give each group to the neediest split."""
    names = ["train", "valid", "test"]
    counts = {pk: group_class_counts(g, num_classes) for pk, g in groups.items()}
    totals = Counter()
    for c in counts.values():
        totals.update(c)

    # rarest classes decided first so their few groups land deliberately
    class_order = [c for c, _ in sorted(totals.items(), key=lambda kv: kv[1])]

    rng = random.Random(seed)
    assigned: Dict[str, str] = {}
    got = {s: Counter() for s in names}          # per-split per-class instances
    n_files = {s: 0 for s in names}
    total_files = sum(len(g["files"]) for g in groups.values())

    def place(pk):
        """Choose the split with the largest unmet share, weighted by ratio."""
        best, best_score = None, None
        for s, r in zip(names, ratios):
            # deficit on the classes this group carries
            deficit = 0.0
            for cid, n in counts[pk].items():
                target = totals[cid] * r
                deficit += (target - got[s][cid]) / max(target, 1.0) * n
            # keep overall split sizes near target too
            size_deficit = (total_files * r - n_files[s]) / max(total_files * r, 1.0)
            score = deficit + size_deficit
            if best_score is None or score > best_score:
                best, best_score = s, score
        return best

    for cid in class_order:
        holders = [pk for pk, c in counts.items() if c.get(cid) and pk not in assigned]
        rng.shuffle(holders)
        # ensure the rarest classes reach every split when enough groups exist
        holders.sort(key=lambda pk: -counts[pk][cid])
        for pk in holders:
            s = place(pk)
            assigned[pk] = s
            got[s].update(counts[pk])
            n_files[s] += len(groups[pk]["files"])

    for pk in groups:                            # groups with no annotations
        if pk not in assigned:
            s = min(names, key=lambda s: n_files[s] / max(ratios[names.index(s)], 1e-9))
            assigned[pk] = s
            n_files[s] += len(groups[pk]["files"])
    return assigned, counts


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-root", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                    metavar=("TRAIN", "VALID", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    class_names = load_class_names(args.names)
    nc = len(class_names)
    groups = collect(args.yolo_root)
    print("patients (groups): %d | files: %d | distinct source images: %d"
          % (len(groups), sum(len(g["files"]) for g in groups.values()),
             len({s for g in groups.values() for s in g["sources"]})))

    assigned, counts = stratified_assign(groups, nc, args.ratios, args.seed)

    # ---- verification -------------------------------------------------------
    files_by_split = defaultdict(list)
    src_by_split = defaultdict(set)
    pat_by_split = defaultdict(set)
    cls_by_split = {s: Counter() for s in ("train", "valid", "test")}
    for pk, g in groups.items():
        s = assigned[pk]
        pat_by_split[s].add(pk)
        for split, stem, img, lbl in g["files"]:
            files_by_split[s].append(img)
            src_by_split[s].add(source_key(stem))
        cls_by_split[s].update(counts[pk])

    problems = []
    for a in ("train", "valid", "test"):
        for b in ("train", "valid", "test"):
            if a >= b:
                continue
            if src_by_split[a] & src_by_split[b]:
                problems.append("source images shared by %s/%s: %d"
                                % (a, b, len(src_by_split[a] & src_by_split[b])))
            if pat_by_split[a] & pat_by_split[b]:
                problems.append("patients shared by %s/%s: %d"
                                % (a, b, len(pat_by_split[a] & pat_by_split[b])))
    missing_train = [class_names[c] for c in range(nc) if not cls_by_split["train"].get(c)]
    if missing_train:
        problems.append("classes absent from TRAIN: %s" % missing_train)

    print("\n%-24s %8s %8s %8s" % ("", "train", "valid", "test"))
    print("%-24s %8d %8d %8d" % ("images", *(len(files_by_split[s]) for s in ("train","valid","test"))))
    print("%-24s %8d %8d %8d" % ("patients", *(len(pat_by_split[s]) for s in ("train","valid","test"))))
    print("%-24s %8d %8d %8d" % ("source images", *(len(src_by_split[s]) for s in ("train","valid","test"))))
    print("%-24s %8d %8d %8d" % ("instances",
          *(sum(cls_by_split[s].values()) for s in ("train","valid","test"))))
    print("%-24s %8d %8d %8d" % ("classes present",
          *(len([c for c in range(nc) if cls_by_split[s].get(c)]) for s in ("train","valid","test"))))

    print("\nper-class instances:")
    print("%-24s %8s %8s %8s" % ("class", "train", "valid", "test"))
    for c in range(nc):
        print("%-24s %8d %8d %8d" % (class_names[c][:24], cls_by_split["train"].get(c,0),
                                     cls_by_split["valid"].get(c,0), cls_by_split["test"].get(c,0)))

    os.makedirs(args.out, exist_ok=True)
    for s in ("train", "valid", "test"):
        with open(os.path.join(args.out, s + ".txt"), "w") as f:
            for p in sorted(files_by_split[s]):
                f.write(os.path.abspath(p) + "\n")
    manifest = {
        "seed": args.seed, "ratios": args.ratios,
        "grouping": "patient (parsed from Roboflow filename); groups source images too",
        "counts": {s: {"images": len(files_by_split[s]),
                       "patients": len(pat_by_split[s]),
                       "source_images": len(src_by_split[s]),
                       "instances": sum(cls_by_split[s].values()),
                       "classes_present": len([c for c in range(nc) if cls_by_split[s].get(c)])}
                   for s in ("train", "valid", "test")},
        "per_class": {class_names[c]: {s: cls_by_split[s].get(c, 0)
                                       for s in ("train", "valid", "test")}
                      for c in range(nc)},
        "verification_problems": problems,
    }
    with open(os.path.join(args.out, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    print()
    if problems:
        print("VERIFICATION FAILED:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print("VERIFIED: no patient and no source image crosses splits.")
    print("wrote %s/{train,valid,test}.txt and split_manifest.json" % args.out)


if __name__ == "__main__":
    main()
