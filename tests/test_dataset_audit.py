#!/usr/bin/env python3
"""Tests for the dataset leakage/integrity audit.

Builds a synthetic 3-split dataset containing, by construction:
  * one byte-identical test<->train duplicate  (exact leak)
  * one JPEG-recompressed+resized test copy of a train image (near-dup leak)
  * one image with no label file, one label with no image
  * one out-of-range class id and one zero-area polygon
  * a COCO json disagreeing with the YOLO labels by one annotation

...then asserts the audit finds exactly those and fails the verdict.
A clean copy of the same dataset must pass.

Run:  python tests/test_dataset_audit.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from tools.dataset_audit import scan_split, find_cross_split_leaks, reconcile_coco  # noqa: E402
from tools.yolo_polygons_to_coco import shoelace_area  # noqa: E402

CLASSES = ["Filling", "Caries", "TAD"]
W, H = 256, 192
GOOD_LABEL = "0 0.10 0.10 0.40 0.10 0.40 0.40 0.10 0.40\n"


def write_img(path, seed):
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 255, (H, W, 3), dtype=np.uint8)
    cv2.imwrite(path, img)
    return img


def build(root, clean):
    for split in ("train", "valid", "test"):
        os.makedirs(os.path.join(root, split, "images"))
        os.makedirs(os.path.join(root, split, "labels"))

    def put(split, name, seed, label=GOOD_LABEL):
        p = os.path.join(root, split, "images", name)
        img = write_img(p, seed)
        if label is not None:
            with open(os.path.join(root, split, "labels",
                                   os.path.splitext(name)[0] + ".txt"), "w") as f:
                f.write(label)
        return p, img

    for i in range(4):
        put("train", "tr%d.jpg" % i, seed=i)
    for i in range(2):
        put("valid", "va%d.jpg" % i, seed=100 + i)
    for i in range(3):
        put("test", "te%d.jpg" % i, seed=200 + i)

    if clean:
        return

    # 1. exact leak: byte-copy train image into test
    shutil.copy(os.path.join(root, "train", "images", "tr0.jpg"),
                os.path.join(root, "test", "images", "leak_exact.jpg"))
    with open(os.path.join(root, "test", "labels", "leak_exact.txt"), "w") as f:
        f.write(GOOD_LABEL)

    # 2. near-dup leak: resize + recompress a train image
    src = cv2.imread(os.path.join(root, "train", "images", "tr1.jpg"))
    resized = cv2.resize(src, (W // 2, H // 2))
    cv2.imwrite(os.path.join(root, "test", "images", "leak_near.jpg"), resized,
                [cv2.IMWRITE_JPEG_QUALITY, 60])
    with open(os.path.join(root, "test", "labels", "leak_near.txt"), "w") as f:
        f.write(GOOD_LABEL)

    # 3. image without label, label without image
    write_img(os.path.join(root, "train", "images", "orphan_img.jpg"), 55)
    with open(os.path.join(root, "train", "labels", "orphan_lbl.txt"), "w") as f:
        f.write(GOOD_LABEL)

    # 4. bad class id + zero-area polygon
    with open(os.path.join(root, "train", "labels", "tr2.txt"), "w") as f:
        f.write(GOOD_LABEL)
        f.write("99 0.1 0.1 0.2 0.1 0.2 0.2\n")            # class out of range
        f.write("1 0.5 0.5 0.5001 0.5 0.5002 0.5\n")        # collinear/zero area


def coco_for(root, split, classes, drop_one):
    """COCO json built from the YOLO labels, optionally missing 1 annotation."""
    img_dir = os.path.join(root, split, "images")
    lbl_dir = os.path.join(root, split, "labels")
    images, anns, aid = [], [], 1
    for iid, fname in enumerate(sorted(os.listdir(img_dir)), 1):
        images.append({"id": iid, "file_name": fname, "width": W, "height": H})
        lp = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
        if not os.path.exists(lp):
            continue
        for ln in open(lp):
            parts = ln.split()
            if not parts:
                continue
            cid = int(parts[0])
            if not 0 <= cid < len(classes):
                continue
            vals = [float(v) for v in parts[1:]]
            if shoelace_area([v * W for v in vals[0::2]],
                             [v * H for v in vals[1::2]]) < 1.0:
                continue  # mirror the audit's validity rule
            anns.append({"id": aid, "image_id": iid, "category_id": cid + 1,
                         "bbox": [10, 10, 50, 50], "area": 2500, "iscrowd": 0,
                         "segmentation": []})   # empty seg, as in the real export
            aid += 1
    if drop_one and anns:
        anns.pop()
    return {"images": images, "annotations": anns,
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(classes)]}


def main():
    # ---------- dirty dataset: every planted defect must be found ----------
    with tempfile.TemporaryDirectory() as tmp:
        build(tmp, clean=False)
        splits = {s: scan_split(os.path.join(tmp, s, "images"),
                                os.path.join(tmp, s, "labels"), CLASSES)
                  for s in ("train", "valid", "test")}

        leaks = find_cross_split_leaks(splits)
        assert len(leaks["exact_cross_split"]) == 1, leaks["exact_cross_split"]
        pair = {f for _, f in leaks["exact_cross_split"][0]}
        assert pair == {"tr0.jpg", "leak_exact.jpg"}, pair
        assert len(leaks["near_duplicate_test"]) == 1, leaks["near_duplicate_test"]
        assert leaks["near_duplicate_test"][0]["test_image"] == "leak_near.jpg"
        assert leaks["near_duplicate_test"][0]["other_image"] == "tr1.jpg"

        tp = splits["train"]["problems"]
        assert tp["image_without_label"] == ["orphan_img.jpg"], dict(tp)
        assert tp["label_without_image"] == ["orphan_lbl.txt"], dict(tp)
        assert len(tp["class_id_out_of_range"]) == 1
        assert len(tp["zero_area_polygon"]) == 1

        # reconciliation: COCO missing one annotation must show up as a diff
        cpath = os.path.join(tmp, "coco_train.json")
        json.dump(coco_for(tmp, "train", CLASSES, drop_one=True), open(cpath, "w"))
        rec = reconcile_coco(cpath, splits["train"], CLASSES)
        assert rec["coco_annotations"] == rec["yolo_annotations"] - 1
        assert rec["per_class_count_diff"], "count disagreement must be reported"
        assert rec["annotations_with_empty_segmentation"] == rec["coco_annotations"]
        assert rec["category_names_match"] is True

    # ---------- clean dataset: audit must pass and exit 0 ----------
    with tempfile.TemporaryDirectory() as tmp:
        build(tmp, clean=True)
        names = os.path.join(tmp, "classes.txt")
        with open(names, "w") as f:
            f.write("\n".join(CLASSES) + "\n")
        cpath = os.path.join(tmp, "coco_test.json")
        json.dump(coco_for(tmp, "test", CLASSES, drop_one=False), open(cpath, "w"))

        out = os.path.join(tmp, "report")
        proc = subprocess.run(
            [sys.executable, os.path.join(ROOT, "tools", "dataset_audit.py"),
             "--root", tmp, "--names", names, "--out", out,
             "--coco", "test=" + cpath],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.returncode == 0, proc.stdout.decode()[-2000:]
        rep = json.load(open(out + ".json"))
        assert rep["verdict"]["leakage_free"] is True
        assert rep["verdict"]["problem_count"] == 0, rep["problems"]
        assert rep["totals"]["train"]["images"] == 4
        assert os.path.exists(out + ".md")

    print("ALL CHECKS PASSED: audit detects exact + near-duplicate leakage, "
          "orphan files, bad labels, COCO/YOLO disagreement; clean data passes")


if __name__ == "__main__":
    main()
