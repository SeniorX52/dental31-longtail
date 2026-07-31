#!/usr/bin/env python3
"""Round-trip test for the YOLO-polygon -> COCO converter.

Builds a synthetic mini-dataset with known polygons (including a degenerate
polygon and a long-tail class distribution), converts it, then verifies:

  1. pycocotools loads the JSON and index-builds without error;
  2. polygon areas match the analytic shoelace values;
  3. feeding the ground truth back as detections scores mAP = 1.0 on BOTH
     bbox and segm metrics (proves the JSON is well-formed for COCOeval);
  4. degenerate polygons are dropped, negative images are kept.

Run:  python tests/test_yolo2coco.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.yolo_polygons_to_coco import convert, shoelace_area  # noqa: E402

W, H = 640, 480
CLASSES = ["Caries", "Crown", "Filling", "TAD"]

# (class_id, normalized polygon) per image. Image 2 has no label file
# (negative sample); image 1 includes a degenerate 2-point line.
LABELS = {
    "img0": [
        (0, [0.10, 0.10, 0.30, 0.10, 0.30, 0.40, 0.10, 0.40]),   # rectangle
        (2, [0.50, 0.50, 0.80, 0.55, 0.70, 0.85]),               # triangle
    ],
    "img1": [
        (1, [0.05, 0.05, 0.45, 0.05, 0.45, 0.25, 0.25, 0.35, 0.05, 0.25]),
        (3, [0.60, 0.60, 0.61, 0.60, 0.62, 0.60]),               # collinear -> zero area
        (2, [0.55, 0.10, 0.90, 0.10, 0.90, 0.30, 0.55, 0.30]),
    ],
}


def build_dataset(root):
    img_dir = os.path.join(root, "images")
    lbl_dir = os.path.join(root, "labels")
    os.makedirs(img_dir)
    os.makedirs(lbl_dir)
    for name in ["img0", "img1", "img2"]:
        cv2.imwrite(os.path.join(img_dir, name + ".jpg"),
                    np.zeros((H, W, 3), dtype=np.uint8))
    for name, insts in LABELS.items():
        with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
            for cls, poly in insts:
                f.write(str(cls) + " " + " ".join("%.6f" % v for v in poly) + "\n")
    return img_dir, lbl_dir


def main():
    tmp = tempfile.mkdtemp(prefix="yolo2coco_test_")
    try:
        img_dir, lbl_dir = build_dataset(tmp)
        result = convert(img_dir, lbl_dir, CLASSES)
        coco_dict = result["coco"]

        # -- structural checks -------------------------------------------------
        assert len(coco_dict["images"]) == 3, "negative image must be kept"
        assert len(coco_dict["annotations"]) == 4, \
            "expected 4 annotations (degenerate dropped), got %d" % len(coco_dict["annotations"])
        assert result["dropped"]["degenerate_polygon"] == 1
        assert result["dropped"]["missing_label_file"] == 1
        assert result["per_class"]["TAD"] == 0

        # -- area check against analytic shoelace ------------------------------
        ann0 = coco_dict["annotations"][0]           # the rectangle on img0
        expect = (0.30 - 0.10) * W * (0.40 - 0.10) * H
        assert abs(ann0["area"] - expect) < 1.0, (ann0["area"], expect)

        # -- pycocotools round trip: GT as detections must score mAP = 1.0 -----
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        gt_path = os.path.join(tmp, "gt.json")
        with open(gt_path, "w") as f:
            json.dump(coco_dict, f)
        coco_gt = COCO(gt_path)

        dets = []
        for ann in coco_dict["annotations"]:
            det = {k: ann[k] for k in ("image_id", "category_id", "bbox", "segmentation")}
            det["score"] = 1.0
            dets.append(det)

        for iou_type in ("bbox", "segm"):
            coco_dt = coco_gt.loadRes([dict(d) for d in dets])
            ev = COCOeval(coco_gt, coco_dt, iou_type)
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            assert abs(ev.stats[0] - 1.0) < 1e-6, \
                "%s self-eval mAP = %.4f, expected 1.0" % (iou_type, ev.stats[0])

        print("\nALL CHECKS PASSED: converter output is valid COCO for bbox AND segm eval")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
