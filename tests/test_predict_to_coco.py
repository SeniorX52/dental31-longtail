#!/usr/bin/env python3
"""Tests for the YOLO-seg -> COCO detections exporter.

Two guarantees are checked:
  1. RLE round trip — masks converted by `masks_to_coco` and scored against a
     GT built from the SAME masks give mask mAP = 1.0 in pycocotools. This is
     what proves the exported segmentations are correctly encoded and aligned.
  2. Class mapping is by NAME, and a name mismatch fails loudly instead of
     silently mis-assigning categories (the classic position-mapping bug).

Run:  python tests/test_predict_to_coco.py
"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from yolov8_seg_longtail.predict_to_coco import (  # noqa: E402
    masks_to_coco, build_class_mapping)

H, W = 120, 160
CATS = [{"id": 1, "name": "Caries"}, {"id": 2, "name": "Crown"}]


def make_case():
    """Two instances per image with known masks/boxes."""
    m1 = np.zeros((H, W), np.uint8); m1[10:50, 20:70] = 1
    m2 = np.zeros((H, W), np.uint8); m2[60:100, 90:140] = 1
    dets = [
        {"cls": 0, "score": 0.9, "bbox_xyxy": [20, 10, 70, 50], "mask": m1},
        {"cls": 1, "score": 0.8, "bbox_xyxy": [90, 60, 140, 100], "mask": m2},
    ]
    return dets


def test_mapping_by_name():
    mapping = build_class_mapping({0: "Crown", 1: "Caries"}, CATS)
    # index 0 is "Crown" -> category 2, NOT category 1
    assert mapping == {0: 2, 1: 1}, mapping

    try:
        build_class_mapping({0: "Dragon"}, CATS)
    except SystemExit:
        pass
    else:
        raise AssertionError("unknown class name must fail loudly")


def test_rle_roundtrip_scores_perfect():
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    dets = make_case()
    mapping = {0: 1, 1: 2}
    coco_dets = masks_to_coco(dets, image_id=1, yolo_idx_to_cat_id=mapping)
    assert len(coco_dets) == 2
    assert coco_dets[0]["bbox"] == [20.0, 10.0, 50.0, 40.0]   # xyxy -> xywh
    assert isinstance(coco_dets[0]["segmentation"]["counts"], str)

    # GT built from the identical masks
    gt_anns = []
    for i, d in enumerate(dets, 1):
        rle = mask_utils.encode(np.asfortranarray(d["mask"]))
        rle["counts"] = rle["counts"].decode("ascii")
        x0, y0, x1, y1 = d["bbox_xyxy"]
        gt_anns.append({"id": i, "image_id": 1, "category_id": mapping[d["cls"]],
                        "segmentation": rle, "iscrowd": 0,
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "area": float(mask_utils.area(rle))})
    gt = {"images": [{"id": 1, "file_name": "a.jpg", "width": W, "height": H}],
          "annotations": gt_anns, "categories": CATS}

    with tempfile.TemporaryDirectory() as tmp:
        gp = os.path.join(tmp, "gt.json")
        json.dump(gt, open(gp, "w"))
        coco_gt = COCO(gp)
        for iou_type in ("bbox", "segm"):
            coco_dt = coco_gt.loadRes([dict(d) for d in coco_dets])
            ev = COCOeval(coco_gt, coco_dt, iou_type)
            ev.evaluate(); ev.accumulate(); ev.summarize()
            assert abs(ev.stats[0] - 1.0) < 1e-6, \
                "%s mAP = %.4f, expected 1.0" % (iou_type, ev.stats[0])


def test_missing_mask_still_exports_box():
    dets = [{"cls": 0, "score": 0.5, "bbox_xyxy": [1, 2, 11, 22], "mask": None}]
    out = masks_to_coco(dets, 3, {0: 1})
    assert len(out) == 1 and "segmentation" not in out[0]
    assert out[0]["bbox"] == [1.0, 2.0, 10.0, 20.0]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS %s" % name)
    print("\nALL CHECKS PASSED: RLE export scores 1.0 on bbox+segm, "
          "class mapping is name-based and fails loudly")
