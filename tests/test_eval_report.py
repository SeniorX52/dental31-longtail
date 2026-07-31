#!/usr/bin/env python3
"""Test the eval report tool on synthetic GT/detections with known quality.

Class 'head' gets perfect detections, class 'tail' gets badly shifted ones —
the report must rank head AP ~1.0, tail AP low, group them correctly from the
TRAIN json counts, and flag the tail class as unstable (<10 eval instances).

Run:  python tests/test_eval_report.py
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(ROOT, "eval", "coco_eval_report.py")


def box_ann(aid, img, cat, x, y, w, h):
    return {"id": aid, "image_id": img, "category_id": cat,
            "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
            "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]]}


def main():
    with tempfile.TemporaryDirectory() as tmp:
        cats = [{"id": 1, "name": "head_cls"}, {"id": 2, "name": "tail_cls"}]
        images = [{"id": i, "file_name": "%d.jpg" % i, "width": 200, "height": 200}
                  for i in range(12)]

        # eval GT: 12 head instances (stable), 4 tail instances (unstable: < 10)
        gt_anns, aid = [], 1
        for i in range(12):
            gt_anns.append(box_ann(aid, i, 1, 10, 10, 50, 50)); aid += 1
            if i < 4:
                gt_anns.append(box_ann(aid, i, 2, 100, 100, 40, 40)); aid += 1
        gt = {"images": images, "annotations": gt_anns, "categories": cats}
        gt_path = os.path.join(tmp, "gt.json")
        json.dump(gt, open(gt_path, "w"))

        # train json defines grouping: head 6000 instances, tail 40
        tr_imgs = [{"id": 0, "file_name": "t.jpg", "width": 10, "height": 10}]
        tr_anns = [dict(box_ann(i + 1, 0, 1, 0, 0, 2, 2)) for i in range(6000)]
        tr_anns += [dict(box_ann(6001 + i, 0, 2, 0, 0, 2, 2)) for i in range(40)]
        train = {"images": tr_imgs, "annotations": tr_anns, "categories": cats}
        train_path = os.path.join(tmp, "train.json")
        json.dump(train, open(train_path, "w"))

        # detections: head perfect, tail shifted far off (IoU ~ 0.09)
        dets = []
        for a in gt_anns:
            d = {"image_id": a["image_id"], "category_id": a["category_id"],
                 "bbox": list(a["bbox"]), "score": 0.9}
            if a["category_id"] == 2:
                d["bbox"][0] += 30
                d["bbox"][1] += 30
            dets.append(d)
        dt_path = os.path.join(tmp, "dt.json")
        json.dump(dets, open(dt_path, "w"))

        out = os.path.join(tmp, "report")
        subprocess.run([sys.executable, SCRIPT, "--gt", gt_path, "--dt", dt_path,
                        "--train-json", train_path, "--iou-type", "bbox",
                        "--out", out], check=True, stdout=subprocess.PIPE)

        rep = json.load(open(out + ".json"))
        by_name = {r["class"]: r for r in rep["per_class"]}
        assert by_name["head_cls"]["group"] == "head"
        assert by_name["tail_cls"]["group"] == "tail"
        assert by_name["head_cls"]["AP"] > 0.99
        assert by_name["tail_cls"]["AP"] < 0.10
        assert by_name["tail_cls"]["unstable"] is True
        assert by_name["head_cls"]["unstable"] is False
        assert rep["group_AP"]["head"] > 0.99 > rep["group_AP"]["tail"]
        assert os.path.exists(out + ".md")

    print("ALL CHECKS PASSED: eval report computes per-class AP, groups, "
          "and stability flags correctly")


if __name__ == "__main__":
    main()
