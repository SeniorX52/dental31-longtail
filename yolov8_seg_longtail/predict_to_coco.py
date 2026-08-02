#!/usr/bin/env python3
"""Run a YOLO-seg model over a split and export COCO-format detections.

Why: ultralytics' own `val` reports its internal mAP, which is NOT numerically
identical to pycocotools COCOeval (different recall interpolation and
matching details). Both projects must be judged by ONE scorer, so we export
YOLO predictions to COCO JSON and score everything with
`eval/coco_eval_report.py`. The ultralytics number is still recorded
alongside as a sanity cross-check.

Masks are exported as uncompressed-safe RLE (pycocotools `encode`), which is
what COCOeval expects for `iouType=segm`.

Usage:
    python yolov8_seg_longtail/predict_to_coco.py \\
        --weights best.pt --gt annotations/instances_test.json \\
        --images data/test/images --out preds_test_seg.json \\
        --imgsz 640 --conf 0.001 --max-det 300 --seed 42
"""
import argparse
import json
import os
from typing import Dict, List

import numpy as np


def masks_to_coco(dets: List[dict], image_id: int,
                  yolo_idx_to_cat_id: Dict[int, int]) -> List[dict]:
    """dets: [{cls, score, bbox_xyxy, mask(HxW uint8|None)}] -> COCO records."""
    from pycocotools import mask as mask_utils

    out = []
    for d in dets:
        cat_id = yolo_idx_to_cat_id.get(int(d["cls"]))
        if cat_id is None:
            continue
        x0, y0, x1, y1 = [float(v) for v in d["bbox_xyxy"]]
        rec = {"image_id": int(image_id), "category_id": int(cat_id),
               "bbox": [x0, y0, x1 - x0, y1 - y0],
               "score": float(d["score"])}
        m = d.get("mask")
        if m is not None:
            rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            rec["segmentation"] = rle
        out.append(rec)
    return out


def build_class_mapping(model_names: Dict[int, str],
                        categories: List[dict]) -> Dict[int, int]:
    """Map YOLO class index -> COCO category id by NAME, not position.

    Position-based mapping is the classic silent bug when the data.yaml order
    and the COCO category order differ; failing loudly here is deliberate.
    """
    name_to_cat = {c["name"]: c["id"] for c in categories}
    lower = {c["name"].lower(): c["id"] for c in categories}
    mapping, missing = {}, []
    for idx, name in model_names.items():
        cid = name_to_cat.get(name, lower.get(str(name).lower()))
        if cid is None:
            missing.append(name)
        else:
            mapping[int(idx)] = cid
    if missing:
        raise SystemExit(
            "model classes not found in the COCO categories: %s\n"
            "class names must match between data.yaml and the COCO json."
            % missing)
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gt", required=True, help="COCO gt json (defines image ids)")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001,
                    help="low by design: mAP needs the full ranked list")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    ap.add_argument("--max-det", type=int, default=100,
                    help="COCOeval's primary mAP uses maxDets=100, so exporting "
                         "more than 100 per image costs mask-encoding time and "
                         "changes nothing")
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    # Checkpoints written by train_seg.py pickle a reference to
    # __main__.LongTailSegModel / BoundaryAwareSegLoss, because that script was
    # __main__ when it saved them. Loading from any other entry point cannot
    # resolve those names, so re-publish them into this process's __main__
    # before torch.load runs. Without this, every ablation arm's weights are
    # unloadable outside the trainer.
    import __main__
    try:
        from yolov8_seg_longtail.train_seg import (  # noqa: F401
            LongTailSegModel, BoundaryAwareSegLoss, LongTailSegTrainer)
        for _cls in (LongTailSegModel, BoundaryAwareSegLoss, LongTailSegTrainer):
            setattr(__main__, _cls.__name__, _cls)
    except Exception as _e:                      # stock checkpoints don't need it
        print("note: long-tail classes not registered (%s)" % _e)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.gt) as f:
        gt = json.load(f)
    file_to_id = {im["file_name"]: im["id"] for im in gt["images"]}

    model = YOLO(args.weights)
    mapping = build_class_mapping(model.names, gt["categories"])

    records, missing_files, n_imgs = [], [], 0
    import time
    t0 = time.time()
    for idx, im in enumerate(gt["images"], 1):
        if args.progress_every and idx % args.progress_every == 0:
            el = time.time() - t0
            rate = idx / max(el, 1e-9)
            print("[%d/%d] %d dets | %.1f img/s | eta %.1f min"
                  % (idx, len(gt["images"]), len(records), rate,
                     (len(gt["images"]) - idx) / max(rate, 1e-9) / 60), flush=True)
        path = os.path.join(args.images, im["file_name"])
        if not os.path.exists(path):
            missing_files.append(im["file_name"])
            continue
        res = model.predict(path, imgsz=args.imgsz, conf=args.conf,
                            iou=args.iou, max_det=args.max_det,
                            device=args.device, verbose=False, retina_masks=True)[0]
        n_imgs += 1
        if res.boxes is None or len(res.boxes) == 0:
            continue
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy()
        masks = None
        if res.masks is not None:
            masks = res.masks.data.cpu().numpy()  # (N, H, W) at image size
        dets = []
        for i in range(len(boxes)):
            m = None
            if masks is not None and i < len(masks):
                m = masks[i]
                if m.shape != (im["height"], im["width"]):
                    import cv2
                    m = cv2.resize(m.astype(np.float32),
                                   (im["width"], im["height"]),
                                   interpolation=cv2.INTER_NEAREST)
                m = (m > 0.5).astype(np.uint8)
            dets.append({"cls": classes[i], "score": scores[i],
                         "bbox_xyxy": boxes[i], "mask": m})
        records.extend(masks_to_coco(dets, file_to_id[im["file_name"]], mapping))

    with open(args.out, "w") as f:
        json.dump(records, f)
    print("wrote %s: %d detections over %d images" % (args.out, len(records), n_imgs))
    if missing_files:
        print("WARNING: %d images in the GT json were not found on disk (e.g. %s)"
              % (len(missing_files), missing_files[:3]))
    print("score with:\n  python eval/coco_eval_report.py --gt %s --dt %s "
          "--train-json <train.json> --iou-type segm --out <name>"
          % (args.gt, args.out))


if __name__ == "__main__":
    main()
