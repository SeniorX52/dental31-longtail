#!/usr/bin/env python3
"""Run a trained DINO checkpoint over a split and write COCO-format detections.

DINO's own `--eval` prints pycocotools numbers, but Project 1 and Project 2
must be judged by ONE scorer (`eval/coco_eval_report.py`) so that per-class and
head/mid/tail figures are computed identically for both. This script therefore
dumps raw detections rather than relying on the repo's internal evaluator.

It imports the DINO repo in place, so run it with the repo on PYTHONPATH:

    PYTHONPATH=$HOME/DINO python dino_longtail/export_dino_preds.py \\
        --dino-root $HOME/DINO \\
        --config $HOME/DINO/config/DINO/DINO_4scale.py \\
        --checkpoint logs/dino_baseline/checkpoint.pth \\
        --coco-path data_coco --split test2017 \\
        --gt data_clean/annotations/instances_test.json \\
        --out preds/dino_baseline_test.json \\
        --options num_classes=32 dn_labelbook_size=32

Scores are kept unthresholded (top-`--max-det` per image) because COCO mAP
integrates over the full ranked list.
"""
import argparse
import json
import os
import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dino-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--coco-path", required=True)
    ap.add_argument("--split", default="test2017",
                    help="image folder name under --coco-path")
    ap.add_argument("--gt", required=True,
                    help="COCO json defining image ids (our instances_*.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-det", type=int, default=100,
                    help="COCOeval's primary mAP uses maxDets=100")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--options", nargs="*", default=[],
                    help="key=value overrides passed to the DINO config")
    ap.add_argument("--clahe", action="store_true",
                    help="apply CLAHE before the transform -- REQUIRED when "
                         "scoring a model trained with lt_clahe, so inference "
                         "sees the same preprocessing as training")
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.abspath(args.dino_root))
    import torch
    from PIL import Image
    import datasets.transforms as T                      # noqa: E402
    from util.slconfig import SLConfig                    # noqa: E402
    from models.registry import MODULE_BUILD_FUNCS        # noqa: E402

    cfg = SLConfig.fromfile(args.config)
    overrides = {}
    for kv in args.options:
        k, _, v = kv.partition("=")
        try:
            v = eval(v, {}, {})       # ints/bools/floats as in DINO's own parser
        except Exception:
            pass
        overrides[k] = v
    cfg.merge_from_dict(overrides)
    # DINO's build functions read these off the args namespace
    cfg.device = args.device
    for k in ("dataset_file", "coco_path"):
        if not hasattr(cfg, k):
            setattr(cfg, k, "coco" if k == "dataset_file" else args.coco_path)

    build_func = MODULE_BUILD_FUNCS.get(cfg.modelname)
    model, _, postprocessors = build_func(cfg)
    model.to(args.device).eval()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("loaded %s | missing %d | unexpected %d"
          % (args.checkpoint, len(missing), len(unexpected)))
    if missing:
        print("  first missing:", list(missing)[:5])

    with open(args.gt) as f:
        gt = json.load(f)
    img_dir = os.path.join(args.coco_path, args.split)

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if args.clahe:
        import cv2 as _cv2
        _clahe_op = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    records, n_missing = [], 0
    import time
    t0 = time.time()
    for idx, im in enumerate(gt["images"], 1):
        path = os.path.join(img_dir, im["file_name"])
        if not os.path.exists(path):
            n_missing += 1
            continue
        image = Image.open(path).convert("RGB")
        if args.clahe:
            import numpy as _np, cv2 as _cv2
            _lab = _cv2.cvtColor(_np.asarray(image), _cv2.COLOR_RGB2LAB)
            _lab[..., 0] = _clahe_op.apply(_lab[..., 0])
            image = Image.fromarray(_cv2.cvtColor(_lab, _cv2.COLOR_LAB2RGB))
        tensor, _ = transform(image, None)
        with torch.no_grad():
            out = model(tensor[None].to(args.device))
        # postprocessor rescales boxes to the ORIGINAL image size
        target_sizes = torch.tensor([[im["height"], im["width"]]], device=args.device)
        res = postprocessors["bbox"](out, target_sizes)[0]

        scores = res["scores"].cpu()
        labels = res["labels"].cpu()
        boxes = res["boxes"].cpu()                       # xyxy, absolute
        keep = scores.argsort(descending=True)[: args.max_det]
        for i in keep.tolist():
            x0, y0, x1, y1 = [float(v) for v in boxes[i]]
            records.append({
                "image_id": int(im["id"]),
                "category_id": int(labels[i]),           # raw id, as DINO trains on
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": float(scores[i]),
            })
        if idx % 100 == 0:
            el = time.time() - t0
            print("[%d/%d] %d dets | %.1f img/s" % (idx, len(gt["images"]),
                                                    len(records), idx / max(el, 1e-9)),
                  flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f)
    print("wrote %s: %d detections over %d images (%d image files missing)"
          % (args.out, len(records), len(gt["images"]), n_missing))
    print("score with eval/coco_eval_report.py --iou-type bbox")


if __name__ == "__main__":
    main()
