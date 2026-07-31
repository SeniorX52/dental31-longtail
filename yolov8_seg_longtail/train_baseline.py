#!/usr/bin/env python3
"""Stock YOLOv8-seg baseline — deliberately unmodified.

This is the control arm. It exists as its own script (rather than a flag on
`train_seg.py`) so that no long-tail modification can leak into the baseline
by accident: this file imports nothing from the method modules and passes
ultralytics' defaults straight through.

Per BASELINE_PROTOCOL.md the baseline gets a *longer* schedule than the
reference ~50-epoch schedule, so the method cannot later win on training
length alone.

Usage:
    python yolov8_seg_longtail/train_baseline.py --data data/data.yaml \\
        --model yolov8s-seg.pt --epochs 100 --imgsz 640 --batch 16 --seed 42
"""
import argparse
import json
import os
import subprocess
from typing import List, Optional


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "not-a-git-repo"


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8s-seg.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--project", default="runs/baseline")
    ap.add_argument("--name", default="yolov8s_seg_baseline")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    overrides = dict(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        seed=args.seed, deterministic=True,
        project=args.project, name=args.name, exist_ok=True,
        # everything below is ultralytics' default, stated explicitly so the
        # record shows the baseline was not quietly tuned
        optimizer="auto", lr0=0.01, lrf=0.01, momentum=0.937,
        weight_decay=0.0005, warmup_epochs=3.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1,
        scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.0, copy_paste=0.0,
    )
    if args.device is not None:
        overrides["device"] = args.device

    model = YOLO(args.model)
    results = model.train(**overrides)

    # provenance record next to the weights, so every number is traceable
    save_dir = str(getattr(results, "save_dir", os.path.join(args.project, args.name)))
    record = {"role": "baseline", "script": os.path.basename(__file__),
              "model": args.model, "commit": git_commit(),
              "overrides": {k: v for k, v in overrides.items()},
              "save_dir": save_dir}
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "provenance.json"), "w") as f:
        json.dump(record, f, indent=1)

    print("\nbaseline weights: %s/weights/best.pt" % save_dir)
    print("next: score it through the shared scorer, not ultralytics' metric:")
    print("  python yolov8_seg_longtail/predict_to_coco.py --weights %s/weights/best.pt \\\n"
          "      --gt annotations/instances_test.json --images data/test/images \\\n"
          "      --out preds/yolo_baseline_test.json" % save_dir)


if __name__ == "__main__":
    main()
