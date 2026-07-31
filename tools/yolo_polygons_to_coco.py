#!/usr/bin/env python3
"""Convert YOLO polygon (segmentation) labels to COCO instance-segmentation JSON.

Why this exists: the dataset ships with YOLO polygon labels, but its COCO JSON
export has `segmentation: []` (boxes only). Mask supervision and mask-mAP
evaluation both require real COCO segmentation records, so this script rebuilds
the COCO JSON from the YOLO labels as the single source of truth.

YOLO polygon label format, one instance per line:
    <class_id> x1 y1 x2 y2 ... xn yn        (all coords normalized to [0,1])

Output follows the COCO instances schema: polygons in absolute pixel coords,
bbox = [x, y, w, h] derived from the polygon, area = polygon area (shoelace,
summed over parts), iscrowd = 0.

Usage:
    python tools/yolo_polygons_to_coco.py \
        --images data/train/images --labels data/train/labels \
        --names data/data.yaml --out annotations/instances_train.json

`--names` accepts either a YOLO data.yaml (reads its `names:` list/dict) or a
plain text file with one class name per line.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def load_class_names(path: str) -> List[str]:
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
            with open(path) as f:
                data = yaml.safe_load(f)
            names = data["names"]
        except ImportError:
            names = _parse_yaml_names_minimal(path)
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names, key=int)]
        return list(names)
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def _parse_yaml_names_minimal(path: str) -> List[str]:
    """Fallback data.yaml `names:` parser for environments without PyYAML.

    Handles the two layouts Ultralytics emits: inline list and indented
    `id: name` mapping.
    """
    with open(path) as f:
        lines = f.readlines()
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if not stripped.startswith("names:"):
            continue
        rest = stripped[len("names:"):].strip()
        if rest.startswith("["):  # inline list
            return [x.strip().strip("'\"") for x in rest.strip("[]").split(",")]
        mapping = {}
        for sub in lines[i + 1:]:
            if not sub.startswith((" ", "\t")):
                break
            sub = sub.strip()
            if ":" not in sub:
                break
            key, _, val = sub.partition(":")
            key = key.strip().lstrip("- ")
            if not key.isdigit():
                break
            mapping[int(key)] = val.strip().strip("'\"")
        if mapping:
            return [mapping[k] for k in sorted(mapping)]
    raise ValueError("could not find a `names:` entry in %s (install pyyaml?)" % path)


def image_size(path: str) -> Tuple[int, int]:
    """Return (width, height) without requiring cv2 for the common formats."""
    try:
        import cv2  # type: ignore
        img = cv2.imread(path)
        if img is None:
            raise IOError("cv2 could not read %s" % path)
        h, w = img.shape[:2]
        return w, h
    except ImportError:
        from PIL import Image  # type: ignore
        with Image.open(path) as im:
            return im.size


def shoelace_area(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    acc = 0.0
    for i in range(n):
        j = (i + 1) % n
        acc += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(acc) / 2.0


def parse_label_file(path: str) -> List[Tuple[int, List[float]]]:
    """Return [(class_id, [x1,y1,...])] with coords still normalized."""
    out = []
    with open(path) as f:
        for lineno, ln in enumerate(f, 1):
            parts = ln.split()
            if not parts:
                continue
            if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                # A valid polygon needs >=3 points; 5 fields would be a bbox
                # label, which means the label dir mixes formats — flag it.
                raise ValueError(
                    "%s:%d has %d fields; expected odd count >=7 "
                    "(class + >=3 xy pairs)" % (path, lineno, len(parts)))
            out.append((int(parts[0]), [float(v) for v in parts[1:]]))
    return out


def convert(images_dir: str,
            labels_dir: str,
            class_names: List[str],
            min_points: int = 3,
            clamp: bool = True) -> Dict:
    images, annotations = [], []
    stats = {name: 0 for name in class_names}
    dropped = {"degenerate_polygon": 0, "bad_class_id": 0, "missing_label_file": 0}

    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS))
    if not files:
        raise SystemExit("no images found in %s" % images_dir)

    ann_id = 1
    for img_id, fname in enumerate(files, 1):
        w, h = image_size(os.path.join(images_dir, fname))
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})

        label_path = os.path.join(labels_dir, os.path.splitext(fname)[0] + ".txt")
        if not os.path.exists(label_path):
            dropped["missing_label_file"] += 1  # negative sample: image kept, no anns
            continue

        for cls_id, coords in parse_label_file(label_path):
            if not 0 <= cls_id < len(class_names):
                dropped["bad_class_id"] += 1
                continue
            xs = coords[0::2]
            ys = coords[1::2]
            if clamp:
                xs = [min(max(x, 0.0), 1.0) for x in xs]
                ys = [min(max(y, 0.0), 1.0) for y in ys]
            xs = [x * w for x in xs]
            ys = [y * h for y in ys]
            area = shoelace_area(xs, ys)
            if len(xs) < min_points or area < 1.0:  # < 1 px^2: unusable for masks
                dropped["degenerate_polygon"] += 1
                continue
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            poly = [round(v, 2) for pair in zip(xs, ys) for v in pair]
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id + 1,  # COCO ids are 1-based
                "segmentation": [poly],
                "bbox": [round(x0, 2), round(y0, 2),
                         round(x1 - x0, 2), round(y1 - y0, 2)],
                "area": round(area, 2),
                "iscrowd": 0,
            })
            stats[class_names[cls_id]] += 1
            ann_id += 1

    coco = {
        "info": {"description": "converted from YOLO polygon labels by yolo_polygons_to_coco.py"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": n, "supercategory": "dental"}
                       for i, n in enumerate(class_names)],
    }
    return {"coco": coco, "per_class": stats, "dropped": dropped}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--names", required=True,
                    help="data.yaml or one-name-per-line txt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    names = load_class_names(args.names)
    result = convert(args.images, args.labels, names)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result["coco"], f)

    n_img = len(result["coco"]["images"])
    n_ann = len(result["coco"]["annotations"])
    print("wrote %s: %d images, %d annotations, %d classes"
          % (args.out, n_img, n_ann, len(names)))
    print("dropped: %s" % result["dropped"])
    print("\nper-class instance counts (ascending — check the tail):")
    for name, cnt in sorted(result["per_class"].items(), key=lambda kv: kv[1]):
        print("  %6d  %s" % (cnt, name))


if __name__ == "__main__":
    main()
