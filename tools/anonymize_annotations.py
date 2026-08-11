#!/usr/bin/env python3
"""Strip patient-identifying prefixes from COCO annotation file names.

Source images in the public Roboflow/Kaggle export carry patient names in at
least two different layouts:

    <uid>-<SURNAME>_<FORENAME>_<YYYY-MM-DD><HHMMSS>_jpg.rf.<sha>.jpg
    cropped_<FORENAME>-<SURNAME>_<DD-MM-YYYY>_<n>_png.rf.<sha>.jpg

Both carry a name and an examination date. Both also end in a Roboflow content
hash that is already unique across the corpus, so dropping everything before it
removes the identifying information while leaving every image individually
addressable:

    rf.<sha>.jpg

Do not try to enumerate the identifying layouts. The check at the end of this
script is a whitelist: every written name must reduce to exactly
``rf.<hash>.<ext>``, and anything else is a failure.

Anyone reconstructing the dataset can match published annotations back to their
own download by that hash, so the anonymised files stay fully reproducible.

Writes the forward mapping to a separate file so the original names remain
recoverable locally. That mapping is identifying data: keep it out of git.

    python3 tools/anonymize_annotations.py \
        --in  data_coco/annotations \
        --out data_coco_public/annotations \
        --map _internal/filename_map.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# Everything before the Roboflow hash, whatever shape it takes. Anchored at the
# start, so a bare "rf.<sha>.jpg" passes through unchanged and the script is
# idempotent.
PREFIX = re.compile(r"^.*?(?=rf\.)")

# The only shape allowed out. Whitelist, not blacklist: a name that fails this
# is rejected regardless of whether it looks identifying to us.
SAFE = re.compile(r"^rf\.[0-9a-f]{6,}\.(jpg|jpeg|png)$", re.IGNORECASE)


def anonymize(name: str) -> str:
    """Reduce one file name to its Roboflow content hash.

    Names with no Roboflow hash fall back to a digest of the original, so an
    unexpected layout is still stripped rather than passed through.
    """
    if "rf." in name:
        return PREFIX.sub("", name)
    ext = Path(name).suffix.lower() or ".jpg"
    return "rf." + hashlib.sha256(name.encode()).hexdigest()[:32] + ext


def process(src: Path, dst: Path, mapping: dict[str, str]) -> tuple[int, int]:
    coco = json.loads(src.read_text())
    images = coco.get("images", [])
    changed = 0

    for image in images:
        original = image["file_name"]
        clean = anonymize(original)
        if clean != original:
            mapping[clean] = original
            image["file_name"] = clean
            changed += 1

    coco.setdefault("info", {})["description"] = (
        "Dental panoramic radiographs, 31 classes. File names reduced to their "
        "Roboflow content hash; patient identifiers removed."
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(coco))
    return len(images), changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", dest="dst", type=Path, required=True)
    ap.add_argument("--map", dest="map", type=Path, required=True)
    args = ap.parse_args()

    mapping: dict[str, str] = {}
    files = sorted(args.src.glob("*.json"))
    if not files:
        print(f"no .json files under {args.src}", file=sys.stderr)
        return 1

    for src in files:
        total, changed = process(src, args.dst / src.name, mapping)
        print(f"{src.name}: {total} images, {changed} renamed")

    args.map.parent.mkdir(parents=True, exist_ok=True)
    args.map.write_text(json.dumps(mapping, indent=1, sort_keys=True))
    print(f"mapping for {len(mapping)} names -> {args.map}")

    # Whitelist check on what was actually written to disk. Every name must
    # reduce to rf.<hash>.<ext>; anything else fails the run.
    checked = 0
    leaked = []
    for out in sorted(args.dst.glob("*.json")):
        for image in json.loads(out.read_text()).get("images", []):
            checked += 1
            if not SAFE.match(image["file_name"]):
                leaked.append(f"{out.name}: {image['file_name']}")

    if leaked:
        print(
            f"REJECTED: {len(leaked)} of {checked} names are not rf.<hash>.<ext>",
            *leaked[:10],
            sep="\n  ",
            file=sys.stderr,
        )
        return 2

    print(f"verified: all {checked} names reduce to rf.<hash>.<ext>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
