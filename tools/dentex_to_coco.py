#!/usr/bin/env python3
"""Convert the DENTEX 2023 disease subset into our COCO vocabulary.

WHY THIS DATASET, AND WHY NOW. Three of the four DENTEX diagnosis classes map
directly onto ours, and two of them are the classes sitting at our metric floor:

    DENTEX 'Caries'            -> our 'Caries'              (0.094 on our valid)
    DENTEX 'Deep Caries'       -> our 'Caries'  (see below)
    DENTEX 'Periapical Lesion' -> our 'Periapical lesion'   (0.031)
    DENTEX 'Impacted'          -> our 'impacted tooth'      (0.490)

That is the comparison the project has been missing. Our caries and periapical
numbers are near zero and 911 annotations of those classes are invisible to
every model we have trained, which is consistent with two very different
stories: the task is intrinsically hard, or our labels are unreliable. DENTEX
is professionally annotated for a MICCAI challenge, so scoring our own model on
it separates the two. Comparable performance means the difficulty is real;
markedly better on DENTEX means the problem is our labels. It also satisfies the
standing requirement that no long-tail claim be made without an external
benchmark.

The impacted-tooth class is the control. It scores well on our corpus, so if the
domain gap alone were responsible for a collapse on DENTEX we would expect to
see it there too; if impacted transfers and caries does not, the gap is not the
explanation.

DEEP CARIES. Our vocabulary has a single undifferentiated 'Caries'; DENTEX
splits caries by depth. Deep caries is caries, so it is merged by default and
the strict mapping is available with --no-merge-deep-caries. Both are reported,
because the merge changes the support materially (2189 -> 2767 instances).

WHAT THIS IS NOT. This is a zero-shot cross-dataset evaluation: different
clinics, different machines, a different annotation protocol and a different
image geometry (2744x1316 against our 1615x840). The number it produces is a
LOWER bound on transferable performance, not an estimate of DENTEX-native
performance, and it should never be quoted as the latter.

Usage:
    python tools/dentex_to_coco.py \\
        --dentex /media/mostafa/EGYPT_SSD/dental31/dentex/extracted \\
        --our-gt data_clean/annotations/instances_valid.json \\
        --out data_external/dentex
"""
import argparse
import json
import os

MAP = {
    "Impacted": "impacted tooth",
    "Caries": "Caries",
    "Periapical Lesion": "Periapical lesion",
    "Deep Caries": "Caries",          # merged unless --no-merge-deep-caries
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dentex", required=True, help="extracted DENTEX root")
    ap.add_argument("--our-gt", required=True,
                    help="one of our COCO files, read only for the vocabulary")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-merge-deep-caries", action="store_true")
    args = ap.parse_args()

    ours = json.load(open(args.our_gt))
    our_id = {c["name"]: c["id"] for c in ours["categories"]}

    src_dir = os.path.join(args.dentex, "training_data",
                           "quadrant-enumeration-disease")
    src_json = os.path.join(src_dir, "train_quadrant_enumeration_disease.json")
    src_imgs = os.path.join(src_dir, "xrays")
    if not os.path.isfile(src_json):
        raise SystemExit("DENTEX disease json not found at %s" % src_json)

    d = json.load(open(src_json))
    dis = {c["id"]: c["name"] for c in d["categories_3"]}

    mapping = dict(MAP)
    if args.no_merge_deep_caries:
        mapping.pop("Deep Caries")

    # keep only our vocabulary, remapped onto OUR ids so that predictions from
    # our models can be scored against this file without any translation
    keep_names = sorted({v for v in mapping.values()})
    missing = [n for n in keep_names if n not in our_id]
    if missing:
        raise SystemExit("target class not in our vocabulary: %s" % missing)

    anns, used_imgs, per_class = [], set(), {}
    dropped = 0
    for a in d["annotations"]:
        name = dis.get(a.get("category_id_3"))
        tgt = mapping.get(name)
        if tgt is None:
            dropped += 1
            continue
        rec = {
            "id": len(anns) + 1,
            "image_id": a["image_id"],
            "category_id": our_id[tgt],
            "bbox": a["bbox"],
            "area": a.get("area", a["bbox"][2] * a["bbox"][3]),
            "iscrowd": a.get("iscrowd", 0),
        }
        seg = a.get("segmentation")
        if seg:
            rec["segmentation"] = seg
        anns.append(rec)
        used_imgs.add(a["image_id"])
        per_class[tgt] = per_class.get(tgt, 0) + 1

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    images, n_link = [], 0
    for im in d["images"]:
        if im["id"] not in used_imgs:
            continue
        images.append({"id": im["id"], "file_name": im["file_name"],
                       "width": im["width"], "height": im["height"]})
        s = os.path.join(src_imgs, im["file_name"])
        t = os.path.join(args.out, "images", im["file_name"])
        if os.path.exists(s) and not os.path.exists(t):
            os.symlink(os.path.abspath(s), t)
            n_link += 1

    # Carry OUR FULL category list, not just the shared three. A model that
    # predicts all 31 classes must be allowed to emit any of them here; the
    # evaluation is then restricted to the shared classes at scoring time.
    out = {"images": images, "annotations": anns,
           "categories": ours["categories"]}
    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, "instances_dentex_disease.json")
    with open(dst, "w") as f:
        json.dump(out, f)

    with_seg = sum(1 for a in anns if "segmentation" in a)
    print("  images  : %d (%d symlinked)" % (len(images), n_link))
    print("  anns    : %d kept, %d dropped as outside our vocabulary"
          % (len(anns), dropped))
    print("  with segmentation: %d (%.0f %%)"
          % (with_seg, 100 * with_seg / max(len(anns), 1)))
    print("  deep caries merged into Caries: %s"
          % ("no" if args.no_merge_deep_caries else "yes"))
    print("  per class (our ids):")
    for n in sorted(per_class, key=lambda k: -per_class[k]):
        print("    %-22s id %-3d %6d" % (n, our_id[n], per_class[n]))
    print("  wrote %s" % dst)
    print("\n  score our model on it with:")
    print("    python yolov8_seg_longtail/predict_to_coco.py --weights <best.pt> \\")
    print("      --gt %s \\" % dst)
    print("      --images %s/images --out preds/dentex_<tag>.json" % args.out)


if __name__ == "__main__":
    main()
