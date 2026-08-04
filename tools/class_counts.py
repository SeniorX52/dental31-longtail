#!/usr/bin/env python3
"""Per-class IMAGE-level and instance-level counts for every split.

The audit reports instance counts. Instance counts overstate how much
independent evidence a class carries: 21 orthodontic-bracket instances spread
over 2 radiographs is two observations, not twenty-one, and a per-class AP
computed from them moves on whether those two images happened to land in the
split. Image-level counts are what determines whether a per-class claim can be
made at all, so both are reported side by side.

Also emits, per class, the number of evaluation images containing it, which is
the denominator any per-class confidence interval is really built on.

Usage:
    python tools/class_counts.py --ann-dir data_clean/annotations --out reports/class_counts
"""
import argparse
import json
import os
from collections import defaultdict

HEAD_MIN = 5000
TAIL_MAX = 100


def counts_for(path):
    with open(path) as f:
        d = json.load(f)
    names = {c["id"]: c["name"] for c in d["categories"]}
    inst = defaultdict(int)
    imgs = defaultdict(set)
    for a in d["annotations"]:
        inst[a["category_id"]] += 1
        imgs[a["category_id"]].add(a["image_id"])
    return names, inst, {k: len(v) for k, v in imgs.items()}, len(d["images"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann-dir", required=True)
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]
    names, inst, imgs, totals = {}, {}, {}, {}
    for s in splits:
        p = os.path.join(args.ann_dir, "instances_%s.json" % s)
        n, i, m, t = counts_for(p)
        names.update(n)
        inst[s], imgs[s], totals[s] = i, m, t

    train_inst = inst.get("train", {})

    def group_of(c):
        n = train_inst.get(c, 0)
        return "head" if n > HEAD_MIN else ("tail" if n < TAIL_MAX else "mid")

    rows = []
    for cid in sorted(names, key=lambda c: -train_inst.get(c, 0)):
        r = {"category_id": cid, "class": names[cid], "group": group_of(cid)}
        for s in splits:
            r["%s_instances" % s] = inst[s].get(cid, 0)
            r["%s_images" % s] = imgs[s].get(cid, 0)
        rows.append(r)

    result = {"source": args.ann_dir,
              "split_image_totals": totals,
              "grouping": {"head_min_train_instances": HEAD_MIN,
                           "tail_max_train_instances": TAIL_MAX},
              "per_class": rows}
    with open(args.out + ".json", "w") as f:
        json.dump(result, f, indent=1)

    L = ["# Per-class counts — `%s`" % args.ann_dir, "",
         "Images per split: " + ", ".join("%s %d" % (s, totals[s]) for s in splits) + ".", "",
         "`inst` = annotation instances. `imgs` = distinct images containing the class; "
         "this is the effective sample size for any per-class claim.", "",
         "| class | group | " + " | ".join("%s inst | %s imgs" % (s, s) for s in splits) + " |",
         "|---|---|" + "---|---|" * len(splits)]
    for r in rows:
        cells = " | ".join("%d | %d" % (r["%s_instances" % s], r["%s_images" % s])
                           for s in splits)
        L.append("| %s | %s | %s |" % (r["class"], r["group"], cells))

    ev = "valid" if "valid" in splits else splits[-1]
    thin = [r for r in rows if r["%s_images" % ev] < 10]
    L += ["", "## Classes that cannot support a per-class claim", "",
          "%d of %d classes appear in fewer than 10 **%s** images. For these, a "
          "per-class AP or Dice is determined by a handful of radiographs and "
          "will not survive a change of split, so they are reported inside the "
          "tail group rather than individually."
          % (len(thin), len(rows), ev), "",
          "| class | %s images | %s instances |" % (ev, ev), "|---|---|---|"]
    for r in sorted(thin, key=lambda r: r["%s_images" % ev]):
        L.append("| %s | %d | %d |" % (r["class"], r["%s_images" % ev],
                                       r["%s_instances" % ev]))
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(L) + "\n")

    print("images per split:", totals)
    print("classes with <10 %s images: %d of %d" % (ev, len(thin), len(rows)))
    print("wrote %s.md and %s.json" % (args.out, args.out))


if __name__ == "__main__":
    main()
