#!/usr/bin/env python3
"""Stage two of the tooth-conditioned pipeline: per-tooth pathology classification.

THE INDUCTIVE BIAS, MADE CONCRETE. A whole-image lesion detector must decide
what and where simultaneously, against lesion outlines we have measured to be
unreliable (911 undetectable by a fifteen-model consensus, 2.7 percent
recovered by the model that fixed caries). This stage restructures the problem
the way the domain does (DENTEX hierarchy, arXiv:2305.19112; HierarchicalDet,
arXiv:2303.06500): teeth are localised first, and pathology becomes a
multi-label property OF each tooth. The label conversion is the point:
"this tooth has caries" is derived from ANY overlapping lesion annotation, so a
sloppy outline, a shifted box or a merged lesion all collapse to the same
correct tooth-level bit. The supervision gets more reliable exactly where the
original labels are least reliable, and the output granularity is the one the
external check already scores us well on (73.6 to 84.1 percent precision).

MECHANICS.
  1. The stage-one tooth detector (trained on the DENTEX enumeration sets by
     build_tooth_corpus.py + the queue driver) is run over our images.
  2. Every detected tooth above --tooth-conf becomes a sample: crop with a
     context margin, multi-label target = which of the four pathology classes
     have a ground-truth lesion whose centre falls inside the tooth box.
  3. A torchvision ResNet-18 (ImageNet init) trains with BCEWithLogits, the
     standard multi-label treatment; single-label softmax would be wrong since
     one tooth can carry caries AND a root canal AND a periapical lesion.
  4. Evaluation on the validation teeth: per-class average precision
     (implemented directly, no sklearn dependency), plus precision/recall at
     0.5, written to a report.

Usage (the queue driver calls these in order):
    python tools/tooth_stage2.py crops --teeth runs/segment/tooth_det/weights/best.pt \\
        --gt data_clean/annotations/instances_train.json \\
        --images data_clean/train/images --out reports/tooth_crops_train.json
    python tools/tooth_stage2.py train --train reports/tooth_crops_train.json \\
        --val reports/tooth_crops_valid.json --epochs 10 \\
        --model-out weights/tooth_stage2_resnet18.pt \\
        --report reports/toothstage_valid
"""
import argparse
import json
import os

import cv2
import numpy as np

PATHOLOGY = ["Caries", "Periapical lesion", "Root Canal Treatment", "Bone Loss"]


def cmd_crops(args):
    import torch
    from ultralytics import YOLO

    gt = json.load(open(args.gt))
    names = {c["id"]: c["name"] for c in gt["categories"]}
    want = {cid for cid, n in names.items() if n in PATHOLOGY}
    idx_of = {n: i for i, n in enumerate(PATHOLOGY)}
    lesions = {}
    for a in gt["annotations"]:
        if a["category_id"] in want:
            lesions.setdefault(a["image_id"], []).append(
                (idx_of[names[a["category_id"]]], a["bbox"]))
    im_info = {im["id"]: im for im in gt["images"]}

    model = YOLO(args.teeth)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    samples, n_pos = [], 0
    ids = sorted(im_info)
    for i in range(0, len(ids), args.batch):
        chunk = ids[i:i + args.batch]
        paths = [os.path.join(args.images, im_info[j]["file_name"]) for j in chunk]
        results = model.predict(paths, imgsz=args.imgsz, conf=args.tooth_conf,
                                verbose=False, device=dev)
        for j, r in zip(chunk, results):
            for b in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = [float(v) for v in b]
                labels = [0, 0, 0, 0]
                for (k, (lx, ly, lw, lh)) in lesions.get(j, []):
                    cx, cy = lx + lw / 2, ly + lh / 2
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        labels[k] = 1
                n_pos += any(labels)
                samples.append({"image": im_info[j]["file_name"],
                                "box": [x1, y1, x2, y2], "labels": labels})
        if (i // args.batch) % 20 == 0:
            print("  %d/%d images, %d teeth so far" % (i + len(chunk), len(ids), len(samples)), flush=True)

    json.dump({"images_dir": os.path.abspath(args.images),
               "classes": PATHOLOGY, "samples": samples},
              open(args.out, "w"))
    print("  %d tooth crops (%d with at least one pathology, %.1f %%) -> %s"
          % (len(samples), n_pos, 100 * n_pos / max(len(samples), 1), args.out))


def average_precision(scores, labels):
    order = np.argsort(-scores)
    l = labels[order]
    if l.sum() == 0:
        return float("nan")
    tp = np.cumsum(l)
    prec = tp / (np.arange(len(l)) + 1)
    return float((prec * l).sum() / l.sum())


def cmd_train(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torchvision

    class Crops(Dataset):
        def __init__(self, manifest, train):
            d = json.load(open(manifest))
            self.dir, self.samples = d["images_dir"], d["samples"]
            self.train = train
            self.cache = {}

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            s = self.samples[i]
            im = cv2.imread(os.path.join(self.dir, s["image"]))
            h, w = im.shape[:2]
            x1, y1, x2, y2 = s["box"]
            mx, my = 0.12 * (x2 - x1), 0.12 * (y2 - y1)   # context margin
            x1, y1 = max(int(x1 - mx), 0), max(int(y1 - my), 0)
            x2, y2 = min(int(x2 + mx), w), min(int(y2 + my), h)
            crop = im[y1:y2, x1:x2]
            if crop.size == 0:
                crop = im
            crop = cv2.resize(crop, (224, 224))
            if self.train and np.random.rand() < 0.5:
                crop = crop[:, ::-1]                       # horizontal flip
            t = torch.from_numpy(crop[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255
            t = (t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) \
                / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return t, torch.tensor(s["labels"], dtype=torch.float32)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = DataLoader(Crops(args.train, True), batch_size=args.batch,
                    shuffle=True, num_workers=4, drop_last=True)
    va = DataLoader(Crops(args.val, False), batch_size=args.batch,
                    shuffle=False, num_workers=4)
    net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    net.fc = nn.Linear(net.fc.in_features, len(PATHOLOGY))
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.BCEWithLogitsLoss()

    for ep in range(args.epochs):
        net.train()
        tot = n = 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = crit(net(x), y)
            loss.backward()
            opt.step()
            tot += float(loss) * len(x); n += len(x)
        sched.step()
        print("  epoch %d/%d  train BCE %.4f" % (ep + 1, args.epochs, tot / max(n, 1)), flush=True)

    net.eval()
    S, L = [], []
    import torch as _t
    with _t.no_grad():
        for x, y in va:
            S.append(_t.sigmoid(net(x.to(dev))).cpu().numpy())
            L.append(y.numpy())
    S, L = np.concatenate(S), np.concatenate(L)

    report = {"classes": PATHOLOGY, "n_val_teeth": int(len(S)), "per_class": {}}
    print("\n  TOOTH-LEVEL validation (%d teeth):" % len(S))
    print("  %-24s %8s %10s %8s %8s %8s" % ("class", "AP", "prevalence", "P@0.5", "R@0.5", "n_pos"))
    for k, name in enumerate(PATHOLOGY):
        ap_ = average_precision(S[:, k], L[:, k])
        pred = S[:, k] >= 0.5
        tp = int((pred & (L[:, k] == 1)).sum())
        p = tp / max(int(pred.sum()), 1)
        r = tp / max(int(L[:, k].sum()), 1)
        report["per_class"][name] = {"AP": ap_, "precision_at_0.5": p,
                                     "recall_at_0.5": r, "n_pos": int(L[:, k].sum())}
        print("  %-24s %8.4f %9.1f%% %7.1f%% %7.1f%% %8d"
              % (name, ap_, 100 * L[:, k].mean(), 100 * p, 100 * r, int(L[:, k].sum())))
    torch_path = args.model_out
    _t.save(net.state_dict(), torch_path)
    json.dump(report, open(args.report + ".json", "w"), indent=1)
    print("  saved %s and %s.json" % (torch_path, args.report))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("crops")
    a.add_argument("--teeth", required=True, help="stage-one tooth detector weights")
    a.add_argument("--gt", required=True)
    a.add_argument("--images", required=True)
    a.add_argument("--imgsz", type=int, default=1024)
    a.add_argument("--tooth-conf", type=float, default=0.5)
    a.add_argument("--batch", type=int, default=16)
    a.add_argument("--out", required=True)
    b = sub.add_parser("train")
    b.add_argument("--train", required=True)
    b.add_argument("--val", required=True)
    b.add_argument("--epochs", type=int, default=10)
    b.add_argument("--batch", type=int, default=64)
    b.add_argument("--model-out", required=True)
    b.add_argument("--report", required=True)
    args = ap.parse_args()
    (cmd_crops if args.cmd == "crops" else cmd_train)(args)


if __name__ == "__main__":
    main()
