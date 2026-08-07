#!/usr/bin/env python3
"""Locate the mask bottleneck: prototype basis, or coefficient head?

YOLOv8-seg does not predict masks directly. It predicts 32 coefficients per
instance and reconstructs the mask as `sigmoid(coeffs @ prototypes)`. Two things
can therefore limit mask quality, and they call for opposite fixes:

  * the learned 32-prototype BASIS cannot represent these shapes, or
  * the basis is fine and the COEFFICIENT HEAD fails to find the right point in
    it.

The measurements so far do not separate these. `mask_resolution_ceiling.py`
bounds what the prototype GRID can represent (mean Dice 0.8963 at input/4), and
the trained model achieves 0.6969 -- a 20-point gap that no intervention closed.
This script attributes that gap.

METHOD. For each ground-truth instance we take the trained model's own
prototypes and solve, in closed form, for the coefficient vector that best
reconstructs that instance:

    c* = argmin_c || P^T c - y ||^2      over the instance's box crop

where P is the (32, H, W) prototype stack and y the ground-truth mask mapped to
{-1, +1}. Thresholding `P^T c*` at zero gives the best mask ANY coefficient
head could produce from this basis. Ordinary least squares, no training.

READING THE RESULT.
  oracle Dice near 0.89  -> the basis spans the space; the COEFFICIENT HEAD is
                            the bottleneck, and the fix is head capacity or
                            supervision, not resolution
  oracle Dice near 0.75  -> the 32-prototype BASIS is inadequate for this data,
                            and more prototypes or a different mask
                            parameterisation is the fix

Either answer is a located finding. The current state -- a 20-point gap of
unknown origin -- is not.

Usage:
    python tools/oracle_coefficients.py \
        --weights runs/segment/abl_S0/weights/best.pt \
        --gt data_clean/annotations/instances_valid.json \
        --images data_clean/valid/images --limit 400 \
        --out reports/oracle_coefficients
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cv2.setNumThreads(1)


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else None


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=400,
                    help="images to sample; the aggregate converges quickly")
    ap.add_argument("--ridge", type=float, default=1e-3,
                    help="tiny ridge term; the normal equations are near-singular "
                         "when an instance covers few pixels")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import __main__
    from yolov8_seg_longtail.train_seg import (  # noqa: F401
        BoundaryAwareSegLoss, HighResProto, LongTailSegModel, LongTailSegTrainer,
        P2Proto, P2ProtoSegModel)
    for cls in (LongTailSegModel, BoundaryAwareSegLoss, LongTailSegTrainer,
                P2ProtoSegModel, P2Proto, HighResProto):
        setattr(__main__, cls.__name__, cls)

    from pycocotools.coco import COCO

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = (ck.get("ema") or ck.get("model")).float().eval().to(dev)

    # capture the prototype stack straight off the head
    grab = {}
    head = model.model[-1]
    head.proto.register_forward_hook(lambda m, i, o: grab.__setitem__("p", o))

    coco = COCO(args.gt)
    img_ids = sorted(coco.getImgIds())[: args.limit]

    rows = []
    for k, iid in enumerate(img_ids):
        info = coco.loadImgs(iid)[0]
        path = os.path.join(args.images, info["file_name"])
        im = cv2.imread(path)
        if im is None:
            continue
        h0, w0 = im.shape[:2]
        x = cv2.resize(im, (args.imgsz, args.imgsz))[:, :, ::-1].copy()
        t = torch.from_numpy(x).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        with torch.no_grad():
            model(t)
        P = grab.get("p")
        if P is None:
            raise SystemExit("prototype hook produced nothing")
        P = P[0].float().cpu().numpy()                 # (32, ph, pw)
        nproto, ph, pw = P.shape

        anns = coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=None))
        for a in anns:
            m = coco.annToMask(a)
            if m.sum() == 0:
                continue
            # ground truth on the prototype grid, exactly where the mask lives
            g = cv2.resize(m.astype(np.float32), (pw, ph),
                           interpolation=cv2.INTER_AREA)
            g = (g >= 0.5).astype(np.float32)
            if g.sum() == 0:
                continue
            bx, by, bw, bh = a["bbox"]
            x1 = max(int(bx / w0 * pw) - 1, 0); x2 = min(int((bx + bw) / w0 * pw) + 1, pw)
            y1 = max(int(by / h0 * ph) - 1, 0); y2 = min(int((by + bh) / h0 * ph) + 1, ph)
            if x2 <= x1 or y2 <= y1:
                continue

            Pc = P[:, y1:y2, x1:x2].reshape(nproto, -1)          # (32, npix)
            gc = g[y1:y2, x1:x2].reshape(-1)                      # (npix,)
            if gc.sum() == 0:
                continue
            y = (2.0 * gc - 1.0)                                  # {-1,+1}

            # closed-form ridge least squares: c = (PP^T + lam I)^-1 P y
            A = Pc @ Pc.T + args.ridge * np.eye(nproto, dtype=np.float32)
            try:
                c = np.linalg.solve(A, Pc @ y)
            except np.linalg.LinAlgError:
                continue
            rec = ((Pc.T @ c) > 0).astype(np.uint8)

            # Score at PROTO resolution: how well the basis spans this shape.
            d_proto = dice(gc.astype(bool), rec.astype(bool))

            # Score at FULL resolution too, by putting the reconstruction back on
            # the image grid. Only this number is comparable with the grid
            # ceiling and with what the model achieves, because both of those
            # are full-resolution measurements. Comparing a proto-resolution
            # Dice against them would credit the basis for avoiding an
            # upsampling penalty it never pays.
            full = np.zeros((ph, pw), np.float32)
            full[y1:y2, x1:x2] = rec.reshape(y2 - y1, x2 - x1)
            full = cv2.resize(full, (w0, h0), interpolation=cv2.INTER_LINEAR)
            d_full = dice(m.astype(bool), (full >= 0.5))
            i_full = iou(m.astype(bool), (full >= 0.5))
            if d_proto is None or d_full is None:
                continue
            rows.append({"cat": a["category_id"], "dice": d_proto,
                         "dice_full": d_full, "iou": i_full,
                         "px": int(gc.sum())})
        if (k + 1) % 100 == 0:
            print("  %d/%d images, %d instances" % (k + 1, len(img_ids), len(rows)),
                  flush=True)

    if not rows:
        raise SystemExit("no instances scored")

    D = np.array([r["dice"] for r in rows])            # proto resolution
    DF = np.array([r["dice_full"] for r in rows])      # full resolution
    I = np.array([r["iou"] for r in rows])
    names = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    GRID_CEILING = 0.8963      # from tools/mask_resolution_ceiling.py, input/4
    MODEL_ACHIEVED = 0.6969    # paired contour comparison, common cases

    res = {
        "weights": args.weights, "instances": len(rows),
        "oracle_mean_dice": float(D.mean()),
        "oracle_median_dice": float(np.median(D)),
        "oracle_mean_iou": float(I.mean()),
        "frac_below_iou_075": float((I < 0.75).mean()),
        "grid_ceiling_dice": GRID_CEILING,
        "model_achieved_dice": MODEL_ACHIEVED,
        "per_class": {},
    }
    for cid in sorted({r["cat"] for r in rows}):
        v = [r["dice"] for r in rows if r["cat"] == cid]
        res["per_class"][names.get(cid, str(cid))] = {
            "mean_dice": float(np.mean(v)), "n": len(v)}

    res["oracle_mean_dice_fullres"] = float(DF.mean())
    # Attribution uses the FULL-resolution oracle, the only one on the same
    # footing as the grid ceiling and the achieved score.
    gap_total = GRID_CEILING - MODEL_ACHIEVED
    gap_basis = GRID_CEILING - float(DF.mean())
    gap_head = float(DF.mean()) - MODEL_ACHIEVED
    res["attribution"] = {
        "total_gap": gap_total,
        "attributable_to_basis": gap_basis,
        "attributable_to_coefficient_head": gap_head,
        "basis_share": gap_basis / gap_total if gap_total else None,
        "head_share": gap_head / gap_total if gap_total else None,
    }

    with open(args.out + ".json", "w") as f:
        json.dump(res, f, indent=1)

    L = ["# Oracle coefficient fit", "",
         "Best mask achievable from the trained model's own prototypes with",
         "optimal coefficients, solved in closed form. No training.", "",
         "| quantity | Dice |", "|---|---|",
         "| grid ceiling (what the resolution allows) | %.4f |" % GRID_CEILING,
         "| **oracle coefficients, full resolution** | **%.4f** |" % DF.mean(),
         "| oracle coefficients, prototype resolution | %.4f |" % D.mean(),
         "| what the trained model achieves | %.4f |" % MODEL_ACHIEVED, "",
         "## Attribution of the %.4f gap" % gap_total, "",
         "| source | Dice lost | share |", "|---|---|---|",
         "| prototype **basis** cannot represent it | %.4f | %.0f %% |"
         % (gap_basis, 100 * gap_basis / gap_total),
         "| **coefficient head** fails to find it | %.4f | %.0f %% |"
         % (gap_head, 100 * gap_head / gap_total), "",
         "%d instances scored. Oracle mean IoU %.4f; %.1f %% of instances still "
         "below IoU 0.75 even with optimal coefficients."
         % (len(rows), I.mean(), 100 * (I < 0.75).mean()), ""]
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(L) + "\n")

    print("\n  instances scored          : %d" % len(rows))
    print("  grid ceiling              : %.4f" % GRID_CEILING)
    print("  ORACLE, full resolution   : %.4f" % DF.mean())
    print("  ORACLE, proto resolution  : %.4f  (basis expressiveness)" % D.mean())
    print("  model achieves            : %.4f" % MODEL_ACHIEVED)
    print("\n  gap attributable to BASIS : %.4f (%.0f %%)"
          % (gap_basis, 100 * gap_basis / gap_total))
    print("  gap attributable to HEAD  : %.4f (%.0f %%)"
          % (gap_head, 100 * gap_head / gap_total))
    print("\nwrote %s.md and %s.json" % (args.out, args.out))


if __name__ == "__main__":
    main()
