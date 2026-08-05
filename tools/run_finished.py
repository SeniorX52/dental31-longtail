#!/usr/bin/env python3
"""Exit 0 if an ultralytics run directory holds a COMPLETED training run.

Why this is not just a row count. `results.csv` is flushed asynchronously, so
for a moment after training returns the file can still be one row behind the
epoch that actually finished. A completion guard keyed on the row count read a
finished 100-epoch run as 99/100, refused to score it, and the queue then
recorded the losing arm as the project's final result.

The checkpoint carries a definitive marker instead: on completion ultralytics
strips the optimizer state and stamps `epoch = -1`. That does not depend on
flush timing.

The checkpoints pickle custom subclasses defined in
`yolov8_seg_longtail/train_seg.py`, so those names must be re-registered into
`__main__` before `torch.load` can resolve them -- the same reason
`predict_to_coco.py` does it. Without the registration the load raises
AttributeError and a naive guard fails closed, silently treating a finished run
as unfinished.

Usage:
    python tools/run_finished.py runs/segment/final_S2_100ep && echo complete
"""
import os
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: run_finished.py <run_dir>", file=sys.stderr)
        return 2
    run_dir = sys.argv[1]
    ckpt = os.path.join(run_dir, "weights", "last.pt")
    if not os.path.isfile(ckpt):
        return 1

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    import __main__
    import torch

    try:
        from yolov8_seg_longtail.train_seg import (  # noqa: E402
            BoundaryAwareSegLoss, LongTailSegModel, LongTailSegTrainer)
        for cls in (LongTailSegModel, BoundaryAwareSegLoss, LongTailSegTrainer):
            setattr(__main__, cls.__name__, cls)
    except Exception as exc:                      # pragma: no cover
        print("could not register pickled classes: %s" % exc, file=sys.stderr)
        return 1

    try:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as exc:
        # a truncated checkpoint lands here; that is genuinely not finished
        print("checkpoint unreadable: %s" % str(exc)[:120], file=sys.stderr)
        return 1

    return 0 if ck.get("epoch", 0) == -1 else 1


if __name__ == "__main__":
    sys.exit(main())
