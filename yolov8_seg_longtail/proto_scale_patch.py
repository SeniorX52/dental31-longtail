"""Teach the ultralytics segment validator that prototypes are not always at input/4.

WHY THIS EXISTS. `--proto-scale 2` moves the prototype grid from input/4 to
input/2, and `train_seg.py` already sets `mask_ratio=2` so the dataloader builds
ground-truth masks at the matching resolution. Validation still crashed:

    RuntimeError: mat1 and mat2 shapes cannot be multiplied
                  (27x57728 and 230912x300)

ultralytics 8.4.108 hardcodes the /4 assumption in two places in
`models/yolo/segment/val.py`, and neither consults `mask_ratio`:

    line 103   imgsz = [4 * x for x in proto.shape[2:]]        # image size FROM proto
    line 128   mask_size = [... else s // 4 for s in prepared_batch["imgsz"]]

At 1312x704 input the first reconstructs the image size as 2x too large, and the
second resizes the ground truth back down to 328x176 (57728 px) while the
predictions come out of the prototype grid at 656x352 (230912 px). The two can
never be compared, so the run dies at its first validation having trained fine.

That is what killed abl_HR1280hp on 10 Aug after 42 minutes: the driver read the
first failure as an out-of-memory error, retried at batch 1, and hit the same
wall. The `mask_ratio=2` override was working correctly the whole time.

The predict path needs no patch. `segment/predict.py` passes explicit shapes to
`process_mask`/`process_mask_native`, so scoring a proto-scale-2 model already
worked; only the in-training validator is wrong.

Both replacements are the upstream bodies with the literal 4 replaced by the
actual prototype stride. Installing is a no-op at the default stride, so the
normal path is untouched.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def install(proto_stride: int) -> bool:
    """Patch SegmentationValidator for a prototype grid at input/`proto_stride`.

    Returns True if a patch is now active. A stride of 4 is the stock layout and
    needs nothing, so it returns False and leaves ultralytics alone.
    """
    if proto_stride == 4:
        return False

    from ultralytics.models.yolo.segment import val as segval
    from ultralytics.utils import ops

    SV = segval.SegmentationValidator
    if getattr(SV, "_lt_proto_stride", None) == proto_stride:
        return True

    base = SV.__mro__[1]  # DetectionValidator, whose methods these two extend

    def postprocess(self, preds):
        proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        out = base.postprocess(self, preds[0])
        # the only change: the prototype grid is input/proto_stride, not input/4
        imgsz = [proto_stride * x for x in proto.shape[2:]]
        for i, pred in enumerate(out):
            coefficient = pred.pop("extra")
            pred["masks"] = self.process(proto[i], coefficient, pred["bboxes"], shape=imgsz)
        return out

    def _prepare_batch(self, si, batch):
        prepared = base._prepare_batch(self, si, batch)
        nl = prepared["cls"].shape[0]
        if self.args.overlap_mask:
            masks = batch["masks"][si]
            index = torch.arange(1, nl + 1, device=masks.device).view(nl, 1, 1)
            masks = (masks == index).float()
        else:
            masks = batch["masks"][batch["batch_idx"] == si]
        if nl:
            # and here: ground truth belongs at the prototype resolution
            mask_size = [
                s if self.process is ops.process_mask_native else s // proto_stride
                for s in prepared["imgsz"]
            ]
            if list(masks.shape[1:]) != list(mask_size):
                masks = F.interpolate(masks[None], mask_size, mode="bilinear",
                                      align_corners=False)[0]
                masks = masks.gt_(0.5)
        prepared["masks"] = masks
        return prepared

    SV.postprocess = postprocess
    SV._prepare_batch = _prepare_batch
    SV._lt_proto_stride = proto_stride
    return True
