#!/usr/bin/env python3
"""Package trained checkpoints into a deliverable the client can actually open.

WHY RE-EXPORT RATHER THAN COPY. Our segmentation runs were trained through
`train_seg.py`, whose model class `LongTailSegModel` is defined in a script that
runs as `__main__`. ultralytics pickles the whole nn.Module into the checkpoint,
so the class name is baked in, and on any machine without this repo on the path:

    AttributeError: Can't get attribute 'LongTailSegModel' on
                    <module '__main__' (built-in)>

`YOLO(best.pt)` fails outright. That is not a deliverable. This already cost the
project once: scoring died mid-queue on exactly this error (see PROGRESS.md,
02 Aug).

The fix is safe because `LongTailSegModel` is a pure subclass of ultralytics'
`SegmentationModel` that overrides one method, `init_criterion`, which is used
only during training. The parameters, the architecture and the forward pass are
identical, so re-pointing the class and dropping the training-only attributes
yields a stock checkpoint with the same weights and the same outputs. Equality
of outputs is verified numerically, not assumed.

Checkpoints trained by stock ultralytics (the baseline) and DINO's `.pth` state
dicts are already portable and are copied unchanged, with a checksum.

Usage:
    python tools/package_weights.py --out deliverable/weights            # this machine
    python tools/package_weights.py --out deliverable/weights --verify-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Training-only state that must not travel with an inference checkpoint. Each is
# read by the custom criterion and by nothing else.
TRAIN_ONLY_ATTRS = (
    "class_weights", "boundary_weight", "mask_aux", "coeff_weight",
    "coeff_ridge", "coeff_width", "bg_gate", "proto_scale", "proto_src",
    "criterion",
)


def sha256(path: str, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensors(obj, acc=None):
    """Flatten every tensor out of ultralytics' nested forward output."""
    acc = [] if acc is None else acc
    if torch.is_tensor(obj):
        acc.append(obj)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            _tensors(o, acc)
    elif isinstance(obj, dict):
        for o in obj.values():
            _tensors(o, acc)
    return acc


def _strip(module) -> None:
    from ultralytics.nn.tasks import SegmentationModel
    if module is None:
        return
    if type(module).__name__ != "SegmentationModel":
        module.__class__ = SegmentationModel
    for a in TRAIN_ONLY_ATTRS:
        if hasattr(module, a):
            try:
                delattr(module, a)
            except Exception:
                setattr(module, a, None)


def export_seg(src: str, dst: str) -> dict:
    """Re-export a custom-class segmentation checkpoint as a stock one."""
    sys.path.insert(0, REPO)
    import yolov8_seg_longtail.train_seg as T

    # The class was pickled while train_seg.py was running as `__main__`, so the
    # checkpoint names `__main__.LongTailSegModel`. Importing the module defines
    # it under its real package path and unpickling still fails. It has to be
    # grafted onto whatever `__main__` is now -- the same re-registration
    # predict_to_coco.py does for the same reason.
    main_mod = sys.modules["__main__"]
    for name in dir(T):
        obj = getattr(T, name)
        if isinstance(obj, type) and not hasattr(main_mod, name):
            setattr(main_mod, name, obj)

    ck = torch.load(src, map_location="cpu", weights_only=False)
    model = ck.get("model")
    before_cls = type(model).__name__
    orig_dtype = next(model.parameters()).dtype

    # capture the forward BEFORE any surgery, on a fixed input, to prove the
    # re-export does not change what the network computes
    torch.manual_seed(0)
    x = torch.randn(1, 3, 640, 640)
    net = (ck.get("ema") or model).float().eval()
    with torch.no_grad():
        ref = _tensors(net(x))

    _strip(ck.get("model"))
    _strip(ck.get("ema"))
    # optimizer state is training-only and roughly doubles the file
    for k in ("optimizer", "updates", "ema_updates"):
        ck.pop(k, None)

    net2 = (ck.get("ema") or ck.get("model")).float().eval()
    with torch.no_grad():
        out = _tensors(net2(x))
    if len(ref) != len(out):
        raise RuntimeError("forward structure changed: %d vs %d tensors" % (len(ref), len(out)))
    max_abs = max((a - b).abs().max().item() for a, b in zip(ref, out))

    # ultralytics stores best.pt with the weights in half precision; loading
    # and re-saving in float32 doubles the file (138 MB -> 288 MB) and changes
    # nothing at inference, so restore the dtype the checkpoint arrived in.
    for key in ("model", "ema"):
        m = ck.get(key)
        if m is not None and orig_dtype == torch.float16:
            ck[key] = m.half()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save(ck, dst)
    return {
        "class_before": before_cls,
        "class_after": type(ck.get("ema") or ck.get("model")).__name__,
        "max_abs_output_diff": max_abs,
        "identical": max_abs == 0.0,
    }


def export_dino(src: str, dst: str) -> str:
    """Copy a DINO checkpoint keeping only what inference needs.

    The training checkpoints carry optimizer and lr_scheduler state, which is
    about two thirds of the file and is useless to anyone loading the model to
    run it. Dropping them takes the four detection checkpoints from 1.9 GB to
    roughly 0.7 GB. `model`, `epoch` and `args` are kept so the checkpoint is
    still self-describing and DINO's own loader accepts it.
    """
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model" not in ck:
        shutil.copy2(src, dst)
        return "copied unchanged (unrecognised layout)"
    keep = {k: ck[k] for k in ("model", "epoch", "args") if k in ck}
    dropped = sorted(set(ck) - set(keep))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save(keep, dst)
    return "kept model/epoch/args, dropped %s" % (", ".join(dropped) or "nothing")


def verify_dino(path: str) -> str:
    """Confirm the packaged detection checkpoint loads and holds real weights."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model"]
        n = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
        return "OK (%d tensors, %.1f M params)" % (len(sd), n / 1e6)
    except Exception as e:
        return "FAILED: %s %s" % (type(e).__name__, str(e)[:100])


def verify_portable(path: str) -> str:
    """Load the file in a CLEAN interpreter with this repo removed from the path.

    Anything less is not a test: importing the repo in-process is exactly the
    condition the client will not have.
    """
    code = (
        "import sys;"
        "sys.path=[p for p in sys.path if 'ML_SOTA' not in p];"
        "from ultralytics import YOLO;"
        "m=YOLO(sys.argv[1]);"
        "print(type(m.model).__name__)"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", code, os.path.abspath(path)], capture_output=True,
                       text=True, env=env, cwd=tempfile.gettempdir())
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()
        return "FAILED: " + (tail[-1][:140] if tail else "unknown")
    return "OK (%s)" % r.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="deliverable/weights")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    spec_path = os.path.join(REPO, "tools", "weights_manifest.json")
    with open(spec_path) as f:
        spec = json.load(f)

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for e in spec["entries"]:
        src = e["src"] if os.path.isabs(e["src"]) else os.path.join(REPO, e["src"])
        dst = os.path.join(args.out, e["file"])
        if not os.path.exists(src):
            if os.path.exists(dst):
                # packaged on the other machine and copied in; verify, do not discard
                status = verify_portable(dst) if e["kind"].startswith("seg") else verify_dino(dst)
                rows.append(dict(e, sha256=sha256(dst), bytes=os.path.getsize(dst),
                                 note="packaged on the other machine", status=status))
                print("  %-28s %-9s %s | already packaged"
                      % (e["name"], "%.0f MB" % (os.path.getsize(dst) / 1e6), status))
                continue
            print("  %-28s SOURCE MISSING (%s)" % (e["name"], src))
            rows.append(dict(e, status="source missing"))
            continue
        if not args.verify_only:
            if e["kind"] == "seg_custom":
                info = export_seg(src, dst)
                note = "re-exported %s -> %s, outputs %s" % (
                    info["class_before"], info["class_after"],
                    "bit-identical" if info["identical"]
                    else "max |diff| %.2e" % info["max_abs_output_diff"])
            elif e["kind"] == "dino":
                note = export_dino(src, dst)
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                note = "copied unchanged (already portable)"
        else:
            note = "not rebuilt"
        status = verify_portable(dst) if e["kind"].startswith("seg") else verify_dino(dst)
        rows.append(dict(e, sha256=sha256(dst), bytes=os.path.getsize(dst),
                         note=note, status=status))
        print("  %-28s %-9s %s | %s" % (e["name"], "%.0f MB" % (os.path.getsize(dst) / 1e6),
                                        status, note))

    with open(os.path.join(args.out, "checksums.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("\n  wrote %s" % os.path.join(args.out, "checksums.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
