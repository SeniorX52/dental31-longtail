#!/usr/bin/env python3
"""Identify and document a DETR/DINO or YOLO checkpoint.

Records exactly what the initialization weights are
(architecture, class count, training args, file hash) so the write-up cites
the real artifact instead of a description of it, and so an independent re-run
starts from a byte-identical file.

Also prints the classification-head parameter names, which is what
`--finetune_ignore` must list when re-heading the model to 31 classes.
Getting that list wrong is the classic silent failure: the run "works" but
initializes the head from COCO logits, or crashes on a shape mismatch.

Usage:
    python tools/inspect_checkpoint.py weights/checkpoint0033_4scale.pth
"""
import argparse
import hashlib
import os
import sys


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--head-substrings", default="class_embed,label_enc",
                    help="comma-separated substrings identifying the class head")
    args = ap.parse_args()

    import torch

    print("file:   %s" % args.checkpoint)
    print("size:   %.1f MB" % (os.path.getsize(args.checkpoint) / 1e6))
    print("sha256: %s" % file_sha256(args.checkpoint))

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        print("\ncheckpoint is not a dict (type=%s)" % type(ckpt).__name__)
        return
    print("\ntop-level keys: %s" % sorted(ckpt.keys()))
    if "epoch" in ckpt:
        print("epoch: %s" % ckpt["epoch"])

    ck_args = ckpt.get("args")
    if ck_args is not None:
        d = vars(ck_args) if hasattr(ck_args, "__dict__") else dict(ck_args)
        interesting = ["dataset_file", "backbone", "num_queries", "num_classes",
                       "num_feature_levels", "hidden_dim", "enc_layers",
                       "dec_layers", "dn_labelbook_size", "epochs", "lr",
                       "batch_size", "modelname", "use_dn", "two_stage_type"]
        print("\ntraining args (selected):")
        for k in interesting:
            if k in d:
                print("  %-22s %s" % (k, d[k]))

    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    if not isinstance(state, dict):
        return
    n_params = sum(v.numel() for v in state.values() if hasattr(v, "numel"))
    print("\nstate_dict: %d tensors, %.1fM parameters" % (len(state), n_params / 1e6))

    backbone_keys = [k for k in state if k.startswith("backbone")]
    print("backbone tensors: %d (first: %s)"
          % (len(backbone_keys), backbone_keys[0] if backbone_keys else "-"))

    subs = [s.strip() for s in args.head_substrings.split(",") if s.strip()]
    print("\nclassification-head tensors (pass these to --finetune_ignore):")
    head_names = []
    for k, v in state.items():
        if any(s in k for s in subs):
            shape = tuple(v.shape) if hasattr(v, "shape") else "?"
            print("  %-52s %s" % (k, shape))
            head_names.append(k)
    if not head_names:
        print("  (none matched %s)" % subs)
        return

    cover = minimal_substring_cover(head_names, [k for k in state if k not in set(head_names)])
    print("\n  --finetune_ignore %s" % " ".join(cover))
    print("  DINO ignores a tensor when any of these strings is a SUBSTRING of"
          "\n  its name. The set above was verified to match all %d head tensors"
          "\n  and none of the %d remaining tensors."
          % (len(head_names), len(state) - len(head_names)))


def minimal_substring_cover(targets, others):
    """Smallest set of substrings matching every target and no other key.

    Guards against the trap that a naive first-component heuristic falls into:
    'transformer' does match the head copies inside the transformer, but it
    also matches every encoder/decoder weight, which would silently discard
    the pretrained backbone-to-head stack.
    """
    candidates = set()
    for name in targets:
        parts = name.split(".")
        for i in range(len(parts)):
            for j in range(i + 1, len(parts) + 1):
                candidates.add(".".join(parts[i:j]))
    # keep only substrings that never appear in a non-head tensor name
    safe = [c for c in candidates if not any(c in o for o in others)]
    safe.sort(key=lambda c: (-sum(1 for t in targets if c in t), len(c)))

    cover, remaining = [], set(targets)
    for c in safe:
        if not remaining:
            break
        hit = {t for t in remaining if c in t}
        if hit:
            cover.append(c)
            remaining -= hit
    if remaining:
        cover.extend(sorted(remaining))  # fall back to exact names
    return sorted(cover)


if __name__ == "__main__":
    main()
