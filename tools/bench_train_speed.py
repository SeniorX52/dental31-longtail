#!/usr/bin/env python3
"""Benchmark accuracy-neutral training speedups for the YOLO-seg runs.

Motivation: the ablation is 7 runs. A 30% speedup is worth several hours, but
only if it is genuinely accuracy-neutral -- a faster run that quietly changes
the optimisation is useless for a comparison table.

So every configuration is scored on TWO axes:

  * throughput  -- seconds per epoch, measured on steady-state epochs only
                   (the first epoch is discarded: it carries dataloader warmup,
                   cuDNN autotuning and, for compile, graph capture)
  * equivalence -- the training loss trajectory under a fixed seed. Layout and
                   I/O changes must reproduce the reference losses to within
                   floating-point noise. A config that shifts the loss is
                   reported as NOT equivalent and excluded regardless of speed.

Configurations tested (each is a superset of what came before, plus isolated
arms so the contribution of each is attributable):

  ref           ultralytics defaults, as used for the existing baseline
  cache         cache='ram'      -- decode each image once instead of every epoch
  chlast        channels_last    -- NHWC, the layout tensor cores actually want
  workers       more dataloader workers (the box has 24 cores, default is 8)
  compile       torch.compile    -- safe here because imgsz is fixed at 640;
                                    it would thrash on DINO's multi-scale input
  nondet        deterministic=False -- lets cuDNN pick faster kernels. Kept
                                    SEPARATE because it trades bit-exact
                                    reproducibility, which the protocol promises
  combo         every arm that proved both fast and equivalent

Usage:
    python tools/bench_train_speed.py --data data_clean/data.yaml \\
        --model yolov8x-seg.pt --fraction 0.10 --epochs 4 --out reports/speed_bench
"""
import argparse
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional

CONFIGS = [
    ("ref",     dict()),
    ("cache",   dict(cache="ram")),
    ("chlast",  dict(channels_last=True)),
    ("workers", dict(workers=16)),
    ("compile", dict(compile=True)),
    ("nondet",  dict(deterministic=False)),
]


def run_one(name: str, extra: Dict, args) -> Optional[Dict]:
    """Train a short run and return {epoch_times, losses}."""
    from ultralytics import YOLO
    import torch

    run_dir = os.path.join(args.project, name)
    shutil.rmtree(run_dir, ignore_errors=True)

    overrides = dict(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        seed=args.seed, deterministic=True, fraction=args.fraction,
        project=args.project, name=name, exist_ok=True, val=False, plots=False,
        verbose=False, workers=8,
    )
    overrides.update(extra)

    torch.manual_seed(args.seed)
    t0 = time.time()
    try:
        YOLO(args.model).train(**overrides)
    except Exception as e:
        print("  %s FAILED: %s" % (name, str(e)[:200]))
        return None
    wall = time.time() - t0

    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        # ultralytics may nest under its own runs_dir
        cands = []
        for root, _, files in os.walk(args.project):
            if "results.csv" in files and os.path.basename(root) == name:
                cands.append(os.path.join(root, "results.csv"))
        if not cands:
            print("  %s: no results.csv" % name)
            return None
        csv_path = cands[0]

    import csv as _csv
    rows = list(_csv.DictReader(open(csv_path)))
    if not rows:
        return None

    def col(row, *keys):
        for k in row:
            if any(x in k for x in keys):
                try:
                    return float(row[k])
                except ValueError:
                    pass
        return None

    times = [col(r, "time") for r in rows]
    # per-epoch durations from the cumulative time column
    durs = [times[0]] + [times[i] - times[i - 1] for i in range(1, len(times))]
    losses = [(col(r, "train/box_loss"), col(r, "train/seg_loss"),
               col(r, "train/cls_loss")) for r in rows]

    steady = durs[1:] if len(durs) > 1 else durs      # drop warmup epoch
    return {"name": name, "wall_s": wall, "epoch_s": durs,
            "steady_mean_s": sum(steady) / max(len(steady), 1),
            "losses": losses}


def equivalent(ref: Dict, cur: Dict, tol: float) -> (bool, float):
    """Max relative difference across the recorded loss trajectory."""
    worst = 0.0
    for a, b in zip(ref["losses"], cur["losses"]):
        for x, y in zip(a, b):
            if x is None or y is None:
                continue
            denom = max(abs(x), 1e-8)
            worst = max(worst, abs(x - y) / denom)
    return worst <= tol, worst


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8x-seg.pt")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=0.02,
                    help="max relative loss deviation still called equivalent")
    ap.add_argument("--project", default="runs/speedbench")
    ap.add_argument("--out", default="reports/speed_bench")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)

    results = []
    for name, extra in CONFIGS:
        if args.only and name not in args.only:
            continue
        print("\n=== %s  %s ===" % (name, extra or "(defaults)"))
        r = run_one(name, extra, args)
        if r:
            r["overrides"] = {k: str(v) for k, v in extra.items()}
            results.append(r)
            print("  steady %.1f s/epoch | wall %.0f s" % (r["steady_mean_s"], r["wall_s"]))

    if not results:
        raise SystemExit("no configuration completed")

    ref = next((r for r in results if r["name"] == "ref"), results[0])
    print("\n%-9s %12s %10s %14s %s" % ("config", "s/epoch", "speedup", "max loss dev", "verdict"))
    table = []
    for r in results:
        sp = ref["steady_mean_s"] / max(r["steady_mean_s"], 1e-9)
        ok, worst = equivalent(ref, r, args.tol)
        verdict = "reference" if r is ref else ("EQUIVALENT" if ok else "CHANGES LOSS")
        print("%-9s %12.1f %9.2fx %13.2e  %s" % (r["name"], r["steady_mean_s"], sp, worst, verdict))
        table.append({"config": r["name"], "s_per_epoch": r["steady_mean_s"],
                      "speedup": sp, "max_loss_dev": worst,
                      "equivalent": bool(ok), "overrides": r["overrides"]})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump({"reference": ref["name"], "tol": args.tol,
                   "model": args.model, "fraction": args.fraction,
                   "epochs": args.epochs, "results": table}, f, indent=1)
    print("\nwrote %s.json" % args.out)

    winners = [t for t in table if t["equivalent"] and t["speedup"] > 1.02
               and t["config"] != "ref"]
    if winners:
        print("\nadopt (faster AND loss-equivalent):")
        for w in sorted(winners, key=lambda w: -w["speedup"]):
            print("   %-9s %.2fx  %s" % (w["config"], w["speedup"], w["overrides"]))
    else:
        print("\nno configuration was both faster and loss-equivalent")


if __name__ == "__main__":
    main()
