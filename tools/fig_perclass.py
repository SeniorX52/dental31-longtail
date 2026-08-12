#!/usr/bin/env python3
"""Per-class effect of restoring native input resolution, held-out test split.

The paper's mechanism claim is that the gain concentrates on small, low-contrast
findings rather than spreading evenly, so the figure has to make baseline AP
legible next to the change. Bars carry the relative change; the annotation to the
right of each label carries the baseline AP the change is measured against.

Numbers are seed 42 against the 100-epoch 640 baseline, both on the rebuilt
held-out split. Source: RESULTS.md section 10.

    python3 tools/fig_perclass.py --out paper/tex/fig_perclass.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# class, test instances, baseline AP, seed-42 AP
ROWS = [
    ("Bone loss",            469, 0.0136, 0.0235),
    ("Root canal treatment", 2866, 0.1150, 0.1888),
    ("Periapical lesion",     797, 0.0214, 0.0351),
    ("Caries",               1604, 0.0855, 0.1295),
    ("Root piece",            392, 0.1464, 0.2002),
    ("Mandibular canal",       92, 0.1116, 0.1383),
    ("Filling",              7352, 0.2756, 0.3382),
    ("Crown",                1687, 0.4490, 0.5201),
    ("Impacted tooth",       4192, 0.5053, 0.5358),
    ("Missing teeth",         527, 0.1528, 0.1462),
    ("Maxillary sinus",        70, 0.2304, 0.1931),
]

GAIN = "#2166AC"   # validated pair, OKLab dE 21.1 protan / 28.7 normal
LOSS = "#B2182B"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d6"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("paper/tex/fig_perclass.pdf"))
    args = ap.parse_args()

    rows = sorted(ROWS, key=lambda r: (r[3] - r[2]) / r[2])
    names = [r[0] for r in rows]
    change = [100 * (r[3] - r[2]) / r[2] for r in rows]
    base = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(3.35, 3.4))
    y = range(len(rows))

    # Thin bars, rounded data-end, 2px surface gap between neighbours.
    ax.barh(list(y), change,
            color=[GAIN if c > 0 else LOSS for c in change],
            height=0.62, linewidth=0)

    ax.axvline(0, color=INK, lw=0.8, zorder=3)
    ax.set_yticks(list(y))
    # Baseline AP rides in the tick label rather than in its own column: a
    # separate column collides with the labels on the negative bars.
    ax.set_yticklabels([f"{n}  ({b:.3f})" for n, b in zip(names, base)],
                       fontsize=7, color=INK)
    ax.set_xlabel("change in per-class AP (%)", fontsize=7.5, color=INK)
    ax.tick_params(axis="x", labelsize=7, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-22, 86)

    # Direct labels: the value on every bar. Only eleven of them, and the exact
    # figures are what the text refers back to.
    for i, c in enumerate(change):
        # Negative bars take their label on the zero side. Placed beyond the bar
        # end it would run into the class label, which is the one collision this
        # layout can produce.
        x = c + 2.0 if c > 0 else 2.0
        ax.text(x, i, f"{c:+.1f}%", va="center", ha="left",
                fontsize=6.5, color=INK)

    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    fig.tight_layout(pad=0.3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
