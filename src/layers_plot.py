#!/usr/bin/env python3
"""
Plot the three sweeps from layers_scaling.sbatch.

    python3 plot_layers.py layers_scaling_12345.csv
"""

import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1]

# (network, depth, width) -> list of s_per_epoch, one per rep
times = defaultdict(list)
with open(path, newline="") as f:
    for row in csv.DictReader(f):
        key = (row["network"], int(row["depth"]), int(row["width"]))
        times[key].append(float(row["s_per_epoch"]))

# average the reps
avg = {k: sum(v) / len(v) for k, v in times.items()}
nets = sorted({net for net, _, _ in avg})

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# 1. depth scaling, width fixed at 32
for net in nets:
    pts = sorted((d, t) for (n, d, w), t in avg.items() if n == net and w == 32)
    ax1.plot([d for d, _ in pts], [t for _, t in pts], "o-", label=net)
ax1.set_xlabel("depth (hidden layers)")
ax1.set_title("Depth slacing (width=32)")
ax1.legend()

# 2. width scaling, depth fixed at 2
for net in nets:
    pts = sorted((w, t) for (n, d, w), t in avg.items() if n == net and d == 2)
    ax2.plot([w for w, _ in pts], [t for _, t in pts], "o-", label=net)
ax2.set_xlabel("width (neurons per layer)")
ax2.set_title("Width scaling (depth=2)")
ax2.legend()

# 3. grid, both binaries: color = width, solid = first net, dashed = second
widths = sorted({w for (n, d, w) in avg})
colors = {w: f"C{i}" for i, w in enumerate(widths)}
styles = ["-", "--"]

drawn = []                      # widths that actually have a grid row
for net, style in zip(nets, styles):
    for width in widths:
        pts = sorted((d, t) for (n, d, w), t in avg.items()
                     if n == net and w == width)
        if len(pts) > 1:
            ax3.plot([d for d, _ in pts], [t for _, t in pts],
                     marker="o", ls=style, color=colors[width])
            if width not in drawn:
                drawn.append(width)
ax3.set_xlabel("depth (hidden layers)")
ax3.set_title("Depth x width (both)")

# two legends: colour = width, line style = binary
keys = [plt.Line2D([], [], color=colors[w], marker="o", label=f"width {w}")
        for w in drawn]
keys += [plt.Line2D([], [], color="gray", ls=s, label=n)
         for n, s in zip(nets, styles)]
ax3.legend(handles=keys, fontsize="small")

for ax in (ax1, ax2, ax3):
    ax.set_ylabel("seconds per epoch")
    ax.grid(True, alpha=0.3)

fig.tight_layout()
out = path.replace(".csv", ".png")
fig.savefig(out, dpi=150)
print("wrote", out)
