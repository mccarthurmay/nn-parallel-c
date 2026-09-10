#!/usr/bin/env python3
"""
Plot the sweeps from scaling.sbatch: hidden-layer architecture and batch size.

    python3 layers_plot.py scaling_1557.csv
"""

import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1]

# (network, hidden, batch) -> list of s_per_epoch, one per rep
times = defaultdict(list)
hidden_order = []
with open(path, newline="") as f:
    for row in csv.DictReader(f):
        key = (row["network"], row["hidden"], int(row["batch"]))
        times[key].append(float(row["s_per_epoch"]))
        if row["hidden"] not in hidden_order:
            hidden_order.append(row["hidden"])

# average the reps
avg = {k: sum(v) / len(v) for k, v in times.items()}
nets = sorted({net for net, _, _ in avg})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 1. architecture scaling, batch fixed at 10
for net in nets:
    pts = [(h, avg[(net, h, 10)]) for h in hidden_order if (net, h, 10) in avg]
    ax1.plot(range(len(pts)), [t for _, t in pts], "o-", label=net)
ax1.set_xticks(range(len(hidden_order)))
ax1.set_xticklabels(hidden_order, rotation=45, ha="right")
ax1.set_xlabel("hidden layers")
ax1.set_title("layers scaling (batch=10)")
ax1.legend()

# 2. batch scaling, hidden fixed at "30"
for net in nets:
    pts = sorted((b, t) for (n, h, b), t in avg.items() if n == net and h == "30")
    ax2.plot([b for b, _ in pts], [t for _, t in pts], "o-", label=net)
ax2.set_xlabel("batch size")
ax2.set_title("batch scaling (hidden=30)")
ax2.legend()

for ax in (ax1, ax2):
    ax.set_ylabel("seconds per epoch")
    ax.grid(True, alpha=0.3)

fig.tight_layout()
out = path.replace(".csv", ".png")
fig.savefig(out, dpi=150)
print("wrote", out)
