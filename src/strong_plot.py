#!/usr/bin/env python3
"""
Plot the strong-scaling sweep from strong_scaling.sbatch.

    python3 strong_plot.py strong_1234.csv
"""

import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1]

# (network, hidden, cores) -> mean seconds per epoch over the reps
times = defaultdict(list)
with open(path, newline="") as f:
    for row in csv.DictReader(f):
        times[(row["network"], row["hidden"], int(row["cores"]))].append(
            float(row["s_per_epoch"]))
avg = {k: sum(v) / len(v) for k, v in times.items()}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for net, hidden in sorted({(n, h) for n, h, _ in avg}):
    pts = sorted((c, s) for (n, h, c), s in avg.items() if n == net and h == hidden)
    cores = [c for c, _ in pts]
    secs = [s for _, s in pts]
    label = f"{net} h={hidden}"

    ax1.plot(cores, secs, "o-", label=label)
    base = secs[0]
    ax2.plot(cores, [base / s for s in secs], "o-", label=label)

cores = sorted({c for _, _, c in avg})
ax2.plot(cores, cores, "k--", alpha=0.4, label="ideal")

ax1.set_ylabel("seconds per epoch")
ax1.set_title("strong scaling: wall time")
ax2.set_ylabel("speedup vs 1 core")
ax2.set_title("speedup")

for ax in (ax1, ax2):
    ax.set_xlabel("cores")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

fig.tight_layout()
out = path.replace(".csv", ".png")
fig.savefig(out, dpi=150)
print("wrote", out)
