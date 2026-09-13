#!/usr/bin/env python3
"""
Plot accuracy from the scaling.sbatch sweeps: architecture and batch size.

    python3 accuracy_plot.py scaling_1827.csv
    python3 accuracy_plot.py scaling_1827.csv network_v3 network_v5

Any networks named after the csv restrict the plot to just those, in the
order given 
"""

import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1]
wanted = sys.argv[2:]

# (network, hidden, batch) -> list of accuracy, one per rep
accs = defaultdict(list)
hidden_order = []
with open(path, newline="") as f:
    for row in csv.DictReader(f):
        key = (row["network"], row["hidden"], int(row["batch"]))
        accs[key].append(float(row["accuracy"]))
        if row["hidden"] not in hidden_order:
            hidden_order.append(row["hidden"])

# average the reps
avg = {k: sum(v) / len(v) for k, v in accs.items()}
nets = sorted({net for net, _, _ in avg})
if wanted:
    missing = [n for n in wanted if n not in nets]
    if missing:
        sys.exit(f"not in {path}: {', '.join(missing)}\navailable: {', '.join(nets)}")
    nets = wanted

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 1. architecture sweep, batch fixed at 10
for net in nets:
    pts = [(h, avg[(net, h, 10)]) for h in hidden_order if (net, h, 10) in avg]
    ax1.plot(range(len(pts)), [a for _, a in pts], "o-", label=net)
ax1.set_xticks(range(len(hidden_order)))
ax1.set_xticklabels(hidden_order, rotation=45, ha="right")
ax1.set_xlabel("hidden layers")
ax1.set_title("accuracy by architecture (batch=10)")
ax1.legend()

# 2. batch sweep, hidden fixed at "30"
for net in nets:
    pts = sorted((b, a) for (n, h, b), a in avg.items() if n == net and h == "30")
    ax2.plot([b for b, _ in pts], [a for _, a in pts], "o-", label=net)
ax2.set_xlabel("batch size")
ax2.set_title("accuracy by batch size (hidden=30)")
ax2.legend()

for ax in (ax1, ax2):
    ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.3)

fig.tight_layout()
out = path.replace(".csv", "_accuracy.png")
fig.savefig(out, dpi=150)
print("wrote", out)
