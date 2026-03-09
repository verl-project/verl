#!/usr/bin/env python3
"""Plot critic/score/mean from a verl training log file.

Usage:
    python examples/grpo_trainer/plot_reward.py 2048_log_20260309_152521.txt
"""

import re
import sys
import matplotlib.pyplot as plt

log_file = sys.argv[1]

steps, scores = [], []
pattern = re.compile(r"step:(\d+).*?critic/score/mean:([-\d.]+)")

with open(log_file) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            steps.append(int(m.group(1)))
            scores.append(float(m.group(2)))

if not steps:
    print("No critic/score/mean found in log.")
    sys.exit(1)

plt.figure(figsize=(10, 4))
plt.plot(steps, scores, marker="o", markersize=3)
plt.xlabel("Step")
plt.ylabel("critic/score/mean")
plt.title("Reward over Training Steps")
plt.grid(True)
plt.tight_layout()
out = log_file.replace(".txt", "_reward.png")
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
