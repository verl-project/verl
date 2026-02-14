#!/usr/bin/env python3
"""Plot score/throughput/memory/loss curves from one train.log."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


STEP_RE = re.compile(r"step:(\d+) -")
PATTERNS: dict[str, re.Pattern[str]] = {
    "score": re.compile(r"critic/score/mean:([-+0-9.eE]+)"),
    "throughput": re.compile(r"perf/throughput:([-+0-9.eE]+)"),
    "memory_reserved_gb": re.compile(r"perf/max_memory_reserved_gb:([-+0-9.eE]+)"),
    # GRPO usually uses actor pg loss, PPO typically has critic vf loss.
    "actor_pg_loss": re.compile(r"actor/pg_loss:([-+0-9.eE]+)"),
    "critic_vf_loss": re.compile(r"critic/vf_loss:([-+0-9.eE]+)"),
}


def parse_log(log_path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m is None:
                continue
            step = int(m.group(1))
            row = rows.setdefault(step, {})
            for key, pattern in PATTERNS.items():
                mm = pattern.search(line)
                if mm is not None:
                    row[key] = float(mm.group(1))
    return rows


def write_tsv(rows: dict[int, dict[str, float]], out_tsv: Path) -> None:
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "step",
                "score",
                "throughput",
                "memory_reserved_gb",
                "actor_pg_loss",
                "critic_vf_loss",
                "loss_used",
            ]
        )
        for step in sorted(rows.keys()):
            row = rows[step]
            loss_used = row.get("actor_pg_loss")
            if loss_used is None:
                loss_used = row.get("critic_vf_loss")
            writer.writerow(
                [
                    step,
                    row.get("score", ""),
                    row.get("throughput", ""),
                    row.get("memory_reserved_gb", ""),
                    row.get("actor_pg_loss", ""),
                    row.get("critic_vf_loss", ""),
                    "" if loss_used is None else loss_used,
                ]
            )


def _collect_xy(rows: dict[int, dict[str, float]], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for step in sorted(rows.keys()):
        val = rows[step].get(key)
        if val is None:
            continue
        xs.append(step)
        ys.append(val)
    return xs, ys


def _collect_loss_xy(rows: dict[int, dict[str, float]]) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for step in sorted(rows.keys()):
        row = rows[step]
        val = row.get("actor_pg_loss")
        if val is None:
            val = row.get("critic_vf_loss")
        if val is None:
            continue
        xs.append(step)
        ys.append(val)
    return xs, ys


def plot(rows: dict[int, dict[str, float]], out_png: Path, title: str | None = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axis_cfg = [
        (axes[0, 0], "score", "critic/score/mean", "Score"),
        (axes[0, 1], "throughput", "perf/throughput", "Tokens/s/GPU"),
        (axes[1, 0], "memory_reserved_gb", "perf/max_memory_reserved_gb", "GB"),
    ]

    for ax, key, panel_title, y_label in axis_cfg:
        xs, ys = _collect_xy(rows, key)
        if xs:
            ax.plot(xs, ys, linewidth=1.8)
        ax.set_title(panel_title)
        ax.set_xlabel("step")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)

    loss_x, loss_y = _collect_loss_xy(rows)
    if loss_x:
        axes[1, 1].plot(loss_x, loss_y, linewidth=1.8)
    axes[1, 1].set_title("loss (actor/pg_loss or critic/vf_loss)")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 4 curves from a single train.log.")
    parser.add_argument("--log", type=Path, required=True, help="Path to train.log")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path",
    )
    parser.add_argument(
        "--metrics-tsv",
        type=Path,
        required=True,
        help="Output parsed metrics TSV",
    )
    parser.add_argument("--title", type=str, default=None, help="Optional figure title")
    args = parser.parse_args()

    rows = parse_log(args.log)
    if not rows:
        raise RuntimeError(f"No step metrics found in {args.log}")

    write_tsv(rows, args.metrics_tsv)
    plot(rows, args.output, args.title)
    print(f"log={args.log}")
    print(f"output_png={args.output}")
    print(f"metrics_tsv={args.metrics_tsv}")
    print(f"num_steps={len(rows)}")


if __name__ == "__main__":
    main()
