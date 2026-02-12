#!/usr/bin/env python3
"""Plot six-case DeepSpeed benchmark curves from train logs.

Expected input layout:
  <run_root>/
    summary.tsv
    <case>/train.log

The script produces:
  - PNG figure with 4 subplots: score / throughput / memory / loss
  - TSV dump of parsed per-step metrics
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


STEP_RE = re.compile(r"step:(\d+) -")
CASE_ORDER = [
    "zero1_no_offload",
    "zero2_no_offload",
    "zero3_no_offload",
    "zero1_cpu_offload",
    "zero2_cpu_offload",
    "zero3_cpu_offload",
]
METRIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "score": re.compile(r"critic/score/mean:([-+0-9.eE]+)"),
    "throughput": re.compile(r"perf/throughput:([-+0-9.eE]+)"),
    "memory_reserved_gb": re.compile(r"perf/max_memory_reserved_gb:([-+0-9.eE]+)"),
    "vf_loss": re.compile(r"critic/vf_loss:([-+0-9.eE]+)"),
}


def parse_summary_cases(summary_path: Path) -> list[str]:
    if not summary_path.exists():
        raise FileNotFoundError(f"summary file not found: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cases = [row.get("case", "").strip() for row in reader]
    cases = [c for c in cases if c]
    if not cases:
        raise RuntimeError(f"no cases found in {summary_path}")
    unknown = [c for c in cases if c not in CASE_ORDER]
    if unknown:
        raise RuntimeError(f"summary contains unsupported case names: {unknown}")
    return sorted(cases, key=CASE_ORDER.index)


def parse_case_log(log_path: Path) -> dict[int, dict[str, float]]:
    if not log_path.exists():
        raise FileNotFoundError(f"log file not found: {log_path}")
    step_rows: dict[int, dict[str, float]] = {}
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            step_match = STEP_RE.search(line)
            if step_match is None:
                continue
            step = int(step_match.group(1))
            row = step_rows.setdefault(step, {})
            for metric_name, pattern in METRIC_PATTERNS.items():
                m = pattern.search(line)
                if m is not None:
                    row[metric_name] = float(m.group(1))
    return step_rows


def write_metrics_tsv(parsed: dict[str, dict[int, dict[str, float]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["case", "step", "score", "throughput", "memory_reserved_gb", "vf_loss"])
        for case in CASE_ORDER:
            rows = parsed.get(case)
            if not rows:
                continue
            for step in sorted(rows.keys()):
                row = rows[step]
                writer.writerow(
                    [
                        case,
                        step,
                        row.get("score", ""),
                        row.get("throughput", ""),
                        row.get("memory_reserved_gb", ""),
                        row.get("vf_loss", ""),
                    ]
                )


def plot_curves(parsed: dict[str, dict[int, dict[str, float]]], out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axis_defs = [
        (axes[0, 0], "score", "critic/score/mean", "Score"),
        (axes[0, 1], "throughput", "perf/throughput", "Tokens/s/GPU"),
        (axes[1, 0], "memory_reserved_gb", "perf/max_memory_reserved_gb", "GB"),
        (axes[1, 1], "vf_loss", "critic/vf_loss", "Loss"),
    ]

    for case in CASE_ORDER:
        rows = parsed.get(case)
        if not rows:
            continue
        steps = sorted(rows.keys())
        for ax, metric_key, title, ylabel in axis_defs:
            xs = []
            ys = []
            for step in steps:
                val = rows[step].get(metric_key)
                if val is None:
                    continue
                xs.append(step)
                ys.append(val)
            if xs:
                ax.plot(xs, ys, label=case, linewidth=1.8)
            ax.set_title(title)
            ax.set_xlabel("step")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

    axes[0, 0].legend(loc="best", fontsize=8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DeepSpeed six-case benchmark curves.")
    parser.add_argument("--run-root", type=Path, required=True, help="Benchmark run root directory")
    parser.add_argument("--summary", type=Path, default=None, help="summary.tsv path (default: <run_root>/summary.tsv)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output PNG path (default: <run_root>/zero_six_curves.png)",
    )
    parser.add_argument(
        "--metrics-tsv",
        type=Path,
        default=None,
        help="parsed metrics TSV path (default: <run_root>/zero_six_curves_metrics.tsv)",
    )
    args = parser.parse_args()

    run_root: Path = args.run_root
    summary_path = args.summary or run_root / "summary.tsv"
    output_png = args.output or run_root / "zero_six_curves.png"
    metrics_tsv = args.metrics_tsv or run_root / "zero_six_curves_metrics.tsv"

    cases = parse_summary_cases(summary_path)
    parsed: dict[str, dict[int, dict[str, float]]] = {}
    for case in cases:
        parsed[case] = parse_case_log(run_root / case / "train.log")

    write_metrics_tsv(parsed, metrics_tsv)
    plot_curves(parsed, output_png)

    print(f"run_root={run_root}")
    print(f"summary={summary_path}")
    print(f"output_png={output_png}")
    print(f"metrics_tsv={metrics_tsv}")


if __name__ == "__main__":
    main()
