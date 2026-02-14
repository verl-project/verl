#!/usr/bin/env python3
"""Summarize timing breakdown across six DeepSpeed ZeRO benchmark cases."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


CASE_ORDER = [
    "zero1_no_offload",
    "zero2_no_offload",
    "zero3_no_offload",
    "zero1_cpu_offload",
    "zero2_cpu_offload",
    "zero3_cpu_offload",
]

TIMING_KEYS = [
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/values",
    "timing_s/update_critic",
    "timing_s/update_actor",
    "timing_s/update_weights",
    "timing_s/step",
]


def parse_step_rows(log_path: Path) -> list[tuple[int, dict[str, float]]]:
    rows: list[tuple[int, dict[str, float]]] = []
    if not log_path.exists():
        return rows
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"step:(\d+) -", line)
            if not m:
                continue
            step = int(m.group(1))
            vals: dict[str, float] = {}
            for key in TIMING_KEYS:
                mm = re.search(re.escape(key) + r":([-+0-9.eE]+)", line)
                if mm:
                    vals[key] = float(mm.group(1))
            rows.append((step, vals))
    return rows


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile timing breakdown from six-case logs.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tail-steps", type=int, default=5, help="Average over last N steps")
    parser.add_argument(
        "--baseline-case",
        type=str,
        default="zero2_no_offload",
        choices=CASE_ORDER,
        help="Case used for slowdown ratio",
    )
    parser.add_argument("--output-tsv", type=Path, default=None)
    args = parser.parse_args()

    summaries: dict[str, dict[str, float]] = {}
    for case in CASE_ORDER:
        rows = parse_step_rows(args.run_root / case / "train.log")
        if not rows:
            continue
        last_step = rows[-1][0]
        start_step = max(1, last_step - args.tail_steps + 1)
        tail = [vals for step, vals in rows if step >= start_step]
        summary = {"last_step": float(last_step), "window_start_step": float(start_step)}
        for key in TIMING_KEYS:
            vals = [row[key] for row in tail if key in row]
            summary[key] = avg(vals) if vals else 0.0
        summaries[case] = summary

    if not summaries:
        raise RuntimeError(f"no step rows found under {args.run_root}")

    baseline = summaries.get(args.baseline_case)
    if baseline is None:
        raise RuntimeError(f"baseline case missing in logs: {args.baseline_case}")

    print(f"run_root={args.run_root}")
    print(f"tail_steps={args.tail_steps}")
    print(f"baseline_case={args.baseline_case}")
    print("")

    header = (
        "case",
        "last_step",
        "avg_step_s",
        "step_ratio_vs_baseline",
        "gen_pct",
        "old_log_prob_pct",
        "values_pct",
        "update_critic_pct",
        "update_actor_pct",
        "update_weights_pct",
    )
    print("\t".join(header))

    rows_out: list[list[str]] = []
    for case in CASE_ORDER:
        s = summaries.get(case)
        if s is None:
            continue
        step = s["timing_s/step"]
        base_step = baseline["timing_s/step"]
        ratio = step / base_step if base_step > 0 else 0.0

        def pct(key: str) -> float:
            return (s[key] / step * 100.0) if step > 0 else 0.0

        row = [
            case,
            f"{int(s['last_step'])}",
            f"{step:.6f}",
            f"{ratio:.3f}",
            f"{pct('timing_s/gen'):.2f}",
            f"{pct('timing_s/old_log_prob'):.2f}",
            f"{pct('timing_s/values'):.2f}",
            f"{pct('timing_s/update_critic'):.2f}",
            f"{pct('timing_s/update_actor'):.2f}",
            f"{pct('timing_s/update_weights'):.2f}",
        ]
        rows_out.append(row)
        print("\t".join(row))

    if args.output_tsv is not None:
        args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_tsv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(header)
            writer.writerows(rows_out)
        print(f"\noutput_tsv={args.output_tsv}")


if __name__ == "__main__":
    main()
