#!/usr/bin/env python3
"""Rank W&B runs by best metric value across all steps."""

from __future__ import annotations

import argparse
import math
import os
from typing import Iterable, Optional, Tuple

import wandb


DEFAULT_ENTITY = "tommaso-bendinelli-eth-zurich"
DEFAULT_PROJECT = "multiple_choice_question_study"
DEFAULT_METRIC = "val-aux/openai/gsm8k/score/mean@1"
DEFAULT_MC_METRIC = "val-aux/openai/gsm8k_mc/score/mean@1"
DEFAULT_FORMAT_OK_METRIC = "val-aux/openai/gsm8k/format_ok/mean@1"
DEFAULT_EVAL_SIZE = 1319
ENV_PATH = "/local/home/tommaben/repo/custom/verl/.env"


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_wandb_api_key(env_path: str = ENV_PATH) -> None:
    if not os.path.exists(env_path):
        return

    api_key = None
    with open(env_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if not line.startswith("WANDB_API_KEY="):
                continue
            _, value = line.split("=", 1)
            api_key = _strip_quotes(value.strip())
            break

    if api_key and not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = api_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank W&B runs by best metric value across all steps."
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument(
        "--mc-metric",
        default=DEFAULT_MC_METRIC,
        help="Metric to capture at the best step (default: %(default)s).",
    )
    parser.add_argument("--format-ok-metric", default=DEFAULT_FORMAT_OK_METRIC)
    parser.add_argument("--eval-size", type=int, default=DEFAULT_EVAL_SIZE)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of runs to display (default: all).",
    )
    return parser.parse_args()


def _iter_history(
    run: wandb.apis.public.Run,
    metric: str,
    format_ok_metric: Optional[str],
    mc_metric: Optional[str],
) -> Iterable[dict]:
    keys = [metric, "training/global_step", "_step"]
    if format_ok_metric:
        keys.append(format_ok_metric)
    if mc_metric:
        keys.append(mc_metric)
    return run.scan_history(keys=keys)


def best_metric_value(
    run: wandb.apis.public.Run,
    metric: str,
    format_ok_metric: Optional[str],
    mc_metric: Optional[str],
) -> Tuple[
    Optional[float], Optional[int], Optional[float], Optional[float], Optional[int]
]:
    best_value: Optional[float] = None
    best_step: Optional[int] = None
    best_format_ok: Optional[float] = None
    best_mc_value: Optional[float] = None
    last_step: Optional[int] = None

    for row in _iter_history(run, metric, format_ok_metric, mc_metric):
        value = row.get(metric)
        if value is None:
            continue
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(value_float) or math.isinf(value_float):
            continue

        step = row.get("training/global_step")
        if step is None:
            step = row.get("_step")
        try:
            step_int = int(step) if step is not None else None
        except (TypeError, ValueError):
            step_int = None
        if step_int is not None and (last_step is None or step_int > last_step):
            last_step = step_int

        if best_value is None or value_float > best_value:
            best_value = value_float
            best_step = step_int
            if format_ok_metric:
                format_ok_value = row.get(format_ok_metric)
                try:
                    format_ok_float = (
                        float(format_ok_value) if format_ok_value is not None else None
                    )
                except (TypeError, ValueError):
                    format_ok_float = None
                if format_ok_float is not None and (
                    math.isnan(format_ok_float) or math.isinf(format_ok_float)
                ):
                    format_ok_float = None
                best_format_ok = format_ok_float
            else:
                best_format_ok = None
            if mc_metric:
                mc_value = row.get(mc_metric)
                try:
                    mc_float = float(mc_value) if mc_value is not None else None
                except (TypeError, ValueError):
                    mc_float = None
                if mc_float is not None and (math.isnan(mc_float) or math.isinf(mc_float)):
                    mc_float = None
                best_mc_value = mc_float
            else:
                best_mc_value = None

    if last_step is None:
        if best_step is not None:
            last_step = best_step
        else:
            last_step = 0

    return best_value, best_step, best_format_ok, best_mc_value, last_step


def should_skip(run: wandb.apis.public.Run) -> bool:
    name = getattr(run, "name", "") or ""
    display_name = getattr(run, "display_name", "") or ""
    return name.startswith("upload_") or display_name.startswith("upload_")


def main() -> int:
    args = parse_args()
    load_wandb_api_key()

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")

    results = []
    for run in runs:
        if should_skip(run):
            continue
        best_value, best_step, best_format_ok, best_mc_value, last_step = best_metric_value(
            run, args.metric, args.format_ok_metric, args.mc_metric
        )
        if best_value is None:
            continue
        if best_step is None:
            steps_since_best = ""
        else:
            steps_since_best = str(max(0, last_step - best_step))
        run_name = run.display_name or run.name or ""
        if best_format_ok is None:
            syntax_errors = None
        else:
            syntax_errors = int(round((1.0 - best_format_ok) * args.eval_size))
        results.append(
            {
                "best_value": best_value,
                "step": best_step,
                "steps_since_best": steps_since_best,
                "mc_score": best_mc_value,
                "format_ok": best_format_ok,
                "syntax_errors": syntax_errors,
                "id": run.id,
                "name": run_name,
            }
        )

    results.sort(key=lambda item: item["best_value"], reverse=True)

    if args.limit is not None:
        results = results[: args.limit]

    header = [
        "rank",
        "best_value",
        "step",
        "steps_since_best",
        "mc_score",
        "format_ok",
        "syntax_errors",
        "run_id",
        "run_name",
    ]
    rows = []
    for idx, item in enumerate(results, start=1):
        rows.append(
            [
                str(idx),
                f"{item['best_value']:.6f}",
                "" if item["step"] is None else str(item["step"]),
                item["steps_since_best"],
                "" if item["mc_score"] is None else f"{item['mc_score']:.6f}",
                ""
                if item["format_ok"] is None
                else f"{item['format_ok']:.6f}",
                ""
                if item["syntax_errors"] is None
                else str(item["syntax_errors"]),
                item["id"],
                item["name"],
            ]
        )

    widths = [len(col) for col in header]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def format_row(values):
        return "  ".join(value.ljust(widths[i]) for i, value in enumerate(values))

    print(format_row(header))
    print(format_row(["-" * width for width in widths]))
    for row in rows:
        print(format_row(row))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
