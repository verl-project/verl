#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd
import wandb

from verl.experimental.fully_async_policy.scaling_metrics import (
    PLOT_SECTIONS,
    iter_plot_pairs,
)


def _parse_project_path(project_path: str) -> tuple[str | None, str]:
    if "/" not in project_path:
        return None, project_path
    entity, project = project_path.split("/", 1)
    return entity, project


def _matches_filters(run: Any, args: argparse.Namespace) -> bool:
    if args.state and run.state != args.state:
        return False
    if args.group and run.group not in set(args.group):
        return False
    if args.tag:
        run_tags = set(run.tags or [])
        if not set(args.tag).issubset(run_tags):
            return False
    if args.name_regex and not re.search(args.name_regex, run.name or ""):
        return False
    return True


def _summary_dict(run: Any) -> dict[str, Any]:
    summary = getattr(run.summary, "_json_dict", None)
    if summary is not None:
        return dict(summary)
    return dict(run.summary)


def collect_runs_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    api = wandb.Api()
    metric_names = sorted({metric for pair in iter_plot_pairs() for metric in pair})
    rows: list[dict[str, Any]] = []
    for run in api.runs(args.source_project):
        if not _matches_filters(run, args):
            continue

        summary = _summary_dict(run)
        row = {
            "run_id": run.id,
            "run_name": run.name,
            "run_group": run.group,
            "run_state": run.state,
            "run_url": run.url,
        }
        for metric_name in metric_names:
            value = summary.get(metric_name)
            if isinstance(value, list | dict):
                continue
            row[metric_name] = value
        rows.append(row)
        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        return pd.DataFrame(
            columns=["run_id", "run_name", "run_group", "run_state", "run_url", *metric_names]
        )

    dataframe = pd.DataFrame.from_records(rows)
    for metric_name in metric_names:
        if metric_name not in dataframe.columns:
            dataframe[metric_name] = pd.NA
    return dataframe


def _make_plot_key(section: str, x_metric: str, y_metric: str) -> str:
    key = f"plots/{section}/{x_metric}__vs__{y_metric}"
    return key.replace(" ", "_")


def log_plots(
    dataframe: pd.DataFrame,
    source_project: str,
    analysis_project: str | None,
    analysis_name: str,
) -> None:
    entity, project = _parse_project_path(analysis_project or source_project)
    run = wandb.init(
        entity=entity,
        project=project,
        job_type="analysis",
        name=analysis_name,
        config={
            "source_project": source_project,
            "plot_sections": {section: list(pairs) for section, pairs in PLOT_SECTIONS.items()},
        },
    )
    try:
        if not dataframe.empty:
            run.log({"tables/fully_async_scaling_runs": wandb.Table(dataframe=dataframe)})

        for section, plot_pairs in PLOT_SECTIONS.items():
            for x_metric, y_metric in plot_pairs:
                plot_df = dataframe[["run_name", "run_id", "run_url", x_metric, y_metric]].dropna()
                if plot_df.empty:
                    continue
                table = wandb.Table(dataframe=plot_df)
                run.log(
                    {
                        _make_plot_key(section, x_metric, y_metric): wandb.plot.scatter(
                            table,
                            x=x_metric,
                            y=y_metric,
                            title=f"{y_metric} vs {x_metric}",
                        )
                    }
                )
    finally:
        run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect fully async scaling summaries from W&B and log the standard scatter plots.",
    )
    parser.add_argument(
        "--source-project",
        required=True,
        help="W&B source project path in the form entity/project.",
    )
    parser.add_argument(
        "--analysis-project",
        default=None,
        help="W&B project path to store the generated analysis run. Defaults to --source-project.",
    )
    parser.add_argument(
        "--analysis-name",
        default="fully-async-scaling-analysis",
        help="Name of the analysis run to create.",
    )
    parser.add_argument(
        "--state",
        default="finished",
        help="Optional W&B run state filter. Use an empty string to disable filtering.",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="Repeatable W&B run group filter.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Repeatable W&B tag filter. All provided tags must be present.",
    )
    parser.add_argument(
        "--name-regex",
        default=None,
        help="Optional regex applied to the W&B run name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of matching runs to inspect.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to write the collected run summary table as CSV.",
    )
    parser.add_argument(
        "--skip-log",
        action="store_true",
        help="Collect the table and optionally write CSV without creating a W&B analysis run.",
    )
    args = parser.parse_args()

    dataframe = collect_runs_dataframe(args)
    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)
        print(f"Wrote {len(dataframe)} rows to {output_path}")
    else:
        print(f"Collected {len(dataframe)} runs from {args.source_project}")

    if not args.skip_log:
        log_plots(
            dataframe=dataframe,
            source_project=args.source_project,
            analysis_project=args.analysis_project,
            analysis_name=args.analysis_name,
        )


if __name__ == "__main__":
    main()
