from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

import wandb

import scripts.wandb_rank_runs as wandb_rank_runs


_EARLY_STOP_CHANGE_CUTOFF = datetime(2026, 2, 3, 18, 59, tzinfo=timezone.utc)
_PATIENCE = 10


def _to_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def test_steps_since_best_on_cpu() -> None:
    wandb_rank_runs.load_wandb_api_key()
    assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY is required for this test."

    api = wandb.Api()
    runs = api.runs(f"{wandb_rank_runs.DEFAULT_ENTITY}/{wandb_rank_runs.DEFAULT_PROJECT}")

    pre_count = 0
    post_count = 0
    for run in runs:
        if wandb_rank_runs.should_skip(run):
            continue
        created_at = _to_datetime(getattr(run, "created_at", None))
        if created_at is None:
            continue
        if created_at < _EARLY_STOP_CHANGE_CUTOFF:
            cutoff_group = "pre"
        else:
            cutoff_group = "post"
            if getattr(run, "state", "") == "running":
                continue
        format_ok_metric = (
            wandb_rank_runs.DEFAULT_FORMAT_OK_METRIC
            if wandb_rank_runs.DEFAULT_FORMAT_OK_METRIC in run.summary
            else None
        )
        mc_metric = (
            wandb_rank_runs.DEFAULT_MC_METRIC
            if wandb_rank_runs.DEFAULT_MC_METRIC in run.summary
            else None
        )
        best_value, best_step, _, _, last_step = wandb_rank_runs.best_metric_value(
            run,
            wandb_rank_runs.DEFAULT_METRIC,
            format_ok_metric,
            mc_metric,
        )
        if best_value is None or best_step is None or last_step is None:
            continue
        assert last_step >= best_step, "last_step should be >= best_step."
        steps_since_best = last_step - best_step
        assert steps_since_best >= 0, "steps_since_best should be >= 0."
        if cutoff_group == "post":
            assert (
                steps_since_best <= _PATIENCE
            ), f"steps_since_best should be <= {_PATIENCE} after cutoff."
            post_count += 1
        else:
            pre_count += 1

    assert pre_count > 0, "No pre-cutoff runs with metric history were found."
    assert post_count > 0, "No post-cutoff runs with metric history were found."
