"""Reward adapter for Tool Gym exported veRL rows."""

from __future__ import annotations

from typing import Any

from tool_gym.train_utils.export.verl.reward_score import (
    compute_score as toolgym_compute_score,
)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    return toolgym_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
