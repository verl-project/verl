"""
Rule-based reward function for BlindTasks rows.
Credits: https://github.com/swiss-ai/vrl/tree/vlmsareblind
"""

import json
from typing import Any

from .scorer import score


def _parse_ground_truth(ground_truth: Any) -> dict[str, Any] | None:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            parsed = json.loads(ground_truth)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _task_name_from_inputs(
    data_source: str,
    extra_info: dict[str, Any] | None,
) -> str | None:
    """Return the BlindTasks registry key used to choose the verifier"""
    if isinstance(extra_info, dict) and extra_info.get("task_name"):
        return str(extra_info["task_name"])
    if isinstance(data_source, str) and data_source.startswith("blindtasks."):
        return data_source
    return None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Score a final display_answers answer against a BlindTasks row.
    NOTE: the original VRL scorer supports partial-credit via `shaped`,
    but we stick to binary reward to avoid reward hacking.
    """
    task_name = _task_name_from_inputs(data_source, extra_info)
    if task_name is None:
        print(
            "Warning: BlindTasks reward requires extra_info['task_name']; "
            "returning 0 reward."
        )
        return 0.0

    gt = _parse_ground_truth(ground_truth)
    if gt is None:
        print(
            "Warning: BlindTasks ground_truth is not a JSON object; "
            "returning 0 reward."
        )
        return 0.0

    prediction = "" if solution_str is None else str(solution_str)
    if not prediction:
        return 0.0

    try:
        return float(
            score(
                task_name,
                gt,
                prediction,
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        print(
            f"Warning: BlindTasks scoring failed for data_source={data_source!r}, "
            f"task_name={task_name!r}; returning 0 reward. Error: {exc}"
        )
        return 0.0
