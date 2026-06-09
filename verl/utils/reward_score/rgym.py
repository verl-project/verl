"""
Rule-based reward function for Reasoning Gym parquet rows.
Credits: https://github.com/EduardDurech/r-gym
"""

import json
from typing import Any

from reasoning_gym import get_score_answer_fn
from reasoning_gym.utils import extract_answer


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Score a veRL completion against the serialized Reasoning Gym entry."""
    entry = json.loads(ground_truth)
    source_dataset = (
        extra_info.get("source_dataset") if isinstance(extra_info, dict) else None
    )
    source_dataset = source_dataset or entry.get("metadata", {}).get("source_dataset") or data_source
    try:
        score_fn = get_score_answer_fn(source_dataset)
    except Exception as exc:
        print(
            f"Warning: Reasoning Gym dataset {source_dataset!r} is not registered; "
            f"returning 0 reward. Error: {exc}"
        )
        return 0.0
    dataset = getattr(score_fn, "__self__", None)

    candidates = []
    extracted = extract_answer(solution_str or "", tag_name="answer")
    if extracted is not None:
        candidates.append(extracted)
    candidates.append(solution_str)

    best_score = 0.0
    for candidate in candidates:
        try:
            if dataset is not None:
                score = dataset.score_answer_cascade(candidate, entry)
            else:
                score = score_fn(candidate, entry)
            best_score = max(best_score, float(score))
        except Exception:
            continue
    return best_score
