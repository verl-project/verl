"""Scoring dispatch for BlindTasks samples."""

from typing import Any

from .circled_letter import _verify_char, _verify_label
from .grid import _verify_joint
from .utils import (
    verify_count,
    verify_direction,
    verify_relation,
    verify_yes_no,
)

_COUNT_TASKS: frozenset[str] = frozenset(
    {
        "blindtasks.nested_squares",
        "blindtasks.nested_squares.hard",
        "blindtasks.circles",
        "blindtasks.lines",
        "blindtasks.lines.hard",
        "blindtasks.olympic",
        "blindtasks.olympic.hard",
        "blindtasks.olympic.interlocking",
        "blindtasks.grid",
        "blindtasks.subway",
        "blindtasks.subway.hard",
    }
)

_YES_NO_TASKS: frozenset[str] = frozenset({"blindtasks.circles.hard"})
_LABEL_TASKS: frozenset[str] = frozenset(
    {
        "blindtasks.circled_letter",
        "blindtasks.size_compare",
        "blindtasks.size_compare.hard",
    }
)
_CHAR_TASKS: frozenset[str] = frozenset({"blindtasks.circled_letter.hard"})
_JOINT_TASKS: frozenset[str] = frozenset(
    {
        "blindtasks.grid.hard",
        "blindtasks.grid.word",
    }
)
_DIRECTION_TASKS: frozenset[str] = frozenset(
    {
        "blindtasks.orientation",
        "blindtasks.orientation.hard",
    }
)
_RELATION_TASKS: frozenset[str] = frozenset(
    {
        "blindtasks.spatial_relation",
        "blindtasks.spatial_relation.hard",
    }
)


def score(
    task_name: str,
    ground_truth: dict[str, Any],
    prediction: str,
) -> float:
    if task_name in _COUNT_TASKS:
        target = int(ground_truth["answer"])
        return verify_count(prediction, target)

    if task_name in _YES_NO_TASKS:
        return verify_yes_no(prediction, str(ground_truth["answer"]))

    if task_name in _LABEL_TASKS:
        return _verify_label(prediction, str(ground_truth["answer"]))

    if task_name in _CHAR_TASKS:
        return _verify_char(prediction, str(ground_truth["answer"]))

    if task_name in _JOINT_TASKS:
        rows = int(ground_truth["rows"])
        cols = int(ground_truth["cols"])
        return _verify_joint(prediction, rows, cols)

    if task_name in _DIRECTION_TASKS:
        target = str(ground_truth["answer"])
        return verify_direction(prediction, target)

    if task_name in _RELATION_TASKS:
        return verify_relation(prediction, str(ground_truth["answer"]))

    raise ValueError(f"Unknown blindtasks task_name: {task_name!r}")
