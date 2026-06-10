from . import circled_letter, circles, grid, lines, nested_squares, olympic, orientation, size_compare, subway

from . import (
    spatial_relation,
)
from .base import TaskType

_TASK_MODULES = (
    circled_letter,
    circles,
    grid,
    lines,
    nested_squares,
    olympic,
    orientation,
    size_compare,
    spatial_relation,
    subway,
)

_ALL_TASKS: list[TaskType] = (
    [m.TASK_EASY for m in _TASK_MODULES]
    + [m.TASK_HARD for m in _TASK_MODULES]
    + [olympic.TASK_INTERLOCKING, grid.TASK_WORD]
)
REGISTRY: dict[str, TaskType] = {t.name: t for t in _ALL_TASKS}

BALANCED_CLASSES: dict[str, tuple] = {
    "blindtasks.nested_squares": tuple(range(2, 7)),
    "blindtasks.nested_squares.hard": tuple(range(2, 6)),
    "blindtasks.lines": tuple(range(2, 16)),
    "blindtasks.lines.hard": (0, 1, 2),
    "blindtasks.circles": tuple(range(2, 9)),
    "blindtasks.circles.hard": ("yes", "no"),
    "blindtasks.circled_letter": ("A", "B", "C", "D"),
    "blindtasks.olympic": tuple(range(3, 7)),
    "blindtasks.olympic.hard": tuple(range(5, 10)),
    "blindtasks.olympic.interlocking": tuple(range(5, 10)),
    "blindtasks.grid": tuple(range(2, 13)),
    "blindtasks.subway": (0, 1, 2),
    "blindtasks.subway.hard": (0, 1, 2, 3),
    "blindtasks.orientation": ("N", "S", "E", "W"),
    "blindtasks.orientation.hard": ("N", "NE", "E", "SE", "S", "SW", "W", "NW"),
    "blindtasks.spatial_relation": ("above", "below", "left", "right"),
    "blindtasks.spatial_relation.hard": ("above", "below", "left", "right"),
    "blindtasks.size_compare": ("A", "B"),
    "blindtasks.size_compare.hard": ("A", "B", "C"),
    # Skipped (combinatorial answer space):
    # - blindtasks.circled_letter.hard (30+ characters)
    # - blindtasks.grid.hard (121 (rows, cols) tuples)
    # - blindtasks.grid.word (49 (rows, cols) tuples)
}


def get(name: str) -> TaskType:
    if name not in REGISTRY:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]
