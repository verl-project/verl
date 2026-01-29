from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable, Optional, Sequence, TypeVar

T = TypeVar("T")


@contextmanager
def patch_bridge_adapter_filter(
    model_bridge,
    extra_predicate: Callable[[str], bool],
):
    """Temporarily extend bridge adapter predicate."""
    if model_bridge is None:
        yield
        return
    orig = getattr(model_bridge, "_is_adapter_param_name", None)
    if orig is None:
        yield
        return

    def _patched(name: str) -> bool:
        return orig(name) or extra_predicate(name)

    model_bridge._is_adapter_param_name = _patched
    try:
        yield
    finally:
        model_bridge._is_adapter_param_name = orig


@contextmanager
def patch_bridge_build_tasks(
    model_bridge,
    task_transform: Callable[[Optional[Sequence[T]]], Optional[Sequence[T]]],
):
    """Temporarily wrap build_conversion_tasks."""
    if model_bridge is None:
        yield
        return
    orig = getattr(model_bridge, "build_conversion_tasks", None)
    if orig is None:
        yield
        return

    def _wrapped(*args, **kwargs):
        tasks = orig(*args, **kwargs)
        return task_transform(tasks)

    model_bridge.build_conversion_tasks = _wrapped
    try:
        yield
    finally:
        model_bridge.build_conversion_tasks = orig


def filter_none_tasks(tasks: Optional[Iterable[T]]) -> Optional[list[T]]:
    if tasks is None:
        return None
    return [task for task in tasks if task is not None]
