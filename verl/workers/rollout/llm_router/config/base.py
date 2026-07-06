"""Base config classes and multi-line repr helpers for KVCAware LLM Router.

This module holds the shared config primitives and the ``_multiline_repr`` helpers
used by every config dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from omegaconf import DictConfig, ListConfig


class ConfigError(ValueError):
    """Raised when config validation fails."""


# ============================================================
# Multi-line repr helpers (shared by all config dataclasses)
# ============================================================


def _format_value(value: Any, indent: int) -> str:
    """Format a value for multi-line repr, recursing into nested containers."""
    if is_dataclass(value) and not isinstance(value, type):
        return _multiline_repr(value, indent)
    if isinstance(value, list | ListConfig):
        items = list(value)
        if not items:
            return "[]"
        inner = ",\n".join(f"{' ' * (indent + 2)}{_format_value(v, indent + 2)}" for v in items)
        return f"[\n{inner},\n{' ' * indent}]"
    if isinstance(value, dict | DictConfig):
        pairs = list(value.items())
        if not pairs:
            return "{}"
        inner = ",\n".join(f"{' ' * (indent + 2)}{k!r}: {_format_value(v, indent + 2)}" for k, v in pairs)
        return "{\n" + inner + f",\n{' ' * indent}}}"
    return repr(value)


def _multiline_repr(obj: Any, indent: int = 0) -> str:
    """Multi-line indented repr for a dataclass instance.

    Each field on its own line; nested dataclasses/lists/dicts recurse with
    deeper indentation.  Produces a readable tree, e.g.::

        KVCAwareConfig(
          strategies=[
            KVCAwareStrategyConfig(
              weight=1.0,
              ...
            ),
          ],
          ...
        )
    """
    pad = " " * indent
    inner_pad = " " * (indent + 2)
    parts = []
    for f in fields(obj):
        if not f.repr:
            continue
        val = getattr(obj, f.name)
        parts.append(f"{f.name}={_format_value(val, indent + 2)}")
    if not parts:
        return f"{type(obj).__name__}()"
    body = ",\n".join(f"{inner_pad}{p}" for p in parts)
    return f"{type(obj).__name__}(\n{body},\n{pad})"


# ============================================================
# Strategy base class
# ============================================================


@dataclass(repr=False)
class StrategyConfig:
    """Base config for routing strategies.

    All strategy configs inherit this and must provide ``weight`` and
    ``collector_names`` (the list of collector names this strategy uses).

    Attributes:
        weight: Multi-strategy weighting coefficient (0 < weight ≤ 1).
        collector_names: Collector names this strategy binds to.
    """

    weight: float
    collector_names: list[str]

    def __post_init__(self) -> None:
        if not (0 < self.weight <= 1):
            raise ConfigError(f"weight must be in (0, 1], got {self.weight}")
        if not isinstance(self.collector_names, list | ListConfig):
            raise ConfigError(f"collector_names must be a list, got {type(self.collector_names).__name__}")
        # Normalize ListConfig → plain list for consistent downstream use
        if isinstance(self.collector_names, ListConfig):
            object.__setattr__(self, "collector_names", list(self.collector_names))

    __repr__ = _multiline_repr
