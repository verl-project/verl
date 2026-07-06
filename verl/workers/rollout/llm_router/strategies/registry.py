"""Runtime strategy registry.

Maps a strategy *config dataclass type* to its *runtime* strategy class, so
the Balancer can instantiate the right strategy from a parsed config without a
name lookup. This is distinct from the config layer's ``_target_`` dispatch
(hydra ``instantiate``), which resolves config dataclasses themselves.
"""

from __future__ import annotations

from typing import ClassVar


class StrategyRegistry:
    """Class-level registry mapping strategy config type → strategy class."""

    _registry: ClassVar[dict[type, type]] = {}

    @classmethod
    def register(cls, config_cls: type, strategy_cls: type) -> None:
        """Register a runtime strategy class for a config dataclass type."""
        cls._registry[config_cls] = strategy_cls

    @classmethod
    def get(cls, config_cls: type) -> type:
        """Look up the runtime strategy class registered for ``config_cls``."""
        if config_cls not in cls._registry:
            raise KeyError(
                f"No strategy registered for config type {config_cls!r}; registered: {list(cls._registry.keys())}"
            )
        return cls._registry[config_cls]
