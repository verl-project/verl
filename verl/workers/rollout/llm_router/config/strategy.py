"""Strategy-specific configs.

Concrete routing strategy configs. The matching runtime strategy classes
(e.g. ``KVCAwareStrategy``) live under ``verl.workers.rollout.llm_router.strategies``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verl.workers.rollout.llm_router.config.base import ConfigError, StrategyConfig, _multiline_repr


@dataclass(repr=False)
class KVCAwareStrategyConfig(StrategyConfig):
    """Config for KVCache-Aware routing strategy.

    S = α × S_cache + (1-α) × S_load
    """

    alpha: float = 0.7
    load_threshold: float = 0.9
    layer_weights: dict[str, float] = field(default_factory=lambda: {"gpu": 0.7, "cpu": 0.2, "ssd": 0.1})

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.load_threshold < 1:
            raise ConfigError(f"load_threshold must be in (0, 1), got {self.load_threshold}")
        valid_keys = {"gpu", "cpu", "ssd"}
        if not set(self.layer_weights.keys()) == valid_keys:
            raise ConfigError(f"layer_weights keys must be {valid_keys} only, got {set(self.layer_weights.keys())}")
        weights_sum = sum(self.layer_weights.values())
        if abs(weights_sum - 1.0) > 1e-6:
            raise ConfigError(f"layer_weights values must sum to 1.0, got {weights_sum}")

    __repr__ = _multiline_repr
