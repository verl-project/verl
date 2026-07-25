# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Top-level KVCAwareConfig and parsing logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from .base import (
    ConfigError,
    StrategyConfig,
    _multiline_repr,
)
from .cache import CacheStoreConfig
from .collector import CollectorConfig

# ============================================================
# Top-level KVCAwareConfig
# ============================================================

_DEFAULT_COLLECTOR = CollectorConfig()
_DEFAULT_CACHE_STORE = CacheStoreConfig()


@dataclass(repr=False)
class KVCAwareConfig:
    """Top-level config for KVCAwareBalancer, parsed from OmegaConf DictConfig.

    VeRL composes the router config under the rollout ``strategy`` node
    (via the ``router@strategy`` hydra defaults entry — see
    ``verl/trainer/config/rollout/router/kvcaware.yaml``) and passes that node
    to the Balancer constructor.  The Balancer calls ``from_config(cfg)`` to
    obtain this fully-resolved dataclass instance.

    Attributes:
        strategies: Polymorphic strategy list (each with ``_target_``).
        collector: Collector module connection-type tuning config.
        cache_store: CacheStore configuration.
    """

    strategies: list[StrategyConfig]  # required, no default
    collector: CollectorConfig = field(default_factory=lambda: _DEFAULT_COLLECTOR)
    cache_store: CacheStoreConfig = field(default_factory=lambda: _DEFAULT_CACHE_STORE)

    @classmethod
    def from_config(cls, cfg: DictConfig | dict) -> KVCAwareConfig:
        """Two-step parsing of VeRL-transmitted config.

        Step 1: OmegaConf.merge for auto-recursive dataclass fields
                (collector, cache_store).

        Step 2: Manual traversal of strategies (list or dict from Hydra
                defaults composition) — instantiate each ``_target_`` entry.
        """
        if not isinstance(cfg, DictConfig | dict):
            raise ConfigError(f"cfg must be DictConfig or dict, got {type(cfg)}")

        cfg = OmegaConf.create(cfg)

        # ── Extract polymorphic sections before merge ──────────────
        # Strategies are polymorphic (_target_); pull them out before the
        # dataclass-typed merge below. Top-level `_target_` is defensively popped
        # for structured-input compatibility.
        strategies_raw = _extract_strategies(cfg)

        # ── Step 1: merge dataclass-typed fields (collector, cache_store) ──
        defaults = OmegaConf.create(
            {
                "collector": OmegaConf.structured(CollectorConfig),
                "cache_store": OmegaConf.structured(CacheStoreConfig),
            }
        )
        kwargs_for_merge = OmegaConf.create(cfg)
        # Remove polymorphic sections to avoid ReadonlyConfigError
        for key in ("strategies", "_target_"):
            if key in kwargs_for_merge:
                OmegaConf.set_struct(kwargs_for_merge, False)
                kwargs_for_merge.pop(key, None)
                OmegaConf.set_struct(kwargs_for_merge, True)

        # Validate non-dict types for collector/cache_store
        if (
            "collector" in kwargs_for_merge
            and kwargs_for_merge.collector is not None
            and not isinstance(kwargs_for_merge.collector, dict | DictConfig)
        ):
            raise ConfigError(f"collector must be a dict, got {type(kwargs_for_merge.collector).__name__}")
        if (
            "cache_store" in kwargs_for_merge
            and kwargs_for_merge.cache_store is not None
            and not isinstance(kwargs_for_merge.cache_store, dict | DictConfig)
        ):
            raise ConfigError(f"cache_store must be a dict, got {type(kwargs_for_merge.cache_store).__name__}")

        merged = OmegaConf.merge(defaults, kwargs_for_merge)
        config_obj = OmegaConf.to_object(merged)

        # Extract resolved dataclass fields
        if isinstance(config_obj, dict):
            collector_cfg = config_obj.get("collector") or CollectorConfig()
            cache_store_cfg = config_obj.get("cache_store") or CacheStoreConfig()
        else:
            collector_cfg = getattr(config_obj, "collector", None) or CollectorConfig()
            cache_store_cfg = getattr(config_obj, "cache_store", None) or CacheStoreConfig()

        # ── Step 2: parse strategies (polymorphic list) ────────────
        if strategies_raw is None:
            raise ConfigError("strategies is required — must be explicitly configured")
        strategies = _parse_polymorphic_list(strategies_raw, StrategyConfig, "strategies")

        # ── Validate and construct ─────────────────────────────────
        result = cls(
            strategies=strategies,
            collector=collector_cfg,
            cache_store=cache_store_cfg,
        )
        result.validate()
        return result

    def validate(self) -> None:
        """Validate the full config. Raises ConfigError with all violations."""
        errors: list[str] = []

        if not self.strategies:
            errors.append("strategies must be non-empty")
        elif not isinstance(self.strategies, list):
            errors.append("strategies must be a list")
        else:
            total_weight = sum(s.weight for s in self.strategies)
            if not (0.9 <= total_weight <= 1.1):
                errors.append(f"sum of strategy weights must be ~1.0, got {total_weight}")

        if errors:
            raise ConfigError("; ".join(errors))

    def __repr__(self) -> str:
        """Multi-line indented repr (delegates to the shared _multiline_repr)."""
        return _multiline_repr(self)


# ============================================================
# Helper functions
# ============================================================


def _extract_strategies(cfg: DictConfig) -> list[Any] | None:
    """Extract strategies from cfg, handling both list and dict formats.

    Hydra defaults composition produces a dict (keyed by strategy name),
    but direct YAML can be a list. Both are supported.
    """
    if "strategies" not in cfg:
        return None
    val = cfg["strategies"]
    if val is None:
        return None
    if isinstance(val, list | ListConfig):
        return list(val)
    if isinstance(val, dict | DictConfig):
        # Hydra defaults composition → dict of {name: strategy_cfg}
        return list(val.values())
    # Not a list or dict — will be caught by validation
    return val


def _parse_polymorphic_list(
    items: list[Any],
    base_class: type,
    list_name: str,
) -> list[Any]:
    """Parse a polymorphic list where each item has ``_target_`` for hydra.instantiate.

    Validates that each instantiated item is a subclass of ``base_class``.
    """
    result: list[Any] = []
    if not items:
        return result

    for i, item in enumerate(items):
        if not isinstance(item, dict | DictConfig):
            raise ConfigError(f"{list_name}[{i}] must be a dict, got {type(item)}")

        item_conf = OmegaConf.create(item) if isinstance(item, dict) else item

        if "_target_" not in item_conf:
            raise ConfigError(f"{list_name}[{i}] must have '_target_' key, got keys: {list(item_conf.keys())}")

        try:
            parsed = instantiate(item_conf)
        except (InstantiationException, ImportError, AttributeError) as e:
            raise ConfigError(f"{list_name}[{i}] failed to instantiate _target_ '{item_conf._target_}': {e}") from e

        if not isinstance(parsed, base_class):
            raise ConfigError(
                f"{list_name}[{i}] _target_ must inherit {base_class.__name__}, got {type(parsed).__name__}"
            )

        result.append(parsed)

    return result
