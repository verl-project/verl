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

"""Strategy-specific configs.

Concrete routing strategy configs. The matching runtime strategy classes
(e.g. ``KVCacheAwareStrategy``) live under ``verl.workers.rollout.router.kvcaware.strategies``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import Layer, OverloadMode, SlowCut
from .base import ConfigError, StrategyConfig, _multiline_repr


@dataclass(repr=False)
class KVCAwareStrategyConfig(StrategyConfig):
    """Config for KVCache-Aware routing strategy.

    S = α × S_cache + (1-α) × S_load
    """

    alpha: float = 0.7
    load_threshold: float = 0.9
    layer_weights: dict[Layer, float] = field(default_factory=lambda: {Layer.GPU: 0.7, Layer.CPU: 0.2, Layer.SSD: 0.1})
    # Sticky short-circuit: when True, a returning session is sent back to its
    # bound replica only if that replica is NOT overloaded (load > load_threshold).
    memory_overload_filter: bool = True
    # Fallback scoring mode used after the sticky short-circuit misses.
    slow_cut: SlowCut = SlowCut.PREFIX_LOAD_AWARE
    # Overload check mode for the sticky short-circuit (independent of slow_cut).
    # ``simple`` = kv_cache_usage_perc > load_threshold; ``blended`` = the
    # original weighted load formula. Default ``blended`` preserves behavior.
    overload_mode: OverloadMode = OverloadMode.KV_LOAD

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.load_threshold < 1:
            raise ConfigError(f"load_threshold must be in (0, 1), got {self.load_threshold}")
        if not isinstance(self.memory_overload_filter, bool):
            raise ConfigError(f"memory_overload_filter must be a bool, got {self.memory_overload_filter!r}")
        # Normalize yaml str → SlowCut (also validates the value is a known mode).
        try:
            self.slow_cut = SlowCut(self.slow_cut)
        except ValueError as exc:
            raise ConfigError(f"slow_cut must be one of {[m.value for m in SlowCut]}, got {self.slow_cut!r}") from exc
        # Normalize yaml str → OverloadMode (also validates the value is a known mode).
        try:
            self.overload_mode = OverloadMode(self.overload_mode)
        except ValueError as exc:
            raise ConfigError(
                f"overload_mode must be one of {[m.value for m in OverloadMode]}, got {self.overload_mode!r}"
            ) from exc
        # Normalize yaml str keys → Layer (also validates each key is a known layer).
        try:
            self.layer_weights = {Layer(k): v for k, v in self.layer_weights.items()}
        except ValueError as exc:
            raise ConfigError(f"layer_weights keys must be layer names, got {set(self.layer_weights)}") from exc
        if set(self.layer_weights.keys()) != {Layer.GPU, Layer.CPU, Layer.SSD}:
            raise ConfigError(
                f"layer_weights must be exactly {{{Layer.GPU}, {Layer.CPU}, {Layer.SSD}}}, "
                f"got {set(self.layer_weights.keys())}"
            )
        weights_sum = sum(self.layer_weights.values())
        if abs(weights_sum - 1.0) > 1e-6:
            raise ConfigError(f"layer_weights values must sum to 1.0, got {weights_sum}")

    __repr__ = _multiline_repr
