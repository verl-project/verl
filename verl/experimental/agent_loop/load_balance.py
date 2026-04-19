# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pluggable load-balance strategies for :class:`GlobalRequestLoadBalancer`."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

_LOAD_BALANCE_STRATEGY_CLASSES: dict[str, type[LoadBalanceStrategy]] = {}


def _register(name: str, cls: type[LoadBalanceStrategy]) -> None:
    if name in _LOAD_BALANCE_STRATEGY_CLASSES:
        raise ValueError(f"Load balance strategy {name!r} is already registered")
    _LOAD_BALANCE_STRATEGY_CLASSES[name] = cls


def register(name: str):
    """Decorator: @register("my_strategy") on a :class:LoadBalanceStrategy subclass (like agent loop @register)."""

    def decorator(cls: type[LoadBalanceStrategy]) -> type[LoadBalanceStrategy]:
        _register(name, cls)
        return cls

    return decorator


def register_load_balance_strategy(name: str, cls: type[LoadBalanceStrategy]) -> None:
    """Register a strategy class without decorator (e.g. plugins). Same as ``@register(name)`` on ``cls``."""
    _register(name, cls)


def create_load_balance_strategy(name: str, server_actor_ids: list[str], **kwargs: Any) -> LoadBalanceStrategy:
    """Instantiate a registered strategy by name."""
    cls = _LOAD_BALANCE_STRATEGY_CLASSES.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown load_balance_strategy {name!r}. Registered: {sorted(_LOAD_BALANCE_STRATEGY_CLASSES)}"
        )
    return cls(server_actor_ids, **kwargs)


class LoadBalanceStrategy(ABC):
    """Stateless or stateful strategy object used for *new* routing decisions (sticky handled elsewhere).

    Subclasses must implement ``pick_server`` and accept
    ``__init__(self, server_actor_ids: list[str], **kwargs)``.
    """

    @abstractmethod
    def pick_server(
        self,
        server_ids: list[str],
        inflight: dict[str, int],
        kv_usage: dict[str, float | None],
    ) -> str:
        """Choose ``server_ids`` member. ``kv_usage`` may contain ``None`` for unknown metrics."""


@register("least_requests")
class LeastRequestsStrategy(LoadBalanceStrategy):
    def __init__(self, server_actor_ids: list[str], **kwargs: Any):
        pass

    def pick_server(
        self,
        server_ids: list[str],
        inflight: dict[str, int],
        kv_usage: dict[str, float | None],
    ) -> str:
        min_inf = min(inflight[s] for s in server_ids)
        for sid in sorted(server_ids):
            if inflight[sid] == min_inf:
                return sid
        raise RuntimeError("unreachable")


@register("least_kv_cache")
class LeastKVCacheStrategy(LoadBalanceStrategy):
    """Prefer lowest ``kv_usage`` scalar; unknown metrics fall back to least in-flight."""

    def __init__(self, server_actor_ids: list[str], **kwargs: Any):
        pass

    def pick_server(
        self,
        server_ids: list[str],
        inflight: dict[str, int],
        kv_usage: dict[str, float | None],
    ) -> str:
        def key(sid: str) -> tuple[float, int, str]:
            kv = kv_usage.get(sid)
            if kv is None:
                return (float("inf"), inflight[sid], sid)
            return (kv, inflight[sid], sid)

        return min(server_ids, key=key)


@register("weighted_rr")
class WeightedRoundRobinStrategy(LoadBalanceStrategy):
    """Smooth weighted round-robin (largest-current-weight), static weights per replica order."""

    def __init__(self, server_actor_ids: list[str], weights: list[float] | None = None, **kwargs: Any):
        self._ids = list(server_actor_ids)
        n = len(self._ids)
        if n == 0:
            raise ValueError("server_actor_ids must be non-empty")
        if weights is None:
            w = [1.0] * n
        else:
            if len(weights) != n:
                raise ValueError(f"load_balance_weights length {len(weights)} != num servers {n}")
            w = [float(x) for x in weights]
        self._weights = w
        self._current = [0.0] * n

    def pick_server(
        self,
        server_ids: list[str],
        inflight: dict[str, int],
        kv_usage: dict[str, float | None],
    ) -> str:
        # Selection ignores inflight; weights define share across replicas.
        pos = {sid: i for i, sid in enumerate(self._ids)}
        indices = [pos[s] for s in server_ids]
        total = sum(self._weights[i] for i in indices)
        for i in indices:
            self._current[i] += self._weights[i]
        best_i = max(indices, key=lambda i: (self._current[i], -i))
        self._current[best_i] -= total
        return self._ids[best_i]


@register("random")
class RandomStrategy(LoadBalanceStrategy):
    def __init__(self, server_actor_ids: list[str], random_seed: int | None = None, **kwargs: Any):
        self._rng = random.Random(random_seed)

    def pick_server(
        self,
        server_ids: list[str],
        inflight: dict[str, int],
        kv_usage: dict[str, float | None],
    ) -> str:
        return self._rng.choice(server_ids)
