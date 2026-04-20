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

import logging
import random
import threading
from abc import ABC, abstractmethod
from typing import Any

from verl.experimental.agent_loop.prometheus_metrics import (
    build_metrics_url,
    fetch_prometheus_text,
    parse_prometheus_metric_value,
)

logger = logging.getLogger(__name__)

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
    """Stateless or stateful strategy object used for new routing decisions (sticky handled elsewhere).

    Subclasses must implement pick_server and accept
    __init__(self, server_actor_ids: list[str], **kwargs)
    """

    @abstractmethod
    def pick_server(
        self,
        server_ids: list[str],
        inflight: dict[str, int],
    ) -> str:
        """Choose server_ids member."""


@register("least_requests")
class LeastRequestsStrategy(LoadBalanceStrategy):
    def __init__(self, server_actor_ids: list[str], **kwargs: Any):
        pass

    def pick_server(
        self,
        server_ids: list[str],
        inflight: dict[str, int],
    ) -> str:
        # O(n): one pass for min inflight; ties broken by lexicographically smallest sid (same as sorted(server_ids)).
        if not server_ids:
            raise ValueError("server_ids must be non-empty")
        server_id = min(server_ids, key=lambda sid: (inflight[sid], sid))
        return server_id


@register("least_kv_cache")
class LeastKVCacheStrategy(LoadBalanceStrategy):
    """Prefer lowest KV usage from periodic HTTP /metrics scrape; else fall back to least in-flight.

    When metric_name is set, a daemon thread scrapes each replica on refresh_interval_s and
    pick_server uses that internal state. When metric_name is unset, no HTTP is performed and
    all KV readings are treated as unknown (same as least in-flight, with lexicographic sid tie-break).
    """

    def __init__(self, server_actor_ids: list[str], **kwargs: Any):
        self._server_actor_ids = list(server_actor_ids)
        self._metric_name: str | None = kwargs.pop("metric_name", None)
        self._metrics_path: str = kwargs.pop("metrics_path", "/metrics")
        self._refresh_interval_s: float = float(kwargs.pop("refresh_interval_s", 2.0))
        self._fetch_timeout_s: float = float(kwargs.pop("fetch_timeout_s", 2.0))

        self._kv_usage: dict[str, float | None] = {sid: None for sid in self._server_actor_ids}
        self._lock = threading.Lock()
        self._metrics_stop = threading.Event()
        self._metrics_thread: threading.Thread | None = None
        self._metrics_fail_warned: set[str] = set()

        if self._metric_name:
            self._metrics_thread = threading.Thread(
                target=self._metrics_background_loop,
                name="verl-lb-least-kv-cache",
                daemon=True,
            )
            self._metrics_thread.start()

    def _metrics_background_loop(self) -> None:
        while not self._metrics_stop.is_set():
            self._refresh_metrics_blocking()
            if self._metrics_stop.wait(timeout=self._refresh_interval_s):
                break

    def _mark_metrics_failed(self, sid: str) -> bool:
        with self._lock:
            self._kv_usage[sid] = None
            if sid not in self._metrics_fail_warned:
                self._metrics_fail_warned.add(sid)
                return True
            return False

    def _mark_metrics_success(self, sid: str, val: float | None) -> bool:
        with self._lock:
            self._kv_usage[sid] = val
            if sid in self._metrics_fail_warned:
                self._metrics_fail_warned.discard(sid)
                return True
            return False

    def _kv_snapshot(self, server_ids: list[str]) -> dict[str, float | None]:
        with self._lock:
            return {sid: self._kv_usage.get(sid) for sid in server_ids}

    def _refresh_metrics_blocking(self) -> None:
        metric_name = self._metric_name
        if not metric_name:
            return

        for sid in self._server_actor_ids:
            url = build_metrics_url(sid, self._metrics_path)
            try:
                text = fetch_prometheus_text(url, self._fetch_timeout_s)
                val = parse_prometheus_metric_value(text, metric_name)
                recovered = self._mark_metrics_success(sid, val)
                if recovered:
                    logger.info("KV cache metrics recovered for replica %s", sid)
            except Exception as e:
                self._mark_metrics_failed(sid)
                logger.warning("Failed to refresh KV metrics for %s from %s: %s", sid, url, e)

    def pick_server(
        self,
        server_ids: list[str],
        inflight: dict[str, int],
    ) -> str:
        if self._metric_name:
            usage = self._kv_snapshot(server_ids)
        else:
            usage = {sid: None for sid in server_ids}

        def key(sid: str) -> tuple[float, int, str]:
            kv = usage.get(sid)
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
    ) -> str:
        return self._rng.choice(server_ids)
