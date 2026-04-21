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
"""Pluggable load-balance strategies (``pick_server``) used by Ray actors in ``load_balance``."""

from __future__ import annotations

import logging
import random
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from verl.experimental.agent_loop.prometheus_metrics import (
    build_metrics_url,
    fetch_prometheus_text,
    parse_prometheus_metric_value,
)

logger = logging.getLogger(__name__)

# Cap parallel /metrics scrapes per refresh
_MAX_KV_METRICS_SCRAPE_WORKERS = 32

_LOAD_BALANCE_STRATEGY_CLASSES: dict[str, type[LoadBalanceStrategy]] = {}


def host_key_for_load_balance(server_address: str) -> str:
    """extract host part from server_address"""
    a = server_address.strip()
    if not a:
        return a
    # [IPv6 address]:port
    if a.startswith("["):
        rb = a.find("]")
        if rb != -1:
            return a[1:rb].strip()
        return a
    # IPv4 address:port
    if a.count(":") == 1:
        host, port = a.rsplit(":", 1)
        if port.isdigit():
            return host.strip()
    return a


def resolve_load_balance_weight(weights: dict[str, float], server_address: str) -> float | None:
    """Return configured weight for server_address, or None if unset.

    Tries exact server_address key first, then host_key_for_load_balance.
    WeightedRoundRobinStrategy treats None as weight 1.0.
    """
    if server_address in weights:
        return float(weights[server_address])
    key = host_key_for_load_balance(server_address)
    if key in weights:
        return float(weights[key])
    return None


def _register(name: str, cls: type[LoadBalanceStrategy]) -> None:
    if name in _LOAD_BALANCE_STRATEGY_CLASSES:
        raise ValueError(f"Load balance strategy {name!r} is already registered")
    _LOAD_BALANCE_STRATEGY_CLASSES[name] = cls


def register(name: str):
    """Decorator: @register("my_strategy") on a :class:`LoadBalanceStrategy` subclass."""

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
    """Strategy used when neither request nor group sticky applies (see ``load_balance`` Ray actors).

    Subclasses must implement pick_server and accept
    ``__init__(self, server_actor_ids: list[str], **kwargs)``.
    """

    @abstractmethod
    def pick_server(self, server_ids: list[str]) -> str:
        """Choose server_ids member."""


@register("least_requests")
class LeastRequestsStrategy(LoadBalanceStrategy):
    def __init__(self, server_actor_ids: list[str], **kwargs: Any):
        self._inflight: dict[str, int] = {sid: 0 for sid in server_actor_ids}

    def pick_server(self, server_ids: list[str]) -> str:
        # O(n): one pass for min inflight; ties broken by lexicographically smallest sid (same as sorted(server_ids)).
        if not server_ids:
            raise ValueError("server_ids must be non-empty")
        return min(server_ids, key=lambda sid: (self._inflight[sid], sid))


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
        self._inflight: dict[str, int] = {sid: 0 for sid in self._server_actor_ids}
        self._metrics_executor: ThreadPoolExecutor | None = None

        if self._metric_name:
            pool_workers = max(1, min(len(self._server_actor_ids), _MAX_KV_METRICS_SCRAPE_WORKERS))
            self._metrics_executor = ThreadPoolExecutor(
                max_workers=pool_workers,
                thread_name_prefix="verl-lb-kv",
            )
            self._metrics_thread = threading.Thread(
                target=self._metrics_background_loop,
                name="verl-lb-least-kv-cache",
                daemon=True,
            )
            self._metrics_thread.start()

    def _metrics_background_loop(self) -> None:
        try:
            while not self._metrics_stop.is_set():
                try:
                    self._refresh_metrics_blocking()
                except Exception:
                    logger.exception("Unexpected error in KV metrics refresh loop")
                if self._metrics_stop.wait(timeout=self._refresh_interval_s):
                    break
        finally:
            ex = self._metrics_executor
            if ex is not None:
                ex.shutdown(wait=True)
                self._metrics_executor = None

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

    def _refresh_one_replica(self, sid: str, metric_name: str) -> None:
        """Fetch /metrics for one replica; safe to run concurrently (state updates use ``_lock``)."""
        url = build_metrics_url(sid, self._metrics_path)
        try:
            text = fetch_prometheus_text(url, self._fetch_timeout_s)
            val = parse_prometheus_metric_value(text, metric_name)
            if self._mark_metrics_success(sid, val):
                logger.info("KV cache metrics recovered for replica %s", sid)
        except Exception as e:
            if self._mark_metrics_failed(sid):
                logger.warning("Failed to refresh KV metrics for %s from %s: %s", sid, url, e)

    def _refresh_metrics_blocking(self) -> None:
        metric_name = self._metric_name
        if not metric_name:
            return

        ids = self._server_actor_ids
        if not ids:
            return

        pool = self._metrics_executor
        if pool is None:
            return

        def worker(sid: str) -> None:
            self._refresh_one_replica(sid, metric_name)

        list(pool.map(worker, ids))

    def pick_server(self, server_ids: list[str]) -> str:
        if self._metric_name:
            usage = self._kv_snapshot(server_ids)
        else:
            usage = {sid: None for sid in server_ids}

        has_none = any(usage.get(sid) is None for sid in server_ids)
        if has_none:  # fallback to least in-flight
            return min(server_ids, key=lambda sid: (self._inflight[sid], sid))

        return min(server_ids, key=lambda sid: (usage[sid], sid))


@register("weighted_rr")
class WeightedRoundRobinStrategy(LoadBalanceStrategy):
    """Smooth weighted round-robin (largest-current-weight).

    Optional weights maps host id (IPv4/IPv6 without port, or full server_actor_id)
    to a positive weight. Missing keys default to 1.0. weights is None means all 1.0.
    """

    def __init__(
        self,
        server_actor_ids: list[str],
        weights: dict[str, float] | None = None,
        **kwargs: Any,
    ):
        self._ids = list(server_actor_ids)
        n = len(self._ids)
        if n == 0:
            raise ValueError("server_actor_ids must be non-empty")
        if weights is None:
            w = [1.0] * n
        else:
            # if weight is not found, use 1.0 as default
            w = [
                resolve_load_balance_weight(weights, sid)
                if resolve_load_balance_weight(weights, sid) is not None
                else 1.0
                for sid in self._ids
            ]
        self._weights = w
        self._current = [0.0] * n
        self._inflight: dict[str, int] = {sid: 0 for sid in self._ids}
        self._pos = {sid: i for i, sid in enumerate(self._ids)}

    def pick_server(self, server_ids: list[str]) -> str:
        # Smooth WRR state lives in _current
        indices = [self._pos[s] for s in server_ids]
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
        self._inflight: dict[str, int] = {sid: 0 for sid in server_actor_ids}

    def pick_server(self, server_ids: list[str]) -> str:
        return self._rng.choice(server_ids)
