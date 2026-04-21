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

# Cap parallel metrics scrapes per refresh.
_MAX_KV_METRICS_SCRAPE_WORKERS = 32

_LOAD_BALANCE_STRATEGY_CLASSES: dict[str, type[LoadBalanceStrategyBase]] = {}


def host_key_for_load_balance(server_address: str) -> str:
    """Extract host part from server_address"""
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


def resolve_load_balance_weight(weights: dict[str, float] | None, server_address: str) -> float | None:
    """Return configured weight for server_address, or None if not set."""
    if not weights:
        return None
    if server_address in weights:
        return float(weights[server_address])
    key = host_key_for_load_balance(server_address)
    if key in weights:
        return float(weights[key])
    return None


def _register(name: str, cls: type[LoadBalanceStrategyBase]) -> None:
    if name in _LOAD_BALANCE_STRATEGY_CLASSES:
        raise ValueError(f"Load balance strategy {name!r} is already registered")
    _LOAD_BALANCE_STRATEGY_CLASSES[name] = cls


def register(name: str):
    """Decorator: @register("strategy_name") on a LoadBalanceStrategyBase subclass."""

    def decorator(cls: type[LoadBalanceStrategyBase]) -> type[LoadBalanceStrategyBase]:
        _register(name, cls)
        return cls

    return decorator


def create_load_balance_strategy(name: str, server_actor_ids: list[str], **kwargs: Any) -> LoadBalanceStrategyBase:
    """Instantiate a registered strategy by name."""
    cls = _LOAD_BALANCE_STRATEGY_CLASSES.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown load_balance_strategy {name!r}. Registered: {sorted(_LOAD_BALANCE_STRATEGY_CLASSES)}"
        )
    return cls(server_actor_ids, **kwargs)


class LoadBalanceStrategyBase(ABC):
    @abstractmethod
    def pick_server(self, server_ids: list[str]) -> str:
        """Choose server_ids member."""

    def close(self) -> None:
        """Release background resources. Strategies without threads override no-op."""
        return None


class InflightAwareStrategy(LoadBalanceStrategyBase):
    def __init__(self, server_actor_ids: list[str], **kwargs: Any):
        if not server_actor_ids:
            raise ValueError("server_actor_ids must be non-empty")
        self._inflight: dict[str, int] = {sid: 0 for sid in server_actor_ids}

    def notify_request_completed(self, server_id: str) -> None:
        """Decrement in-flight count for server_id."""
        if server_id not in self._inflight:
            raise ValueError(f"Invalid server_id for notify_request_completed: {server_id}")
        if self._inflight[server_id] <= 0:
            raise ValueError(f"notify_request_completed: no inflight on server {server_id}")
        self._inflight[server_id] -= 1

    def notify_request_started(self, server_id: str) -> None:
        """Increment in-flight count for server_id."""
        if server_id not in self._inflight:
            raise ValueError(f"Invalid server_id for notify_request_started: {server_id}")
        if self._inflight[server_id] < 0:
            raise ValueError(f"notify_request_started: negative inflight on server {server_id}")
        self._inflight[server_id] += 1


@register("random")
class RandomStrategy(LoadBalanceStrategyBase):
    """Uniform random choice among candidates."""

    def __init__(self, server_actor_ids: list[str], random_seed: int | None = None, **kwargs: Any):
        self._rng = random.Random(random_seed)

    def pick_server(self, server_ids: list[str]) -> str:
        if not server_ids:
            raise ValueError("server_ids must be non-empty")
        return self._rng.choice(server_ids)


@register("weighted_rr")
class WeightedRoundRobinStrategy(LoadBalanceStrategyBase):
    """Smooth weighted round-robin (largest-current-weight)."""

    def __init__(
        self,
        server_actor_ids: list[str],
        weights: dict[str, float] | None = None,
        **kwargs: Any,
    ):
        if not weights:
            raise ValueError("weights must be non-empty")
        self._server_actor_ids = list(server_actor_ids)
        self._weights = [resolve_load_balance_weight(weights, sid) for sid in self._server_actor_ids]
        self._current = [0.0] * len(self._server_actor_ids)
        self._pos = {sid: i for i, sid in enumerate(self._server_actor_ids)}

    def pick_server(self, server_ids: list[str]) -> str:
        if not server_ids:
            raise ValueError("server_ids must be non-empty")
        indices = [self._pos[s] for s in server_ids]
        total = sum(self._weights[i] for i in indices)
        for i in indices:
            self._current[i] += self._weights[i]
        best_i = max(indices, key=lambda i: (self._current[i], -i))
        self._current[best_i] -= total
        return self._server_actor_ids[best_i]


@register("least_requests")
class LeastRequestsStrategy(InflightAwareStrategy):
    """Prefer server with lowest in-flight requests."""

    def pick_server(self, server_ids: list[str]) -> str:
        return min(server_ids, key=lambda sid: (self._inflight[sid], sid))


@register("least_kv_cache")
class LeastKVCacheStrategy(InflightAwareStrategy):
    """Prefer lowest KV usage from periodic HTTP /metrics scrape."""

    def __init__(self, server_actor_ids: list[str], **kwargs: Any):
        super().__init__(server_actor_ids, **kwargs)
        self._server_actor_ids = list(server_actor_ids)
        self._metric_name: str | None = kwargs.pop("metric_name", None)
        self._metrics_path: str = kwargs.pop("metrics_path", "/metrics")
        self._refresh_interval_s: float = float(kwargs.pop("refresh_interval_s", 2.0))
        self._fetch_timeout_s: float = float(kwargs.pop("fetch_timeout_s", 2.0))

        self._kv_usage: dict[str, float | None] = {sid: None for sid in self._server_actor_ids}
        self._lock = threading.Lock()
        self._metrics_stop = threading.Event()
        self._metrics_thread: threading.Thread | None = None
        self._metrics_executor: ThreadPoolExecutor | None = None
        self._metrics_close_lock = threading.Lock()
        self._metrics_close_requested = False
        self._metrics_close_join_timeout_s: float = 30.0

        if self._metric_name:
            pool_workers = max(1, min(len(self._server_actor_ids), _MAX_KV_METRICS_SCRAPE_WORKERS))
            self._metrics_executor = ThreadPoolExecutor(max_workers=pool_workers)
            self._metrics_thread = threading.Thread(target=self._metrics_background_loop, daemon=True)
            self._metrics_thread.start()

    def close(self) -> None:
        """Stop the metrics scrape thread and shut down its executor (idempotent)."""
        if not self._metric_name:
            return None
        with self._metrics_close_lock:
            if self._metrics_close_requested:
                return None
            self._metrics_close_requested = True
        self._metrics_stop.set()
        self._metrics_executor.shutdown(wait=False, cancel_futures=True)
        self._metrics_thread.join(timeout=self._metrics_close_join_timeout_s)

    def _metrics_background_loop(self) -> None:
        try:
            while not self._metrics_stop.is_set():
                self._refresh_metrics_blocking()
                self._metrics_stop.wait(timeout=self._refresh_interval_s)
        finally:
            self._metrics_executor.shutdown(wait=False, cancel_futures=True)

    def _mark_metrics(self, sid: str, val: float | None) -> None:
        with self._lock:
            self._kv_usage[sid] = val

    def _refresh_one_replica(self, sid: str, metric_name: str) -> None:
        """Fetch /metrics for one replica; safe to run concurrently (state updates use ``_lock``)."""
        url = build_metrics_url(sid, self._metrics_path)
        try:
            text = fetch_prometheus_text(url, self._fetch_timeout_s)
            val = parse_prometheus_metric_value(text, metric_name)
            self._mark_metrics(sid, val)
        except Exception:
            self._mark_metrics(sid, None)
            logger.exception("Failed to refresh KV metrics for %s from %s", sid, url)

    def _refresh_metrics_blocking(self) -> None:
        metric_name = self._metric_name
        if not metric_name:
            return

        pool = self._metrics_executor
        if pool is None:
            return

        def worker(sid: str) -> None:
            self._refresh_one_replica(sid, metric_name)

        list(pool.map(worker, self._server_actor_ids))

    def pick_server(self, server_ids: list[str]) -> str:
        if not server_ids:
            raise ValueError("server_ids must be non-empty")
        if self._metric_name:
            usage = {sid: self._kv_usage.get(sid) for sid in server_ids}
        else:
            usage = {sid: None for sid in server_ids}

        max_usage = max([v for v in usage.values() if v is not None], default=0.0)

        def key_func(sid: str) -> tuple[float, int, str]:
            if usage[sid] is None:
                return (max_usage, self._inflight[sid], sid)
            return (usage[sid], self._inflight[sid], sid)

        return min(server_ids, key=key_func)
