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
    """Return configured weight for server_address, or None if unset.

    Tries exact server_address key first, then host_key_for_load_balance.
    WeightedRoundRobinStrategy treats None as weight 1.0.
    """
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
    """Decorator: @register("my_strategy") on a LoadBalanceStrategyBase subclass."""

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

    @abstractmethod
    def close(self) -> None:
        """Release background resources (threads, HTTP scrapers)."""


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
        self._server_actor_ids = list(server_actor_ids)
        n = len(self._server_actor_ids)
        if n == 0:
            raise ValueError("server_actor_ids must be non-empty")
        self._weights = []
        for sid in self._server_actor_ids:
            v = resolve_load_balance_weight(weights, sid)
            self._weights.append(1.0 if v is None else v)
        self._current = [0.0] * n
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
    def pick_server(self, server_ids: list[str]) -> str:
        return min(server_ids, key=lambda sid: (self._inflight[sid], sid))


@register("least_kv_cache")
class LeastKVCacheStrategy(InflightAwareStrategy):
    """Prefer lowest KV usage from periodic HTTP /metrics scrape;

    When metric_name is set, a daemon thread scrapes each replica on refresh_interval_s and
    pick_server uses that internal state. If the metric is not available, we use p2c algorithm to choose the server.
    """

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
        self._metrics_fail_warned: set[str] = set()
        self._metrics_executor: ThreadPoolExecutor | None = None
        self._metrics_close_lock = threading.Lock()
        self._metrics_close_requested = False
        self._metrics_close_join_timeout_s: float = float(kwargs.pop("metrics_close_join_timeout_s", 30.0))

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

    def close(self) -> None:
        """Stop the metrics scrape thread and shut down its executor (idempotent)."""
        if not self._metric_name:
            return None
        with self._metrics_close_lock:
            if self._metrics_close_requested:
                return None
            self._metrics_close_requested = True
        self._metrics_stop.set()
        ex = self._metrics_executor
        if ex is not None:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                logger.exception("Error shutting down KV metrics ThreadPoolExecutor")
        thr = self._metrics_thread
        if thr is not None and thr.is_alive():
            thr.join(timeout=self._metrics_close_join_timeout_s)
            if thr.is_alive():
                logger.warning(
                    "LeastKVCacheStrategy metrics thread did not exit within %ss",
                    self._metrics_close_join_timeout_s,
                )
        return None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

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
                try:
                    ex.shutdown(wait=True)
                except Exception:
                    logger.exception("Error in KV metrics executor shutdown from background thread")
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
        if not server_ids:
            raise ValueError("server_ids must be non-empty")
        if self._metric_name:
            usage = self._kv_snapshot(server_ids)
        else:
            usage = {sid: None for sid in server_ids}

        max_usage = max([v for v in usage.values() if v is not None], default=0.0)

        def key_func(sid: str) -> tuple[float, int, str]:
            val = usage[sid]
            if val is None:
                return (max_usage, self._inflight[sid], sid)
            return (val, self._inflight[sid], sid)

        return min(server_ids, key=key_func)
