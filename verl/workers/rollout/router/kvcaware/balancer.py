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

"""KVCAwareBalancer — orchestration shell for the KV-cache-aware router.

A pure framework shell: it wires Config / Strategy / collectors, manages their
lifecycle, and delegates each request to ``route()``. It contains no routing
algorithm. Registered in ``LoadBalancerRegistry`` under "kvcaware" and
instantiated by ``get_router_handle``; a plain class, directly constructible
and unit-testable, satisfying the ``RequestLoadBalancer`` Protocol.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import ray

from ..base import LoadBalancerRegistry
from .collectors import CollectorManager
from .config import KVCAwareConfig
from .logging import get_router_logger
from .store import DataStore
from .strategies import (
    ReplicaInfo,
    StrategyRegistry,
    route,
)

logger = get_router_logger("balancer")


@LoadBalancerRegistry.register("kvcaware")
class KVCAwareBalancer:
    """Pure-framework router shell. See module docstring."""

    # Emit a route() latency aggregate every N route() calls to bound log volume.
    _ROUTE_LOG_EVERY = 64

    def __init__(self, servers: dict[str, Any], config: Optional[dict] = None) -> None:
        if not servers:
            raise ValueError("servers must be non-empty")
        self._config = KVCAwareConfig.from_config(config)
        logger.info(f"KVCAwareBalancer, config={self._config}")
        self._strategies: list[tuple[Any, float]] = [
            (StrategyRegistry.get(type(cfg)).from_config(cfg), cfg.weight) for cfg in self._config.strategies
        ]
        self._strategy_summary = self._build_strategy_summary(self._config.strategies)
        self._servers: dict[str, Any] = dict(servers)
        max_num_seqs = self._resolve_max_num_seqs()
        max_num_batched_tokens = self._resolve_max_num_batched_tokens()
        for strategy, _ in self._strategies:
            if hasattr(strategy, "set_capacity"):
                strategy.set_capacity(max_num_seqs, max_num_batched_tokens)
        logger.info(f"KVCAwareBalancer: max_num_seqs={max_num_seqs}, max_num_batched_tokens={max_num_batched_tokens}")
        self._route_calls = 0
        # route() latency profiling — cumulative stats flushed every _ROUTE_LOG_EVERY calls.
        # The flush trigger derives from _route_calls % _ROUTE_LOG_EVERY (a separate
        # since-flush counter would just track it in lockstep).
        self._route_time_total_s = 0.0
        self._route_time_max_ms = 0.0
        self._callbacks: dict[str, list[Callable]] = {
            "on_acquire": [],
            "on_release": [],
            "on_servers_removed": [],
        }
        self._store = DataStore()
        self._init_manager()

    @staticmethod
    def _build_strategy_summary(strategies: list[Any]) -> str:
        """One-line summary of the first (primary) strategy's key params.

        Only the KVCacheAware tuning knobs that affect routing decisions are
        surfaced: alpha / load_threshold / memory_overload_filter / slow_cut.
        """
        cfg = strategies[0]
        name = type(cfg).__name__
        bits = []
        for k in ("alpha", "load_threshold", "memory_overload_filter"):
            if hasattr(cfg, k):
                bits.append(f"{k}={getattr(cfg, k)}")
        if hasattr(cfg, "slow_cut"):
            bits.append(f"slow_cut={cfg.slow_cut.value}")
        return f"{name}({', '.join(bits)})"

    def _resolve_rollout_config_int(self, attr: str, default: int) -> int:
        """Read a positive int ``attr`` from the first server's rollout config.

        Fail-closed to ``default`` when no server exposes ``get_rollout_config``,
        the RPC fails, or the value is missing / non-positive. Shared by the
        ``max_num_seqs`` (per-step request cap) and ``max_num_batched_tokens``
        (per-step token budget) resolvers.
        """
        for handle in self._servers.values():
            if not hasattr(handle, "get_rollout_config"):
                continue
            try:
                rollout_cfg = ray.get(handle.get_rollout_config.remote())
            except Exception as e:  # noqa: BLE001
                logger.warning(f"get_rollout_config failed ({e}); using default {attr}={default}")
                return default
            value = getattr(rollout_cfg, attr, default)
            if value is None or value <= 0:
                logger.warning(f"server returned non-positive {attr}={value}; using default={default}")
                return default
            return value
        return default

    def _resolve_max_num_seqs(self, default: int = 256) -> int:
        """Per-step sequence cap — denominator for the in-flight request load."""
        return self._resolve_rollout_config_int("max_num_seqs", default)

    def _resolve_max_num_batched_tokens(self, default: int = 2048) -> int:
        """Per-step token budget — denominator for the in-flight token load."""
        return self._resolve_rollout_config_int("max_num_batched_tokens", default)

    def _init_manager(self) -> None:
        """Resolve per-server endpoints from Ray actor handles, then start collectors.

        Non-actor handles (plain strings in unit tests) have no
        ``get_server_address``; discovery is skipped and collectors fall back
        to configured/default endpoints.
        """
        collection_names = sorted({name for cfg in self._config.strategies for name in cfg.collector_names})
        server_addresses: dict[str, str] = {}
        kv_event_endpoints: dict[str, list[str]] = {}
        addr_futures = []
        ep_futures = []
        active_replicas = []
        for replica_id, handle in self._servers.items():
            if not hasattr(handle, "get_server_address"):
                logger.warning(
                    f"server '{replica_id}' handle has no get_server_address remote "
                    f"(type={type(handle).__name__}); skipping dynamic endpoint discovery",
                )
                continue
            active_replicas.append(replica_id)
            addr_futures.append(handle.get_server_address.remote())
            ep_futures.append(handle.get_kv_events_endpoints.remote())

        if active_replicas:
            ips_ports = ray.get(addr_futures)
            endpoints_list = ray.get(ep_futures)
            for replica_id, (ip, port), endpoints in zip(active_replicas, ips_ports, endpoints_list, strict=False):
                server_addresses[replica_id] = f"{ip}:{port}"
                if endpoints is None:
                    continue
                # Pad verl's [sub, replay] to the [sub, replay, publisher, topic] ZMQTransport expects.
                if len(endpoints) == 2:
                    endpoints = [*endpoints, "zmq", "kv-events"]
                kv_event_endpoints[replica_id] = endpoints
        self._manager = CollectorManager(
            self._config.collector,
            collection_names,
            server_addresses=server_addresses,
            kv_event_endpoints=kv_event_endpoints,
            balancer_handler=self,
        )
        self._manager.start()

    # ── Callback registry (opt-in hook points for statistic collectors) ──

    def register_call_back(self, event: str, fn: Callable) -> None:
        """Append ``fn`` to the listeners for ``event``.

        Opt-in hook points: ``on_acquire`` / ``on_release`` / ``on_servers_removed``.
        """
        self._callbacks.setdefault(event, []).append(fn)

    def un_register_call_back(self, event: str, fn: Callable) -> None:
        """Remove ``fn`` from ``event``'s callback list (idempotent)."""
        lst = self._callbacks.get(event, [])
        if fn in lst:
            lst.remove(fn)

    def _fire(self, event: str, *args: Any) -> None:
        """Invoke every registered callback for ``event``; errors are swallowed."""
        for fn in self._callbacks.get(event, []):
            try:
                fn(*args)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"callback {event} failed: {type(exc).__name__}: {exc}")

    def get_all_servers(self) -> list[str]:
        """List all active server ids."""
        return list(self._servers.keys())

    def get_status(self) -> dict:
        """Construction + routing snapshot for debugging."""
        return {
            "servers": list(self._servers.keys()),
            "manager": type(self._manager).__name__,
            "strategies": [{"type": type(s).__name__, "weight": w} for s, w in self._strategies],
            "route_calls": self._route_calls,
            "sticky_size": self._store.sticky_status()["size"],
        }

    def release_server(self, server_id: str, prompt_len: int = 0) -> None:
        """Release a server after a request completes; fires ``on_release``.

        ``prompt_len`` (``len(prompt_ids)`` of the completing request) mirrors the
        value passed at ``acquire_server`` time, so the in-flight token gauge is
        decremented by exactly what acquire added. Defaults to 0 for callers that
        do not track it (the token gauge simply stays unchanged on release).
        """
        self._fire("on_release", server_id, prompt_len)

    def acquire_server(self, request_id: str, prompt_ids: list[int] | None = None) -> tuple[str, Any]:
        """Delegate to ``route()`` for a best-first ranking, return ``(top, handle)``.

        Raises ``RuntimeError`` if no replica is available. ``request_id`` is
        forwarded so strategies can short-circuit to a sticky-bound replica;
        ``on_acquire`` then refreshes the binding.
        """
        replicas = [ReplicaInfo(replica_id=sid) for sid in self._servers]
        self._route_calls += 1
        t0 = time.perf_counter()
        ranking = route(
            self._strategies,
            prompt_ids,
            self._store,
            replicas,
            request_id,
        )
        dt_ms = (time.perf_counter() - t0) * 1000
        self._route_time_total_s += dt_ms / 1000.0
        self._route_time_max_ms = max(self._route_time_max_ms, dt_ms)
        if not ranking:
            raise RuntimeError("no available replica to route to")
        server_id = ranking[0]
        self._fire("on_acquire", request_id, server_id, prompt_ids)
        logger.info(
            f"request={request_id} routed to server={server_id} (ranking={ranking}, pool={list(self._servers)}, "
            f"route={dt_ms:.2f}ms, strategy=[{self._strategy_summary}])"
        )
        if self._route_calls % self._ROUTE_LOG_EVERY == 0:
            logger.info(
                f"route-stats: calls={self._ROUTE_LOG_EVERY} total={self._route_time_total_s:.3f}s "
                f"mean={self._route_time_total_s * 1000 / max(self._route_calls, 1):.2f}ms "
                f"max={self._route_time_max_ms:.2f}ms (flushed every {self._ROUTE_LOG_EVERY} calls) "
                f"strategy=[{self._strategy_summary}]"
            )
        return server_id, self._servers[server_id]

    def add_servers(self, servers: dict[str, Any]) -> None:
        """Bulk-add servers to the pool (manager is keyed by init-time addresses, untouched here)."""
        for sid, handle in servers.items():
            self._servers[sid] = handle

    def remove_servers(self, server_ids: list[str]) -> None:
        """Bulk-remove servers; fires ``on_servers_removed`` to invalidate sticky bindings."""
        for sid in server_ids:
            self._servers.pop(sid, None)
        if server_ids:
            self._fire("on_servers_removed", server_ids)
