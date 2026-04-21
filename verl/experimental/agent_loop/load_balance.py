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
"""
Ray global load balancers:
- Concrete actors use @ray.remote.
- load_balancer_actor_class maps rollout.load_balance_sticky_mode to the actor class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import ray
from cachetools import LRUCache

from verl.experimental.agent_loop.load_balance_strategy import create_load_balance_strategy
from verl.workers.config import RolloutConfig

DEFAULT_ROUTING_CACHE_SIZE = 10000

_LOAD_BALANCER_ACTOR_CLASSES: dict[str, Any] = {}


def load_balancer_actor_class(rollout_config: RolloutConfig) -> Any:
    """Resolve the Ray actor class from rollout_config.load_balance_sticky_mode."""
    mode = rollout_config.load_balance_sticky_mode
    cls = _LOAD_BALANCER_ACTOR_CLASSES.get(mode)
    if cls is None:
        raise ValueError(
            f"Unknown load_balance_sticky_mode {mode!r}. Registered: {sorted(_LOAD_BALANCER_ACTOR_CLASSES)}"
        )
    return cls


def register_load_balancer_class(name: str, remote_actor_cls: Any) -> None:
    """Register a Ray actor class without decorator (remote_actor_cls must be ray.remote-wrapped)."""
    if name in _LOAD_BALANCER_ACTOR_CLASSES:
        raise ValueError(f"Load balancer actor {name!r} is already registered")
    _LOAD_BALANCER_ACTOR_CLASSES[name] = remote_actor_cls


class GlobalRequestLoadBalancer(ABC):
    """Abstract base (not a Ray actor). Subclasses are @ray.remote and listed in _LOAD_BALANCER_ACTOR_CLASSES."""

    @staticmethod
    def init_bundle_from_rollout(rollout_config: RolloutConfig) -> dict[str, Any]:
        """Map RolloutConfig load-balance fields to constructor fields (for tests and internal use)."""
        kv = rollout_config.kv_cache_metrics
        strategy_init_kwargs: dict[str, Any] = {}
        if rollout_config.load_balance_weights is not None:
            strategy_init_kwargs["weights"] = dict(rollout_config.load_balance_weights)
        if rollout_config.load_balance_random_seed is not None:
            strategy_init_kwargs["random_seed"] = rollout_config.load_balance_random_seed
        if rollout_config.load_balance_strategy == "least_kv_cache":
            strategy_init_kwargs["metric_name"] = kv.metric_name
            strategy_init_kwargs["metrics_path"] = kv.metrics_path
            strategy_init_kwargs["refresh_interval_s"] = kv.refresh_interval_s
            strategy_init_kwargs["fetch_timeout_s"] = kv.fetch_timeout_s
        return {
            "max_cache_size": DEFAULT_ROUTING_CACHE_SIZE,
            "load_balance_strategy": rollout_config.load_balance_strategy,
            "strategy_init_kwargs": strategy_init_kwargs,
        }

    def _init_from_rollout(self, server_actor_ids: list[str], rollout_config: RolloutConfig) -> None:
        if not server_actor_ids:
            raise ValueError("server_actor_ids must be non-empty")
        bundle = GlobalRequestLoadBalancer.init_bundle_from_rollout(rollout_config)
        load_balance_strategy = bundle["load_balance_strategy"]
        self._routing_cache_size = bundle["max_cache_size"]
        self._server_actor_ids = list(server_actor_ids)
        sk = bundle["strategy_init_kwargs"] or {}
        self.strategy = create_load_balance_strategy(
            load_balance_strategy,
            server_actor_ids=self._server_actor_ids,
            **sk,
        )

    @abstractmethod
    def acquire_server(self, request_id: str, request_group_id: str | None = None) -> str:
        """Acquire a server. request_group_id ties GRPO / rollout-n repeats to one replica (see trajectory)."""

    def release_server(self, server_id: str) -> None:
        """Release a server after a request completes, decrementing strategy in-flight counts."""
        inflight = getattr(self.strategy, "_inflight", None)
        if inflight is None:
            return  # strategy does not expose _inflight; cannot release_server
        if server_id not in inflight:
            raise ValueError(f"Invalid server_id for release: {server_id}")
        if inflight[server_id] <= 0:
            raise ValueError(f"Release called with no inflight requests on server {server_id}")
        inflight[server_id] -= 1

    def update_inflight(self, server_id: str, delta: int) -> None:
        inflight = getattr(self.strategy, "_inflight", None)
        if inflight is not None and server_id in inflight:
            inflight[server_id] += delta

    def close(self) -> None:
        """Stop strategy background work (e.g. least_kv_cache metrics scraper). Safe to call multiple times."""
        self.strategy.close()


@ray.remote
class RequestStickyLoadBalancer(GlobalRequestLoadBalancer):
    """Sticky by request_id only; request_group_id is ignored."""

    def __init__(self, server_actor_ids: list[str], rollout_config: RolloutConfig):
        self._init_from_rollout(server_actor_ids, rollout_config)
        self._request_id_to_server: LRUCache = LRUCache(maxsize=self._routing_cache_size)

    def acquire_server(self, request_id: str, request_group_id: str | None = None) -> str:
        if request_id in self._request_id_to_server:
            server_id = self._request_id_to_server[request_id]
            self.update_inflight(server_id, 1)
            return server_id
        server_id = self.strategy.pick_server(list(self._server_actor_ids))
        self._request_id_to_server[request_id] = server_id
        self.update_inflight(server_id, 1)
        return server_id


@ray.remote
class GroupStickyLoadBalancer(GlobalRequestLoadBalancer):
    """
    GRPO / rollout-n group sticky only: request_id is not used for routing.

    An LRU maps request_group_id (e.g. f"{global_step}:{sample_index}") to a replica so
    all repeats for that sample share one server. Unknown groups go through strategy.pick_server,
    then the mapping is recorded.
    """

    def __init__(self, server_actor_ids: list[str], rollout_config: RolloutConfig):
        self._init_from_rollout(server_actor_ids, rollout_config)
        self._request_group_to_server: LRUCache[str, str] = LRUCache(maxsize=self._routing_cache_size)

    def acquire_server(self, request_id: str, request_group_id: str | None = None) -> str:
        _ = request_id  # API compatibility; group sticky does not route by request_id.
        if not request_group_id:
            raise ValueError("request_group_id is required for group sticky")
        if request_group_id in self._request_group_to_server:
            server_id = self._request_group_to_server[request_group_id]
            self.update_inflight(server_id, 1)
            return server_id
        server_id = self.strategy.pick_server(list(self._server_actor_ids))
        self._request_group_to_server[request_group_id] = server_id
        self.update_inflight(server_id, 1)
        return server_id


_LOAD_BALANCER_ACTOR_CLASSES["request"] = RequestStickyLoadBalancer
_LOAD_BALANCER_ACTOR_CLASSES["group"] = GroupStickyLoadBalancer
