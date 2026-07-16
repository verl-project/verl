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

"""Base class and registry for rollout load-balancer strategies.

Mirrors the pattern in ``verl/workers/engine/base.py`` — the Protocol (like
``BaseEngine``) and ``LoadBalancerRegistry`` (like ``EngineRegistry``) live
together in one zero-dependency module.
"""

import logging
import os
from typing import Any, Optional, Protocol, runtime_checkable

import ray

from verl.workers.config import RolloutConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class RequestLoadBalancer(Protocol):
    """Protocol for rollout inference load balancers (structural subtyping)."""

    def __init__(self, servers: dict[str, Any], config: Optional[dict] = None) -> None:
        ...

    def acquire_server(self, request_id: str, prompt_ids: list[int] | None = None) -> tuple[str, Any]:
        ...

    def release_server(self, server_id: str) -> None:
        ...

    def add_servers(self, servers: dict[str, Any]) -> None:
        ...

    def remove_servers(self, server_ids: list[str]) -> None:
        ...

    def get_all_servers(self) -> list[str]:
        ...

    def get_status(self) -> dict:
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class LoadBalancerRegistry:
    """Registry for load-balancer strategy classes.

    Strategies are registered by name via the :meth:`register` decorator and
    instantiated through ``get_router_handle``.
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator that registers a balancer class under ``name``.

        Usage::

            @LoadBalancerRegistry.register("global_sticky_inflight")
            @ray.remote
            class GlobalRequestLoadBalancer:
                ...

            @LoadBalancerRegistry.register("kvcaware")
            class KVCAwareBalancer:
                ...
        """
        def decorator(balancer_cls):
            if name in cls._registry:
                raise ValueError(
                    f"Load balancer '{name}' is already registered. "
                    f"Existing: {cls._registry[name]}"
                )
            cls._registry[name] = balancer_cls
            logger.info("Registered load balancer strategy: %s", name)
            return balancer_cls
        return decorator

    @classmethod
    def get_cls(cls, name: str) -> type:
        """Look up a registered balancer class by name."""
        if name not in cls._registry:
            raise ValueError(
                f"Unknown load balancer strategy: '{name}'. "
                f"Available strategies: {cls.list_strategies()}"
            )
        return cls._registry[name]

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List all registered strategy names."""
        return sorted(cls._registry.keys())


def _is_ray_actor_class(cls: type) -> bool:
    """Return True if *cls* is a ``@ray.remote`` actor class."""
    return hasattr(cls, "remote") and hasattr(cls, "__ray_metadata__")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _resolve_router_strategy(rollout_config: RolloutConfig) -> str:
    """Return the ``router_strategy`` field, defaulting to ``global_sticky_inflight``."""
    return rollout_config.get("router_strategy", "global_sticky_inflight")


def get_router_handle(servers: dict[str, Any], rollout_config: RolloutConfig) -> Any:
    """Create a load balancer instance from router configuration."""
    strategy = _resolve_router_strategy(rollout_config)
    cls = LoadBalancerRegistry.get_cls(strategy)
    config = rollout_config.get("router_config", None) or {}
    config["full_determinism"] = getattr(rollout_config, "full_determinism", False)

    if _is_ray_actor_class(cls):
        return cls.remote(servers=servers, config=config)
    return ray.remote(cls).remote(servers=servers, config=config)
