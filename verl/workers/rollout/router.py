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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import logging
import os
from typing import Any, Callable, Protocol

import ray
from cachetools import LRUCache
from omegaconf import OmegaConf

from verl.workers.config import RolloutConfig
from verl.workers.config.rollout import RouterConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000


class RequestLoadBalancer(Protocol):
    """Protocol for rollout inference load balancers.

    All strategies must satisfy this interface via structural subtyping.
    """

    def acquire_server(self, request_id: str, prompt_ids: list[int] | None = None) -> tuple[str, Any]:
        """Acquire a server for the given request.

        Args:
            request_id: Request identifier for sticky session routing.
            prompt_ids: Prompt token ids for content-aware routing.

        Returns:
            A ``(server_id, actor_handle)`` tuple.

        Raises:
            RuntimeError: If no servers are available in the pool.
        """
        ...

    def release_server(self, server_id: str) -> None:
        """Release a server after a request completes.

        Args:
            server_id: Identifier of the server to release.
        """
        ...

    def add_servers(self, servers: dict[str, Any]) -> None:
        """Bulk-add servers to the load balancer pool.

        Args:
            servers: Mapping from ``server_id`` to ``actor_handle``.
        """
        ...

    def remove_servers(self, server_ids: list[str]) -> None:
        """Bulk-remove servers from the load balancer pool.

        Args:
            server_ids: List of server identifiers to remove.
        """
        ...

    def get_all_servers(self) -> list[str]:
        """List all active server IDs.

        Returns:
            List of server identifier strings.
        """
        ...

    def get_status(self) -> dict:
        """Return current load balancer state for debugging.

        Returns:
            A dictionary with ``servers``, ``total_inflight``,
            and ``active_servers`` keys.
        """
        ...


@ray.remote
class GlobalRequestLoadBalancer:
    """Global sticky-session + in-flight load balancer shared by all AgentLoopWorkers.

    When a sticky session points to a removed server, the cache entry is
    automatically invalidated and a new server is selected.

    Key features:
    - **Atomic acquire**: ``acquire_server()`` returns ``(server_id, handle)``
    - **Sticky Session**: Uses LRUCache to map request_id → server_id, ensuring
      multi-turn conversations route to the same server.
    - **Least-loaded Selection**: When no sticky session exists, selects the
      server with the fewest in-flight requests.
    - **Deterministic Routing**: When ``full_determinism=True``, tie-breaking
      among equally-loaded servers uses ``hash(request_id)`` so the same
      request always routes to the same server across runs.
    - **Dynamic Server Management**: Supports add/remove servers at runtime
      for hybrid scaling.
    """

    def __init__(
        self,
        servers: dict[str, ray.actor.ActorHandle],
        max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE,
        full_determinism: bool = False,
    ):
        if not servers:
            raise ValueError("servers must be non-empty")

        self._servers: dict[str, ray.actor.ActorHandle] = dict(servers)
        self._inflight_requests: dict[str, int] = {sid: 0 for sid in servers}
        self._request_id_to_server: LRUCache = LRUCache(maxsize=max_cache_size)
        self._full_determinism = full_determinism

    def acquire_server(self, request_id: str, prompt_ids: list[int] | None = None) -> tuple[str, ray.actor.ActorHandle]:
        """Acquire a server for the given request (sticky + least-loaded).

        Returns:
            A tuple of ``(server_id, actor_handle)`` in a single atomic call.
        """
        # Try sticky session first
        if request_id in self._request_id_to_server:
            server_id = self._request_id_to_server[request_id]
            # Check if server is still in the active pool
            if server_id in self._inflight_requests:
                self._inflight_requests[server_id] += 1
                return server_id, self._servers[server_id]
            # Server was removed, clear stale cache entry and re-select
            del self._request_id_to_server[request_id]

        # Select new server (least-loaded among available)
        if not self._inflight_requests:
            raise RuntimeError("No available servers in load balancer")

        min_count = min(self._inflight_requests.values())
        candidates = [sid for sid, count in self._inflight_requests.items() if count == min_count]
        if len(candidates) == 1:
            server_id = candidates[0]
        elif self._full_determinism:
            # Deterministic tie-breaking: same request_id → same server across runs
            server_id = candidates[hash(request_id) % len(candidates)]
        else:
            server_id = candidates[0]
        self._request_id_to_server[request_id] = server_id
        self._inflight_requests[server_id] += 1
        return server_id, self._servers[server_id]

    def release_server(self, server_id: str) -> None:
        """Release a server after a request completes."""
        if server_id not in self._inflight_requests:
            return
        if self._inflight_requests[server_id] > 0:
            self._inflight_requests[server_id] -= 1

    def add_servers(self, servers: dict[str, ray.actor.ActorHandle]) -> None:
        """Atomically add multiple servers to the load balancer pool.

        This is more efficient than calling :meth:`add_server` in a loop
        because it performs a single bulk update on the internal state.

        Args:
            servers: Dict mapping server_id → actor_handle for all servers
                to register.
        """
        for sid, handle in servers.items():
            self._inflight_requests[sid] = 0
            self._servers[sid] = handle
        logger.info(f"[GlobalLoadBalancer] added {len(servers)} servers")

    def remove_servers(self, server_ids: list[str]) -> None:
        """Atomically remove multiple servers from the load balancer pool.

        More efficient than calling :meth:`remove_server` in a loop.

        Args:
            server_ids: List of server identifiers to remove.
        """
        for sid in server_ids:
            self._inflight_requests.pop(sid, None)
            self._servers.pop(sid, None)
        logger.info(f"[GlobalLoadBalancer] removed {len(server_ids)} servers")

    def get_inflight_count(self, server_id: str) -> int:
        """Get number of in-flight requests for a server."""
        return self._inflight_requests.get(server_id, 0)

    def get_all_servers(self) -> list[str]:
        """Get list of all active server IDs."""
        return list(self._inflight_requests.keys())

    def get_status(self) -> dict:
        """Return current load balancer state for debugging."""
        return {
            "servers": dict(self._inflight_requests),
            "total_inflight": sum(self._inflight_requests.values()),
            "active_servers": len(self._inflight_requests),
            "registered_handles": list(self._servers.keys()),
        }


class LoadBalancerRegistry:
    """Registry for load-balancer strategy factory functions.

    Strategies are registered by name and looked up via :meth:`get`.
    The ``plugin_extension`` strategy dynamically imports
    a user-defined class from ``router.router_class``.
    """

    _registry: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., Any]) -> None:
        """Register a load-balancer strategy factory function."""
        if name in cls._registry:
            raise ValueError(f"Load balancer '{name}' is already registered. Existing factory: {cls._registry[name]}")

        cls._registry[name] = factory
        logger.info("Registered load balancer strategy: %s", name)

    @classmethod
    def get(cls, name: str) -> Callable[..., Any]:
        """Look up a registered factory by name."""
        if name not in cls._registry:
            raise ValueError(f"Unknown load balancer strategy: '{name}'. Available strategies: {cls.list_strategies()}")
        return cls._registry[name]

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List all registered strategy names."""
        return sorted(cls._registry.keys())


def _create_global_sticky_inflight(
    servers: dict[str, Any],
    rollout_config: RolloutConfig,
):
    """Factory for the default sticky-session + least-inflight strategy."""

    return GlobalRequestLoadBalancer.remote(
        servers=servers,
        max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
        full_determinism=getattr(rollout_config, "full_determinism", False),
    )


def _load_router_yaml(router_config: RouterConfig) -> dict:
    """Load a router YAML configuration file."""
    config_path = router_config.get("router_config_path", None)
    if not config_path:
        raise ValueError("The 'plugin_extension' strategy requires 'router_config_path' pointing to a YAML file.")

    try:
        cfg = OmegaConf.load(config_path)
        return OmegaConf.to_container(cfg, resolve=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Router config file not found: {config_path}") from e


def _resolve_router_class(yaml_config: dict) -> type:
    """Validate and import a router class from YAML config.
    Extracts ``router_class`` FQN, validates format, imports the module,
    and returns the class object.
    """
    router_class = yaml_config.get("router_class", None)
    if not router_class:
        raise ValueError(
            "External router YAML must contain 'router_class'. "
            "Example: router_class: uni_agent.llm_router.KvcAwareRouter"
        )

    try:
        module_path, class_name = router_class.rsplit(".", 1)
    except ValueError as e:
        raise ValueError(
            f"Invalid fully-qualified class name: '{router_class}'. Expected format: 'module_path.ClassName'"
        ) from e

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}' for '{class_name}'. Original error: {e}") from e

    cls = getattr(module, class_name)  # AttributeError propagates if missing

    if not callable(cls):
        raise TypeError(
            f"'{router_class}' is not callable (type: {type(cls).__name__}). "
            f"Expected a class with a .remote() constructor."
        )

    return cls


def _create_plugin_extension(
    servers: dict[str, Any],
    rollout_config: RolloutConfig,
):
    """Factory for user-defined load balancer via external YAML configuration.

    Loads the class specified by ``router_config_path``, imports it
    dynamically, and instantiates it as a Ray actor with ``servers``
    and YAML kwargs.
    """

    yaml_config = _load_router_yaml(rollout_config.router)
    cls = _resolve_router_class(yaml_config)

    ray_cls = cls if isinstance(cls, ray.actor.ActorClass) else ray.remote(cls)

    logger.info(
        "Creating plugin load balancer: class=%s, servers=%d, kwargs=%s",
        yaml_config["router_class"],
        len(servers),
        yaml_config,
    )
    return ray_cls.remote(servers, yaml_config)


LoadBalancerRegistry.register("global_sticky_inflight", _create_global_sticky_inflight)
LoadBalancerRegistry.register("plugin_extension", _create_plugin_extension)


def get_router_handle(servers: dict[str, Any], rollout_config: RolloutConfig) -> Any:
    """Create a load balancer instance from router configuration."""
    router_config = rollout_config.get("router", None)
    if router_config is None:
        strategy = "global_sticky_inflight"
    else:
        strategy = router_config.get("router_strategy", "global_sticky_inflight")

    factory = LoadBalancerRegistry.get(strategy)
    return factory(servers=servers, rollout_config=rollout_config)
