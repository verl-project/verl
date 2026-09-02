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

import logging
import os
import random
import zlib
from typing import Any, Protocol

import ray
from cachetools import LRUCache
from omegaconf import OmegaConf

from verl.utils.import_utils import load_class_from_fqn, resolve_config_path

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000


class RequestLoadBalancer(Protocol):
    """Protocol for rollout inference load balancers.

    All strategies must satisfy this interface via structural subtyping.
    """

    def acquire_server(self, request_id: str, **extra) -> tuple[str, Any]:
        """Acquire a server for the given request.

        Args:
            request_id: Request identifier for sticky session routing.
            **extra: Keyword fields declared via :meth:`require_acquire_fields`
                (e.g. ``prompt_ids`` for content-aware routing). Only the
                declared fields are serialized into this RPC.

        Returns:
            A ``(server_id, actor_handle)`` tuple.

        Raises:
            RuntimeError: If no servers are available in the pool.
        """
        ...

    def require_acquire_fields(self) -> list[str]:
        """``generate()`` kwargs this router consumes at acquire time; only
        these are serialized into the ``acquire_server`` RPC (e.g.
        ``["prompt_ids"]``; ``[]`` if routing on ``request_id`` alone)."""
        ...

    def require_release_fields(self) -> list[str]:
        """Identity fields this router consumes at release time (currently
        only ``"request_id"``); ``[]`` if counting by ``server_id`` alone."""
        ...

    def release_server(self, server_id: str, request_id: str | None = None) -> None:
        """Release a server after a request completes.

        Args:
            server_id: Identifier of the server to release.
            request_id: Request identifier. Content-aware balancers that need
                the prompt length at release time look it up by this id from
                their own acquire-time bookkeeping, so the full token list is
                not re-serialized over RPC.
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

    def clear_sticky_cache(self) -> dict:
        """Clear sticky-session state to force request redistribution.

        Called by trainers on training/rollout phase switches and by
        fully-async rebalancing, so all strategies (plugins included)
        must implement it.

        Returns:
            A diagnostics dict, conventionally with ``cleared_entries``
            and ``server_loads`` keys.
        """
        ...

    def get_total_inflight(self) -> int:
        """Return the total in-flight requests across registered servers.

        Polled by fully-async drain/rebalance loops to wait for the
        pool to quiesce.
        """
        ...


class GlobalRequestLoadBalancer:
    """Global sticky-session + in-flight load balancer shared by all AgentLoopWorkers.

    When a sticky session points to a removed server, the cache entry is
    automatically invalidated and a new server is selected.

    This is a plain Python class (not a Ray actor). It is wrapped with
    ``ray.remote(...)`` at instantiation time so callers can subclass it and
    override :meth:`acquire_server` before registering the subclass as an actor.

    Key features:
    - **Atomic acquire**: ``acquire_server()`` returns ``(server_id, handle)``
    - **Sticky Session**: Uses LRUCache to map request_id → server_id, ensuring
      multi-turn conversations route to the same server.
    - **Least-loaded Selection**: When no sticky session exists, selects the
      server with the fewest in-flight requests.
    - **Deterministic Routing**: When ``full_determinism=True``, routes every
      request by a stable platform-independent hash (``zlib.crc32``) over sorted
      server IDs so the same request always routes to the same replica across
      processes and runs.
    - **Dynamic Server Management**: Supports add/remove servers at runtime
      for hybrid scaling.
    """

    def __init__(
        self,
        servers: dict[str, ray.actor.ActorHandle],
        max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE,
        full_determinism: bool = False,
    ):
        # Allow empty initial servers: in dynamic-resource-scheduling mode all
        # replicas are hybrid and will be registered later via add_servers().

        self._servers: dict[str, ray.actor.ActorHandle] = dict(servers)
        self._inflight_requests: dict[str, int] = {sid: 0 for sid in servers}
        self._request_id_to_server: LRUCache = LRUCache(maxsize=max_cache_size)
        self._full_determinism = full_determinism

    def acquire_server(self, request_id: str) -> tuple[str, ray.actor.ActorHandle]:
        """Acquire a server for the given request (sticky + least-loaded).

        Args:
            request_id: Request identifier for sticky session routing.

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

        if self._full_determinism:
            # Full-hash routing: same request_id always lands on the same replica
            # across runs and across processes. Least-loaded selection depends on async
            # arrival timing, which varies run-to-run, so it is bypassed entirely here.
            # Using sorted server keys and crc32 ensures platform and process independence
            # without relying on PYTHONHASHSEED or dict insertion order.
            sorted_servers = sorted(self._servers.keys())
            server_id = sorted_servers[zlib.crc32(request_id.encode("utf-8")) % len(sorted_servers)]
        else:
            min_count = min(self._inflight_requests.values())
            candidates = [sid for sid, count in self._inflight_requests.items() if count == min_count]
            server_id = random.choice(candidates)
        self._request_id_to_server[request_id] = server_id
        self._inflight_requests[server_id] += 1
        return server_id, self._servers[server_id]

    def release_server(self, server_id: str, request_id: str | None = None) -> None:
        """Release a server after a request completes.

        ``request_id`` is accepted for signature parity with content-aware
        balancers (which look up the prompt length from their own
        acquire-time bookkeeping); this balancer tracks request counts only
        and ignores it.
        """
        if server_id not in self._inflight_requests:
            return
        if self._inflight_requests[server_id] > 0:
            self._inflight_requests[server_id] -= 1

    def require_acquire_fields(self) -> list[str]:
        """Sticky + least-inflight routing keys on ``request_id`` alone."""
        return []

    def require_release_fields(self) -> list[str]:
        """Counts by ``server_id``; no identity needed at release."""
        return []

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

    def clear_sticky_cache(self) -> dict:
        """Clear the sticky-session cache to force request redistribution.

        After clearing, all subsequent ``acquire_server()`` calls will select
        the least-loaded server (based on ``_inflight_requests``), which
        naturally balances load across all active replicas — including newly
        added ones with zero in-flight requests.

        Returns:
            A dict with ``cleared_entries`` (number of cache entries dropped)
            and ``server_loads`` (current per-server inflight counts for
            diagnostics).
        """
        cleared = len(self._request_id_to_server)
        self._request_id_to_server.clear()
        logger.info(
            f"[GlobalLoadBalancer] Sticky cache cleared: {cleared} entries dropped. "
            f"Server loads: {dict(self._inflight_requests)}"
        )
        return {
            "cleared_entries": cleared,
            "server_loads": dict(self._inflight_requests),
        }

    def get_status(self) -> dict:
        """Return current load balancer state for debugging."""
        return {
            "servers": dict(self._inflight_requests),
            "total_inflight": sum(self._inflight_requests.values()),
            "active_servers": len(self._inflight_requests),
            "registered_handles": list(self._servers.keys()),
        }

    def get_total_inflight(self) -> int:
        """Return the sum of in-flight requests across all currently registered servers."""
        return sum(self._inflight_requests.values())


def _create_global_sticky_inflight(
    servers: dict[str, Any],
    full_determinism: bool = False,
    load_balancer_cls: type | None = None,
):
    """Factory for the default sticky-session + least-inflight strategy.

    Args:
        servers: ``{server_address: actor_handle}`` mapping.
        full_determinism: Rollout-level ``rollout.full_determinism`` flag;
            enables full-hash deterministic routing.
        load_balancer_cls: Class to use as the routing actor. A subclass overrides
            :meth:`acquire_server` and takes full control of routing, so
            ``full_determinism`` is not forwarded to it.
    """
    kwargs = dict(servers=servers, max_cache_size=DEFAULT_ROUTING_CACHE_SIZE)
    # The default GlobalRequestLoadBalancer honors the full_determinism flag
    # in acquire_server. A custom subclass overrides acquire_server and takes
    # full control of routing, so the flag is not forwarded to it.
    if load_balancer_cls is GlobalRequestLoadBalancer:
        kwargs["full_determinism"] = full_determinism
    return ray.remote(load_balancer_cls).remote(**kwargs)


def _load_router_yaml(router_config_path: str) -> dict:
    """Load a router YAML config into a plain dict.

    Hydra ``defaults`` blocks are rejected: inline the referenced configs or
    compose them inside the router plugin's constructor.
    """
    full_path = resolve_config_path(router_config_path)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Router config file not found: {full_path}")

    cfg = OmegaConf.load(full_path)
    if "defaults" in cfg:
        raise ValueError(
            f"Router config {full_path} uses a Hydra 'defaults' block, which is not "
            "supported. Inline the referenced configs or compose them inside the "
            "router plugin's constructor."
        )
    return OmegaConf.to_container(cfg, resolve=True)


def _resolve_router_class(router_class: str) -> type:
    """Validate and import a load-balancer class from an FQN string.

    Raises:
        TypeError: If the resolved object is not callable.
    """
    cls = load_class_from_fqn(router_class, "load balancer class")
    if not callable(cls):
        raise TypeError(
            f"'{router_class}' is not callable (type: {type(cls).__name__}). "
            f"Expected a class with a .remote() constructor."
        )
    return cls


def _create_plugin_extension(
    servers: dict[str, Any],
    router_config_path: str,
):
    """Factory for a user-defined load balancer loaded from an external YAML.

    The file must contain ``router_class`` (FQN); the whole loaded dict is
    passed to the constructor.

    Args:
        servers: ``{server_address: actor_handle}`` mapping.
        router_config_path: Path to the external YAML file.

    Raises:
        ValueError: If ``router_config_path`` is missing or the YAML lacks
            ``router_class``.
        ImportError: If a module or package cannot be imported.
        AttributeError: If the class does not exist in the module.
    """
    yaml_config = _load_router_yaml(router_config_path)
    router_class = yaml_config.get("router_class", None)
    if not router_class:
        raise ValueError(
            "External router YAML must contain 'router_class'. "
            "Example: router_class: uni_agent.llm_router.KvcAwareRouter"
        )
    cls = _resolve_router_class(router_class)
    logger.info(
        "Creating plugin load balancer from YAML: class=%s, servers=%d, config=%s",
        router_class,
        len(servers),
        yaml_config,
    )
    ray_cls = cls if isinstance(cls, ray.actor.ActorClass) else ray.remote(cls)
    return ray_cls.remote(servers, yaml_config)


def get_router_handle(
    servers: dict[str, Any],
    router_config_path: str | None = None,
    full_determinism: bool = False,
    load_balancer_cls: type | None = None,
) -> Any:
    """Create a load balancer instance from router configuration.

    Args:
        servers: ``{server_address: actor_handle}`` mapping.
        router_config_path: Optional external router YAML path. When set, a
            user-defined plugin is loaded; otherwise the default sticky-session
            + least-inflight strategy is used.
        full_determinism: Rollout-level ``rollout.full_determinism`` flag,
            forwarded to strategies that support deterministic routing.
        load_balancer_cls: Optional subclass of the default strategy's load
            balancer. Takes precedence over the config-selected strategy; not
            applicable to the YAML plugin.
    """
    if router_config_path and load_balancer_cls is None:
        return _create_plugin_extension(servers=servers, router_config_path=router_config_path)

    # Programmatic injection (e.g. verl-omni's Deterministic* subclasses)
    # overrides the config-selected strategy. If both are set, the YAML is
    # ignored — warn so the mismatch doesn't fail silently.
    if router_config_path and load_balancer_cls is not None:
        logger.warning(
            "Both router_config_path=%r and load_balancer_cls=%s are set; "
            "the YAML plugin is ignored in favor of the injected subclass.",
            router_config_path,
            getattr(load_balancer_cls, "__name__", load_balancer_cls),
        )
    return _create_global_sticky_inflight(
        servers=servers,
        full_determinism=full_determinism,
        load_balancer_cls=load_balancer_cls or GlobalRequestLoadBalancer,
    )
