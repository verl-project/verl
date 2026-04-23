# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from typing import Any

import ray
from cachetools import LRUCache

DEFAULT_ROUTING_CACHE_SIZE = 10000
_GLOBAL_LOAD_BALANCER_CLASSES: dict[str, Any] = {}


def get_global_load_balancer(mode: str) -> Any:
    """Resolve the Ray actor class from mode."""
    cls = _GLOBAL_LOAD_BALANCER_CLASSES.get(mode)
    if cls is None:
        raise ValueError(f"Unknown load balancer mode {mode!r}. Registered: {sorted(_GLOBAL_LOAD_BALANCER_CLASSES)}")
    return cls


class GlobalRequestLoadBalancer:
    """Global sticky-session + in-flight load balancer shared by all AgentLoopWorkers."""

    def __init__(self, server_actor_ids: list[str], max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE) -> None:
        if not server_actor_ids:
            raise ValueError("server_actor_ids must be non-empty")
        self._server_actor_ids = server_actor_ids
        self._inflight_requests: dict[str, int] = {sid: 0 for sid in server_actor_ids}

    def acquire_server(self, request_id: str, request_group_id: str | None = None) -> str:
        """Acquire a server for the given request, reusing the same server for multi-turn conversations."""
        raise NotImplementedError("Subclasses must implement this method")

    def release_server(self, server_id: str) -> None:
        """Release a server after a request completes"""
        if server_id not in self._inflight_requests:
            raise ValueError(f"Invalid server_id for release: {server_id}")
        if self._inflight_requests[server_id] <= 0:
            raise ValueError(f"Release called with no inflight requests on server {server_id}")
        self._inflight_requests[server_id] -= 1


@ray.remote
class RequestStickyLoadBalancer(GlobalRequestLoadBalancer):
    """Request-level sticky session + in-flight load balancer shared by each AgentLoopWorker."""

    def __init__(self, server_actor_ids: list[str], max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE) -> None:
        super().__init__(server_actor_ids=server_actor_ids, max_cache_size=max_cache_size)
        self._request_id_to_server: LRUCache[str, str] = LRUCache(maxsize=max_cache_size)

    def acquire_server(self, request_id: str, request_group_id: str | None = None) -> str:
        if request_id in self._request_id_to_server:
            server_id = self._request_id_to_server[request_id]
            self._inflight_requests[server_id] += 1
            return server_id

        server_id = min(self._inflight_requests, key=self._inflight_requests.get)
        self._request_id_to_server[request_id] = server_id
        self._inflight_requests[server_id] += 1
        return server_id


@ray.remote
class GroupStickyLoadBalancer(GlobalRequestLoadBalancer):
    """Group-level sticky session + in-flight load balancer shared by all AgentLoopWorkers."""

    def __init__(self, server_actor_ids: list[str], max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE) -> None:
        super().__init__(server_actor_ids=server_actor_ids, max_cache_size=max_cache_size)
        self._request_group_id_to_server: LRUCache[str, str] = LRUCache(maxsize=max_cache_size)

    def acquire_server(self, request_id: str, request_group_id: str | None = None) -> str:
        if request_group_id is None:
            server_id = min(self._inflight_requests, key=self._inflight_requests.get)
            self._inflight_requests[server_id] += 1
            return server_id
        if request_group_id in self._request_group_id_to_server:
            server_id = self._request_group_id_to_server[request_group_id]
            self._inflight_requests[server_id] += 1
            return server_id

        server_id = min(self._inflight_requests, key=self._inflight_requests.get)
        self._request_group_id_to_server[request_group_id] = server_id
        self._inflight_requests[server_id] += 1
        return server_id


_GLOBAL_LOAD_BALANCER_CLASSES["request"] = RequestStickyLoadBalancer
_GLOBAL_LOAD_BALANCER_CLASSES["group"] = GroupStickyLoadBalancer
