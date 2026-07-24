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
"""Unit tests for verl.workers.rollout.router"""

from typing import Any

import pytest
import ray

from verl.workers.config.rollout import RolloutConfig
from verl.workers.rollout.router import get_router_handle
from verl.workers.rollout.router.base import LoadBalancerRegistry


@ray.remote
class _MockLoadBalancer:
    """Ray actor implementing the RequestLoadBalancer Protocol.

    Used for Protocol structural conformance checks.
    """

    def __init__(self, servers: dict[str, Any], router_kwargs: dict):
        self._servers = dict(servers)
        self._inflight: dict[str, int] = {sid: 0 for sid in self._servers}
        self._router_kwargs = dict(router_kwargs)

    def acquire_server(self, request_id: str, prompt_ids: list[int] | None = None) -> tuple[str, Any]:
        if not self._inflight:
            raise RuntimeError("No available servers")

        sid = min(self._inflight, key=self._inflight.get)
        self._inflight[sid] += 1
        return sid, self._servers[sid]

    def release_server(self, server_id: str) -> None:
        if server_id in self._inflight and self._inflight[server_id] > 0:
            self._inflight[server_id] -= 1

    def add_servers(self, servers: dict[str, Any]) -> None:
        for sid, handle in servers.items():
            self._servers[sid] = handle
            self._inflight[sid] = 0

    def remove_servers(self, server_ids: list[str]) -> None:
        for sid in server_ids:
            self._inflight.pop(sid, None)
            self._servers.pop(sid, None)

    def get_all_servers(self) -> list[str]:
        return list(self._inflight.keys())

    def get_status(self) -> dict:
        return {
            "servers": dict(self._inflight),
            "total_inflight": sum(self._inflight.values()),
            "active_servers": len(self._inflight),
        }

    def get_router_kwargs(self) -> dict:
        """Return the router_kwargs passed to the constructor."""
        return dict(self._router_kwargs)


class TestRequestLoadBalancer:
    def test_protocol_methods_present(self):
        """All six Protocol methods are callable on _MockLoadBalancer."""
        for name in (
            "acquire_server",
            "release_server",
            "add_servers",
            "remove_servers",
            "get_all_servers",
            "get_status",
        ):
            assert callable(getattr(_MockLoadBalancer, name, None)), f"'{name}' missing or not callable"


class TestLoadBalancerRegistry:
    def test_register_new_strategy(self):
        name = "test_register_new_strategy"

        @LoadBalancerRegistry.register(name)
        class _MockBalancer:
            pass

        try:
            assert LoadBalancerRegistry.get_cls(name) is _MockBalancer
        finally:
            LoadBalancerRegistry._registry.pop(name, None)

    def test_register_duplicate_raises(self):
        name = "test_register_duplicate_raises"

        @LoadBalancerRegistry.register(name)
        class _MockBalancer1:
            pass

        try:
            with pytest.raises(ValueError, match="already registered"):

                @LoadBalancerRegistry.register(name)
                class _MockBalancer2:
                    pass
        finally:
            LoadBalancerRegistry._registry.pop(name, None)

    def test_get_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown load balancer strategy"):
            LoadBalancerRegistry.get_cls("nonexistent_strategy_xyz_123")

    def test_get_returns_class_for_builtins(self):
        """Only 'global_sticky_inflight' is registered as a built-in strategy."""
        for strategy in ("global_sticky_inflight",):
            cls = LoadBalancerRegistry.get_cls(strategy)
            assert isinstance(cls, type) or hasattr(cls, "remote")

    def test_list_strategies_is_sorted(self):
        strategies = LoadBalancerRegistry.list_strategies()
        assert "global_sticky_inflight" in strategies
        assert strategies == sorted(strategies)

    def test_list_strategies_includes_runtime_registration(self):
        name = "test_list_strategies_includes_runtime"

        @LoadBalancerRegistry.register(name)
        class _MockBalancer:
            pass

        try:
            assert name in LoadBalancerRegistry.list_strategies()
        finally:
            LoadBalancerRegistry._registry.pop(name, None)


@pytest.fixture(scope="module")
def ray_session():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


class TestGetRouterHandle:
    """Tests for get_router_handle with the default global_sticky_inflight strategy."""

    def test_default_rollout_config_uses_sticky_inflight(self, ray_session):
        """When RolloutConfig uses default router_strategy, it should create a GlobalRequestLoadBalancer."""
        config = RolloutConfig()
        lb = get_router_handle(servers={"s0": None, "s1": None}, rollout_config=config)
        status = ray.get(lb.get_status.remote())
        assert status["active_servers"] == 2
        assert status["total_inflight"] == 0

    def test_rollout_config_with_explicit_strategy(self, ray_session):
        config = RolloutConfig(router_strategy="global_sticky_inflight")
        lb = get_router_handle(servers={"a": None, "b": None}, rollout_config=config)
        status = ray.get(lb.get_status.remote())
        assert status["active_servers"] == 2

    def test_unknown_strategy_raises(self, ray_session):
        config = RolloutConfig(router_strategy="unknown_strategy_xyz")
        with pytest.raises(ValueError, match="Unknown load balancer strategy"):
            get_router_handle(servers={"s0": None}, rollout_config=config)

    def test_router_strategy_defaults_to_sticky_inflight(self, ray_session):
        """When RolloutConfig does not explicitly set router_strategy, the default is global_sticky_inflight."""
        config = RolloutConfig()
        lb = get_router_handle(servers={"s0": None, "s1": None}, rollout_config=config)
        status = ray.get(lb.get_status.remote())
        assert status["active_servers"] == 2
