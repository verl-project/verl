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
import yaml

from verl.workers.config.rollout import RouterConfig
from verl.workers.rollout.router import (
    LoadBalancerRegistry,
    get_router_handle,
)


@ray.remote
class _MockPluginLoadBalancer:
    """Ray actor implementing RequestLoadBalancer Protocol.

    Used as the ``router_class`` for plugin_extension tests via
    ``importlib`` dynamic loading, and directly for Protocol
    structural checks."""

    def __init__(self, servers: dict[str, Any], router_kwargs: dict):
        self._servers = dict(servers)
        self._inflight: dict[str, int] = {sid: 0 for sid in self._servers}
        self._router_kwargs = dict(router_kwargs)

    def acquire_server(
        self, request_id: str, prompt_ids: list[int] | None = None
    ) -> tuple[str, Any]:
        if not prompt_ids:
            raise RuntimeError("No available prompt_ids")
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
        """All six Protocol methods are callable on _MockPluginLoadBalancer."""
        for name in (
            "acquire_server", "release_server", "add_servers",
            "remove_servers", "get_all_servers", "get_status",
        ):
            assert callable(getattr(_MockPluginLoadBalancer, name, None)), (
                f"'{name}' missing or not callable"
            )


class TestLoadBalancerRegistry:
    def test_register_new_strategy(self):
        name = "test_register_new_strategy"
        factory = lambda servers, config: None  # noqa: E731
        try:
            LoadBalancerRegistry.register(name, factory)
            assert LoadBalancerRegistry.get(name) is factory
        finally:
            LoadBalancerRegistry._registry.pop(name, None)

    def test_register_duplicate_raises(self):
        name = "test_register_duplicate_raises"
        LoadBalancerRegistry.register(name, lambda: None)
        try:
            with pytest.raises(ValueError, match="already registered"):
                LoadBalancerRegistry.register(name, lambda: None)
        finally:
            LoadBalancerRegistry._registry.pop(name, None)

    def test_get_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown load balancer strategy"):
            LoadBalancerRegistry.get("nonexistent_strategy_xyz_123")

    def test_get_returns_callable_for_builtins(self):
        for strategy in ("global_sticky_inflight", "plugin_extension"):
            factory = LoadBalancerRegistry.get(strategy)
            assert callable(factory)

    def test_list_strategies_is_sorted(self):
        strategies = LoadBalancerRegistry.list_strategies()
        assert "global_sticky_inflight" in strategies
        assert "plugin_extension" in strategies
        assert strategies == sorted(strategies)

    def test_list_strategies_includes_runtime_registration(self):
        name = "test_list_strategies_includes_runtime"
        LoadBalancerRegistry.register(name, lambda: None)
        try:
            assert name in LoadBalancerRegistry.list_strategies()
        finally:
            LoadBalancerRegistry._registry.pop(name, None)


@pytest.fixture(scope="module")
def ray_session():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


class TestGetRouterHandleDefault:
    def test_none_config_defaults_to_sticky_inflight(self, ray_session):
        lb = get_router_handle(servers={"s0": None, "s1": None}, router_config=None)
        status = ray.get(lb.get_status.remote())
        assert status["active_servers"] == 2
        assert status["total_inflight"] == 0

    def test_router_config_with_explicit_strategy(self, ray_session):
        config = RouterConfig(router_strategy="global_sticky_inflight")
        lb = get_router_handle(servers={"a": None, "b": None}, router_config=config)
        status = ray.get(lb.get_status.remote())
        assert status["active_servers"] == 2

    def test_unknown_strategy_raises(self, ray_session):
        config = RouterConfig(router_strategy="unknown_strategy_xyz")
        with pytest.raises(ValueError, match="Unknown load balancer strategy"):
            get_router_handle(servers={"s0": None}, router_config=config)


class TestGetRouterHandlePluginExtension:
    """Tests for plugin_extension strategy via external YAML router config."""

    @staticmethod
    def _write_router_yaml(tmp_path, router_class, **kwargs):
        """Helper: write a temporary router YAML file and return its path."""
        content = {"router_class": router_class, **kwargs}
        yaml_path = tmp_path / "router.yaml"
        yaml_path.write_text(yaml.dump(content))
        return str(yaml_path)

    def test_missing_router_config_path_raises(self, ray_session):
        """When router_config_path is null, plugin_extension should raise."""
        config = RouterConfig(
            router_strategy="plugin_extension",
            router_config_path=None,
        )
        with pytest.raises(ValueError, match="requires 'router_config_path'"):
            get_router_handle(servers={"s0": None}, router_config=config)

    def test_missing_yaml_file_raises(self, ray_session):
        """When router_config_path points to non-existent file."""
        config = RouterConfig(
            router_strategy="plugin_extension",
            router_config_path="/nonexistent/path/router.yaml",
        )
        with pytest.raises(FileNotFoundError, match="Router config file not found"):
            get_router_handle(servers={"s0": None}, router_config=config)

    def test_yaml_missing_router_class_raises(self, ray_session, tmp_path):
        """YAML file exists but doesn't contain router_class."""
        yaml_path = tmp_path / "no_class.yaml"
        yaml_path.write_text(yaml.dump({"some_key": "some_value"}))
        config = RouterConfig(
            router_strategy="plugin_extension",
            router_config_path=str(yaml_path),
        )
        with pytest.raises(ValueError, match="must contain 'router_class'"):
            get_router_handle(servers={"s0": None}, router_config=config)

    def test_invalid_module_raises(self, ray_session, tmp_path):
        """YAML has router_class with non-existent module."""
        yaml_path = self._write_router_yaml(
            tmp_path, "nonexistent.module.ClassName"
        )
        config = RouterConfig(
            router_strategy="plugin_extension",
            router_config_path=yaml_path,
        )
        with pytest.raises(ImportError, match="Failed to import module"):
            get_router_handle(servers={"s0": None}, router_config=config)

    def test_acquire_least_loaded(self, ray_session, tmp_path):
        yaml_path = self._write_router_yaml(
            tmp_path, __name__ + "._MockPluginLoadBalancer"
        )
        config = RouterConfig(
            router_strategy="plugin_extension",
            router_config_path=yaml_path,
        )
        lb = get_router_handle(
            servers={"s0": None, "s1": None, "s2": None}, router_config=config
        )
        s_a, _ = ray.get(lb.acquire_server.remote("a", prompt_ids=[1]))
        s_b, _ = ray.get(lb.acquire_server.remote("b", prompt_ids=[1]))
        s_c, _ = ray.get(lb.acquire_server.remote("c", prompt_ids=[1]))
        assert len({s_a, s_b, s_c}) == 3

    def test_add_remove_get_all_servers(self, ray_session, tmp_path):
        yaml_path = self._write_router_yaml(
            tmp_path, __name__ + "._MockPluginLoadBalancer"
        )
        config = RouterConfig(
            router_strategy="plugin_extension",
            router_config_path=yaml_path,
        )
        lb = get_router_handle(servers={"s0": None}, router_config=config)
        ray.get(lb.add_servers.remote({"s1": None, "s2": None}))
        assert sorted(ray.get(lb.get_all_servers.remote())) == ["s0", "s1", "s2"]
        ray.get(lb.remove_servers.remote(["s0"]))
        assert ray.get(lb.get_all_servers.remote()) == ["s1", "s2"]

    def test_release_and_get_status(self, ray_session, tmp_path):
        yaml_path = self._write_router_yaml(
            tmp_path, __name__ + "._MockPluginLoadBalancer"
        )
        config = RouterConfig(
            router_strategy="plugin_extension",
            router_config_path=yaml_path,
        )
        lb = get_router_handle(servers={"s0": None, "s1": None}, router_config=config)
        ray.get(lb.acquire_server.remote("a", prompt_ids=[1]))  # s0: 1
        ray.get(lb.acquire_server.remote("a", prompt_ids=[1]))  # s0: 2
        ray.get(lb.acquire_server.remote("b", prompt_ids=[1]))  # s1: 1
        assert ray.get(lb.get_status.remote())["total_inflight"] == 3
        ray.get(lb.release_server.remote("s0"))
        assert ray.get(lb.get_status.remote())["total_inflight"] == 2

    def test_empty_pool_raises(self, ray_session, tmp_path):
        yaml_path = self._write_router_yaml(
            tmp_path, __name__ + "._MockPluginLoadBalancer"
        )
        config = RouterConfig(
            router_strategy="plugin_extension",
            router_config_path=yaml_path,
        )
        lb = get_router_handle(servers={"s0": None}, router_config=config)
        ray.get(lb.remove_servers.remote(["s0"]))
        with pytest.raises(ray.exceptions.RayTaskError, match="No available servers"):
            ray.get(lb.acquire_server.remote("req", prompt_ids=[1]))

    def test_plugin_forwards_kwargs_from_yaml(self, ray_session, tmp_path):
        """Extra keys in the YAML are forwarded as kwargs to the constructor."""
        fqn = __name__ + "._MockPluginLoadBalancer"
        yaml_path = self._write_router_yaml(
            tmp_path, fqn, extra_param="hello", another_param=42
        )
        config = RouterConfig(
            router_strategy="plugin_extension",
            router_config_path=yaml_path,
        )
        lb = get_router_handle(servers={"s0": None}, router_config=config)
        kwargs = ray.get(lb.get_router_kwargs.remote())
        assert kwargs.get("extra_param") == "hello"
        assert kwargs.get("another_param") == 42
