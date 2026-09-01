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
"""Unit tests for verl.workers.rollout.router"""

import logging
from typing import Any

import pytest
import ray
import yaml

from verl.workers.rollout.router import GlobalRequestLoadBalancer, get_router_handle


@ray.remote
class _MockPluginLoadBalancer:
    """Ray actor implementing RequestLoadBalancer Protocol.

    Used as the ``router_class`` for plugin tests via ``importlib`` dynamic
    loading, and directly for Protocol structural checks."""

    def __init__(self, servers: dict[str, Any], router_kwargs: dict):
        self._servers = dict(servers)
        self._inflight: dict[str, int] = {sid: 0 for sid in self._servers}
        self._router_kwargs = dict(router_kwargs)
        self.releases: list[tuple] = []
        self.acquire_calls: list[tuple] = []

    def get_router_kwargs(self) -> dict:
        """Return the kwargs dict passed to the constructor."""
        return dict(self._router_kwargs)

    def get_releases(self) -> list[tuple]:
        """Return recorded release_server calls (server_id, request_id)."""
        return list(self.releases)

    def get_acquire_calls(self) -> list[tuple]:
        """Return recorded acquire_server calls as (request_id, prompt_ids)."""
        return list(self.acquire_calls)

    def require_acquire_fields(self) -> list[str]:
        """Protocol: this content-aware mock routes on prompt token ids."""
        return ["prompt_ids"]

    def require_release_fields(self) -> list[str]:
        """Protocol: attribute releases by request_id."""
        return ["request_id"]

    def acquire_server(self, request_id: str, prompt_ids: list[int] | None = None) -> tuple[str, Any]:
        self.acquire_calls.append((request_id, prompt_ids))
        if not prompt_ids:
            raise RuntimeError("No available prompt_ids")
        if not self._inflight:
            raise RuntimeError("No available servers")

        sid = min(self._inflight, key=self._inflight.get)
        self._inflight[sid] += 1
        return sid, self._servers[sid]

    def release_server(self, server_id: str, request_id: str | None = None) -> None:
        self.releases.append((server_id, request_id))
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

    def clear_sticky_cache(self) -> dict:
        """Protocol: clear sticky state (mock has none) and report loads."""
        return {"cleared_entries": 0, "server_loads": dict(self._inflight)}

    def get_total_inflight(self) -> int:
        """Protocol: total in-flight across servers."""
        return sum(self._inflight.values())


class TestRequireFields:
    """Balancers declare which generate() kwargs they consume at acquire time.
    LLMServerClient._acquire_server packs all keyword args locally (free,
    same-process) and only serializes the declared fields into the RPC."""

    def test_default_balancer_needs_nothing(self):
        """The default strategy routes on request_id only — no extra payload."""
        lb = GlobalRequestLoadBalancer(servers={"s0": None})
        assert lb.require_acquire_fields() == []
        assert lb.require_release_fields() == []

    def test_plugin_declares_prompt_ids(self, ray_session, tmp_path):
        yaml_path = TestGetRouterHandlePluginExtensionYaml._write_router_yaml(
            tmp_path, __name__ + "._MockPluginLoadBalancer"
        )
        lb = get_router_handle(servers={"s0": None}, router_config_path=yaml_path)
        assert ray.get(lb.require_acquire_fields.remote()) == ["prompt_ids"]
        assert ray.get(lb.require_release_fields.remote()) == ["request_id"]

    def test_client_acquire_serializes_only_declared_fields(self, ray_session, tmp_path):
        """_acquire_server lazily queries both field declarations (two light
        RPCs issued concurrently, then cached) and filters the packed
        generate() kwargs by the acquire declaration; sampling_params and
        friends never cross the wire."""
        import asyncio

        from omegaconf import OmegaConf

        from verl.workers.rollout.llm_server import LLMServerClient

        yaml_path = TestGetRouterHandlePluginExtensionYaml._write_router_yaml(
            tmp_path, __name__ + "._MockPluginLoadBalancer"
        )
        lb = get_router_handle(servers={"s0": None}, router_config_path=yaml_path)
        client = LLMServerClient(config=OmegaConf.create({}), load_balancer_handle=lb)
        assert client._lb_require_acquire_fields is None  # not queried yet
        sid, _ = asyncio.run(
            client._acquire_server(
                "req-1",
                prompt_ids=[1, 2, 3],
                sampling_params={"temperature": 1.0},
                image_data=["img-bytes"],
            )
        )
        # Both declarations queried and cached on first acquire, so the
        # release path never has to block the event loop on a lookup.
        assert client._lb_require_acquire_fields == ["prompt_ids"]
        assert client._lb_require_release_fields == ["request_id"]
        # The mock received only (request_id, prompt_ids) — sampling_params /
        # image_data stayed in-process.
        calls = ray.get(lb.get_acquire_calls.remote())
        assert calls == [("req-1", [1, 2, 3])]

    def test_client_acquire_with_empty_declaration_sends_request_id_only(self, ray_session):
        """The default balancer declares [] — the acquire RPC carries
        request_id only and matches the single-arg acquire signature."""
        import asyncio

        from omegaconf import OmegaConf

        from verl.workers.rollout.llm_server import LLMServerClient

        lb = ray.remote(GlobalRequestLoadBalancer).remote(servers={"s0": None, "s1": None})
        client = LLMServerClient(config=OmegaConf.create({}), load_balancer_handle=lb)
        sid, _ = asyncio.run(
            client._acquire_server(
                "req-2",
                prompt_ids=[1, 2, 3],  # packed but never serialized: [] declaration
                sampling_params={"temperature": 1.0},
            )
        )
        assert client._lb_require_acquire_fields == []
        assert client._lb_require_release_fields == []
        # Routed fine on request_id alone (sticky/least-inflight).
        assert ray.get(lb.get_status.remote())["total_inflight"] == 1

    def test_legacy_signatures_survive_client_path(self, ray_session):
        """A subclass overriding acquire/release with main's pre-plugin
        signatures keeps working through the client: the default ([], [])
        declaration sends request_id/server_id only."""
        import asyncio
        import time

        from omegaconf import OmegaConf

        from verl.workers.rollout.llm_server import LLMServerClient

        @ray.remote
        class _LegacyBalancer(GlobalRequestLoadBalancer):
            def acquire_server(self, request_id):  # main's single-arg signature
                return super().acquire_server(request_id)

            def release_server(self, server_id):  # main's single-arg signature
                return super().release_server(server_id)

        lb = _LegacyBalancer.remote(servers={"s0": None, "s1": None})
        client = LLMServerClient(config=OmegaConf.create({}), load_balancer_handle=lb)
        sid, _ = asyncio.run(
            client._acquire_server(
                "req-1",
                prompt_ids=[1, 2, 3],  # packed but never serialized: ([], [])
                sampling_params={"temperature": 1.0},
            )
        )
        client._release_server(sid, request_id="req-1")  # fire-and-forget
        time.sleep(1)  # let the actor process the release
        # Both legacy overrides executed (no swallowed TypeError): the counter
        # went up on acquire and back down on release.
        assert ray.get(lb.get_status.remote())["total_inflight"] == 0


@pytest.fixture(scope="module")
def ray_session():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


class TestGetRouterHandleDefault:
    def test_none_config_defaults_to_sticky_inflight(self, ray_session):
        lb = get_router_handle(servers={"s0": None, "s1": None}, router_config_path=None)
        status = ray.get(lb.get_status.remote())
        assert status["active_servers"] == 2
        assert status["total_inflight"] == 0

    def test_empty_router_config_defaults_to_sticky_inflight(self, ray_session):
        lb = get_router_handle(servers={"a": None, "b": None}, router_config_path="")
        status = ray.get(lb.get_status.remote())
        assert status["active_servers"] == 2


class TestGetRouterHandlePluginExtensionYaml:
    """Plugin router via external YAML (router_config_path), with Hydra
    ``defaults`` composition and pkg:// package-relative resolution."""

    @staticmethod
    def _write_router_yaml(tmp_path, router_class, **kwargs):
        """Write a temporary router YAML and return its path."""
        content = {"router_class": router_class, **kwargs}
        yaml_path = tmp_path / "router.yaml"
        yaml_path.write_text(yaml.dump(content))
        return str(yaml_path)

    def test_missing_yaml_file_raises(self, ray_session):
        config = "/nonexistent/path/router.yaml"
        with pytest.raises(FileNotFoundError, match="Router config file not found"):
            get_router_handle(servers={"s0": None}, router_config_path=config)

    def test_yaml_missing_router_class_raises(self, ray_session, tmp_path):
        yaml_path = tmp_path / "no_class.yaml"
        yaml_path.write_text(yaml.dump({"some_key": "value"}))
        config = str(yaml_path)
        with pytest.raises(ValueError, match="must contain 'router_class'"):
            get_router_handle(servers={"s0": None}, router_config_path=config)

    def test_acquire_least_loaded(self, ray_session, tmp_path):
        yaml_path = self._write_router_yaml(tmp_path, __name__ + "._MockPluginLoadBalancer")
        config = yaml_path
        lb = get_router_handle(servers={"s0": None, "s1": None, "s2": None}, router_config_path=config)
        s_a, _ = ray.get(lb.acquire_server.remote("a", prompt_ids=[1]))
        s_b, _ = ray.get(lb.acquire_server.remote("b", prompt_ids=[1]))
        s_c, _ = ray.get(lb.acquire_server.remote("c", prompt_ids=[1]))
        assert len({s_a, s_b, s_c}) == 3

    def test_add_remove_get_all_servers(self, ray_session, tmp_path):
        yaml_path = self._write_router_yaml(tmp_path, __name__ + "._MockPluginLoadBalancer")
        config = yaml_path
        lb = get_router_handle(servers={"s0": None}, router_config_path=config)
        ray.get(lb.add_servers.remote({"s1": None, "s2": None}))
        assert sorted(ray.get(lb.get_all_servers.remote())) == ["s0", "s1", "s2"]
        ray.get(lb.remove_servers.remote(["s0"]))
        assert ray.get(lb.get_all_servers.remote()) == ["s1", "s2"]

    def test_release_and_get_status(self, ray_session, tmp_path):
        yaml_path = self._write_router_yaml(tmp_path, __name__ + "._MockPluginLoadBalancer")
        config = yaml_path
        lb = get_router_handle(servers={"s0": None, "s1": None}, router_config_path=config)
        ray.get(lb.acquire_server.remote("a", prompt_ids=[1]))  # s0: 1
        ray.get(lb.acquire_server.remote("a", prompt_ids=[1]))  # s0: 2
        ray.get(lb.acquire_server.remote("b", prompt_ids=[1]))  # s1: 1
        assert ray.get(lb.get_status.remote())["total_inflight"] == 3
        ray.get(lb.release_server.remote("s0"))
        assert ray.get(lb.get_status.remote())["total_inflight"] == 2

    def test_empty_pool_raises(self, ray_session, tmp_path):
        yaml_path = self._write_router_yaml(tmp_path, __name__ + "._MockPluginLoadBalancer")
        config = yaml_path
        lb = get_router_handle(servers={"s0": None}, router_config_path=config)
        ray.get(lb.remove_servers.remote(["s0"]))
        with pytest.raises(ray.exceptions.RayTaskError, match="No available servers"):
            ray.get(lb.acquire_server.remote("req", prompt_ids=[1]))

    def test_yaml_forwards_composed_dict_to_constructor(self, ray_session, tmp_path):
        """The whole composed YAML dict (router_class included) is passed as kwargs."""
        yaml_path = self._write_router_yaml(tmp_path, __name__ + "._MockPluginLoadBalancer", extra_param="hello")
        config = yaml_path
        lb = get_router_handle(servers={"s0": None}, router_config_path=config)
        kwargs = ray.get(lb.get_router_kwargs.remote())
        assert kwargs.get("extra_param") == "hello"
        assert kwargs.get("router_class") == __name__ + "._MockPluginLoadBalancer"

    def test_yaml_defaults_block_rejected(self, ray_session, tmp_path):
        """A Hydra 'defaults' block is not composed — reject it with guidance
        instead of silently passing it through as a plain field."""
        main = tmp_path / "composed.yaml"
        main.write_text(
            yaml.dump({"defaults": [{"strategy": "kvc"}], "router_class": __name__ + "._MockPluginLoadBalancer"})
        )
        with pytest.raises(ValueError, match="defaults.*not.*supported|not supported"):
            get_router_handle(servers={"s0": None}, router_config_path=str(main))


class TestResolveConfigPath:
    """Path resolution via the shared verl.utils.import_utils.resolve_config_path:
    absolute → CWD → verl project root → verl package dir."""

    def test_absolute_path_passthrough(self):
        from verl.utils.import_utils import resolve_config_path

        assert resolve_config_path("/abs/path/router.yaml") == "/abs/path/router.yaml"

    def test_relative_path_found_in_cwd(self, tmp_path, monkeypatch):
        import os

        from verl.utils.import_utils import resolve_config_path

        (tmp_path / "router.yaml").write_text("router_class: x.Y")
        monkeypatch.chdir(tmp_path)
        assert resolve_config_path("router.yaml") == os.path.join(tmp_path, "router.yaml")

    def test_relative_path_not_found_raises(self, tmp_path, monkeypatch):
        from verl.utils.import_utils import resolve_config_path

        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="configuration file not found"):
            resolve_config_path("no_such_router.yaml")


class TestReleaseServerSignature:
    """release_server carries only request_id; content-aware balancers look up the
    prompt length from their own acquire-time bookkeeping instead of re-receiving
    the full token list over RPC."""

    def test_release_accepts_request_id_only(self, ray_session, tmp_path):
        """release_server takes (server_id, request_id) — no prompt_ids."""
        yaml_path = TestGetRouterHandlePluginExtensionYaml._write_router_yaml(
            tmp_path, __name__ + "._MockPluginLoadBalancer"
        )
        config = yaml_path
        lb = get_router_handle(servers={"s0": None, "s1": None}, router_config_path=config)
        sid, _ = ray.get(lb.acquire_server.remote("req-1", prompt_ids=[1, 2, 3]))
        ray.get(lb.release_server.remote(sid, request_id="req-1"))
        recs = ray.get(lb.get_releases.remote())
        assert recs == [(sid, "req-1")]
        # In-flight decremented alongside the recording
        assert ray.get(lb.get_status.remote())["total_inflight"] == 0

    def test_default_balancer_release_ignores_request_id(self, ray_session):
        lb = ray.remote(GlobalRequestLoadBalancer).remote(servers={"s0": None, "s1": None})
        sid, _ = ray.get(lb.acquire_server.remote("req-1"))
        ray.get(lb.release_server.remote(sid, request_id="req-1"))
        assert ray.get(lb.get_status.remote())["total_inflight"] == 0


def _write_plugin_yaml(tmp_path, router_class):
    """Write a plugin router YAML and return its path."""
    path = tmp_path / "router.yaml"
    path.write_text(yaml.dump({"router_class": router_class}))
    return str(path)


class TestGetRouterHandlePrecedence:
    """``load_balancer_cls`` overrides ``router_config_path``; the YAML path is
    ignored with a warning. Downstream config schemas (e.g. verl-omni's
    DiffusionRolloutConfig) may omit ``router_config_path`` entirely — it must
    be read defensively."""

    def test_both_set_warns_and_uses_subclass(self, ray_session, tmp_path, caplog):
        """load_balancer_cls takes precedence over router_config_path; the
        YAML is ignored and a warning is logged."""
        # A YAML that would, if loaded, point at a *different* class.
        bogus_yaml = _write_plugin_yaml(tmp_path, "nonexistent.BogusRouter")
        caplog.set_level(logging.WARNING, logger="verl.workers.rollout.router")

        lb = get_router_handle(
            servers={"s0": None, "s1": None},
            router_config_path=bogus_yaml,
            load_balancer_cls=GlobalRequestLoadBalancer,
        )
        # Subclass (not the YAML's BogusRouter) was instantiated.
        status = ray.get(lb.get_status.remote())
        assert status["active_servers"] == 2
        # The bogus YAML was never loaded (no BogusRouter import error), and a
        # precedence warning fired.
        assert any("load_balancer_cls" in r.message and "ignored" in r.message for r in caplog.records), (
            f"expected precedence warning; got {[r.message for r in caplog.records]}"
        )

    def test_router_config_path_missing_from_struct_config_does_not_raise(self, ray_session):
        """A struct/DictConfig rollout config without ``router_config_path``
        (e.g. verl-omni's DiffusionRolloutConfig) must not raise when the
        manager reads it via getattr-default."""
        from omegaconf import OmegaConf

        # Simulate a downstream rollout config that predates this field.
        rollout_config = OmegaConf.create({"full_determinism": False})  # no router_config_path key

        # This is the exact access pattern in LLMServerManager._init_global_load_balancer.
        path = getattr(rollout_config, "router_config_path", None)
        assert path is None

        lb = get_router_handle(
            servers={"s0": None, "s1": None},
            router_config_path=path,
            full_determinism=getattr(rollout_config, "full_determinism", False),
        )
        assert ray.get(lb.get_status.remote())["active_servers"] == 2
