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
"""CPU smoke tests for the actorized LLMServerManager (zero-CPU Ray actor).

The factories create the manager over RPC; every accessor is exercised as an
actor RPC from both async and sync call sites (the two driver-side patterns
used by the trainers). The manager subclass and its stub replica class are
function-local so cloudpickle ships them by value — a pytest module is not
importable from Ray worker processes, while real subclasses (fully_async)
live in the verl package and stay importable there.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import ray
import yaml

from verl.utils.tracking import RLInsightLogger
from verl.workers.rollout import llm_server
from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager
from verl.workers.rollout.router import GlobalRequestLoadBalancer

EMBEDDED_PLUGIN_FQN = __name__ + "._EmbeddedPluginBalancer"
ACTOR_PLUGIN_FQN = __name__ + "._ActorPluginBalancer"


def _define_embedded_plugin() -> type:
    """Plain-class plugin resolved from a router YAML — the embedded flavor.

    Defined inside a factory (and only aliased at module level for the YAML
    FQN) so cloudpickle ships it by value: a pytest module is not importable
    from Ray worker processes, unlike real plugins which live in installed
    modules and ship by reference.

    Plain (not ``@ray.remote``): the manager actor is the host process, so the
    plugin is constructed in-process from ``(servers, router_kwargs)``. Note
    the constructor signature: it takes the whole loaded YAML dict, and
    deliberately NOT ``full_determinism`` (only the default strategy's
    constructor receives that flag).
    """

    class _EmbeddedPluginBalancer:
        def __init__(self, servers, router_kwargs):
            self._servers = dict(servers)
            self._router_kwargs = dict(router_kwargs)

        def get_status(self):
            return {"plugin": True, "router_kwargs": dict(self._router_kwargs)}

    return _EmbeddedPluginBalancer


_EmbeddedPluginBalancer = _define_embedded_plugin()
_ActorPluginBalancer = ray.remote(_EmbeddedPluginBalancer)


def _make_manager_cls() -> type[LLMServerManager]:
    """Build a manager subclass with a stub replica class (engine imports stay out)."""

    class _StubReplica:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _ManagerForTests(LLMServerManager):
        rollout_replica_class = _StubReplica  # pre-set: skip engine-class resolution

    return _ManagerForTests


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                name="vllm",
                nnodes=1,
                disaggregation=SimpleNamespace(enabled=False),
            ),
            model=SimpleNamespace(),
        )
    )


def test_create_empty_actor_rpc_surface():
    """Async driver pattern: factories return an actor handle; accessors are
    RPCs returning values (addresses, registry, client). The manager is also
    the load-balancer host: the embedded balancer's surface is the manager's
    own RPC surface."""
    ray.init(ignore_reinit_error=True)
    manager_cls = _make_manager_cls()

    class _RecordingClient(LLMServerClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.recorded = True

    async def scenario():
        handle = await manager_cls.create_empty(config=_config())
        assert await handle.get_addresses.remote() == []
        assert await handle.get_replicas.remote() == []
        assert await handle.get_server_registry.remote() == {}

        status = await handle.get_status.remote()
        assert status["active_servers"] == 0

        # A client_cls subclass crosses the RPC and is constructed inside the
        # actor (the trainers pass FullyAsyncLLMServerClient this way).
        client = await handle.get_client.remote(client_cls=_RecordingClient)
        assert isinstance(client, _RecordingClient)
        assert client.recorded
        # The client routes through the manager's own actor handle.
        assert client._load_balancer == handle
        balancer_status = await client._load_balancer.get_status.remote()
        assert balancer_status["active_servers"] == 0

        ray.kill(handle)

    asyncio.run(scenario())


def test_load_balancer_plane_is_embedded_in_the_manager():
    """The merged balancing plane: add/acquire/release go through the
    manager actor's RPC surface and hit the same embedded balancer state."""
    ray.init(ignore_reinit_error=True)
    manager_cls = _make_manager_cls()

    @ray.remote(num_cpus=0)
    class _StubServer:
        pass

    server_actor = _StubServer.remote()
    manager = manager_cls.create_empty(config=_config())
    ray.get(manager.add_servers.remote({"addr-0": server_actor}))
    server_id, server = ray.get(manager.acquire_server.remote(request_id="req-1"))
    assert server_id == "addr-0"
    # ActorHandle identity survives the acquire round-trip.
    assert server == server_actor
    ray.get(manager.release_server.remote(server_id=server_id, request_id="req-1"))
    assert ray.get(manager.get_total_inflight.remote()) == 0
    assert ray.get(manager.get_all_servers.remote()) == ["addr-0"]
    ray.kill(manager)


def test_bad_router_config_fails_fast_on_driver(tmp_path):
    """A bad router config raises at the factory call, on the caller's node —
    before the actor is launched and before any replica initialization runs:
    a missing YAML, and a YAML whose plugin is a Ray actor class (an embedded
    balancer is constructed in-process, so it cannot host one)."""
    ray.init(ignore_reinit_error=True)
    manager_cls = _make_manager_cls()
    config = _config()
    config.actor_rollout_ref.rollout.router_config_path = "/nonexistent/router.yaml"

    with pytest.raises(FileNotFoundError, match="Router config file not found"):
        manager_cls.create_empty(config=config)

    yaml_path = tmp_path / "router.yaml"
    yaml_path.write_text(yaml.dump({"router_class": ACTOR_PLUGIN_FQN}))
    config.actor_rollout_ref.rollout.router_config_path = str(yaml_path)
    with patch.object(llm_server, "_as_actor", wraps=llm_server._as_actor) as launch:
        with pytest.raises(TypeError, match="embedded load balancer must be"):
            manager_cls.create_empty(config=config)
    launch.assert_not_called()


def test_release_honors_the_strategy_release_fields():
    """The embedded plane must honor the strategy's ``require_release_fields``
    declaration: declared fields are forwarded, undeclared ones are omitted —
    legacy ``release_server(self, server_id)`` signatures keep working through
    the manager's RPC surface."""
    ray.init(ignore_reinit_error=True)
    manager_cls = _make_manager_cls()

    @ray.remote(num_cpus=0)
    class _StubServer:
        pass

    server_actor = _StubServer.remote()

    def _drain_once(balancer_cls):
        manager = manager_cls.create_empty(config=_config(), load_balancer_cls=balancer_cls)
        ray.get(manager.add_servers.remote({"addr-0": server_actor}))
        server_id, _ = ray.get(manager.acquire_server.remote(request_id="req-1"))
        ray.get(manager.release_server.remote(server_id=server_id, request_id="req-1"))
        assert ray.get(manager.get_total_inflight.remote()) == 0
        ray.kill(manager)

    class _LegacyBalancer(GlobalRequestLoadBalancer):
        def release_server(self, server_id):  # main's single-arg signature
            return super().release_server(server_id)

    # Undeclared fields are omitted: before the fix the wrapper unconditionally
    # passed request_id=None and this single-arg signature raised TypeError.
    _drain_once(_LegacyBalancer)

    class _DemandingBalancer(GlobalRequestLoadBalancer):
        def require_release_fields(self) -> list[str]:
            return ["request_id"]

        def release_server(self, server_id, request_id=None):
            if request_id is None:
                raise ValueError("content-aware balancer requires request_id")
            return super().release_server(server_id, request_id)

    # Declared fields are forwarded: a dropped request_id would surface as a
    # RayTaskError(ValueError) inside _drain_once.
    _drain_once(_DemandingBalancer)


def test_yaml_plugin_resolves_on_driver_and_embeds_via_factory(tmp_path):
    """End-to-end contract for the multi-node fix: the factory reads the YAML
    and imports the FQN on the caller's node, ships the RouterSpec into the
    actor, and the plugin is constructed in-process from
    ``(servers, router_kwargs)`` — the actor itself never reads the YAML."""
    ray.init(ignore_reinit_error=True)
    yaml_path = tmp_path / "router.yaml"
    yaml_path.write_text(yaml.dump({"router_class": EMBEDDED_PLUGIN_FQN, "strategy_param": "probe"}))

    config = _config()
    config.actor_rollout_ref.rollout.router_config_path = str(yaml_path)
    manager = _make_manager_cls().create_empty(config=config)

    # The embedded balancer is the plugin, and the whole loaded YAML dict
    # reached its constructor — the spec (class + dict) crossed the RPC.
    status = ray.get(manager.get_status.remote())
    assert status["plugin"] is True
    assert status["router_kwargs"] == {"router_class": EMBEDDED_PLUGIN_FQN, "strategy_param": "probe"}
    ray.kill(manager)


def test_create_initializes_replicas_and_returns_value_snapshots(monkeypatch):
    """The non-empty factory: replicas initialize inside the actor and every
    accessor returns a by-value snapshot — the contract that replaced live
    object references."""
    ray.init(ignore_reinit_error=True)
    # Hermetic: never let a host rl-insight env toggle flip the metrics guard. The driver-side
    # patch covers this process; the manager runs in its own worker, so pin the toggle off in
    # the actor's runtime environment as well (with disable_log_stats=True the guard raises).
    monkeypatch.setattr(RLInsightLogger, "enabled", classmethod(lambda cls: False))
    monkeypatch.setattr(
        llm_server,
        "_as_actor",
        lambda cls: ray.remote(cls).options(num_cpus=0, runtime_env={"env_vars": {RLInsightLogger.ENABLE_ENV: "0"}}),
    )

    @ray.remote(num_cpus=0)
    class _StubEngine:
        pass

    class _StandaloneReplica:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def init_standalone(self, resource_pool=None):
            self._server_handle = _StubEngine.remote()
            self._server_address = "stub-engine-0"

    class _ManagerForCreate(LLMServerManager):
        rollout_replica_class = _StandaloneReplica

    config = _config()
    rollout = config.actor_rollout_ref.rollout
    rollout.n_gpus_per_node = 1
    rollout.tensor_model_parallel_size = 1
    rollout.data_parallel_size = 1
    rollout.pipeline_model_parallel_size = 1
    rollout.disable_log_stats = True
    rollout.prometheus = SimpleNamespace(enable=False)

    manager = _ManagerForCreate.create(config=config)  # sync call site (auto_await)

    assert ray.get(manager.get_addresses.remote()) == ["stub-engine-0"]
    registry = ray.get(manager.get_server_registry.remote())
    assert set(registry) == {"stub-engine-0"}

    # Data plane through the default embedded balancer: acquire returns the
    # same actor handle the registry carries (identity survives the round-trip).
    server_id, handle = ray.get(manager.acquire_server.remote(request_id="req-1"))
    assert server_id == "stub-engine-0"
    assert handle == registry["stub-engine-0"]
    ray.get(manager.release_server.remote(server_id=server_id, request_id="req-1"))

    # Snapshot contract: each get_replicas RPC returns a fresh by-value copy.
    first = ray.get(manager.get_replicas.remote())
    second = ray.get(manager.get_replicas.remote())
    assert len(first) == len(second) == 1
    assert first[0] is not second[0]
    assert first[0].kwargs["replica_rank"] == 0
    ray.kill(manager)


def test_full_determinism_flag_reaches_the_default_balancer():
    """``rollout.full_determinism`` must still reach the embedded default
    balancer (same getattr extraction as the pre-refactor get_router_handle
    call). Strict A/B: with the flag on, hash routing ignores in-flight load
    and re-acquires the same server; with it off, least-loaded deterministically
    switches to the zero-load peer."""
    ray.init(ignore_reinit_error=True)

    def _acquire_after_sticky_clear(full_determinism: bool) -> tuple[str, str]:
        config = _config()
        config.actor_rollout_ref.rollout.full_determinism = full_determinism
        manager = _make_manager_cls().create_empty(config=config)

        @ray.remote(num_cpus=0)
        class _StubServer:
            pass

        ray.get(manager.add_servers.remote({"srv-a": _StubServer.remote(), "srv-b": _StubServer.remote()}))
        first = ray.get(manager.acquire_server.remote(request_id="determinism-probe"))
        # Drop the sticky mapping but keep the in-flight counts: the two
        # strategies now disagree deterministically on the second acquire.
        ray.get(manager.clear_sticky_cache.remote())
        second = ray.get(manager.acquire_server.remote(request_id="determinism-probe"))
        ray.kill(manager)
        return first[0], second[0]

    # Flag forwarded: same server despite its higher in-flight load.
    first, second = _acquire_after_sticky_clear(True)
    assert first == second
    # Flag absent: least-loaded has exactly one zero-load candidate — the peer.
    first, second = _acquire_after_sticky_clear(False)
    assert first != second
