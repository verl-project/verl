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

"""End-to-end integration tests for KVCAwareBalancer via VeRL's drop-in path.

These exercise the **wiring** — VeRL ``get_router_handle`` → ``ray.remote`` →
KVCAwareBalancer Protocol methods across the Ray actor boundary — NOT the
routing quality. ``route()`` is a placeholder ranking (input order); the real
KV-aware algorithm lands with the strategy-module design. So acquire tests
assert handle-correctness for whatever id comes back, not which id is chosen.
Per detailed_balancer.md §5.3B.
"""

from __future__ import annotations

import pytest
import ray
from omegaconf import OmegaConf

from verl.workers.rollout.router import get_router_handle

pytestmark = [pytest.mark.st, pytest.mark.cpu]


# Minimal kvcaware router config, inlined to mirror what Hydra composes under
# ``actor_rollout_ref.rollout.strategy`` when ``rollout/router/kvcaware.yaml`` is
# selected via the ``router@strategy`` defaults entry. ``get_router_handle``
# reads ``strategy`` (then ``router_strategy`` for dispatch) and the factory
# passes the ``strategy`` node to ``KVCAwareConfig.from_config``.
_KVCAWARE_STRATEGY = OmegaConf.create(
    {
        "strategies": [
            {
                "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                "weight": 1.0,
                "collector_names": ["vllm_zmq", "vllm_metrics", "sticky_stat", "inflight_stat"],
            },
        ],
        "collector": {
            "http_polling": {"polling_interval": 1, "http_timeout": 10},
            "long_connection": {
                "base_retry_delay": 1.0,
                "max_retry_delay": 30.0,
                "max_retry_attempts": 5,
                "retry_backoff_factor": 2.0,
            },
        },
        "cache_store": {"kv_cache_store_type": "list", "ttl": 30},
    }
)


@ray.remote
class _MockServer:
    """Minimal Ray actor standing in for a vLLMHttpServer handle.

    Exposes the handler-getter methods the Balancer probes at construction
    (``get_server_address``, ``get_kv_events_endpoints``).
    """

    def get_server_address(self):
        return ("127.0.0.1", 8000)

    def get_kv_events_endpoints(self):
        return None


def _mk(*ids: str) -> dict:
    """Build a {server_id: _MockServer handle} pool — real Ray actor handles."""
    return {s: _MockServer.remote() for s in ids}


@pytest.fixture(scope="session")
def ray_runtime():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def _router_config():
    """Build a rollout-shaped config carrying the kvcaware ``router_config`` node.

    Mirrors the production path: ``get_router_handle`` reads ``router_strategy``
    (→ "kvcaware"), then passes the ``router_config`` node to the Balancer, which
    runs ``KVCAwareConfig.from_config`` on it. ``max_num_seqs`` is resolved inside
    the Balancer via each server's ``get_rollout_config`` RPC (``_MockServer``
    exposes none → default capacity); these are flow tests, not scheduling tests.
    """
    return OmegaConf.create(
        {
            "router_strategy": "kvcaware",
            "router_config": _KVCAWARE_STRATEGY,
        }
    )


class TestKVCAwareEndToEnd:
    """I01-I05: full VeRL dispatch flow over a Ray actor (flow, not scheduling)."""

    def test_i01_actor_lists_servers_and_acquires_valid_handle(self, ray_runtime):
        """
        Feature: a freshly created actor lists its servers and acquire returns a valid handle
        Description: get_router_handle → get_all_servers; then acquire → (id, handle)
        Expectation: get_all_servers returns the provided ids; acquire returns an id in the pool
                     whose handle matches
        """
        servers = _mk("s0", "s1", "s2")
        handle = get_router_handle(servers, _router_config())
        assert set(ray.get(handle.get_all_servers.remote())) == set(servers)
        server_id, actor_handle = ray.get(handle.acquire_server.remote("r1", [1, 2, 3]))
        assert server_id in servers
        assert actor_handle in servers.values()

    def test_i02_acquire_release_acquire_over_ray(self, ray_runtime):
        """
        Feature: release is a no-op and does not break subsequent routing
        Description: acquire → release → acquire over the Ray actor
        Expectation: release returns None; both acquires return valid (id, handle)
        """
        servers = _mk("s0")
        handle = get_router_handle(servers, _router_config())
        sid1, h1 = ray.get(handle.acquire_server.remote("r1", [1, 2]))
        rel = ray.get(handle.release_server.remote(sid1))
        sid2, h2 = ray.get(handle.acquire_server.remote("r2", [1, 2]))
        assert rel is None
        assert {sid1, sid2} <= {"s0"}
        assert h1 in servers.values() and h2 in servers.values()

    def test_i03_add_remove_reflected_in_pool(self, ray_runtime):
        """
        Feature: pool mutations are visible through the actor
        Description: add_servers then get_all_servers; remove_servers then get_all_servers
        Expectation: pool reflects the add and the remove
        """
        handle = get_router_handle(_mk("s0"), _router_config())
        ray.get(handle.add_servers.remote(_mk("s3")))
        assert "s3" in ray.get(handle.get_all_servers.remote())
        ray.get(handle.remove_servers.remote(["s0"]))
        assert "s0" not in ray.get(handle.get_all_servers.remote())

    def test_i04_concurrent_acquires_both_valid(self, ray_runtime):
        """
        Feature: a single actor handles concurrent acquire calls without crashing
        Description: two concurrent acquire_server.remote() resolved together
        Expectation: both return a valid (id, handle) pair (v1 does not track inflight)
        """
        servers = _mk("s0", "s1")
        handle = get_router_handle(servers, _router_config())
        (sid1, h1), (sid2, h2) = ray.get(
            [handle.acquire_server.remote("r1", [1]), handle.acquire_server.remote("r2", [2])]
        )
        for sid, h in [(sid1, h1), (sid2, h2)]:
            assert sid in servers
            assert h in servers.values()

    def test_i05_construction_state_and_route_invocation(self, ray_runtime):
        """
        Feature: construction wires provider/strategies, and acquire invokes route()
        Description: get_router_handle → get_status (construction) → acquire → get_status (route called)
        Expectation: before acquire, provider=CollectorManager, strategy materialized, route_calls=0;
                     after one acquire, route_calls=1
        """
        handle = get_router_handle(_mk("s0", "s1"), _router_config())
        status = ray.get(handle.get_status.remote())
        assert status["manager"] == "CollectorManager"
        assert status["strategies"] == [{"type": "KVCacheAwareStrategy", "weight": 1.0}]
        assert set(status["servers"]) == {"s0", "s1"}
        assert status["route_calls"] == 0
        ray.get(handle.acquire_server.remote("r1", [1, 2, 3]))
        assert ray.get(handle.get_status.remote())["route_calls"] == 1
