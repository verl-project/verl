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

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from verl.workers.rollout.router.kvcaware.balancer import KVCAwareBalancer
from verl.workers.rollout.router.kvcaware.strategies.kvc_aware import KVCacheAwareStrategy

from ._helpers import (
    _make_balancer,
    _router_config,
)

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


# ============================================================
# 5.1 / construction
# ============================================================


class TestKVCAwareBalancerConstruction:
    """B01-Bnn: __init__ wiring and validation."""

    def test_b01_normal_construction_wires_components(self):
        """
        Feature: construction wires config/provider/strategies/servers
        Description: KVCAwareBalancer({"s0": h0}, router_config)
        Expectation: _provider built and started; _strategies wired with weight
        """
        balancer = KVCAwareBalancer({"s0": "h0"}, _router_config())
        assert balancer._manager.started is True
        assert balancer._manager.collection_names == ["inflight_stat", "sticky_stat", "vllm_zmq"]
        assert len(balancer._strategies) == 1
        strat, weight = balancer._strategies[0]
        assert isinstance(strat, KVCacheAwareStrategy)
        assert weight == 1.0

    def test_b02_empty_servers_raises_value_error(self):
        """
        Feature: empty servers pool is rejected
        Description: KVCAwareBalancer({}, router_config)
        Expectation: raises ValueError
        """
        with pytest.raises(ValueError):
            KVCAwareBalancer({}, _router_config())

    def test_b02b_missing_strategies_raises_config_error(self):
        """
        Feature: a config missing strategies is rejected (delegated to from_config)
        Description: KVCAwareBalancer with an empty router_config
        Expectation: raises ConfigError
        """
        from verl.workers.rollout.router.kvcaware.config.base import ConfigError

        with pytest.raises(ConfigError):
            KVCAwareBalancer({"s0": "h0"}, OmegaConf.create({}))

    def test_b03_construction_starts_provider(self):
        """
        Feature: construction starts the provider (lifecycle)
        Description: construct balancer (autouse _FakeCollectorManager) and check start()
        Expectation: the provider's start() is invoked during __init__
        """
        balancer = KVCAwareBalancer({"s0": "h0"}, _router_config())
        assert balancer._manager.started is True


# ============================================================
# trivial methods: get_all_servers / get_status / release_server
# ============================================================


class TestTrivialMethods:
    """B04-B06: the no-algorithm Protocol methods."""

    def test_b04_get_all_servers_returns_ids(self):
        """
        Feature: get_all_servers returns the server ids in the pool
        Description: get_all_servers() on a two-server balancer
        Expectation: returns exactly the pool's ids
        """
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})
        assert set(balancer.get_all_servers()) == {"s0", "s1"}

    def test_b05_get_status_reports_construction_state(self):
        """
        Feature: get_status reports the balancer's construction state
        Description: get_status() on a freshly constructed balancer
        Expectation: reports provider type, materialized strategy, pool ids, route_calls=0
        """
        balancer = _make_balancer({"s0": "h0"})
        status = balancer.get_status()
        # provider is the injected _FakeCollectorManager in unit tests; real env reports
        # "CollectorManager". Assert it matches the constructed provider's type.
        assert status["manager"] == type(balancer._manager).__name__
        assert status["strategies"] == [{"type": "KVCacheAwareStrategy", "weight": 1.0}]
        assert status["servers"] == ["s0"]
        assert status["route_calls"] == 0

    def test_b06_release_server_is_noop(self):
        """
        Feature: release_server is a no-op (v1 does not track inflight)
        Description: release_server on a known and an unknown id
        Expectation: returns None for both; pool unchanged
        """
        balancer = _make_balancer({"s0": "h0"})
        assert balancer.release_server("s0") is None
        assert balancer.release_server("s999") is None
        assert set(balancer.get_all_servers()) == {"s0"}


# ============================================================
# acquire_server — route() delegation
# ============================================================


class TestAcquireServer:
    """B07-Bnn: acquire_server delegates to route() and maps back to a handle."""

    def test_b07_returns_top_ranked_server_and_handle(self, monkeypatch):
        """
        Feature: acquire_server returns the top-ranked server and its handle
        Description: mock route() to return ["s0","s1","s2"]
        Expectation: returns (s0, h0)
        """
        import verl.workers.rollout.router.kvcaware.balancer as balancer_mod

        monkeypatch.setattr(balancer_mod, "route", lambda *a, **k: ["s0", "s1", "s2"])
        balancer = _make_balancer({"s0": "h0", "s1": "h1", "s2": "h2"})
        assert balancer.acquire_server("r1", [1, 2, 3]) == ("s0", "h0")

    def test_b08_prompt_ids_and_replicas_passed_to_route(self, monkeypatch):
        """
        Feature: prompt_ids and pool replicas are forwarded to route()
        Description: spy on route() and capture its arguments
        Expectation: prompt_ids matches the call; replicas carry every pool id
        """
        import verl.workers.rollout.router.kvcaware.balancer as balancer_mod

        seen = {}

        def fake_route(strategies, prompt_ids, provider, replicas, *args, **kwargs):
            seen["prompt_ids"] = prompt_ids
            seen["replica_ids"] = [r.replica_id for r in replicas]
            return ["s0"]

        monkeypatch.setattr(balancer_mod, "route", fake_route)
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})
        balancer.acquire_server("r1", [7, 8, 9])
        assert seen["prompt_ids"] == [7, 8, 9]
        assert set(seen["replica_ids"]) == {"s0", "s1"}

    def test_b09_maps_returned_id_to_handle(self, monkeypatch):
        """
        Feature: the returned top id maps to its actor handle
        Description: mock route() to return ["s1","s0"]
        Expectation: returns (s1, h1) — not the first pool entry
        """
        import verl.workers.rollout.router.kvcaware.balancer as balancer_mod

        monkeypatch.setattr(balancer_mod, "route", lambda *a, **k: ["s1", "s0"])
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})
        assert balancer.acquire_server("r1", [1]) == ("s1", "h1")

    def test_b10_empty_ranking_raises_runtime_error(self, monkeypatch):
        """
        Feature: an empty ranking (no available / all blacklisted) raises
        Description: mock route() to return []
        Expectation: raises RuntimeError
        """
        import verl.workers.rollout.router.kvcaware.balancer as balancer_mod

        monkeypatch.setattr(balancer_mod, "route", lambda *a, **k: [])
        balancer = _make_balancer({"s0": "h0"})
        with pytest.raises(RuntimeError):
            balancer.acquire_server("r1", [1])

    def test_b11_none_prompt_ids_passes_through(self, monkeypatch):
        """
        Feature: prompt_ids=None is forwarded unchanged (strategy degrades to load)
        Description: acquire_server("r1", None); spy on route()
        Expectation: route receives prompt_ids is None; acquire returns normally
        """
        import verl.workers.rollout.router.kvcaware.balancer as balancer_mod

        seen = {}

        def fake_route(strategies, prompt_ids, provider, replicas, *args, **kwargs):
            seen["prompt_ids"] = prompt_ids
            return ["s0"]

        monkeypatch.setattr(balancer_mod, "route", fake_route)
        balancer = _make_balancer({"s0": "h0"})
        assert balancer.acquire_server("r1", None) == ("s0", "h0")
        assert seen["prompt_ids"] is None


# ============================================================
# add_servers / remove_servers — pool mutations
# (provider is global / not keyed by the pool, so only _servers is touched)
# ============================================================


class TestServerPoolMutations:
    """B12-Bnn: add/remove mutate the server pool."""

    def test_b12_add_servers_grows_pool(self):
        """
        Feature: add_servers grows the pool
        Description: add_servers({"s1":h1,"s2":h2}) on a one-server balancer
        Expectation: pool has s0/s1/s2
        """
        balancer = _make_balancer({"s0": "h0"})
        balancer.add_servers({"s1": "h1", "s2": "h2"})
        assert set(balancer.get_all_servers()) == {"s0", "s1", "s2"}

    def test_b13_add_servers_empty_dict_is_noop(self):
        """
        Feature: adding an empty dict changes nothing
        Description: add_servers({}) on a one-server balancer
        Expectation: pool unchanged
        """
        balancer = _make_balancer({"s0": "h0"})
        balancer.add_servers({})
        assert set(balancer.get_all_servers()) == {"s0"}

    def test_b14_add_servers_duplicate_overwrites_handle(self):
        """
        Feature: adding an existing id overwrites its handle (bulk-add semantics)
        Description: add_servers({"s0": new}) when s0 already in pool
        Expectation: handle overwritten; no error raised
        """
        balancer = _make_balancer({"s0": "h0"})
        balancer.add_servers({"s0": "h0_new"})
        assert balancer._servers["s0"] == "h0_new"

    def test_b15_remove_servers_shrinks_pool(self):
        """
        Feature: remove_servers shrinks the pool
        Description: remove_servers(["s0"]) on a two-server balancer
        Expectation: pool keeps only s1
        """
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})
        balancer.remove_servers(["s0"])
        assert set(balancer.get_all_servers()) == {"s1"}

    def test_b16_remove_servers_unknown_id_is_noop(self):
        """
        Feature: removing an unknown id changes nothing and does not raise
        Description: remove_servers(["s999"]) on a one-server balancer
        Expectation: pool unchanged
        """
        balancer = _make_balancer({"s0": "h0"})
        balancer.remove_servers(["s999"])
        assert set(balancer.get_all_servers()) == {"s0"}


# ============================================================
# 5.3A end-to-end flows (balancer directly, route() mocked)
# ============================================================


class TestEndToEndFlows:
    """B17-B18: multi-step flows over the balancer (route() mocked)."""

    def test_b17_acquire_release_acquire(self, monkeypatch):
        """
        Feature: release does not affect subsequent routing
        Description: acquire → release → acquire with route() returning s0
        Expectation: both acquires return a valid server; release is None
        """
        import verl.workers.rollout.router.kvcaware.balancer as balancer_mod

        monkeypatch.setattr(balancer_mod, "route", lambda *a, **k: ["s0"])
        balancer = _make_balancer({"s0": "h0"})
        first = balancer.acquire_server("r1", [1, 2])
        assert balancer.release_server("s0") is None
        second = balancer.acquire_server("r2", [1, 2])
        assert first == ("s0", "h0")
        assert second == ("s0", "h0")

    def test_b18_dynamic_add_remove_then_route(self, monkeypatch):
        """
        Feature: pool mutations take effect for subsequent routing
        Description: route() returns pool replicas in order; add then remove between acquires
        Expectation: after remove, the removed id is no longer routable
        """
        import verl.workers.rollout.router.kvcaware.balancer as balancer_mod

        def fake_route(strategies, prompt_ids, provider, replicas, *args, **kwargs):
            return [r.replica_id for r in replicas]

        monkeypatch.setattr(balancer_mod, "route", fake_route)
        balancer = _make_balancer({"s0": "h0"})

        balancer.add_servers({"s3": "h3"})
        assert balancer.acquire_server("r1", [1])[0] in {"s0", "s3"}

        balancer.remove_servers(["s0"])
        assert "s0" not in balancer.get_all_servers()
        assert balancer.acquire_server("r2", [1])[0] == "s3"

    def test_b19_router_class_fqn_importable(self):
        """
        Feature: the YAML router_class FQN resolves to KVCAwareBalancer
        Description: importlib.import_module + getattr on the documented FQN
        Expectation: returns the KVCAwareBalancer class (VeRL's drop-in lookup step)
        """
        import importlib

        mod = importlib.import_module("verl.workers.rollout.router.kvcaware.balancer")
        assert mod.KVCAwareBalancer is KVCAwareBalancer


# ============================================================
# Sticky-session end-to-end (real strategy, no route monkeypatch)
# ============================================================


class _MetricsProvider:
    """Metrics-aware fake provider for sticky e2e tests.

    Configured per-replica metrics; returns real KV/load numbers so the real
    KVCacheAwareStrategy can compute s_load and decide overload + stickiness.
    get_layer_prefix_hit_rate returns 0.0 → combined scoring degrades to
    load-only (no cache term), which is fine for sticky behavior: the deciding
    factor is whether the bound replica is overloaded.
    """

    def __init__(self, metrics: dict[str, dict] | None = None):
        self._metrics = metrics or {}

    def start(self):
        pass

    def stop(self):
        pass

    def get_metrics(self, replica_id):
        return dict(self._metrics.get(replica_id, {}))

    def get_metric(self, replica_id, key):
        return self.get_metrics(replica_id).get(key, 0.0)

    def get_layer_prefix_hit_rate(self, replica_id, prompt_ids, layer):
        return 0.0

    def kv_cache_load(self, replica_id):
        return 0.0


def _kv_metrics(per_replica: dict[str, dict]) -> dict[str, dict]:
    """Normalize {sid: {kv, running, waiting}} into MetricKey-keyed dicts.

    Defaults: kv=0.3 (→ load=0.12, NOT overloaded under load_threshold 0.9),
    running=0, waiting=0.
    """
    from verl.workers.rollout.router.kvcaware.types import MetricKey

    out = {}
    for sid, m in per_replica.items():
        out[sid] = {
            MetricKey.KV_CACHE_USAGE_PERC: m.get("kv", 0.3),
            MetricKey.NUM_REQUESTS_RUNNING: m.get("running", 0),
            MetricKey.NUM_REQUESTS_WAITING: m.get("waiting", 0),
        }
    return out
