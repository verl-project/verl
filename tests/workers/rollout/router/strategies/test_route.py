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

"""Tests for llm_router strategy seams (runtime strategy classes + registry).

These are the minimal collaborator seams the Balancer constructs against
(see detailed_balancer.md §2.3). Construction, registry dispatch, and routing
are tested here.
"""

from __future__ import annotations

import pytest

from verl.workers.rollout.router.kvcaware.config.strategy import KVCAwareStrategyConfig
from verl.workers.rollout.router.kvcaware.strategies import ReplicaInfo, StrategyRegistry, route
from verl.workers.rollout.router.kvcaware.strategies.kvc_aware import KVCacheAwareStrategy

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


# ============================================================
# StrategyRegistry
# ============================================================


class TestKVCacheAwareStrategy:
    """R02-Rnn: KVCacheAwareStrategy construction seam."""

    def test_r02_from_config_returns_instance_carrying_config(self):
        """
        Feature: from_config constructs a strategy instance from its config
        Description: KVCacheAwareStrategy.from_config(KVCAwareStrategyConfig(weight=0.7))
        Expectation: returns a KVCacheAwareStrategy carrying the config's strategy fields
        """
        cfg = KVCAwareStrategyConfig(weight=0.7, alpha=0.7, load_threshold=0.1, collector_names=["vllm_zmq"])
        strategy = KVCacheAwareStrategy.from_config(cfg)
        assert isinstance(strategy, KVCacheAwareStrategy)
        assert strategy.alpha == 0.7
        assert strategy.load_threshold == 0.1


# ============================================================
# ReplicaInfo
# ============================================================


class TestReplicaInfo:
    """R03: ReplicaInfo value type."""

    def test_r03_carries_only_replica_id(self):
        """
        Feature: ReplicaInfo carries a replica_id and no actor handle
        Description: construct ReplicaInfo(replica_id="s0")
        Expectation: ri.replica_id == "s0" and has no handle attribute
        """
        ri = ReplicaInfo(replica_id="s0")
        assert ri.replica_id == "s0"
        assert not hasattr(ri, "handle")


# ============================================================
# route() — real ranking
# ============================================================


# ---- Strategy registry + route (comprehensive, from test_strategy.py) ----


class TestStrategyRegistry:
    def test_builtin_registered(self):
        """
        Feature: built-in KVCacheAwareStrategy is pre-registered for KVCAwareStrategyConfig
        Description: look up KVCAwareStrategyConfig in the registry
        Expectation: returns KVCacheAwareStrategy class
        """
        from verl.workers.rollout.router.kvcaware.config.strategy import KVCAwareStrategyConfig

        assert StrategyRegistry.get(KVCAwareStrategyConfig) is KVCacheAwareStrategy

    def test_register_and_get(self):
        """
        Feature: custom strategy can be registered and retrieved by config type
        Description: register a dummy config class, then call get() with it
        Expectation: get() returns the registered strategy class
        """

        class _DummyConfig:
            pass

        class _DummyStrategy:
            def score(self, prompt_ids, provider, replicas, request_id=None, sticky_table=None):
                return [0.0] * len(replicas)

        StrategyRegistry.register(_DummyConfig, _DummyStrategy)
        try:
            assert StrategyRegistry.get(_DummyConfig) is _DummyStrategy
        finally:
            StrategyRegistry._registry.pop(_DummyConfig, None)

    def test_get_unknown_raises(self):
        """
        Feature: looking up an unregistered config type raises KeyError
        Description: call get() with a class that was never registered
        Expectation: raises KeyError
        """

        class _UnknownConfig:
            pass

        with pytest.raises(KeyError):
            StrategyRegistry.get(_UnknownConfig)


# --------------------------------------------------------------------------- #
# Test doubles for route() composition tests
# --------------------------------------------------------------------------- #


class FakeRouteDataProvider:
    def __init__(self, data=None):
        self._data = data or {}

    def get_metrics(self, replica_id):
        return {}

    def get_metric(self, replica_id, key):
        return 0.0

    def get_layer_prefix_hit_rate(self, replica_id, prompt_ids, layer):
        return 0.0

    def kv_cache_load(self, replica_id):
        return 0.0


class ConstantStrategy:
    def __init__(self, scores):
        self._scores = scores

    def score(self, prompt_ids, provider, replicas, request_id=None, sticky_table=None):
        return list(self._scores)


class BadLengthStrategy:
    def score(self, prompt_ids, provider, replicas, request_id=None, sticky_table=None):
        return [1.0]


class RaisingStrategy:
    def score(self, prompt_ids, provider, replicas, request_id=None, sticky_table=None):
        raise KeyError("boom")


def _replicas(*ids):
    return [ReplicaInfo(replica_id=rid) for rid in ids]


PROMPT_IDS = [1, 2, 3]


def _strat(**kwargs):
    defaults = dict(
        alpha=0.7,
        load_threshold=0.9,
        layer_weights={"gpu": 0.7, "cpu": 0.2, "ssd": 0.1},
        collector_names=["vllm_zmq"],
        weight=1.0,
        load_weights=(0.4, 0.2, 0.1, 0.3),
    )
    defaults.update(kwargs)
    strat = KVCacheAwareStrategy(**defaults)
    strat.set_capacity(64, 1024)
    return strat


# --------------------------------------------------------------------------- #
# route() — real ranking
# --------------------------------------------------------------------------- #


class TestRoute:
    def test_single_strategy_descending(self):
        """
        Feature: route() returns replica ids sorted by score descending
        """
        provider = FakeRouteDataProvider({})
        ranking = route(
            [(ConstantStrategy([0.2, 0.5, 0.1]), 1.0)],
            PROMPT_IDS,
            provider,
            _replicas("rep_a", "rep_b", "rep_c"),
        )
        assert ranking == ["rep_b", "rep_a", "rep_c"]

    def test_multi_strategy_weighted_sum(self):
        """
        Feature: multiple strategies are combined by weighted sum
        """
        provider = FakeRouteDataProvider({})
        strategies = [
            (ConstantStrategy([1.0, 2.0, 3.0]), 0.5),
            (ConstantStrategy([3.0, 1.0, 0.0]), 0.5),
        ]
        ranking = route(strategies, PROMPT_IDS, provider, _replicas("rep_a", "rep_b", "rep_c"))
        assert ranking[0] == "rep_a"

    def test_overloaded_present_in_ranking(self):
        """
        Feature: all replicas remain present in the ranking
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "rep_a": {
                    "kv_cache_usage_perc": 0.3,
                    "num_requests_running": 1,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 80,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "rep_b": {
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 1,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 30,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "overloaded": {
                    "kv_cache_usage_perc": 0.92,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 90,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
            }
        )
        ranking = route([(strat, 1.0)], PROMPT_IDS, provider, _replicas("rep_a", "rep_b", "overloaded"))
        assert set(ranking) == {"rep_a", "rep_b", "overloaded"}

    def test_empty_pool_raises(self):
        """
        Feature: route() raises RuntimeError when the replica list is empty
        """
        with pytest.raises(RuntimeError):
            route([(ConstantStrategy([]), 1.0)], PROMPT_IDS, FakeRouteDataProvider({}), [])

    def test_length_mismatch_falls_back_to_random(self):
        """
        Feature: route() falls back to random order when score() returns wrong-length list
        """
        ranking = route(
            [(BadLengthStrategy(), 1.0)],
            PROMPT_IDS,
            FakeRouteDataProvider({}),
            _replicas("rep_a", "rep_b"),
        )
        assert set(ranking) == {"rep_a", "rep_b"}

    def test_strategy_exception_falls_back_to_random(self):
        """
        Feature: exceptions from score() cause route() to fall back to random order
        """
        ranking = route([(RaisingStrategy(), 1.0)], PROMPT_IDS, FakeRouteDataProvider({}), _replicas("rep_a", "rep_b"))
        assert set(ranking) == {"rep_a", "rep_b"}

    def test_nan_score_ranked_last(self):
        """
        Feature: non-finite (NaN) scores are ranked last
        """
        provider = FakeRouteDataProvider({})
        ranking = route(
            [(ConstantStrategy([float("nan"), 0.1, 0.5]), 1.0)],
            PROMPT_IDS,
            provider,
            _replicas("nan_rep", "low_rep", "high_rep"),
        )
        assert ranking[0] == "high_rep"
        assert ranking[-1] == "nan_rep"


# --------------------------------------------------------------------------- #
# from_config() classmethod
# --------------------------------------------------------------------------- #
