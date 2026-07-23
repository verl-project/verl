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

"""Unit tests for the LLM router strategy module (strategies/ package).

Unified combined score (one pass, no fast/slow branching):
    S = α·S_cache + (1-α)·S_load
    S_cache = w_gpu·gpu_hit + w_cpu·cpu_hit + w_ssd·ssd_hit   (weights sum to 1)
    S_load  = 1 - load                                         (bigger = less loaded)
    load    = a·kv + b·min(1, running/max_num_seqs) + c·min(1, waiting/max_num_seqs)
              + d·min(1, inflight/max_num_seqs)
              (a+b+c+d=1; default 0.4/0.2/0.1/0.3; bigger = more loaded)

Overload (used only by the sticky short-circuit): ``load > load_threshold``
(default 0.9). Combined scoring never consults overload.
Default cache weights: {gpu:0.7, cpu:0.2, ssd:0.1}.
"""

from __future__ import annotations

import pytest

from verl.workers.rollout.router.kvcaware.strategies import route
from verl.workers.rollout.router.kvcaware.strategies.base import ReplicaInfo
from verl.workers.rollout.router.kvcaware.strategies.kvc_aware import (
    DEFAULT_LOAD_WEIGHTS,
    STICKY_TOP_SCORE,
    KVCacheAwareStrategy,
    StrategyError,
)
from verl.workers.rollout.router.kvcaware.strategies.routing import RoutingStrategy
from verl.workers.rollout.router.kvcaware.types import Layer, MetricKey, SlowCut

pytestmark = [pytest.mark.ut, pytest.mark.cpu]
# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _strat(**kwargs) -> KVCacheAwareStrategy:
    """Build a KVCacheAwareStrategy with required boilerplate fields filled in.

    Calls ``set_capacity(64)`` so the load formula's running/waiting terms
    are deterministic — mimics what the Balancer does after construction.
    """
    defaults = dict(
        alpha=0.7,
        load_threshold=0.9,
        layer_weights={"gpu": 0.7, "cpu": 0.2, "ssd": 0.1},
        collector_names=["vllm_zmq"],
        weight=1.0,
        # Fixed test baseline — intentionally decoupled from DEFAULT_LOAD_WEIGHTS
        # so behavior tests stay stable when the production default changes.
        load_weights=(0.4, 0.2, 0.1, 0.3),
    )
    defaults.update(kwargs)
    strat = KVCacheAwareStrategy(**defaults)
    strat.set_capacity(64)
    return strat


def _replicas(*ids: str) -> list[ReplicaInfo]:
    return [ReplicaInfo(replica_id=rid) for rid in ids]


PROMPT_IDS = [1, 2, 3]


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class FakeRouteDataProvider:
    """In-memory replica metrics for unit tests.

    Each replica entry is a plain dict with the following optional keys:
      kv_cache_usage_perc  – KV cache usage ratio (default 1.0)
      num_requests_running – requests in flight (default 0)
      num_requests_waiting – requests in the queue (default 0)
      inflight_count       – in-flight acquire/release counter (default 0)
      gpu_hit_pct          – GPU prefix cache hit percent 0-100 (default 0)
      tiers                – dict mapping tier name to hit rate (default {})
    """

    def __init__(self, data: dict[str, dict], sticky: dict[str, str] | None = None):
        self._data = data
        self._sticky = sticky or {}

    def get_sticky_binding(self, request_id: str) -> str | None:
        return self._sticky.get(request_id)

    def put_sticky_binding(self, request_id: str, replica_id: str) -> None:
        self._sticky[request_id] = replica_id

    def get_metric(self, replica_id: str, key: str) -> float | int:
        entry = self._data.get(replica_id, {})
        if key == MetricKey.KV_CACHE_USAGE_PERC:
            return entry.get("kv_cache_usage_perc", 1.0)
        if key == MetricKey.NUM_REQUESTS_RUNNING:
            return entry.get("num_requests_running", 0)
        if key == MetricKey.NUM_REQUESTS_WAITING:
            return entry.get("num_requests_waiting", 0)
        if key == MetricKey.INFLIGHT_COUNT:
            return entry.get("inflight_count", 0)
        return entry.get(key, 0.0)

    def get_metrics(self, replica_id: str) -> dict:
        entry = self._data.get(replica_id, {})
        return {
            MetricKey.KV_CACHE_USAGE_PERC: entry.get("kv_cache_usage_perc", 1.0),
            MetricKey.NUM_REQUESTS_RUNNING: entry.get("num_requests_running", 0),
            MetricKey.NUM_REQUESTS_WAITING: entry.get("num_requests_waiting", 0),
            MetricKey.INFLIGHT_COUNT: entry.get("inflight_count", 0),
        }

    def get_layer_prefix_hit_rate(self, replica_id: str, prompt_ids: list[int], layer: str) -> float:
        entry = self._data.get(replica_id, {})
        if layer == Layer.GPU:
            return entry.get("gpu_hit_pct", 0) / 100.0
        return entry.get("tiers", {}).get(layer, 0.0)

    def kv_cache_load(self, replica_id: str) -> float | None:
        # Unit tests key the load signal on kv_cache_usage_perc (no kv-events /
        # retained blocks simulated); mirror it so the load formula sees it.
        return self._data.get(replica_id, {}).get("kv_cache_usage_perc", 1.0)


class ConstantStrategy:
    """Returns a fixed per-replica score list (for route() composition tests)."""

    def __init__(self, scores: list[float]):
        self._scores = scores

    def score(self, prompt_ids, provider, replicas, request_id=None, sticky_table=None) -> list[float]:
        return list(self._scores)


class BadLengthStrategy:
    """Returns a wrong-length list to exercise the contract check in route()."""

    def score(self, prompt_ids, provider, replicas, request_id=None, sticky_table=None) -> list[float]:
        return [1.0]


class RaisingStrategy:
    """Raises inside score() to exercise route()'s exception wrapping."""

    def score(self, prompt_ids, provider, replicas, request_id=None, sticky_table=None) -> list[float]:
        raise KeyError("boom")


# --------------------------------------------------------------------------- #
# Unified combined score (one pass: α·S_cache + (1-α)·S_load)
# --------------------------------------------------------------------------- #

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


class TestKVCAwareCombinedScore:
    def test_three_layer_cache_weighted_sum(self):
        """
        Feature: S_cache is a three-layer weighted sum; load term from S_load
        Description: two light-load replicas (running=0); rep_a has gpu+cpu+ssd hits
        Expectation: scores = [0.766, 0.322]; rep_a ranks first
          rep_a: load=0.4·0.2=0.08 → s_load=0.92; s_cache=0.70; score=0.7·0.70+0.3·0.92=0.766
          rep_b: load=0.4·0.4=0.16 → s_load=0.84; s_cache=0.10; score=0.7·0.10+0.3·0.84=0.322
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "rep_a": {
                    "kv_cache_usage_perc": 0.2,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 80,
                    "tiers": {"cpu": 0.6, "ssd": 0.2},
                },
                "rep_b": {
                    "kv_cache_usage_perc": 0.4,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.3, "ssd": 0.4},
                },
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b"))
        assert scores == pytest.approx([0.766, 0.322])
        assert route([(strat, 1.0)], PROMPT_IDS, provider, _replicas("rep_a", "rep_b")) == ["rep_a", "rep_b"]

    def test_gpu_dominates_when_tiers_empty(self):
        """
        Feature: with no tier hits, S_cache = w_gpu·gpu_hit; load light
        Description: rep_a gpu_hit_pct=70; rep_b none; both running=0
        Expectation: scores = [0.619, 0.252]
          rep_a: load=0.08→s_load=0.92; s_cache=0.49; score=0.7·0.49+0.3·0.92=0.619
          rep_b: load=0.16→s_load=0.84; s_cache=0;    score=0.3·0.84=0.252
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "rep_a": {
                    "kv_cache_usage_perc": 0.2,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 70,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "rep_b": {
                    "kv_cache_usage_perc": 0.4,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b"))
        assert scores == pytest.approx([0.619, 0.252])

    def test_high_load_penalizes_but_full_formula_applied(self):
        """
        Feature: a saturated replica (load>0.9) gets the FULL formula (no zeroing);
        its s_load≈0 drags the score down despite high cache.
        Description: "loaded" kv=1,r=64,w=1000,inflight=64 (load=1.0); "light" kv=0.2,r=0 (load=0.08)
        Expectation: light outranks loaded; loaded score still reflects its cache term (not zeroed)
          loaded: waiting/inflight clamped (1000/64→1.0, 64/64→1.0) → load=0.4+0.2+0.1+0.3=1.0→s_load=0; s_cache=0.63;
                  score=0.7·0.63+0.3·0=0.441  (cache term 0.441 present despite saturation)
          light:  load=0.08→s_load=0.92; s_cache=0.35; score=0.7·0.35+0.3·0.92=0.521
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "loaded": {
                    "kv_cache_usage_perc": 1.0,
                    "num_requests_running": 64,
                    "num_requests_waiting": 1000,
                    "inflight_count": 64,
                    "gpu_hit_pct": 90,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "light": {
                    "kv_cache_usage_perc": 0.2,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 50,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("loaded", "light"))
        assert scores == pytest.approx([0.441, 0.521], abs=1e-4)
        assert scores[1] > scores[0]  # high load penalizes below light
        # loaded's score equals the full formula — cache term (0.441) is NOT zeroed
        # load=1.0 (waiting + inflight clamped) → s_load=0 → score = 0.7·0.63 + 0.3·0 = 0.441
        assert scores[0] == pytest.approx(0.7 * 0.63 + 0.3 * 0.0, abs=1e-4)

    def test_no_cache_pure_load(self):
        """
        Feature: with no cache hits, score collapses to (1-α)·s_load = (1-α)·(1-load)
        Description: idle (load=0) vs kv-full (load=0.4); both no cache, running=0
        Expectation: scores = [0.30, 0.18]
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "idle": {
                    "kv_cache_usage_perc": 0.0,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "full": {
                    "kv_cache_usage_perc": 1.0,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("idle", "full"))
        assert scores == pytest.approx([0.30, 0.18])


# --------------------------------------------------------------------------- #
# StrategyRegistry
# --------------------------------------------------------------------------- #


class TestKVCAwareLoad:
    def test_load_formula_monotonic_in_kv(self):
        """
        Feature: higher kv_usage → higher load → lower s_load → lower score
        Description: three replicas with kv 0 / 0.5 / 1.0 (running=0); no cache
        Expectation: scores decrease as kv rises
          idle:   load=0    → s_load=1.0  → score=0.30
          mid:    load=0.2  → s_load=0.8  → score=0.24
          loaded: load=0.6  → s_load=0.4  → score=0.12   (kv=1,running=64: load=0.4+0.2=0.6)
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "idle": {
                    "kv_cache_usage_perc": 0.0,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "mid": {
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "loaded": {
                    "kv_cache_usage_perc": 1.0,
                    "num_requests_running": 64,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("idle", "mid", "loaded"))
        assert scores == pytest.approx([0.30, 0.24, 0.12])
        assert scores[0] > scores[1] > scores[2]

    def test_running_increases_load(self):
        """
        Feature: running/max_num_seqs contributes to load; clamped to 1.0
        Description: kv=0.5 fixed; running 0 / 32 / 64; no cache
        Expectation: scores decrease as running rises
          r=0:  load=0.2        → s_load=0.80 → score=0.24
          r=32: load=0.2+0.1=0.3  → s_load=0.70 → score=0.21
          r=64: load=0.2+0.2=0.4  → s_load=0.60 → score=0.18
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "r0": {
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "r32": {
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 32,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "r64": {
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 64,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("r0", "r32", "r64"))
        assert scores == pytest.approx([0.24, 0.21, 0.18])
        assert scores[0] > scores[1] > scores[2]

    def test_waiting_increases_load(self):
        """
        Feature: min(1, waiting/max_num_seqs) contributes to load
        Description: kv=0,running=0; waiting 0 vs 10; no cache
        Expectation: waiting replica scores lower
          w=0:  load=0 → s_load=1.0 → score=0.30
          w=10: load=0.1·(10/64)=0.015625 → s_load=0.984375 → score=0.2953
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "w0": {
                    "kv_cache_usage_perc": 0.0,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "w10": {
                    "kv_cache_usage_perc": 0.0,
                    "num_requests_running": 0,
                    "num_requests_waiting": 10,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("w0", "w10"))
        assert scores == pytest.approx([0.30, 0.3 * (1 - 0.1 * (10 / 64))])
        assert scores[0] > scores[1]

    def test_missing_metrics_defaults_to_high_load(self):
        """
        Feature: unknown replica defaults to kv=1.0 → load=0.4 (not 1.0); no cache
        Description: score a replica whose id is absent from the provider
        Expectation: load=0.4·1.0=0.4 → s_load=0.6 → score=0.3·0.6=0.18
        """
        strat = _strat()
        provider = FakeRouteDataProvider({})
        scores = strat.score(PROMPT_IDS, provider, _replicas("ghost"))
        assert scores == pytest.approx([0.18])


# --------------------------------------------------------------------------- #
# _resolve_kv_usage: kv_cache_load drives the load formula
# --------------------------------------------------------------------------- #
class TestResolveKVUsage:
    def test_kv_cache_load_drives_load_formula(self):
        """
        Feature: _resolve_kv_usage uses kv_cache_load (not kv_cache_usage_perc)
        Description: data kv_cache_usage_perc=0.9 but kv_cache_load=0.1
        Expectation: load uses kv_cache_load (0.1): load=0.4·0.1=0.04, s_load=0.96,
                     s_cache=0 → score=0.3·0.96=0.288 (not 0.192 from kv=0.9)
        """

        class _LoadProvider(FakeRouteDataProvider):
            def __init__(self, data, load):
                super().__init__(data)
                self._load = load

            def kv_cache_load(self, replica_id):
                return self._load.get(replica_id)

        strat = _strat()
        provider = _LoadProvider(
            {"rep": {"kv_cache_usage_perc": 0.9, "num_requests_running": 0, "num_requests_waiting": 0}},
            {"rep": 0.1},
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep"))
        assert scores == pytest.approx([0.288])

    def test_kv_cache_load_zero_drives_zero_load(self):
        """
        Feature: kv_cache_load 0.0 (retained unavailable) → load=0
        Description: kv_cache_load returns 0.0 (no kv-events / retained blocks)
        Expectation: kv=0 → load=0 → s_load=1.0, s_cache=0 → score=0.3·1.0=0.3
        """

        class _ZeroLoadProvider(FakeRouteDataProvider):
            def kv_cache_load(self, replica_id):
                return 0.0

        strat = _strat()
        provider = _ZeroLoadProvider(
            {"rep": {"kv_cache_usage_perc": 0.5, "num_requests_running": 0, "num_requests_waiting": 0}}
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep"))
        assert scores == pytest.approx([0.3])


# --------------------------------------------------------------------------- #
# _cache_score: three-layer weighted hit (gpu + cpu + ssd)
# --------------------------------------------------------------------------- #
class TestKVCAwareCacheScore:
    def test_three_layer_weighted_sum(self):
        """
        Feature: _cache_score = w_gpu·gpu + w_cpu·cpu + w_ssd·ssd
        Description: gpu_hit_pct=80, cpu=0.6, ssd=0.2 with default weights
        Expectation: 0.7*0.8 + 0.2*0.6 + 0.1*0.2 = 0.70
        """
        strat = _strat()
        provider = FakeRouteDataProvider({"rep": {"gpu_hit_pct": 80, "tiers": {"cpu": 0.6, "ssd": 0.2}}})
        s_cache, _ = strat._cache_score(provider, ReplicaInfo("rep"), PROMPT_IDS)
        assert s_cache == pytest.approx(0.70)

    def test_gpu_only_when_tier_none(self):
        """
        Feature: None tier hit rate is treated as 0.0
        Description: provider returns None for tiers (mooncake placeholder); gpu_hit_pct=80
        Expectation: 0.7*0.8 + 0 + 0 = 0.56
        """

        class _NoneProvider(FakeRouteDataProvider):
            def get_layer_prefix_hit_rate(self, replica_id, prompt_ids, layer):
                if layer == Layer.GPU:
                    return super().get_layer_prefix_hit_rate(replica_id, prompt_ids, layer)
                return None

        strat = _strat()
        provider = _NoneProvider({"rep": {"gpu_hit_pct": 80, "tiers": {}}})
        s_cache, _ = strat._cache_score(provider, ReplicaInfo("rep"), PROMPT_IDS)
        assert s_cache == pytest.approx(0.56)

    def test_no_hit_returns_zero(self):
        """
        Feature: no gpu hit and no tier hits → _cache_score = 0.0
        Description: replica absent from gpu_hit_pct; tiers all 0
        Expectation: 0.0
        """
        strat = _strat()
        provider = FakeRouteDataProvider({"rep": {"tiers": {"cpu": 0.0, "ssd": 0.0}}})
        s_cache, _ = strat._cache_score(provider, ReplicaInfo("rep"), PROMPT_IDS)
        assert s_cache == pytest.approx(0.0)

    def test_custom_weights_respected(self):
        """
        Feature: _cache_score honors custom layer_weights
        Description: weights {gpu:0.5,cpu:0.3,ssd:0.2}; all hits = 1.0 (gpu_hit_pct=100)
        Expectation: 0.5 + 0.3 + 0.2 = 1.0
        """
        strat = _strat(layer_weights={"gpu": 0.5, "cpu": 0.3, "ssd": 0.2})
        provider = FakeRouteDataProvider({"rep": {"gpu_hit_pct": 100, "tiers": {"cpu": 1.0, "ssd": 1.0}}})
        s_cache, _ = strat._cache_score(provider, ReplicaInfo("rep"), PROMPT_IDS)
        assert s_cache == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Tier weights in the cache term
# --------------------------------------------------------------------------- #
class TestKVCAwareTierWeights:
    def test_cpu_weight_higher_than_ssd(self):
        """
        Feature: cpu tier weight (0.2) > ssd tier weight (0.1) in the cache term
        Description: two light-load replicas; one has cpu hit, other ssd hit
        Expectation: cpu-hit replica scores higher
          cpu_hit: load=0.2→s_load=0.8; s_cache=0.2·0.6=0.12; score=0.7·0.12+0.3·0.8=0.324
          ssd_hit: load=0.2→s_load=0.8; s_cache=0.1·0.8=0.08; score=0.7·0.08+0.3·0.8=0.296
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "cpu_hit": {
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.6, "ssd": 0.0},
                },
                "ssd_hit": {
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.8},
                },
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("cpu_hit", "ssd_hit"))
        assert scores == pytest.approx([0.324, 0.296])
        assert scores[0] > scores[1]

    def test_tier_none_treated_as_zero(self):
        """
        Feature: None return from get_layer_prefix_hit_rate is treated as 0.0
        Description: provider returns None for cpu/ssd layer hit rate
        Expectation: score = (1-α)·s_load (S_cache=0), no TypeError
          rep: load=0.2→s_load=0.8; s_cache=0; score=0.3·0.8=0.24
        """

        class _NoneProvider(FakeRouteDataProvider):
            def get_layer_prefix_hit_rate(self, replica_id, prompt_ids, layer):
                if layer == Layer.GPU:
                    return super().get_layer_prefix_hit_rate(replica_id, prompt_ids, layer)
                return None

        strat = _strat()
        provider = _NoneProvider(
            {
                "rep": {
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {},
                }
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep"))
        assert scores == pytest.approx([0.24])


# --------------------------------------------------------------------------- #
# Construction validation
# --------------------------------------------------------------------------- #
class TestKVCAwareConstruction:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"alpha": 1.5},
            {"alpha": -0.1},
            {"load_threshold": 0},
            {"load_threshold": 1.0},
            {"layer_weights": {"gpu": 0.7, "cpu": 0.2, "ssd": -0.1}},
            {"layer_weights": {"nvme": 1.0}},
            {"layer_weights": {"gpu": 1.0, "cpu": 0.2, "ssd": 0.1}},
            {"layer_weights": {"gpu": 0.7, "cpu": 0.3}},
            {"load_weights": (0.5, 0.3)},
            {"load_weights": (0.5, 0.5, 0.5)},
            {"load_weights": (-0.1, 0.6, 0.5, 0.0)},
        ],
    )
    def test_invalid_construction_raises(self, kwargs):
        """
        Feature: invalid constructor arguments raise StrategyError
        Description: construct KVCacheAwareStrategy with each invalid kwarg
        Expectation: raises StrategyError for each case
        """
        with pytest.raises(StrategyError):
            _strat(**kwargs)

    def test_valid_three_key_weights_accepted(self):
        strat = _strat(layer_weights={"gpu": 0.5, "cpu": 0.3, "ssd": 0.2})
        assert strat.layer_weights == {"gpu": 0.5, "cpu": 0.3, "ssd": 0.2}


# --------------------------------------------------------------------------- #
# set_capacity
# --------------------------------------------------------------------------- #
class TestSetCapacity:
    def test_set_capacity_updates_max_num_seqs(self):
        strat = KVCacheAwareStrategy(
            alpha=0.7,
            load_threshold=0.9,
            layer_weights={"gpu": 0.7, "cpu": 0.2, "ssd": 0.1},
            collector_names=["vllm_zmq"],
            weight=1.0,
        )
        strat.set_capacity(16)
        assert strat._max_num_seqs == 16

    def test_set_capacity_rejects_zero(self):
        strat = KVCacheAwareStrategy(
            alpha=0.7,
            load_threshold=0.9,
            layer_weights={"gpu": 0.7, "cpu": 0.2, "ssd": 0.1},
            collector_names=["vllm_zmq"],
            weight=1.0,
        )
        with pytest.raises(StrategyError):
            strat.set_capacity(0)

    def test_set_capacity_rejects_negative(self):
        strat = KVCacheAwareStrategy(
            alpha=0.7,
            load_threshold=0.9,
            layer_weights={"gpu": 0.7, "cpu": 0.2, "ssd": 0.1},
            collector_names=["vllm_zmq"],
            weight=1.0,
        )
        with pytest.raises(StrategyError):
            strat.set_capacity(-1)

    def test_compute_load_raises_before_set_capacity(self):
        strat = KVCacheAwareStrategy(
            alpha=0.7,
            load_threshold=0.9,
            layer_weights={"gpu": 0.7, "cpu": 0.2, "ssd": 0.1},
            collector_names=["vllm_zmq"],
            weight=1.0,
        )
        with pytest.raises(StrategyError, match="set_capacity"):
            strat._compute_load(0.5, 0, 0)


# --------------------------------------------------------------------------- #
# Interface contract
# --------------------------------------------------------------------------- #
class TestStrategyContract:
    def test_protocol_satisfied(self):
        """
        Feature: KVCacheAwareStrategy satisfies the RoutingStrategy Protocol
        """
        strat = _strat()
        assert isinstance(strat, RoutingStrategy)

    def test_output_length_matches_replicas(self):
        """
        Feature: score() returns a list with same length as replicas
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "rep_a": {
                    "kv_cache_usage_perc": 0.3,
                    "num_requests_running": 1,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 90,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "rep_b": {
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 2,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 10,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
            }
        )
        replicas = _replicas("rep_a", "rep_b")
        scores = strat.score(PROMPT_IDS, provider, replicas)
        assert len(scores) == len(replicas)

    def test_stateless_repeatable(self):
        """
        Feature: calling score() twice on the same inputs produces identical results
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
                    "num_requests_running": 2,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.5, "ssd": 0.0},
                },
            }
        )
        replicas = _replicas("rep_a", "rep_b")
        assert strat.score(PROMPT_IDS, provider, replicas) == pytest.approx(strat.score(PROMPT_IDS, provider, replicas))


# --------------------------------------------------------------------------- #
# route() composition
# --------------------------------------------------------------------------- #


class TestFromConfig:
    def test_from_config_correct_fields(self):
        from verl.workers.rollout.router.kvcaware.config.strategy import KVCAwareStrategyConfig

        cfg = KVCAwareStrategyConfig(
            alpha=0.6,
            load_threshold=0.85,
            layer_weights={"gpu": 0.6, "cpu": 0.3, "ssd": 0.1},
            weight=0.9,
            collector_names=["vllm_zmq"],
        )
        strat = KVCacheAwareStrategy.from_config(cfg)
        assert strat.alpha == pytest.approx(0.6)
        assert strat.load_threshold == pytest.approx(0.85)
        assert strat.layer_weights == {"gpu": 0.6, "cpu": 0.3, "ssd": 0.1}
        assert strat._max_num_seqs is None  # not set until set_capacity()
        assert strat.load_weights == (0.5, 0.0, 0.0, 0.5)

    def test_from_config_scores_match_direct(self):
        from verl.workers.rollout.router.kvcaware.config.strategy import KVCAwareStrategyConfig

        cfg = KVCAwareStrategyConfig(
            alpha=0.7,
            load_threshold=0.9,
            layer_weights={"gpu": 0.7, "cpu": 0.2, "ssd": 0.1},
            weight=1.0,
            collector_names=["vllm_zmq"],
        )
        strat_from_cfg = KVCacheAwareStrategy.from_config(cfg)
        strat_from_cfg.set_capacity(64)
        # Align load_weights with from_config (which lands on DEFAULT_LOAD_WEIGHTS);
        # _strat()'s own baseline differs, so override here for apples-to-apples.
        strat_direct = _strat(load_weights=DEFAULT_LOAD_WEIGHTS)
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
                    "kv_cache_usage_perc": 0.92,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
            }
        )
        replicas = _replicas("rep_a", "rep_b")
        assert strat_from_cfg.score(PROMPT_IDS, provider, replicas) == pytest.approx(
            strat_direct.score(PROMPT_IDS, provider, replicas)
        )


# --------------------------------------------------------------------------- #
# Sticky-session short-circuit (is_overloaded uses load > load_threshold)
# --------------------------------------------------------------------------- #
class TestStickyShortCircuit:
    """Sticky replica wins when bound + present + not overloaded; else fall through.

    Overload now means ``load > load_threshold`` (default 0.9) — i.e. the bound
    replica is genuinely saturated (kv≈1, running≈max_num_seqs, big backlog). With
    the default four-term load weights (0.4/0.2/0.1/0.3) the kv+running+waiting
    terms cap at 0.7, so the inflight term (weight 0.3) must be >0 to push load
    past 0.9; the "saturated" cases below therefore feed inflight=max_num_seqs.
    """

    def _provider(self, sticky=None, **per_replica):
        """Build a FakeRouteDataProvider from {rep_id: metrics_dict} + optional sticky."""
        return FakeRouteDataProvider(per_replica, sticky=sticky)

    # ── is_overloaded ──────────────────────────────────────────────────────
    def test_is_overloaded_true_when_saturated(self):
        """Feature: is_overloaded True when load > load_threshold (0.9).
        Description: kv=1.0, running=64 (mns), waiting=1000, inflight=64 → load=1.0 > 0.9
        Expectation: overloaded
        """
        strat = _strat(load_threshold=0.9)
        provider = self._provider(
            rep_a={
                "kv_cache_usage_perc": 1.0,
                "num_requests_running": 64,
                "num_requests_waiting": 1000,
                "inflight_count": 64,
            }
        )
        assert strat.is_overloaded(provider, ReplicaInfo("rep_a")) is True

    def test_is_overloaded_false_when_light(self):
        """Feature: is_overloaded False when load <= load_threshold.
        Description: kv=0.3, running=0, waiting=0 → load=0.12 < 0.9
        Expectation: not overloaded
        """
        strat = _strat(load_threshold=0.9)
        provider = self._provider(rep_a={"kv_cache_usage_perc": 0.3, "num_requests_running": 0})
        assert strat.is_overloaded(provider, ReplicaInfo("rep_a")) is False

    # ── score() sticky short-circuit ───────────────────────────────────────
    def test_sticky_hit_not_overloaded_short_circuits(self):
        """Feature: bound + present + not overloaded → sticky replica gets top score.
        Description: sticky binds r1→rep_b; rep_b light (load=0.12); rep_a has better
        combined score but must NOT win.
        Expectation: scores = [0.0, STICKY_TOP_SCORE]; route() picks rep_b
        """
        strat = _strat(load_threshold=0.9)
        provider = self._provider(
            sticky={"r1": "rep_b"},
            rep_a={"kv_cache_usage_perc": 0.2, "num_requests_running": 0, "gpu_hit_pct": 80},
            rep_b={"kv_cache_usage_perc": 0.3, "num_requests_running": 0, "gpu_hit_pct": 0},
        )
        replicas = _replicas("rep_a", "rep_b")
        scores = strat.score(PROMPT_IDS, provider, replicas, "r1")
        assert scores == [0.0, STICKY_TOP_SCORE]
        ranking = route([(strat, 1.0)], PROMPT_IDS, provider, replicas, "r1")
        assert ranking[0] == "rep_b"

    def test_sticky_hit_overloaded_falls_back_to_combined(self):
        """Feature: bound but saturated (load>0.9) → no short-circuit, combined scoring.
        Description: sticky binds r1→rep_b; rep_b saturated (kv=1,r=64,w=1000,inflight=64 → load=1.0);
        rep_a light with gpu hit.
        Expectation: rep_a wins (combined), not the saturated sticky rep_b
        """
        strat = _strat(load_threshold=0.9)
        provider = self._provider(
            sticky={"r1": "rep_b"},
            rep_a={"kv_cache_usage_perc": 0.2, "num_requests_running": 0, "gpu_hit_pct": 80},
            rep_b={
                "kv_cache_usage_perc": 1.0,
                "num_requests_running": 64,
                "num_requests_waiting": 1000,
                "inflight_count": 64,
                "gpu_hit_pct": 0,
            },
        )
        replicas = _replicas("rep_a", "rep_b")
        ranking = route([(strat, 1.0)], PROMPT_IDS, provider, replicas, "r1")
        assert ranking[0] == "rep_a"

    def test_sticky_no_binding_cold_start_combined(self):
        """Feature: no sticky binding → combined scoring (cold start).
        Expectation: best combined replica wins (rep_a)
        """
        strat = _strat(load_threshold=0.9)
        provider = self._provider(
            rep_a={"kv_cache_usage_perc": 0.2, "num_requests_running": 0, "gpu_hit_pct": 80},
            rep_b={"kv_cache_usage_perc": 0.3, "num_requests_running": 0, "gpu_hit_pct": 0},
        )
        replicas = _replicas("rep_a", "rep_b")
        ranking = route([(strat, 1.0)], PROMPT_IDS, provider, replicas, "r1")
        assert ranking[0] == "rep_a"

    def test_sticky_bound_replica_removed_falls_back(self):
        """Feature: bound replica no longer in pool → fall back to combined.
        Expectation: rep_a wins (combined), no KeyError/crash
        """
        strat = _strat(load_threshold=0.9)
        provider = self._provider(
            sticky={"r1": "rep_gone"},  # bound replica not in pool
            rep_a={"kv_cache_usage_perc": 0.2, "num_requests_running": 0, "gpu_hit_pct": 80},
            rep_b={"kv_cache_usage_perc": 0.3, "num_requests_running": 0, "gpu_hit_pct": 0},
        )
        replicas = _replicas("rep_a", "rep_b")
        ranking = route([(strat, 1.0)], PROMPT_IDS, provider, replicas, "r1")
        assert ranking[0] == "rep_a"

    def test_sticky_none_request_id_combined(self):
        """Feature: request_id=None → combined scoring (no sticky lookup).
        Expectation: combined scoring, rep_a wins
        """
        strat = _strat(load_threshold=0.9)
        provider = self._provider(
            rep_a={"kv_cache_usage_perc": 0.2, "num_requests_running": 0, "gpu_hit_pct": 80},
            rep_b={"kv_cache_usage_perc": 0.3, "num_requests_running": 0, "gpu_hit_pct": 0},
        )
        replicas = _replicas("rep_a", "rep_b")
        ranking = route([(strat, 1.0)], PROMPT_IDS, provider, replicas, None)
        assert ranking[0] == "rep_a"


# --------------------------------------------------------------------------- #
# _compute_load (load formula) — each term exercised with an explicit weight
# vector, decoupled from DEFAULT_LOAD_WEIGHTS so config changes don't erode
# the formula-coverage assertions.
# --------------------------------------------------------------------------- #
class TestLoadFormula:
    def test_idle_replica_is_zero(self):
        assert _strat()._compute_load(0.0, 0, 0, 0) == pytest.approx(0.0)

    def test_kv_only_contribution(self):
        s = _strat(load_weights=(1.0, 0.0, 0.0, 0.0))
        assert s._compute_load(0.5, 0, 0, 0) == pytest.approx(0.5)

    def test_running_and_kv(self):
        s = _strat(load_weights=(0.5, 0.5, 0.0, 0.0))
        assert s._compute_load(0.5, 32, 0, 0) == pytest.approx(0.5)

    def test_running_clamped_to_one(self):
        s = _strat(load_weights=(0.0, 1.0, 0.0, 0.0))
        assert s._compute_load(0.8, 128, 0, 0) == pytest.approx(1.0)

    def test_waiting_term(self):
        s = _strat(load_weights=(0.0, 0.0, 1.0, 0.0))
        assert s._compute_load(0.0, 0, 10, 0) == pytest.approx(10 / 64)

    def test_inflight_term(self):
        s = _strat(load_weights=(0.0, 0.0, 0.0, 1.0))
        assert s._compute_load(0.0, 0, 0, 32) == pytest.approx(0.5)

    def test_custom_weights_change_load(self):
        strat = _strat(load_weights=(0.6, 0.2, 0.2, 0.0))
        load = strat._compute_load(0.5, 32, 0, 0)
        assert load == pytest.approx(0.4)
        assert load != pytest.approx(0.35)

    def test_near_saturated_exceeds_threshold(self):
        load = _strat()._compute_load(1.0, 64, 1000, 64)
        assert load > 0.9
        assert load <= 1.0


class TestDefaultWeights:
    def test_default_weights_tuple(self):
        assert DEFAULT_LOAD_WEIGHTS == (0.5, 0.0, 0.0, 0.5)
        assert sum(DEFAULT_LOAD_WEIGHTS) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Fallback modes: memory_overload_filter (sticky overload gate) + slow_cut (fallback scoring)
# --------------------------------------------------------------------------- #
class TestFallbackModes:
    """The two formerly-coupled ``USE_VERL_STICKY`` behaviors are now independent
    config knobs: ``memory_overload_filter`` gates the sticky overload check, and
    ``slow_cut`` selects the fallback scoring (``least-inflight`` mirrors verl
    GlobalRequestLoadBalancer)."""

    def test_sticky_hit_ignores_overload(self):
        """memory_overload_filter=False: bound replica wins even when saturated."""
        strat = _strat(load_threshold=0.9, memory_overload_filter=False)
        provider = FakeRouteDataProvider(
            {
                "rep_a": {"kv_cache_usage_perc": 1.0, "num_requests_running": 64, "num_requests_waiting": 1000},
                "rep_b": {"kv_cache_usage_perc": 0.3},
            },
            sticky={"r1": "rep_a"},
        )
        ranking = route([(strat, 1.0)], PROMPT_IDS, provider, _replicas("rep_a", "rep_b"), "r1")
        assert ranking[0] == "rep_a"  # sticky wins despite load≈0.7 (overload check disabled)

    def test_miss_routes_to_least_inflight(self):
        """slow_cut=least-inflight: pick the replica with the fewest in-flight requests."""
        strat = _strat(slow_cut=SlowCut.LEAST_INFLIGHT)
        provider = FakeRouteDataProvider(
            {"rep_a": {"inflight_count": 5}, "rep_b": {"inflight_count": 2}},
        )
        ranking = route([(strat, 1.0)], PROMPT_IDS, provider, _replicas("rep_a", "rep_b"), "r1")
        assert ranking[0] == "rep_b"

    def test_stale_binding_falls_back_to_least_inflight(self):
        """slow_cut=least-inflight: bound replica no longer in pool → fallback."""
        strat = _strat(slow_cut=SlowCut.LEAST_INFLIGHT)
        provider = FakeRouteDataProvider(
            {"rep_a": {"inflight_count": 5}, "rep_b": {"inflight_count": 1}},
            sticky={"r1": "rep_gone"},
        )
        ranking = route([(strat, 1.0)], PROMPT_IDS, provider, _replicas("rep_a", "rep_b"), "r1")
        assert ranking[0] == "rep_b"

    def test_inflight_tie_keeps_pool_order(self):
        """slow_cut=least-inflight tie-break: equal inflight → first replica in pool order."""
        strat = _strat(slow_cut=SlowCut.LEAST_INFLIGHT)
        provider = FakeRouteDataProvider(
            {"rep_a": {"inflight_count": 3}, "rep_b": {"inflight_count": 3}},
        )
        ranking = route([(strat, 1.0)], PROMPT_IDS, provider, _replicas("rep_a", "rep_b"), "r1")
        assert ranking[0] == "rep_a"
