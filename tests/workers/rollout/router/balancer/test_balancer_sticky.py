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

"""Sticky-session end-to-end with the REAL sticky_stat collector (no mocking).

``_router_config`` lists ``sticky_stat`` in ``collector_names``, so the patched
``_FakeCollectorManager`` builds a real ``Collector(CallbackTransport(self),
StickyDecoder)`` that registers ``on_acquire`` / ``on_servers_removed`` on the
Balancer — exactly the production path. The test does NOT register callbacks by
hand; it only injects metrics into the real ``DataStore`` and asserts routing
behaviour. This is the test that proves the Phase-1 statistic chain works
outside the unit-test seam.
"""

from __future__ import annotations

import pytest

from verl.workers.rollout.router.kvcaware.balancer import KVCAwareBalancer
from verl.workers.rollout.router.kvcaware.types import MetricKey

from ._helpers import (
    _router_config,
)

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


def _kv_metrics(per_replica: dict[str, dict]) -> dict[str, dict]:
    """Normalize {sid: {kv, running, waiting, inflight}} into MetricKey-keyed dicts.

    Defaults: kv=0.3, running=0, waiting=0, inflight=0 (light load). NOTE: ``kv``
    here sets ``KV_CACHE_USAGE_PERC``; the load formula actually uses
    ``kv_cache_load`` (retained_blocks/num_gpu_blocks), which tests drive via
    ``add_kv_blocks`` + ``NUM_GPU_BLOCKS`` when they need overload. ``inflight``
    maps to ``INFLIGHT_COUNT`` — the load formula's 4th term (weight 0.3); with
    inflight=0 the kv+running+waiting ceiling is 0.7, so it must be >0 to push
    load past ``load_threshold`` (0.9).
    """
    out = {}
    for sid, m in per_replica.items():
        out[sid] = {
            MetricKey.KV_CACHE_USAGE_PERC: m.get("kv", 0.3),
            MetricKey.NUM_REQUESTS_RUNNING: m.get("running", 0),
            MetricKey.NUM_REQUESTS_WAITING: m.get("waiting", 0),
            MetricKey.INFLIGHT_COUNT: m.get("inflight", 0),
        }
    return out


class TestStickyEndToEnd:
    """Real KVCacheAwareStrategy + real route() + real sticky_stat collector."""

    def _make_balancer(self, servers, metrics):
        """Build a balancer and seed its real DataStore with per-replica metrics."""
        balancer = KVCAwareBalancer(servers, _router_config())
        # capacity=64 matches the overload scenario (running=64 → load=1.0).
        for strategy, _ in balancer._strategies:
            if hasattr(strategy, "set_capacity"):
                strategy.set_capacity(64, 1024)
        balancer._store.refresh_metrics(metrics)
        return balancer

    def test_second_turn_same_request_stays_sticky(self):
        """Feature: bound + not overloaded → second turn routes to same server.
        Description: turn1 acquires (cold start, tie-break picks s0); turn2 same
        request_id with that server still healthy must return the SAME server.
        Expectation: turn1.sid == turn2.sid (sticky hit via real sticky_stat collector)
        """
        balancer = self._make_balancer(
            {"s0": "h0", "s1": "h1"},
            _kv_metrics({"s0": {}, "s1": {}}),
        )
        sid1, _ = balancer.acquire_server("r1", [1, 2])
        sid2, _ = balancer.acquire_server("r1", [1, 2])
        assert sid1 == sid2

    def test_overloaded_sticky_falls_back_to_healthy(self):
        """Feature: bound replica becomes saturated (load>0.9) → rebind to a healthy one.
        Description: turn1 binds r1→s0; then s0 saturated so load exceeds
        ``load_threshold`` (0.9). load = 0.4·kv_load + 0.2·running/max +
        0.1·waiting/max + 0.3·inflight/max with ``max_num_seqs=64``; to push past
        0.9 every weighted term must contribute — in particular inflight (weight
        0.3) must be >0, since with inflight=0 the kv+running+waiting ceiling is
        0.7. Here s0: kv_load=1.0 (retained=10/num_gpu_blocks=10), running=64,
        waiting=1000, inflight=64 → load=1.0; s1 healthy (kv_load=0 → load=0).
        Expectation: turn2 routes to s1 (not the saturated s0), and rebinds r1→s1
        """
        balancer = self._make_balancer(
            {"s0": "h0", "s1": "h1"},
            _kv_metrics({"s0": {}, "s1": {}}),
        )
        sid1, _ = balancer.acquire_server("r1", [1, 2])
        # saturate s0: kv_load=1.0 (retained/num_gpu_blocks) + running/waiting/
        # inflight all maxed so load (0.4·kv+0.2·run+0.1·wait+0.3·inflight) = 1.0
        balancer._store.refresh_metrics(
            _kv_metrics({"s0": {"kv": 1.0, "running": 64, "waiting": 1000, "inflight": 64}, "s1": {"kv": 0.3}})
        )
        balancer._store.add_kv_blocks("s0", [f"b{i}" for i in range(10)])
        balancer._store.refresh_metrics({"s0": {MetricKey.NUM_GPU_BLOCKS: 10}})
        sid2, _ = balancer.acquire_server("r1", [1, 2])
        assert sid2 == "s1", f"expected fallback to s1, got {sid2} (turn1 was {sid1})"

    def test_removed_sticky_server_reselects(self):
        """Feature: bound server removed → reselect from remaining pool.
        Description: turn1 binds r1→s0; remove s0 (fires on_servers_removed →
        sticky invalidate_replica); turn2 must pick from {s1,s2}.
        Expectation: turn2 sid in {s1,s2}, no crash, sticky rebound
        """
        balancer = self._make_balancer(
            {"s0": "h0", "s1": "h1", "s2": "h2"},
            _kv_metrics({"s0": {}, "s1": {}, "s2": {}}),
        )
        balancer.acquire_server("r1", [1, 2])
        balancer.remove_servers(["s0"])
        sid2, _ = balancer.acquire_server("r1", [1, 2])
        assert sid2 in {"s1", "s2"}
        assert balancer._store.get_sticky_binding("r1") in {"s1", "s2"}

    def test_get_status_reports_sticky_size(self):
        """Feature: get_status() includes sticky_size.
        Description: acquire two distinct request_ids; check status
        Expectation: sticky_size == 2
        """
        balancer = self._make_balancer(
            {"s0": "h0", "s1": "h1"},
            _kv_metrics({"s0": {}, "s1": {}}),
        )
        balancer.acquire_server("r1", [1])
        balancer.acquire_server("r2", [1])
        assert balancer.get_status()["sticky_size"] == 2
