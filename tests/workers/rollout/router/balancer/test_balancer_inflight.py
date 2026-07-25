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

"""Inflight + dispatch counters end-to-end with the REAL inflight_stat collector.

``collector_names`` lists ``inflight_stat``, so the patched
``_FakeCollectorManager`` builds a real ``Collector(CallbackTransport(self),
InflightDecoder)`` that registers ``on_acquire`` / ``on_release`` on the
Balancer. acquire bumps the chosen replica's INFLIGHT_COUNT (+1, mirroring verl
``GlobalRequestLoadBalancer._inflight_requests``) and DISPATCHED_COUNT, and
records the per-request turn (``PerRequestStore``) attributed to the receiving
replica's TURN_SUM; release decrements INFLIGHT and bumps COMPLETED_COUNT.
``TestInflightEndToEnd`` covers the gauge + dispatched/completed counters;
``TestTurnTracking`` covers per-request turn + per-replica turn_sum attribution.
"""

from __future__ import annotations

import pytest

from verl.workers.rollout.router.kvcaware.types import MetricKey

from ._helpers import _make_balancer

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


class TestInflightEndToEnd:
    """Real inflight_stat collector: acquire ±1, release ∓1, per-replica isolation."""

    def test_inflight_defaults_to_zero(self):
        balancer = _make_balancer({"s0": "h0"})
        assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_COUNT) == 0

    def test_acquire_increments_inflight(self):
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})
        sid, _ = balancer.acquire_server("r1", [1])
        assert balancer._store.get_metric(sid, MetricKey.INFLIGHT_COUNT) == 1

    def test_release_decrements_inflight(self):
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})
        sid, _ = balancer.acquire_server("r1", [1])
        balancer.release_server(sid)
        assert balancer._store.get_metric(sid, MetricKey.INFLIGHT_COUNT) == 0

    def test_acquire_release_symmetric_returns_to_zero(self):
        balancer = _make_balancer({"s0": "h0"})
        balancer.acquire_server("r1", [1])
        balancer.acquire_server("r2", [1])
        assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_COUNT) == 2
        balancer.release_server("s0")
        balancer.release_server("s0")
        assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_COUNT) == 0

    def test_dispatched_count_is_cumulative_per_replica(self):
        # DISPATCHED_COUNT is the monotonic sibling of the inflight gauge — it
        # only ever climbs (+1 per acquire), never decremented on release.
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})
        balancer.acquire_server("r1", [1])  # → one of s0/s1
        balancer.acquire_server("r2", [1])
        balancer.release_server("s0")
        balancer.release_server("s1")
        balancer.acquire_server("r3", [1])
        total = sum(balancer._store.get_metric(s, MetricKey.DISPATCHED_COUNT) for s in ("s0", "s1"))
        assert total == 3

    def test_completed_count_tracks_releases_per_replica(self):
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})
        sid0, _ = balancer.acquire_server("r1", [1])
        sid1, _ = balancer.acquire_server("r2", [1])
        balancer.release_server(sid0)
        balancer.release_server(sid1)
        # Cold-start tie-break may send both requests to the same replica, so
        # assert on the global tally (sum across replicas), not per-replica ==1.
        total_completed = sum(balancer._store.get_metric(s, MetricKey.COMPLETED_COUNT) for s in ("s0", "s1"))
        assert total_completed == 2
        # dispatched ≥ completed everywhere (completed is a subset of dispatched)
        for s in ("s0", "s1"):
            assert balancer._store.get_metric(s, MetricKey.DISPATCHED_COUNT) >= balancer._store.get_metric(
                s, MetricKey.COMPLETED_COUNT
            )

    def test_inflight_isolated_per_replica(self):
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})
        balancer.acquire_server("r1", [1])  # cold-start tie-break → s0
        assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_COUNT) == 1
        assert balancer._store.get_metric("s1", MetricKey.INFLIGHT_COUNT) == 0


def _total(balancer, key: str) -> int:
    """Sum a per-replica metric across both replicas (distribution-agnostic)."""
    return sum(balancer._store.get_metric(s, key) for s in ("s0", "s1"))


class TestTurnTracking:
    """Per-request turn + per-replica turn_sum, via the real inflight_stat collector."""

    def test_cold_request_starts_at_turn_one(self):
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})
        sid, _ = balancer.acquire_server("r1", [1])
        assert balancer._store.get_per_request("r1", "turn", 0) == 1
        assert balancer._store.get_metric(sid, MetricKey.TURN_SUM) == 1
        assert balancer._store.get_metric(sid, MetricKey.DISPATCHED_COUNT) == 1

    def test_re_dispatch_of_same_request_climbs_turn(self):
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})
        # route() is deterministic; re-acquiring the same request_id climbs its
        # turn regardless of which replica each dispatch lands on.
        for _ in range(3):
            balancer.acquire_server("r1", [1])  # turns 1, 2, 3
        assert balancer._store.get_per_request("r1", "turn", 0) == 3
        # turn_sum across replicas = 1+2+3 = 6 (each turn attributed to its receiver).
        assert _total(balancer, MetricKey.TURN_SUM) == 6
        assert _total(balancer, MetricKey.DISPATCHED_COUNT) == 3

    def test_distinct_requests_each_track_their_own_turn(self):
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})
        balancer.acquire_server("r1", [1])  # r1 turn 1
        balancer.acquire_server("r2", [1])  # r2 turn 1
        balancer.acquire_server("r1", [1])  # r1 turn 2
        assert balancer._store.get_per_request("r1", "turn", 0) == 2
        assert balancer._store.get_per_request("r2", "turn", 0) == 1

    def test_release_bumps_completed_count(self):
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})
        sid, _ = balancer.acquire_server("r1", [1])
        balancer.release_server(sid)
        assert balancer._store.get_metric(sid, MetricKey.COMPLETED_COUNT) == 1
        assert balancer._store.get_metric(sid, MetricKey.DISPATCHED_COUNT) == 1

    def test_turn_attributed_to_receiving_replica(self):
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})
        # r1 dispatched 3 times (turns 1,2,3) + r2 once (turn 1).
        for _ in range(3):
            balancer.acquire_server("r1", [1])
        balancer.acquire_server("r2", [1])
        # Per-request turn is global; the turn VALUES are attributed per-replica,
        # so the cross-replica turn_sum = 1+2+3+1 = 7 and dispatched = 4.
        assert _total(balancer, MetricKey.TURN_SUM) == 7
        assert _total(balancer, MetricKey.DISPATCHED_COUNT) == 4
