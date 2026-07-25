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

"""Tests for PerReplicaStore.incr and DataStore sticky-session delegation.

Covers incremental writes (inflight ±1, keeping the decoder stateless) and the
DataStore facade storing sticky bindings in PerRequestStore.
"""

from __future__ import annotations

import pytest

from verl.workers.rollout.router.kvcaware.store.data_store import DataStore
from verl.workers.rollout.router.kvcaware.store.per_replica_store import PerReplicaStore
from verl.workers.rollout.router.kvcaware.store.per_request_store import PerRequestStore
from verl.workers.rollout.router.kvcaware.types import MetricKey

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


# ── PerReplicaStore.incr (plain instances — isolated, not the singleton) ──


class TestPerReplicaStoreIncr:
    def test_incr_from_default(self):
        s = PerReplicaStore()
        s.incr("n0", MetricKey.INFLIGHT_COUNT)
        assert s.get("n0", MetricKey.INFLIGHT_COUNT) == 1
        s.incr("n0", MetricKey.INFLIGHT_COUNT, 1)
        assert s.get("n0", MetricKey.INFLIGHT_COUNT) == 2

    def test_incr_negative_delta(self):
        s = PerReplicaStore()
        s.incr("n0", MetricKey.INFLIGHT_COUNT, 5)
        s.incr("n0", MetricKey.INFLIGHT_COUNT, -2)
        assert s.get("n0", MetricKey.INFLIGHT_COUNT) == 3

    def test_incr_default_delta_is_one(self):
        s = PerReplicaStore()
        s.incr("n0", MetricKey.INFLIGHT_COUNT)
        assert s.get("n0", MetricKey.INFLIGHT_COUNT) == 1

    def test_incr_isolates_nodes(self):
        s = PerReplicaStore()
        s.incr("n0", MetricKey.INFLIGHT_COUNT)
        s.incr("n1", MetricKey.INFLIGHT_COUNT, 3)
        assert s.get("n0", MetricKey.INFLIGHT_COUNT) == 1
        assert s.get("n1", MetricKey.INFLIGHT_COUNT) == 3

    def test_incr_does_not_clobber_other_keys(self):
        s = PerReplicaStore()
        s.refresh({"n0": {MetricKey.NUM_REQUESTS_RUNNING: 7}})
        s.incr("n0", MetricKey.INFLIGHT_COUNT)
        assert s.get("n0", MetricKey.NUM_REQUESTS_RUNNING) == 7
        assert s.get("n0", MetricKey.INFLIGHT_COUNT) == 1

    def test_incr_unknown_key_raises(self):
        s = PerReplicaStore()
        with pytest.raises(KeyError):
            s.incr("n0", "not_a_real_key")

    def test_incr_many_applies_all_deltas(self):
        """Batched write matches per-key incr semantics (defaults, sums, isolation)."""
        s = PerReplicaStore()
        s.incr_many(
            "n0",
            {
                MetricKey.INFLIGHT_COUNT: 1,
                MetricKey.DISPATCHED_COUNT: 1,
                MetricKey.PROMPT_LEN_SUM: 2163,
            },
        )
        assert s.get("n0", MetricKey.INFLIGHT_COUNT) == 1
        assert s.get("n0", MetricKey.DISPATCHED_COUNT) == 1
        assert s.get("n0", MetricKey.PROMPT_LEN_SUM) == 2163

    def test_incr_many_does_not_clobber_other_keys(self):
        s = PerReplicaStore()
        s.refresh({"n0": {MetricKey.NUM_REQUESTS_RUNNING: 7}})
        s.incr_many("n0", {MetricKey.INFLIGHT_COUNT: 2, MetricKey.DISPATCHED_COUNT: 2})
        assert s.get("n0", MetricKey.NUM_REQUESTS_RUNNING) == 7
        assert s.get("n0", MetricKey.INFLIGHT_COUNT) == 2

    def test_incr_many_empty_is_noop(self):
        s = PerReplicaStore()
        s.incr_many("n0", {})  # no KeyError, no node created
        assert s.get("n0", MetricKey.INFLIGHT_COUNT) == 0

    def test_incr_many_unknown_key_raises(self):
        s = PerReplicaStore()
        with pytest.raises(KeyError):
            s.incr_many("n0", {MetricKey.INFLIGHT_COUNT: 1, "not_a_real_key": 1})


# ── DataStore sticky-session delegation + inflight (singleton-backed) ──


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Isolate each test from the global PerReplicaStore / PerRequestStore singletons."""
    PerRequestStore._instance = None
    PerReplicaStore._instance = None
    yield
    PerRequestStore._instance = None
    PerReplicaStore._instance = None


class TestDataStoreStickyDelegation:
    def test_put_then_get(self):
        ds = DataStore()
        ds.put_sticky_binding("r1", "s0")
        assert ds.get_sticky_binding("r1") == "s0"

    def test_get_missing_is_none(self):
        assert DataStore().get_sticky_binding("ghost") is None

    def test_invalidate_binding(self):
        ds = DataStore()
        ds.put_sticky_binding("r1", "s0")
        ds.invalidate_sticky_binding("r1")
        assert ds.get_sticky_binding("r1") is None

    def test_invalidate_replica_clears_all_bound(self):
        ds = DataStore()
        ds.put_sticky_binding("r1", "s0")
        ds.put_sticky_binding("r2", "s1")
        ds.put_sticky_binding("r3", "s0")
        ds.invalidate_sticky_replica("s0")
        assert ds.get_sticky_binding("r1") is None
        assert ds.get_sticky_binding("r3") is None
        assert ds.get_sticky_binding("r2") == "s1"

    def test_sticky_status_reports_size(self):
        ds = DataStore()
        ds.put_sticky_binding("r1", "s0")
        ds.put_sticky_binding("r2", "s1")
        assert ds.sticky_status()["size"] == 2

    def test_put_refresh_overwrites_binding(self):
        ds = DataStore()
        ds.put_sticky_binding("r1", "s0")
        ds.put_sticky_binding("r1", "s1")  # overload-fallback re-routes
        assert ds.get_sticky_binding("r1") == "s1"

    def test_invalidate_missing_binding_is_noop(self):
        ds = DataStore()
        ds.invalidate_sticky_binding("ghost")  # must not raise
        assert ds.get_sticky_binding("ghost") is None

    def test_invalidate_replica_with_no_bindings_is_noop(self):
        ds = DataStore()
        ds.put_sticky_binding("r1", "s0")
        ds.invalidate_sticky_replica("sX")  # no binding points at sX
        assert ds.get_sticky_binding("r1") == "s0"
