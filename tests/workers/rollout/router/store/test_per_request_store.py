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

"""Tests for PerRequestStore — generic per-request key/value state."""

from __future__ import annotations

import pytest

from verl.workers.rollout.router.kvcaware.store.data_store import DataStore
from verl.workers.rollout.router.kvcaware.store.per_request_store import (
    DEFAULT_PER_REQUEST_MAX_SIZE,
    PerRequestStore,
)

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


# ── PerRequestStore (plain instances — isolated, not the singleton) ──


class TestPerRequestStore:
    def test_incr_cold_starts_at_delta(self):
        s = PerRequestStore()
        assert s.incr("r1", "turn") == 1
        assert s.incr("r1", "turn", 2) == 3  # delta is arbitrary

    def test_incr_climbs(self):
        s = PerRequestStore()
        assert s.incr("r1", "turn") == 1
        assert s.incr("r1", "turn") == 2
        assert s.incr("r1", "turn") == 3

    def test_distinct_requests_isolated(self):
        s = PerRequestStore()
        s.incr("r1", "turn")
        s.incr("r1", "turn")
        s.incr("r2", "turn")
        assert s.get("r1", "turn") == 2
        assert s.get("r2", "turn") == 1

    def test_distinct_keys_within_a_request_are_independent(self):
        # A request can carry several unrelated per-request values — the store is
        # generic, not locked to one (e.g. "turn") counter.
        s = PerRequestStore()
        s.incr("r1", "turn")
        s.incr("r1", "turn")
        s.incr("r1", "retries")
        assert s.get("r1", "turn") == 2
        assert s.get("r1", "retries") == 1

    def test_set_and_get_roundtrip(self):
        s = PerRequestStore()
        s.set("r1", "first_replica", "s0")
        assert s.get("r1", "first_replica") == "s0"

    def test_get_unset_returns_default(self):
        s = PerRequestStore()
        assert s.get("ghost", "turn") is None
        assert s.get("ghost", "turn", 0) == 0

    def test_invalid_max_size_raises(self):
        with pytest.raises(ValueError):
            PerRequestStore(max_size=0)

    def test_default_max_size(self):
        assert PerRequestStore().max_size == DEFAULT_PER_REQUEST_MAX_SIZE

    def test_lru_eviction_drops_request_state(self):
        # max_size=2: touching a third distinct request evicts the LRU; the
        # evicted request's next access starts cold again.
        s = PerRequestStore(max_size=2)
        s.incr("r1", "turn")
        s.incr("r2", "turn")
        s.incr("r3", "turn")  # evicts r1 (LRU)
        assert s.incr("r1", "turn") == 1  # cold again after eviction

    def test_reset_clears_state(self):
        s = PerRequestStore()
        s.incr("r1", "turn")
        s.reset()
        assert s.get("r1", "turn", 0) == 0


# ── DataStore per-request delegation (singleton-backed) ──


@pytest.fixture(autouse=True)
def _reset_singleton():
    PerRequestStore._instance = None
    yield
    PerRequestStore._instance = None


class TestDataStorePerRequestDelegation:
    def test_incr_per_request_returns_value_via_facade(self):
        ds = DataStore()
        assert ds.incr_per_request("r1", "turn") == 1
        assert ds.incr_per_request("r1", "turn") == 2
        assert ds.get_per_request("r1", "turn", 0) == 2

    def test_get_per_request_unknown_returns_default(self):
        assert DataStore().get_per_request("ghost", "turn", 0) == 0
