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

"""Tests for the Balancer-callback statistic path.

Covers the Phase-1 components that turn Balancer callbacks into store writes:
``StatisticEvent`` (pack contract), ``StickyDecoder`` / ``InflightDecoder``
(decoders), and ``CallbackTransport`` (the pure-forwarder transport that
registers on the Balancer). Together these mirror what the network collectors
do, but driven by the Balancer's own request-path hooks.
"""

from __future__ import annotations

import asyncio

import pytest

from verl.workers.rollout.router.kvcaware.collectors.decoder import MetricsUpdate, StickyUpdate
from verl.workers.rollout.router.kvcaware.collectors.decoder.basic.inflight import InflightDecoder
from verl.workers.rollout.router.kvcaware.collectors.decoder.basic.sticky import StickyDecoder
from verl.workers.rollout.router.kvcaware.collectors.transport.callback import (
    CallbackTransport,
    StatisticEvent,
)
from verl.workers.rollout.router.kvcaware.types import MetricKey

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


class TestStatisticEvent:
    def test_defaults_and_fields(self):
        ev = StatisticEvent("on_acquire", request_id="r1", replica_id="s0")
        assert ev.event == "on_acquire"
        assert ev.request_id == "r1"
        assert ev.replica_id == "s0"
        assert ev.server_ids == ()
        assert ev.prompt_len == 0  # default — no prompt forwarded

    def test_frozen(self):
        ev = StatisticEvent("on_acquire")
        with pytest.raises(AttributeError):
            ev.event = "x"  # type: ignore[misc]

    def test_on_servers_removed_carries_tuple(self):
        ev = StatisticEvent("on_servers_removed", server_ids=("s0", "s1"))
        assert ev.server_ids == ("s0", "s1")


class TestStickyDecoder:
    def test_on_acquire_emits_put(self):
        upd = StickyDecoder().decode(StatisticEvent("on_acquire", request_id="r1", replica_id="s0"), "")
        assert isinstance(upd, StickyUpdate)
        assert upd.action == "put"
        assert upd.request_id == "r1"
        assert upd.replica_id == "s0"

    def test_on_servers_removed_emits_invalidate_replica(self):
        upd = StickyDecoder().decode(StatisticEvent("on_servers_removed", server_ids=["s0", "s1"]), "")
        assert isinstance(upd, StickyUpdate)
        assert upd.action == "invalidate_replica"
        assert upd.replica_ids == ("s0", "s1")

    def test_on_release_returns_none(self):
        assert StickyDecoder().decode(StatisticEvent("on_release", replica_id="s0"), "") is None

    def test_on_acquire_missing_fields_returns_none(self):
        assert StickyDecoder().decode(StatisticEvent("on_acquire"), "") is None

    def test_non_event_payload_returns_none(self):
        d = StickyDecoder()
        assert d.decode(b"bytes", "") is None
        assert d.decode("str", "") is None


class TestInflightDecoder:
    def test_on_acquire_emits_inflight_plus_dispatched_delta(self):
        upd = InflightDecoder().decode(StatisticEvent("on_acquire", request_id="r1", replica_id="s0"), "")
        assert isinstance(upd, MetricsUpdate)
        assert upd.node_id == "s0"
        assert upd.metrics == {
            MetricKey.INFLIGHT_COUNT: 1,
            MetricKey.INFLIGHT_TOKENS: 0,  # no prompt forwarded → 0 token delta
            MetricKey.DISPATCHED_COUNT: 1,
            MetricKey.PROMPT_LEN_SUM: 0,  # no prompt forwarded → 0 length delta
        }
        assert upd.is_delta is True
        assert upd.request_id == "r1"  # carried so the collector attributes the dispatch's turn

    def test_on_acquire_forwards_prompt_len_delta(self):
        upd = InflightDecoder().decode(
            StatisticEvent("on_acquire", request_id="r1", replica_id="s0", prompt_len=42), ""
        )
        assert upd.metrics[MetricKey.PROMPT_LEN_SUM] == 42  # len(prompt_ids) at dispatch
        assert upd.metrics[MetricKey.INFLIGHT_TOKENS] == 42  # gauge +prompt_len on acquire

    def test_on_release_emits_inflight_minus_completed_delta(self):
        upd = InflightDecoder().decode(StatisticEvent("on_release", replica_id="s0", prompt_len=42), "")
        assert isinstance(upd, MetricsUpdate)
        assert upd.metrics == {
            MetricKey.INFLIGHT_COUNT: -1,
            MetricKey.INFLIGHT_TOKENS: -42,  # gauge -prompt_len, mirrors the acquire
            MetricKey.COMPLETED_COUNT: 1,
        }
        assert upd.is_delta is True
        assert upd.request_id is None  # on_release carries no request_id

    def test_on_release_without_prompt_len_leaves_token_gauge_unchanged(self):
        # Callers that don't track prompt_len release with the default 0, so the
        # token gauge simply isn't decremented (no spurious negative drift).
        upd = InflightDecoder().decode(StatisticEvent("on_release", replica_id="s0"), "")
        assert upd.metrics[MetricKey.INFLIGHT_TOKENS] == 0

    def test_on_servers_removed_is_noop(self):
        # Faithful to verl: removal must NOT zero the counter — release
        # symmetry maintains it (zeroing would let a later release drive it
        # negative). Cumulative counts are lifetime totals — also left intact.
        assert InflightDecoder().decode(StatisticEvent("on_servers_removed", server_ids=["s0"]), "") is None

    def test_non_event_payload_returns_none(self):
        assert InflightDecoder().decode(b"bytes", "") is None

    def test_missing_replica_returns_none(self):
        assert InflightDecoder().decode(StatisticEvent("on_acquire"), "") is None


class _FakeBalancer:
    """Minimal Balancer stand-in exposing register/un_register_call_back."""

    def __init__(self):
        self.callbacks: dict[str, list] = {}

    def register_call_back(self, event, fn):
        self.callbacks.setdefault(event, []).append(fn)

    def un_register_call_back(self, event, fn):
        lst = self.callbacks.get(event, [])
        if fn in lst:
            lst.remove(fn)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestCallbackTransport:
    def test_is_async_false(self):
        assert CallbackTransport(_FakeBalancer()).is_async is False

    def test_subscribe_registers_three_hooks_and_forwards(self):
        balancer = _FakeBalancer()
        transport = CallbackTransport(balancer)
        received: list = []
        _run(transport.subscribe(lambda raw, nid: received.append(raw)))

        assert set(balancer.callbacks) == {"on_acquire", "on_release", "on_servers_removed"}
        # one callback per hook
        assert all(len(lst) == 1 for lst in balancer.callbacks.values())

        balancer.callbacks["on_acquire"][0]("r1", "s0")
        balancer.callbacks["on_release"][0]("s0", 7)  # release forwards prompt_len
        balancer.callbacks["on_servers_removed"][0](["s1", "s2"])

        assert received == [
            StatisticEvent("on_acquire", request_id="r1", replica_id="s0"),
            StatisticEvent("on_release", replica_id="s0", prompt_len=7),
            StatisticEvent("on_servers_removed", server_ids=("s1", "s2")),
        ]

    def test_stop_unregisters_all(self):
        balancer = _FakeBalancer()
        transport = CallbackTransport(balancer)
        _run(transport.subscribe(lambda raw, nid: None))
        transport.stop()
        assert all(not lst for lst in balancer.callbacks.values())

    def test_stop_is_idempotent(self):
        transport = CallbackTransport(_FakeBalancer())
        _ = transport.subscribe  # method exists; not yet subscribed
        transport.stop()  # must not raise even with empty registry


class TestCollectorCallbackIntegration:
    """End-to-end: Collector(CallbackTransport, decoder) → handler → DataStore.

    Exercises the is_async=False start path (tmp loop runs the loop-free
    subscribe), the handler's StickyUpdate/MetricsUpdate dispatch, and the
    store writes — the whole Phase-1 statistic chain.
    """

    @pytest.fixture(autouse=True)
    def _reset_singletons(self):
        from verl.workers.rollout.router.kvcaware.store.per_replica_store import PerReplicaStore
        from verl.workers.rollout.router.kvcaware.store.per_request_store import PerRequestStore

        PerReplicaStore._instance = None
        PerRequestStore._instance = None
        yield
        PerReplicaStore._instance = None
        PerRequestStore._instance = None

    def test_sticky_collector_writes_binding_on_acquire(self):
        from verl.workers.rollout.router.kvcaware.collectors.collector import Collector
        from verl.workers.rollout.router.kvcaware.store.data_store import DataStore

        balancer = _FakeBalancer()
        collector = Collector(CallbackTransport(balancer), StickyDecoder())
        collector.start()
        try:
            balancer.callbacks["on_acquire"][0]("r1", "s0")
            assert DataStore().get_sticky_binding("r1") == "s0"
        finally:
            collector.stop()

    def test_inflight_collector_applies_acquire_release_delta(self):
        from verl.workers.rollout.router.kvcaware.collectors.collector import Collector
        from verl.workers.rollout.router.kvcaware.store.data_store import DataStore

        balancer = _FakeBalancer()
        collector = Collector(CallbackTransport(balancer), InflightDecoder())
        collector.start()
        try:
            balancer.callbacks["on_acquire"][0]("r1", "s0", [1, 2, 3])  # +1 inflight, +3 tokens
            balancer.callbacks["on_acquire"][0]("r2", "s0", list(range(10)))  # +1 inflight, +10 tokens
            balancer.callbacks["on_release"][0]("s0", 3)  # -1 inflight, -3 tokens, +1 completed
            assert DataStore().get_metric("s0", MetricKey.INFLIGHT_COUNT) == 1
            assert DataStore().get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 10  # 3 + 10 - 3
            assert DataStore().get_metric("s0", MetricKey.DISPATCHED_COUNT) == 2
            assert DataStore().get_metric("s0", MetricKey.COMPLETED_COUNT) == 1
        finally:
            collector.stop()

    def test_inflight_collector_accumulates_prompt_len_sum(self):
        from verl.workers.rollout.router.kvcaware.collectors.collector import Collector
        from verl.workers.rollout.router.kvcaware.store.data_store import DataStore

        balancer = _FakeBalancer()
        collector = Collector(CallbackTransport(balancer), InflightDecoder())
        collector.start()
        try:
            # on_acquire(request_id, chosen, prompt_ids) — the third arg is the
            # prompt token-id list; its length is attributed to PROMPT_LEN_SUM.
            balancer.callbacks["on_acquire"][0]("r1", "s0", [1, 2, 3])  # len 3
            balancer.callbacks["on_acquire"][0]("r2", "s0", list(range(10)))  # len 10
            balancer.callbacks["on_acquire"][0]("r3", "s1", None)  # no prompt → 0
            ds = DataStore()
            assert ds.get_metric("s0", MetricKey.PROMPT_LEN_SUM) == 13  # 3 + 10
            assert ds.get_metric("s1", MetricKey.PROMPT_LEN_SUM) == 0  # no prompt forwarded
        finally:
            collector.stop()

    def test_inflight_collector_tracks_turns_and_turn_sum(self):
        from verl.workers.rollout.router.kvcaware.collectors.collector import Collector
        from verl.workers.rollout.router.kvcaware.store.data_store import DataStore

        balancer = _FakeBalancer()
        collector = Collector(CallbackTransport(balancer), InflightDecoder())
        collector.start()
        try:
            # r1 dispatched three times (turns 1,2,3) to s0,s1,s0; r2 once (turn 1)
            # to s1. Each acquire also bumps INFLIGHT/DISPATCHED; the dispatch's
            # turn is attributed to the receiving replica's TURN_SUM (the inflight
            # collector carries request_id and does the PerRequestStore/TURN_SUM work).
            balancer.callbacks["on_acquire"][0]("r1", "s0")
            balancer.callbacks["on_acquire"][0]("r1", "s1")
            balancer.callbacks["on_acquire"][0]("r1", "s0")
            balancer.callbacks["on_acquire"][0]("r2", "s1")

            ds = DataStore()
            # per-request turn is global (Nth dispatch of that request_id overall)
            assert ds.get_per_request("r1", "turn", 0) == 3
            assert ds.get_per_request("r2", "turn", 0) == 1
            # ...but each dispatch's turn is attributed to the receiving replica's
            # TURN_SUM (per-replica, in PerReplicaStore): s0 got r1's turn-1 + turn-3 = 4;
            # s1 got r1's turn-2 + r2's turn-1 = 3.
            assert ds.get_metric("s0", MetricKey.TURN_SUM) == 4
            assert ds.get_metric("s1", MetricKey.TURN_SUM) == 3
        finally:
            collector.stop()

    def test_request_id_without_dispatch_does_not_record_turn(self):
        """Turn is gated on DISPATCHED_COUNT (a dispatch), not request_id presence.

        A hypothetical per-request delta that carries request_id but does NOT
        bump DISPATCHED_COUNT must not touch the turn table or TURN_SUM — guards
        against a future request_id-carrying update overloading the turn path.
        """
        from verl.workers.rollout.router.kvcaware.collectors.collector import Collector
        from verl.workers.rollout.router.kvcaware.collectors.decoder import MetricsUpdate
        from verl.workers.rollout.router.kvcaware.store.data_store import DataStore

        collector = Collector(CallbackTransport(_FakeBalancer()), InflightDecoder())
        # request_id present, but no DISPATCHED_COUNT in the delta → not a dispatch.
        collector._write_metrics_update(
            MetricsUpdate(
                node_id="s0",
                metrics={MetricKey.INFLIGHT_COUNT: 1},
                is_delta=True,
                request_id="r1",
            )
        )
        ds = DataStore()
        assert ds.get_per_request("r1", "turn", 0) == 0  # no dispatch → no turn recorded
        assert ds.get_metric("s0", MetricKey.TURN_SUM) == 0
