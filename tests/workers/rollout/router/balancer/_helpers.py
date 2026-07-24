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

"""Helpers for balancer unit tests.

Defines ``_FakeCollectorManager`` (real statistic collectors + stubbed network
collectors) and helper functions. Patching is done by ``conftest.py`` via a
session-scoped autouse fixture (``_conditional_patch``) that only fires when
balancer ut tests are selected, so it never leaks to Ray workers in other test
directories.
"""

from __future__ import annotations

from omegaconf import OmegaConf


class _FakeCollectorManager:
    """Stand-in for ``CollectorManager`` — stubs NETWORK collectors but builds
    REAL statistic collectors (anything with ``is_async=False``).

    The Balancer-callback → CallbackTransport → StickyDecoder/InflightDecoder →
    DataStore chain is the heart of the Phase-1 refactor, so it must run
    end-to-end (NOT mocked). Network collectors (vllm_metrics / vllm_zmq) need
    live endpoints and are stubbed; tests inject metrics directly via
    ``balancer._store.refresh_metrics(...)``.
    """

    def __init__(
        self, collectors_config, collection_names, server_addresses=None, kv_event_endpoints=None, balancer_handler=None
    ):
        self.collectors_config = collectors_config
        self.collection_names = collection_names
        self.server_addresses = server_addresses
        self.kv_event_endpoints = kv_event_endpoints
        self.balancer_handler = balancer_handler
        self.started = False
        self.stopped = False
        # Build every collector through the real factory; keep only the statistic
        # ones (is_async=False). Network collectors are built but not started.
        from verl.workers.rollout.router.kvcaware.collectors.collector import get_collector

        self._statistic_collectors = [
            c
            for c in (
                get_collector(name, collectors_config, balancer_handler=balancer_handler) for name in collection_names
            )
            if not getattr(c._transport, "is_async", True)
        ]

    def start(self):
        self.started = True
        for c in self._statistic_collectors:
            c.start()

    def stop(self):
        self.stopped = True
        for c in self._statistic_collectors:
            c.stop()


def _router_config(weight: float = 1.0):
    """Build a minimal router_config (OmegaConf) the Balancer accepts."""
    return OmegaConf.create(
        {
            "strategies": [
                {
                    "_target_": "verl.workers.rollout.router.kvcaware.config.strategy.KVCAwareStrategyConfig",
                    "weight": weight,
                    "collector_names": ["vllm_zmq", "sticky_stat", "inflight_stat"],
                },
            ],
        }
    )


def _fake_init_manager(self):
    """Replacement for KVCAwareBalancer._init_manager in unit tests.

    Injects a ``_FakeCollectorManager`` that builds REAL statistic collectors
    (sticky_stat/inflight_stat → the real Balancer-callback chain) while
    stubbing network collectors. The real ``DataStore`` is KEPT (not swapped
    for a stub) so the real statistic collectors
    (``Collector._data_store = DataStore()``) and strategy reads share the SAME
    singleton-backed store — that's what makes the sticky path actually work.
    Tests inject metrics via ``balancer._store.refresh_metrics(...)``.
    """
    collection_names = sorted({name for cfg in self._config.strategies for name in cfg.collector_names})
    self._manager = _FakeCollectorManager(
        self._config.collector,
        collection_names,
        balancer_handler=self,
    )
    self._manager.start()


def _make_balancer(servers=None, max_num_seqs=None):
    """Build a balancer over the given servers (default two).

    ``max_num_seqs`` overrides the capacity the Balancer resolved at construction
    (tests pass plain-string servers with no ``get_rollout_config``, so the
    Balancer's RPC resolution falls back to its default). Applied the same way
    the Balancer applies it in ``__init__``: via ``strategy.set_capacity(...)``.
    """
    from verl.workers.rollout.router.kvcaware.balancer import KVCAwareBalancer

    if servers is None:
        servers = {"s0": "h0", "s1": "h1"}
    balancer = KVCAwareBalancer(servers, _router_config())
    if max_num_seqs is not None:
        for strategy, _ in balancer._strategies:
            if hasattr(strategy, "set_capacity"):
                strategy.set_capacity(max_num_seqs)
    return balancer
