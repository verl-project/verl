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

"""Tests for vLLM HTTP metrics collection with real vLLM service.

Test flow:
1. Launch a real vLLM model service (Qwen3-4B).
2. Create a Collector(HTTPTransport, VLLMMetricsDecoder) via get_collector().
3. Call start() to begin metrics polling; the collector writes to the store.
4. Verify that expected metrics exist via DataStore.
"""

from __future__ import annotations

import time

import pytest
from conftest import NODE_ID

from verl.workers.rollout.router.kvcaware.collectors.collector import get_collector
from verl.workers.rollout.router.kvcaware.config.collector import CollectorConfig
from verl.workers.rollout.router.kvcaware.store.data_store import DataStore
from verl.workers.rollout.router.kvcaware.types import MetricKey

POLL_INTERVAL = 2.0
HTTP_TIMEOUT = 10.0


def _make_collector():
    cfg = CollectorConfig(
        http_polling={"polling_interval": POLL_INTERVAL, "http_timeout": HTTP_TIMEOUT},
    )
    return get_collector(
        "vllm_metrics",
        cfg,
        server_addresses={NODE_ID: NODE_ID},
    )


@pytest.mark.st
@pytest.mark.gpu
class TestVLLMMetricsCollectorWithRealService:
    """Integration tests: vLLM HTTP metrics collector against a live vLLM server."""

    def test_start_and_metrics_exist(self, vllm_service):
        """
        Feature: Collector writes real metrics to the store after start()
        Expectation:
            DataStore contains NODE_ID after one polling cycle.
            kv_cache_usage_perc → float, num_requests_running/waiting → int.
        """
        store = DataStore()
        collector = _make_collector()

        collector.start()
        time.sleep(POLL_INTERVAL + 3.0)
        collector.stop()

        assert NODE_ID in store.get_metric_node_ids(), (
            f"Expected node_id '{NODE_ID}' in store, got {store.get_metric_node_ids()}"
        )
        assert isinstance(store.get_metric(NODE_ID, MetricKey.KV_CACHE_USAGE_PERC), float)
        assert isinstance(store.get_metric(NODE_ID, MetricKey.NUM_REQUESTS_RUNNING), int)
        assert isinstance(store.get_metric(NODE_ID, MetricKey.NUM_REQUESTS_WAITING), int)

    def test_metrics_values_are_sane(self, vllm_service):
        """
        Feature: Collected metric values are within reasonable bounds
        Expectation:
            kv_cache_usage_perc >= 0.0
            num_requests_running >= 0
            num_requests_waiting >= 0
        """
        store = DataStore()
        collector = _make_collector()

        collector.start()
        time.sleep(POLL_INTERVAL + 3.0)
        collector.stop()

        assert store.get_metric(NODE_ID, MetricKey.KV_CACHE_USAGE_PERC) >= 0.0
        assert store.get_metric(NODE_ID, MetricKey.NUM_REQUESTS_RUNNING) >= 0
        assert store.get_metric(NODE_ID, MetricKey.NUM_REQUESTS_WAITING) >= 0

    def test_store_get_node_dict(self, vllm_service):
        """
        Feature: DataStore.get_metrics(node_id) returns the full node metrics dict
        Expectation:
            Dict contains kv_cache_usage_perc, num_requests_running, num_requests_waiting.
        """
        store = DataStore()
        collector = _make_collector()

        collector.start()
        time.sleep(POLL_INTERVAL + 3.0)
        collector.stop()

        node_metrics = store.get_metrics(NODE_ID)
        assert isinstance(node_metrics, dict)
        assert MetricKey.KV_CACHE_USAGE_PERC in node_metrics
        assert MetricKey.NUM_REQUESTS_RUNNING in node_metrics
        assert MetricKey.NUM_REQUESTS_WAITING in node_metrics

    def test_multiple_poll_cycles_refresh(self, vllm_service):
        """
        Feature: Multiple polling cycles refresh the store with updated values
        Expectation:
            After 3 polling cycles the store contains data and values are reasonable.
        """
        store = DataStore()
        collector = _make_collector()

        collector.start()
        time.sleep(POLL_INTERVAL * 3 + 2.0)
        collector.stop()

        assert len(store.get_metrics(NODE_ID)) > 0, "Store should have metrics after multiple poll cycles"
