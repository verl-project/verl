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

"""Tests for vLLM ZMQ KV-cache event collection with real vLLM service.

Test flow:
1. Launch a real vLLM model service (Qwen3-4B) with kv-events-config enabled.
2. Create a Collector(ZMQTransport, VLLMKVDecoder) via get_collector().
3. Call start() — the collector writes KV events to the store.
4. Send an inference request to trigger BlockStored events.
5. Verify that KV cache data is accessible via DataStore.
"""

from __future__ import annotations

import time

import pytest
from conftest import NODE_ID, VLLM_MODEL, ZMQ_REPLAY_PORT, ZMQ_SUB_PORT, send_inference_request

from verl.workers.rollout.router.kvcaware.collectors.collector import get_collector
from verl.workers.rollout.router.kvcaware.config.collector import CollectorConfig
from verl.workers.rollout.router.kvcaware.store.data_store import DataStore


def _make_collector():
    cfg = CollectorConfig()  # default long_connection knobs
    return get_collector(
        "vllm_zmq",
        cfg,
        kv_event_endpoints={NODE_ID: [f"127.0.0.1:{ZMQ_SUB_PORT}", f"127.0.0.1:{ZMQ_REPLAY_PORT}", "zmq", "kv-events"]},
    )


@pytest.mark.st
@pytest.mark.gpu
class TestVLLMKVEventCollectorWithRealService:
    """Integration tests: vLLM ZMQ KV-cache collector against a live vLLM ZMQ publisher."""

    def test_start_and_kv_store_updated(self, vllm_kv_service):
        """
        Feature: Collector receives ZMQ events and updates KV cache store
        Expectation:
            block_size is set (learned from first event).
            At least one block is cached.
            NODE_ID appears in at least one cached block.
        """
        store = DataStore()
        collector = _make_collector()

        collector.start()
        time.sleep(5.0)
        send_inference_request(vllm_kv_service, VLLM_MODEL, "hello world")
        time.sleep(5.0)
        collector.stop()

        assert store.get_block_size() is not None, "block_size should be learned from KV events"
        assert store.get_block_size() > 0
        assert store.get_kv_block_count() > 0, "KV cache should have blocks after BlockStored events"
        assert store.kv_node_has_blocks(NODE_ID), f"Expected NODE_ID '{NODE_ID}' in at least one cached block"

    def test_block_size_learned(self, vllm_kv_service):
        """
        Feature: block_size is learned from the first BlockStored KV event
        Expectation:
            block_size is a positive integer (vLLM default is 16).
        """
        store = DataStore()
        collector = _make_collector()

        collector.start()
        time.sleep(5.0)
        send_inference_request(vllm_kv_service, VLLM_MODEL)
        time.sleep(5.0)
        collector.stop()

        assert isinstance(store.get_block_size(), int)
        assert store.get_block_size() > 0
        assert store.get_block_size() == 16, f"Expected block_size=16 (vLLM default), got {store.get_block_size()}"

    def test_multiple_inferences_accumulate_blocks(self, vllm_kv_service):
        """
        Feature: Multiple inference requests accumulate more blocks in the store
        Expectation:
            After multiple requests, the KV cache has entries.
        """
        store = DataStore()
        collector = _make_collector()

        collector.start()
        time.sleep(5.0)
        for prompt in [
            "What is machine learning?",
            "Explain quantum computing briefly.",
            "Tell me about deep reinforcement learning.",
        ]:
            send_inference_request(vllm_kv_service, VLLM_MODEL, prompt)
            time.sleep(3.0)
        time.sleep(3.0)
        collector.stop()

        assert store.get_kv_block_count() > 0, "Expected blocks after multiple inferences"

    def test_clear_kv_node_removes_all_blocks(self, vllm_kv_service):
        """
        Feature: DataStore.clear_kv_node removes all blocks for a node
        Expectation:
            After clear_kv_node, NODE_ID no longer appears in any cached block.
        """
        store = DataStore()
        collector = _make_collector()

        collector.start()
        time.sleep(5.0)
        send_inference_request(vllm_kv_service, VLLM_MODEL)
        time.sleep(5.0)
        collector.stop()

        if store.get_kv_block_count() == 0:
            pytest.skip("No blocks received from KV events")

        store.clear_kv_node(NODE_ID)

        assert not store.kv_node_has_blocks(NODE_ID), (
            f"NODE_ID '{NODE_ID}' should not appear in any block after clear_kv_node"
        )

    def test_decoder_hash_mapping_populated(self, vllm_kv_service):
        """
        Feature: VLLMKVDecoder.remote_to_local_block_hash is populated after events
        Description:
            Verify that the decoder's hash mapping tracks remote→local block hashes,
            and that every local hash appears in the KV cache store.
        Expectation:
            remote_to_local_block_hash is non-empty.
            All local hashes are present in the KV cache store.
        """
        store = DataStore()
        collector = _make_collector()

        collector.start()
        time.sleep(5.0)
        send_inference_request(vllm_kv_service, VLLM_MODEL)
        time.sleep(5.0)
        collector.stop()

        mapping = collector._decoder.remote_to_local_block_hash
        assert len(mapping) > 0, "remote_to_local_block_hash should have entries after processing events"
        for remote_bh, local_bh in mapping.items():
            assert isinstance(remote_bh, str)
            assert isinstance(local_bh, str)
            assert store.has_kv_block(local_bh), f"Local hash '{local_bh}' from mapping not found in KV cache store"
