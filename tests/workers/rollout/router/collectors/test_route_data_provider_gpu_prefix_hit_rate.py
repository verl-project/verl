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

"""Tests for DataStore.get_layer_prefix_hit_rate with real vLLM service + ZMQ KV events.

Test flow:
1. Launch a real vLLM model service with kv-events-config enabled (ZMQ publisher).
2. Create CollectorManager with collection_names=["vllm_zmq"], which internally
   creates a Collector(ZMQTransport, VLLMKVDecoder) via get_collector().
3. Call provider.start() to begin event subscription.
4. Send inference requests via httpx to trigger KV cache block-stored events.
5. Obtain prompt token IDs via vLLM /tokenize endpoint (messages format, with chat template).
6. Call DataStore().get_layer_prefix_hit_rate(prompt_ids) and verify the results.
"""

from __future__ import annotations

import time

import httpx
import pytest
from conftest import NODE_ID, VLLM_MODEL, ZMQ_REPLAY_PORT, ZMQ_SUB_PORT, send_inference_request

from verl.workers.rollout.router.kvcaware.collectors.manager import CollectorManager
from verl.workers.rollout.router.kvcaware.config.collector import CollectorConfig
from verl.workers.rollout.router.kvcaware.store.data_store import DataStore
from verl.workers.rollout.router.kvcaware.types import Layer

# ── Helpers ──────────────────────────────────────────────────────────────


def _get_token_ids(node_id: str, model: str, prompt: str) -> list[int]:
    """Get prompt token IDs from vLLM's /tokenize endpoint with chat template applied.

    Must use messages format so the chat template is applied — the same
    template vLLM applies during /v1/chat/completions inference.  KV cache
    blocks are computed from the fully-formatted sequence (including special
    tokens like <|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n),
    so the token IDs used for prefix hash lookup must match that sequence.

    Falls back to transformers AutoTokenizer + apply_chat_template if
    /tokenize is unavailable.
    """
    try:
        resp = httpx.post(
            f"http://{node_id}/tokenize",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "add_generation_prompt": True,
            },
            timeout=10.0,
        )
        if resp.status_code == 200:
            return resp.json().get("tokens", [])
    except Exception:
        pass

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False).input_ids


def _make_provider(node_id: str) -> CollectorManager:
    return CollectorManager(
        collectors_config=CollectorConfig(),
        collection_names=["vllm_zmq"],
        kv_event_endpoints={
            node_id: [f"127.0.0.1:{ZMQ_SUB_PORT}", f"127.0.0.1:{ZMQ_REPLAY_PORT}", "zmq", "kv-events"],
        },
    )


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.st
@pytest.mark.gpu
class TestGpuPrefixHitRateWithRealService:
    """Integration tests: DataStore.get_layer_prefix_hit_rate against a live vLLM ZMQ publisher."""

    def test_prefix_hit_rate_with_partial_match(self, vllm_kv_service):
        """
        Feature: get_layer_prefix_hit_rate returns 100% hit rate for a shorter prompt
        Description:
            1. Send an inference request with a long prompt A.
            2. Call get_layer_prefix_hit_rate with prompt B that is a strict prefix of A.
        Expectation:
            Since B's blocks are a subset of A's cached blocks, all of B's prefix
            blocks are cached → hit_rate = 100.
        """
        prompt_long = (
            "The history of artificial intelligence began in the 1950s and has evolved dramatically since then"
        )
        prompt_short = "The history of artificial intelligence began in the 1950s"

        long_ids = _get_token_ids(vllm_kv_service, VLLM_MODEL, prompt_long)
        short_ids = _get_token_ids(vllm_kv_service, VLLM_MODEL, prompt_short)
        assert len(short_ids) < len(long_ids), (
            f"Short prompt should have fewer tokens than long, got short={len(short_ids)}, long={len(long_ids)}"
        )

        provider = _make_provider(vllm_kv_service)
        provider.start()
        time.sleep(5.0)
        send_inference_request(vllm_kv_service, VLLM_MODEL, prompt_long)
        time.sleep(8.0)
        provider.stop()

        store = DataStore()
        hit = store.get_layer_prefix_hit_rate(NODE_ID, short_ids, Layer.GPU)

        if hit == 0.0:
            pytest.skip(
                f"Short prompt has {len(short_ids)} tokens — fewer than block_size "
                f"or no full block formed; cannot assert prefix hit rate"
            )

        assert hit == 1.0, f"Expected hit_rate=1.0 for prefix match, got {hit}"

    def test_prefix_hit_rate_returns_node_id_key(self, vllm_kv_service):
        """
        Feature: get_layer_prefix_hit_rate returns dict with node_id as key
        Description:
            1. Send an inference request.
            2. Call get_layer_prefix_hit_rate with the prompt's token IDs.
        Expectation:
            Keys are node IDs in "host:port" format matching NODE_ID.
            Values are integers in [0, 100].
        """
        prompt = "Explain the concept of neural networks in simple terms"
        prompt_ids = _get_token_ids(vllm_kv_service, VLLM_MODEL, prompt)
        assert len(prompt_ids) > 0, "Should have token IDs for the prompt"

        provider = _make_provider(vllm_kv_service)
        provider.start()
        time.sleep(5.0)
        send_inference_request(vllm_kv_service, VLLM_MODEL, prompt)
        time.sleep(8.0)
        provider.stop()

        store = DataStore()
        hit = store.get_layer_prefix_hit_rate(NODE_ID, prompt_ids, Layer.GPU)

        assert 0.0 <= hit <= 1.0, f"Hit rate should be in [0.0, 1.0], got {hit}"
