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

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

pytest.importorskip("ray")
pytest.importorskip("vllm")

from verl.trainer.ppo.ray_trainer import compute_spec_decode_metrics
from verl.workers.rollout.vllm_rollout import vllm_async_server


class _FakeMtpEngine:
    """Minimal model of the MTP failure mode seen in hybrid sleep."""

    def __init__(self):
        self.vllm_config = SimpleNamespace(speculative_config=None)
        self.mtp_drafter_available = True
        self.sleep_levels_that_discard_mtp_drafter = {2}

    async def sleep(self, level: int):
        if level in self.sleep_levels_that_discard_mtp_drafter:
            self.mtp_drafter_available = False

    async def reset_encoder_cache(self):
        pass

    def sync_actor_weights(self):
        pass

    def generate_spec_decode_stats(self):
        num_draft_tokens = 3
        num_accepted_tokens = num_draft_tokens if self.mtp_drafter_available else 0
        num_verify_steps = 1
        return num_draft_tokens, num_accepted_tokens, num_verify_steps


def test_mtp_hybrid_sleep_keeps_drafter_available_for_nonzero_acceptance(monkeypatch):
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)

    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = SimpleNamespace(mtp=SimpleNamespace(enable=True, enable_rollout=True))
    server.model_config = SimpleNamespace(lora_rank=0, lora={})
    server.engine = _FakeMtpEngine()

    asyncio.run(server._sleep_hybrid())
    server.engine.sync_actor_weights()
    drafts, accepts, verifies = server.engine.generate_spec_decode_stats()
    metrics = compute_spec_decode_metrics(
        spec_drafts=np.array([drafts]),
        spec_accepts=np.array([accepts]),
        spec_verifies=np.array([verifies]),
    )

    assert metrics["rollout/spec_accept_rate"] > 0.0
    assert metrics["rollout/spec_accept_length"] > 1.0


@pytest.mark.parametrize(
    ("method", "draft_config", "lora_rank", "expected_level"),
    [
        pytest.param("draft_model", "separate", 0, 1, id="separate-draft"),
        pytest.param("eagle3", "separate", 0, 1, id="eagle-draft"),
        pytest.param(None, "none", 0, 2, id="no-speculation"),
        pytest.param("ngram", "target", 0, 2, id="ngram"),
        pytest.param("ngram_gpu", "target", 0, 2, id="ngram-gpu"),
        pytest.param("suffix", "none", 0, 2, id="suffix"),
        pytest.param("custom_class", "target", 0, 2, id="custom-proposer"),
        pytest.param("extract_hidden_states", "separate", 0, 2, id="hidden-state-extraction"),
        pytest.param("mtp", "separate", 0, 2, id="mtp-without-verl-mtp-guard"),
        pytest.param(None, "none", 8, 1, id="lora-adapter"),
    ],
)
def test_hybrid_sleep_level_uses_resolved_draft_config(monkeypatch, method, draft_config, lora_rank, expected_level):
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)
    target_model_config = SimpleNamespace(model="target")
    draft_model_config = {
        "none": None,
        "target": target_model_config,
        "separate": SimpleNamespace(model="draft"),
    }[draft_config]
    speculative_config = (
        SimpleNamespace(
            method=method,
            target_model_config=target_model_config,
            draft_model_config=draft_model_config,
        )
        if method is not None
        else None
    )

    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = SimpleNamespace(mtp=None)
    server.model_config = SimpleNamespace(lora_rank=lora_rank, lora={})
    server.engine = SimpleNamespace(
        vllm_config=SimpleNamespace(speculative_config=speculative_config),
        sleep=AsyncMock(),
        reset_encoder_cache=AsyncMock(),
    )

    asyncio.run(server._sleep_hybrid())

    server.engine.sleep.assert_awaited_once_with(level=expected_level)
    server.engine.reset_encoder_cache.assert_awaited_once()
