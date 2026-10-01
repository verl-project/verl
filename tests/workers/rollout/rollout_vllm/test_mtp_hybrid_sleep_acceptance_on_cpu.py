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

import numpy as np
import pytest

pytest.importorskip("ray")
pytest.importorskip("vllm")

from verl.trainer.ppo.ray_trainer import compute_spec_decode_metrics
from verl.workers.rollout.vllm_rollout import vllm_async_server


class _FakeMtpEngine:
    """Minimal model of the MTP failure mode seen in hybrid sleep."""

    def __init__(self):
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


class _FakeSleepEngine:
    """Records sleep level and collective_rpc method names."""

    def __init__(self):
        self.sleep_levels = []
        self.rpcs = []

    async def sleep(self, level: int):
        self.sleep_levels.append(level)

    async def reset_encoder_cache(self):
        pass

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.rpcs.append(method)


def _hybrid_server(model_config, actor_strategy=None):
    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = SimpleNamespace(mtp=None)
    server.model_config = model_config
    server.actor_strategy = actor_strategy
    server.engine = _FakeSleepEngine()
    return server


def test_hybrid_full_param_sleep_discards_weights(monkeypatch):
    """No LoRA: hybrid sleep stays at level 2. The full checkpoint sync refills discarded params."""
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)
    server = _hybrid_server(SimpleNamespace(lora_rank=0, lora={"rank": 0}), actor_strategy="fsdp")

    asyncio.run(server._sleep_hybrid())

    assert server.engine.sleep_levels == [2]
    assert server.engine.rpcs == []


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2", "fsdp_turbo"])
def test_hybrid_fsdp_merged_lora_sleep_keeps_weights(monkeypatch, strategy):
    """FSDP merged LoRA uses a PEFT export, which is not a full refill (#7904)."""
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)
    server = _hybrid_server(
        SimpleNamespace(lora_rank=64, lora={"rank": 64, "merge": True}),
        actor_strategy=strategy,
    )

    asyncio.run(server._sleep_hybrid())

    assert server.engine.sleep_levels == [1]
    assert server.engine.rpcs == []


@pytest.mark.parametrize("strategy", ["megatron", "veomni", "torchtitan", None])
def test_hybrid_megatron_merged_lora_sleep_discards_weights(monkeypatch, strategy):
    """Megatron merged LoRA exports a full HF state dict, so level 2 stays valid.

    An unset strategy also stays at level 2. Only the FSDP PEFT-export path opts into level 1.
    """
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)
    server = _hybrid_server(
        SimpleNamespace(lora_rank=0, lora={"rank": 64, "merge": True}),
        actor_strategy=strategy,
    )

    asyncio.run(server._sleep_hybrid())

    assert server.engine.sleep_levels == [2]
    assert server.engine.rpcs == []


def test_hybrid_lora_adapter_sleep_keeps_weights(monkeypatch):
    """Adapter sleep stays at level 1 on every backend. Only the adapter is synced."""
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)
    server = _hybrid_server(
        SimpleNamespace(lora_rank=64, lora={"rank": 64, "merge": False}),
        actor_strategy="megatron",
    )

    asyncio.run(server._sleep_hybrid())

    assert server.engine.sleep_levels == [1]
    assert server.engine.rpcs == []
