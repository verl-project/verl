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

"""Regression: disaggregated weight sync must deliver LoRA adapters, not the base.

With LoRA and `checkpoint_engine.backend != "naive"`, the disaggregated branch of
ActorRolloutRefWorker.update_weights used to call get_per_tensor_param() with its
defaults, pinning base_sync_done=False. That collects the frozen base and skips
every `lora_` tensor, so the adapter never reached the rollout engine: the rollout
policy silently stayed at the initial checkpoint for the whole run.

These tests pin the two halves of the fix without a GPU or Ray:
  - the trainer runs the two-phase protocol (base once, adapters afterwards; with a
    real load_format the very first push is already adapter-only),
  - the rollout worker recognises an adapter push from the payload itself and
    forwards peft_config / base_sync_done, while base pushes stay untouched and
    every tensor is forwarded in order.
"""

import asyncio
from types import SimpleNamespace

import torch

from verl.checkpoint_engine.base import CheckpointEngineWorker
from verl.workers.engine_workers import ActorRolloutRefWorker


def _unwrap(method):
    """verl decorates worker methods with @register; tests call the raw coroutine."""
    return getattr(method, "__wrapped__", method)


class _RolloutConfig:
    """Minimal stand-in for the rollout config the worker reads."""

    def __init__(self, load_format="safetensors", backend="nccl"):
        self.load_format = load_format
        self.checkpoint_engine = SimpleNamespace(backend=backend)

    def get(self, key, default=None):
        return getattr(self, key, default)


def _make_rollout_worker(tensor_names, wire_format="named_tensors", lora_rank=32):
    """Build a stub `self` for CheckpointEngineWorker.update_weights."""

    async def receive_weights(global_steps=None):
        for name in tensor_names:
            yield name, torch.zeros(1)

    captured = {}

    class _ServerAdapter:
        async def update_weights(self, weights, global_steps=None, **kwargs):
            captured["kwargs"] = kwargs
            captured["names"] = [name async for name, _ in weights]

    # A real instance without __init__ (which would need Ray): helper methods then
    # resolve through the class, so this test exercises the shipped code rather than
    # a hand-wired copy of it.
    worker = object.__new__(CheckpointEngineWorker)
    worker.checkpoint_engine = SimpleNamespace(receive_weights=receive_weights, wire_format=wire_format)
    worker.server_adapter = _ServerAdapter()
    worker.model_config = SimpleNamespace(
        lora_rank=lora_rank,
        lora_alpha=2 * lora_rank,
        target_modules=["q_proj", "v_proj"],
        exclude_modules=None,
    )
    worker.update_weights = _unwrap(CheckpointEngineWorker.update_weights).__get__(worker)
    return worker, captured


class TestRolloutSideAdapterDetection:
    """CheckpointEngineWorker.update_weights annotates adapter pushes."""

    def test_adapter_push_is_annotated(self):
        names = [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
        ]
        worker, captured = _make_rollout_worker(names)
        asyncio.run(worker.update_weights(global_steps=1))

        assert captured["names"] == names, "the peek must forward every tensor, in order"
        assert captured["kwargs"].get("base_sync_done") is True
        peft_config = captured["kwargs"].get("peft_config")
        assert peft_config is not None
        assert peft_config["r"] == 32
        assert peft_config["lora_alpha"] == 64
        assert peft_config["target_modules"] == ["q_proj", "v_proj"]

    def test_base_push_is_untouched(self):
        names = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.base_layer.weight",
        ]
        worker, captured = _make_rollout_worker(names)
        asyncio.run(worker.update_weights(global_steps=1))

        assert captured["names"] == names
        assert "peft_config" not in captured["kwargs"], "base pushes must keep the previous behavior"
        assert "base_sync_done" not in captured["kwargs"]

    def test_full_finetuning_push_is_untouched(self):
        """No LoRA configured: the payload carries no `lora_` tensor either way."""
        names = ["model.layers.0.self_attn.q_proj.weight"]
        worker, captured = _make_rollout_worker(names, lora_rank=0)
        asyncio.run(worker.update_weights(global_steps=1))

        assert captured["names"] == names
        assert captured["kwargs"] == {"wire_format": "named_tensors"}

    def test_delta_wire_format_is_not_inspected(self):
        """Delta engines drive their own sync state machine; leave them alone."""
        names = ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
        worker, captured = _make_rollout_worker(names, wire_format="sharded_delta")
        asyncio.run(worker.update_weights(global_steps=1))

        assert captured["names"] == names
        assert "peft_config" not in captured["kwargs"]


class TestTrainerSideTwoPhaseProtocol:
    """The disaggregated branch must drive base_sync_done like the colocated one."""

    @staticmethod
    def _make_trainer_worker(load_format="safetensors", peft_config=None):
        calls = []

        def get_per_tensor_param(layered_summon=False, base_sync_done=False):
            calls.append(base_sync_done)
            return iter(()), peft_config

        async def send_weights(per_tensor_param, global_steps=None):
            return {}

        worker = SimpleNamespace(
            config=SimpleNamespace(rollout=_RolloutConfig(load_format=load_format)),
            actor=SimpleNamespace(engine=SimpleNamespace(get_per_tensor_param=get_per_tensor_param)),
            checkpoint_engine=SimpleNamespace(send_weights=send_weights),
        )
        bound = _unwrap(ActorRolloutRefWorker.update_weights).__get__(worker)
        return bound, calls

    def test_real_load_format_pushes_adapters_from_the_first_sync(self):
        update_weights, calls = self._make_trainer_worker(peft_config={"r": 32})
        asyncio.run(update_weights(global_steps=1, mode="nccl"))
        asyncio.run(update_weights(global_steps=2, mode="nccl"))

        # The rollout engine already holds real base weights, so no base push is needed.
        assert calls == [True, True]

    def test_dummy_load_format_pushes_base_first(self):
        update_weights, calls = self._make_trainer_worker(load_format="dummy", peft_config={"r": 32})
        asyncio.run(update_weights(global_steps=1, mode="nccl"))
        asyncio.run(update_weights(global_steps=2, mode="nccl"))

        # Base first (weights are placeholders), adapters from then on.
        assert calls == [False, True]

    def test_full_finetuning_never_flips_the_flag(self):
        update_weights, calls = self._make_trainer_worker(load_format="dummy", peft_config=None)
        asyncio.run(update_weights(global_steps=1, mode="nccl"))
        asyncio.run(update_weights(global_steps=2, mode="nccl"))

        # Without a peft_config there are no adapters to switch to.
        assert calls == [False, False]
