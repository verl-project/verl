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
import torch

from verl import DataProto
from verl.experimental.one_step_off_policy.ray_trainer import OneStepOffRayTrainer, PrecomputedReward


class _AsyncRolloutManager:
    async def clear_kv_cache(self):
        return None


def _trainer_for_fit_generate():
    trainer = OneStepOffRayTrainer.__new__(OneStepOffRayTrainer)
    trainer.metrics = {}
    trainer.timing_raw = {}
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                temperature=1.0,
            ),
        ),
    )
    trainer.is_last_step = True
    trainer.async_rollout_manager = _AsyncRolloutManager()
    trainer._fit_update_weights = lambda: None
    trainer.reward_tensor = None
    trainer.reward_extra_infos_dict = {}
    return trainer


def _batch():
    return DataProto.from_dict(
        tensors={
            "responses": torch.zeros((2, 2), dtype=torch.long),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
        },
        non_tensors={},
        meta_info={"timing": {"agent_loop/generate_sequences/mean": 1.0}},
    )


class _DynamicRolloutManager:
    def __init__(self):
        self.calls = 0

    async def generate_sequences(self, gen_batch):
        self.calls += 1
        if self.calls == 1:
            reward_values = [[0, 1], [0, 2], [0, 5], [0, 5]]
            acc = [0, 1, 1, 1]
        else:
            reward_values = [[0, 0], [0, 0], [0, 3], [0, 4]]
            acc = [0, 0, 0, 1]

        return DataProto.from_dict(
            tensors={
                "responses": torch.zeros((len(gen_batch), 2), dtype=torch.long),
                "attention_mask": torch.ones((len(gen_batch), 3), dtype=torch.long),
                "rm_scores": torch.tensor(reward_values, dtype=torch.float32),
            },
            non_tensors={"acc": np.array(acc)},
            meta_info={"timing": {"agent_loop/generate_sequences/mean": 1.0}, "reward_extra_keys": ["acc"]},
        )


def test_fit_generate_sets_precomputed_reward_payload():
    trainer = _trainer_for_fit_generate()
    reward_tensor = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
    reward_payload = PrecomputedReward(reward_tensor, {"acc": [0, 1]})

    async def run():
        future = asyncio.Future()
        future.set_result(
            ({"dynamic_sampling/kept_prompts": 1.0}, {"generate_async": 2.0}, 0, _batch(), reward_payload)
        )
        batch, next_future = await trainer._fit_generate(future, iter(()))
        return batch, next_future

    batch, next_future = asyncio.run(run())

    assert next_future is None
    assert trainer.reward_tensor is reward_tensor
    assert trainer.reward_extra_infos_dict == {"acc": [0, 1]}
    assert trainer.metrics["dynamic_sampling/kept_prompts"] == 1.0
    assert trainer.timing_raw["generate_async"] == 2.0
    assert "timing" not in batch.meta_info


def test_fit_compute_reward_reuses_precomputed_reward():
    trainer = OneStepOffRayTrainer.__new__(OneStepOffRayTrainer)
    trainer.reward_tensor = torch.ones((1, 1))
    batch = _batch()

    assert trainer._fit_compute_reward(batch) is batch


def test_dynamic_sampling_enable_check_handles_missing_config():
    trainer = OneStepOffRayTrainer.__new__(OneStepOffRayTrainer)
    trainer.config = SimpleNamespace(algorithm={})

    assert not trainer._is_dynamic_sampling_enabled()

    trainer.config = SimpleNamespace(algorithm={"filter_groups": {"enable": True}})
    assert trainer._is_dynamic_sampling_enabled()


def test_async_dynamic_sampling_backfills_until_train_prompt_batch_is_ready():
    trainer = OneStepOffRayTrainer.__new__(OneStepOffRayTrainer)
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=2)),
        algorithm=SimpleNamespace(filter_groups={"metric": "acc", "max_num_gen_batches": 3}),
        data=SimpleNamespace(train_batch_size=2),
        trainer=SimpleNamespace(balance_batch=False),
    )
    trainer.global_steps = 3
    trainer.use_rm = False
    trainer.async_rollout_manager = _DynamicRolloutManager()
    trainer._get_gen_batch = lambda batch: DataProto.from_dict(
        tensors={"prompts": torch.zeros((len(batch), 1), dtype=torch.long)}
    )

    continuous_iterator = iter(
        [
            (0, {"prompts": torch.ones((2, 1), dtype=torch.long)}),
            (0, {"prompts": torch.full((2, 1), 2, dtype=torch.long)}),
        ]
    )

    metrics, timing_raw, epoch, batch, reward_payload = asyncio.run(
        trainer._async_gen_next_dynamic_batch(continuous_iterator)
    )

    assert epoch == 0
    assert trainer.async_rollout_manager.calls == 2
    assert len(batch) == 4
    assert isinstance(reward_payload, PrecomputedReward)
    assert reward_payload.tensor.sum(dim=-1).tolist() == [1, 2, 3, 4]
    assert reward_payload.extra_infos["acc"].tolist() == [0, 1, 0, 1]
    assert metrics["dynamic_sampling/num_gen_batches"] == 2.0
    assert metrics["dynamic_sampling/num_prompt_in_batch"] == 2.0
    assert timing_raw["generate_async"] >= 0.0
