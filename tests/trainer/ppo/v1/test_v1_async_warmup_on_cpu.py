# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""CPU tests for V1 async warmup-after-resume gating."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from omegaconf import OmegaConf

from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync


def _stub(
    *,
    skip_rollout_tq: bool = False,
    restored_tq_prompt_count: int = 0,
    num_warmup_batches: int = 2,
    train_batch_size: int = 4,
):
    stub = SimpleNamespace(
        config=OmegaConf.create(
            {
                "skip": {"rollout_tq": {"enable": skip_rollout_tq}},
                "data": {"train_batch_size": train_batch_size},
                "trainer": {
                    "v1": {
                        "colocate_async": {"num_warmup_batches": num_warmup_batches},
                        "separate_async": {"num_warmup_batches": num_warmup_batches},
                    }
                },
            }
        ),
        _restored_tq_prompt_count=restored_tq_prompt_count,
    )
    stub._add_batch_to_generate = MagicMock()
    stub._add_prompts_to_generate = MagicMock()
    stub._add_async_warmup_batches = PPOTrainer._add_async_warmup_batches.__get__(stub)
    return stub


def test_fresh_start_submits_warmup_batches():
    stub = _stub(restored_tq_prompt_count=0, num_warmup_batches=3)
    stub._add_async_warmup_batches(3)
    stub._add_batch_to_generate.assert_not_called()
    stub._add_prompts_to_generate.assert_called_once_with(12)


def test_full_restored_prompt_pool_skips_warmup():
    stub = _stub(restored_tq_prompt_count=12, num_warmup_batches=3)
    stub._add_async_warmup_batches(3)
    stub._add_batch_to_generate.assert_not_called()
    stub._add_prompts_to_generate.assert_not_called()


def test_partial_restored_prompt_pool_is_topped_up():
    stub = _stub(restored_tq_prompt_count=6, num_warmup_batches=3)
    stub._add_async_warmup_batches(3)
    stub._add_batch_to_generate.assert_not_called()
    stub._add_prompts_to_generate.assert_called_once_with(6)


def test_overfilled_restored_prompt_pool_does_not_add_warmup():
    stub = _stub(restored_tq_prompt_count=13, num_warmup_batches=3)
    stub._add_async_warmup_batches(3)
    stub._add_batch_to_generate.assert_not_called()
    stub._add_prompts_to_generate.assert_not_called()


def test_skip_rollout_tq_skips_warmup_even_without_restored_prompts():
    stub = _stub(skip_rollout_tq=True, restored_tq_prompt_count=0, num_warmup_batches=3)
    stub._add_async_warmup_batches(3)
    stub._add_batch_to_generate.assert_not_called()
    stub._add_prompts_to_generate.assert_not_called()


def test_colocate_async_on_train_begin_uses_shared_helper():
    stub = _stub(restored_tq_prompt_count=12, num_warmup_batches=3)
    stub.on_train_begin = PPOTrainerColocateAsync.on_train_begin.__get__(stub)
    stub.on_train_begin()
    stub._add_batch_to_generate.assert_not_called()
    stub._add_prompts_to_generate.assert_not_called()


def test_separate_async_on_train_begin_uses_shared_helper():
    stub = _stub(restored_tq_prompt_count=0, num_warmup_batches=2)
    stub.on_train_begin = PPOTrainerSeparateAsync.on_train_begin.__get__(stub)
    stub.on_train_begin()
    stub._add_batch_to_generate.assert_not_called()
    stub._add_prompts_to_generate.assert_called_once_with(8)
