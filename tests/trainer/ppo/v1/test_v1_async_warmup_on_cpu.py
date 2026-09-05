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


def _stub(*, skip_rollout_tq: bool = False, resumed_inflight_prompts: int = 0, num_warmup_batches: int = 2):
    stub = SimpleNamespace(
        config=OmegaConf.create(
            {
                "skip": {"rollout_tq": {"enable": skip_rollout_tq}},
                "trainer": {
                    "v1": {
                        "colocate_async": {"num_warmup_batches": num_warmup_batches},
                        "separate_async": {"num_warmup_batches": num_warmup_batches},
                    }
                },
            }
        ),
        _resumed_inflight_prompts=resumed_inflight_prompts,
    )
    stub._add_batch_to_generate = MagicMock()
    stub._add_async_warmup_batches = PPOTrainer._add_async_warmup_batches.__get__(stub)
    return stub


def test_fresh_start_submits_warmup_batches():
    stub = _stub(resumed_inflight_prompts=0, num_warmup_batches=3)
    stub._add_async_warmup_batches(3)
    assert stub._add_batch_to_generate.call_count == 3


def test_reissued_inflight_skips_warmup():
    stub = _stub(resumed_inflight_prompts=4, num_warmup_batches=3)
    stub._add_async_warmup_batches(3)
    stub._add_batch_to_generate.assert_not_called()


def test_skip_rollout_tq_skips_warmup_even_without_reissue():
    stub = _stub(skip_rollout_tq=True, resumed_inflight_prompts=0, num_warmup_batches=3)
    stub._add_async_warmup_batches(3)
    stub._add_batch_to_generate.assert_not_called()


def test_colocate_async_on_train_begin_uses_shared_helper():
    stub = _stub(resumed_inflight_prompts=2, num_warmup_batches=3)
    stub.on_train_begin = PPOTrainerColocateAsync.on_train_begin.__get__(stub)
    stub.on_train_begin()
    stub._add_batch_to_generate.assert_not_called()


def test_separate_async_on_train_begin_uses_shared_helper():
    stub = _stub(resumed_inflight_prompts=0, num_warmup_batches=2)
    stub.on_train_begin = PPOTrainerSeparateAsync.on_train_begin.__get__(stub)
    stub.on_train_begin()
    assert stub._add_batch_to_generate.call_count == 2
