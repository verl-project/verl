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
"""CPU tests for V1 ESI / save_freq checkpoint gating.

``PPOTrainer._maybe_save_checkpoint`` is the default-trainer counterpart of the
V0 / experimental ``should_save_ckpt_esi`` save condition. These tests bind the
method to a stub so they do not construct a full trainer.
"""

import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from omegaconf import OmegaConf

from verl.trainer.ppo.v1.trainer_base import PPOTrainer


def _stub(*, save_freq: int, global_steps: int, max_steps_duration: float, esi_redundant_time: float = 0):
    stub = SimpleNamespace(
        config=OmegaConf.create({"trainer": {"save_freq": save_freq, "esi_redundant_time": esi_redundant_time}}),
        global_steps=global_steps,
        max_steps_duration=max_steps_duration,
        timing_raw={},
    )
    stub._save_checkpoint = MagicMock()
    stub._maybe_save_checkpoint = PPOTrainer._maybe_save_checkpoint.__get__(stub)
    return stub


def _clear_esi_env():
    os.environ.pop("MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP", None)
    os.environ.pop("SAGEMAKER_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP", None)


def test_save_freq_hit_writes_checkpoint():
    _clear_esi_env()
    stub = _stub(save_freq=10, global_steps=20, max_steps_duration=30)
    assert stub._maybe_save_checkpoint(is_last_step=False) is True
    stub._save_checkpoint.assert_called_once()


def test_save_freq_miss_does_not_write():
    _clear_esi_env()
    stub = _stub(save_freq=10, global_steps=3, max_steps_duration=30)
    assert stub._maybe_save_checkpoint(is_last_step=False) is False
    stub._save_checkpoint.assert_not_called()


def test_last_step_writes_even_when_not_on_freq():
    _clear_esi_env()
    stub = _stub(save_freq=10, global_steps=3, max_steps_duration=30)
    assert stub._maybe_save_checkpoint(is_last_step=True) is True
    stub._save_checkpoint.assert_called_once()


def test_save_freq_disabled_skips_esi_force_save():
    os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(time.time() + 90)
    stub = _stub(save_freq=-1, global_steps=3, max_steps_duration=30, esi_redundant_time=30)
    assert stub._maybe_save_checkpoint(is_last_step=False) is False
    stub._save_checkpoint.assert_not_called()
    _clear_esi_env()


def test_esi_expiry_force_saves_off_freq():
    os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(time.time() + 90)
    stub = _stub(save_freq=100, global_steps=3, max_steps_duration=30, esi_redundant_time=30)
    assert stub._maybe_save_checkpoint(is_last_step=False) is True
    stub._save_checkpoint.assert_called_once()
    _clear_esi_env()


def test_esi_far_future_does_not_force_save():
    os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(time.time() + 10_000)
    stub = _stub(save_freq=100, global_steps=3, max_steps_duration=30, esi_redundant_time=30)
    assert stub._maybe_save_checkpoint(is_last_step=False) is False
    stub._save_checkpoint.assert_not_called()
    _clear_esi_env()
