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

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager as RewardLoopDAPORewardManager
from verl.workers.reward_manager.dapo import DAPORewardManager


class _DummyTokenizer:
    eos_token = "</s>"

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(int(token_id)) for token_id in token_ids)


def _constant_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return 0.5


def _overlong_buffer_cfg(enable: bool, length: int = 128):
    return OmegaConf.create({"enable": enable, "len": length, "penalty_factor": 1.0, "log": False})


def _make_data(batch_size: int = 2, seq_len: int = 4) -> DataProto:
    return DataProto.from_dict(
        tensors={
            "prompts": torch.ones(batch_size, seq_len, dtype=torch.long),
            "responses": torch.ones(batch_size, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 2 * seq_len, dtype=torch.long),
        },
        non_tensors={
            "reward_model": np.array([{"ground_truth": "1"}] * batch_size, dtype=object),
            "data_source": np.array(["unit_test"] * batch_size, dtype=object),
        },
    )


def test_construct_with_overlong_buffer_disabled():
    """max_resp_len is not required when the overlong penalty is disabled. See issue #5858."""
    reward_manager = DAPORewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=_constant_compute_score,
        max_resp_len=None,
        overlong_buffer_cfg=_overlong_buffer_cfg(enable=False),
    )
    assert reward_manager.max_resp_len is None


def test_construct_with_overlong_buffer_enabled_requires_max_resp_len():
    with pytest.raises(AssertionError, match="max_resp_len must be provided"):
        DAPORewardManager(
            tokenizer=_DummyTokenizer(),
            num_examine=0,
            compute_score=_constant_compute_score,
            max_resp_len=None,
            overlong_buffer_cfg=_overlong_buffer_cfg(enable=True),
        )


def test_construct_with_overlong_buffer_enabled_rejects_short_max_resp_len():
    with pytest.raises(AssertionError, match="max_resp_len must be larger"):
        DAPORewardManager(
            tokenizer=_DummyTokenizer(),
            num_examine=0,
            compute_score=_constant_compute_score,
            max_resp_len=64,
            overlong_buffer_cfg=_overlong_buffer_cfg(enable=True, length=128),
        )


def test_call_without_overlong_buffer_cfg():
    """The default overlong_buffer_cfg=None must not crash at __call__ time."""
    reward_manager = DAPORewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=_constant_compute_score,
    )
    reward_tensor = reward_manager(_make_data())
    assert torch.all(reward_tensor[:, -1] == 0.5)


def test_call_with_overlong_buffer_enabled_applies_penalty():
    reward_manager = DAPORewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=_constant_compute_score,
        max_resp_len=4,
        overlong_buffer_cfg=_overlong_buffer_cfg(enable=True, length=2),
    )
    reward_tensor = reward_manager(_make_data(seq_len=4))
    # exceed_len = 4 - (4 - 2) = 2, so overlong_reward = -2 / 2 * 1.0 = -1.0
    assert torch.all(reward_tensor[:, -1] == 0.5 - 1.0)


def test_reward_loop_construct_with_overlong_buffer_disabled():
    """The experimental reward loop manager accepts a disabled overlong buffer without max_resp_len."""
    config = OmegaConf.create(
        {
            "reward": {
                "reward_kwargs": {
                    "overlong_buffer_cfg": {"enable": False, "len": 128, "penalty_factor": 1.0, "log": False},
                    "max_resp_len": None,
                }
            }
        }
    )
    reward_manager = RewardLoopDAPORewardManager(
        config=config, tokenizer=_DummyTokenizer(), compute_score=_constant_compute_score
    )
    assert reward_manager.max_resp_len is None
