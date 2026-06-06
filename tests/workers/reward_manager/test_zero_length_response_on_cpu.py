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

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.workers.reward_manager.batch import BatchRewardManager
from verl.workers.reward_manager.dapo import DAPORewardManager
from verl.workers.reward_manager.naive import NaiveRewardManager
from verl.workers.reward_manager.prime import PrimeRewardManager


class FakeTokenizer:
    eos_token = "<eos>"

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token.item()) for token in token_ids)

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return [self.decode(row, skip_special_tokens=skip_special_tokens) for row in token_ids]


def fixed_score(**kwargs):
    return {"score": 3.0, "acc": 1.0}


def fixed_batch_scores(**kwargs):
    return [{"score": 3.0, "acc": 1.0}, {"score": 4.0, "acc": 1.0}]


def make_data():
    return DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[1, 2], [3, 4]]),
            "responses": torch.tensor([[0, 0, 0], [5, 6, 0]]),
            # The first sample has no valid response tokens; the second has two.
            "attention_mask": torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]]),
        },
        non_tensors={
            "reward_model": np.array([{"ground_truth": "a"}, {"ground_truth": "b"}], dtype=object),
            "data_source": np.array(["unit", "unit"], dtype=object),
        },
    )


@pytest.mark.parametrize(
    "manager",
    [
        NaiveRewardManager(FakeTokenizer(), num_examine=0, compute_score=fixed_score),
        DAPORewardManager(FakeTokenizer(), num_examine=0, compute_score=fixed_score),
    ],
)
def test_token_reward_managers_skip_zero_length_response(manager):
    reward_tensor = manager(make_data())

    assert reward_tensor[0].tolist() == [0.0, 0.0, 0.0]
    assert reward_tensor[1].tolist() == [0.0, 3.0, 0.0]


def test_prime_reward_manager_skips_zero_length_response(monkeypatch):
    manager = PrimeRewardManager(FakeTokenizer(), num_examine=0)
    monkeypatch.setattr(manager, "verify", lambda data: [3.0, 4.0])

    reward_tensor = manager(make_data())

    assert reward_tensor[0].tolist() == [0.0, 0.0, 0.0]
    assert reward_tensor[1].tolist() == [0.0, 4.0, 0.0]


def test_dapo_reward_manager_skips_zero_length_response_with_overlong_buffer():
    manager = DAPORewardManager(
        FakeTokenizer(),
        num_examine=0,
        compute_score=fixed_score,
        max_resp_len=3,
        overlong_buffer_cfg=SimpleNamespace(enable=True, len=1, penalty_factor=1.0, log=True),
    )

    result = manager(make_data(), return_dict=True)

    assert result["reward_tensor"][0].tolist() == [0.0, 0.0, 0.0]
    assert result["reward_tensor"][1].tolist() == [0.0, 3.0, 0.0]
    assert result["reward_extra_info"]["overlong"] == [False, False]


def test_batch_reward_manager_skips_zero_length_response():
    manager = BatchRewardManager(FakeTokenizer(), num_examine=0, compute_score=fixed_batch_scores)

    reward_tensor = manager(make_data())

    assert reward_tensor[0].tolist() == [0.0, 0.0, 0.0]
    assert reward_tensor[1].tolist() == [0.0, 4.0, 0.0]
