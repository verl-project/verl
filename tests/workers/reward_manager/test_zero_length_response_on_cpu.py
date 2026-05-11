# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.workers.reward_manager.naive import NaiveRewardManager


class Tokenizer:
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return " ".join(str(token) for token in ids if token != self.pad_token_id)


def _make_data() -> DataProto:
    return DataProto(
        batch=TensorDict(
            {
                "prompts": torch.tensor([[11, 12], [11, 12]], dtype=torch.long),
                "responses": torch.tensor([[0, 0, 0], [101, 102, 0]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.long),
            },
            batch_size=2,
        ),
        non_tensor_batch={
            "reward_model": np.array(
                [{"ground_truth": "empty-high"}, {"ground_truth": "normal"}],
                dtype=object,
            ),
            "data_source": np.array(["probe", "probe"], dtype=object),
            "extra_info": np.array([{}, {}], dtype=object),
        },
    )


def test_naive_reward_manager_does_not_write_reward_for_zero_length_response():
    def compute_score(data_source, solution_str, ground_truth, extra_info):
        return 10.0 if ground_truth == "empty-high" else 1.0

    reward_tensor = NaiveRewardManager(
        Tokenizer(),
        num_examine=0,
        compute_score=compute_score,
    )(_make_data())

    assert torch.equal(reward_tensor[0], torch.zeros_like(reward_tensor[0]))
    assert torch.equal(reward_tensor[1], torch.tensor([0.0, 1.0, 0.0]))


def test_reward_loop_assemble_rm_scores_skips_zero_length_response():
    rm_scores = RewardManagerBase.assemble_rm_scores(_make_data(), [10.0, 1.0])

    assert torch.equal(rm_scores[0], torch.zeros_like(rm_scores[0]))
    assert torch.equal(rm_scores[1], torch.tensor([0.0, 1.0, 0.0]))
