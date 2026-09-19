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
import pytest
import torch

from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.v1.utils import compute_advantage, compute_advantage_for_multi_trajectories


@pytest.fixture
def batch_data() -> DataProto:
    tensors = {
        "token_level_rewards": torch.tensor(
            [
                [100, 0, 0, 0],
                [1, 2, 3, 4],
                [200, 200, 0, 0],
                [150, 0, 0, 150],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
            dtype=torch.float32,
        ),
        "response_mask": torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 1, 0, 1],
                [0, 0, 0, 0],
            ],
            dtype=torch.long,
        ),
    }
    non_tensors = {
        "uid": np.array(["prompt_a"] * 6, dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def test_compute_advantage_for_single_trajectory(batch_data: DataProto):
    result = compute_advantage_for_multi_trajectories(
        data=batch_data,
        batch_keys=[f"prompt_a_{i}_0" for i in range(len(batch_data))],
        adv_estimator=AdvantageEstimator.GRPO,
    )
    expected = compute_advantage(
        batch_data,
        adv_estimator=AdvantageEstimator.GRPO,
    )
    assert torch.equal(result.batch["advantages"], expected.batch["advantages"])
    assert torch.equal(result.batch["returns"], expected.batch["returns"])
    assert torch.equal(result.batch["response_mask"], expected.batch["response_mask"])


def test_compute_advantage_for_multi_trajectories(batch_data: DataProto):
    result = compute_advantage_for_multi_trajectories(
        data=batch_data,
        batch_keys=["prompt_a_0_0", "prompt_a_0_1", "prompt_a_2_0", "prompt_a_2_1", "prompt_a_3_0", "prompt_a_4_0"],
        adv_estimator=AdvantageEstimator.GRPO,
    )
    expected = compute_advantage(
        batch_data.select_idxs([1, 3, 4, 5]),
        adv_estimator=AdvantageEstimator.GRPO,
    )
    gather_row_indices = [0, 0, 1, 1, 2, 3]
    gather_col_indices = [0, 0, 1, 1, 0, 0]
    adv_expected = (
        expected.batch["advantages"][gather_row_indices, gather_col_indices].unsqueeze(-1)
        * result.batch["response_mask"]
    )
    assert torch.equal(result.batch["advantages"], adv_expected)
    assert torch.equal(result.batch["returns"], adv_expected)


def test_compute_advantage_masks_zero_variance_grpo_groups():
    tensors = {
        "token_level_rewards": torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "response_mask": torch.ones(4, 2, dtype=torch.long),
    }
    non_tensors = {
        "uid": np.array(["zero-var", "zero-var", "mixed", "mixed"], dtype=object),
    }
    data = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

    result = compute_advantage(data, adv_estimator=AdvantageEstimator.GRPO)

    assert torch.equal(result.batch["response_mask"][0], torch.zeros(2, dtype=torch.long))
    assert torch.equal(result.batch["response_mask"][1], torch.zeros(2, dtype=torch.long))
    assert torch.equal(result.batch["response_mask"][2], torch.ones(2, dtype=torch.long))
    assert torch.equal(result.batch["response_mask"][3], torch.ones(2, dtype=torch.long))
    assert torch.equal(result.batch["advantages"][0], torch.zeros(2, dtype=torch.float32))
    assert torch.equal(result.batch["advantages"][1], torch.zeros(2, dtype=torch.float32))
    assert result.meta_info["zero_variance_metrics"]["training/zero_variance/filtered_prompt_count"] == 1.0
    assert result.meta_info["zero_variance_metrics"]["training/zero_variance/num_prompts"] == 2.0
    assert result.meta_info["zero_variance_metrics"]["training/zero_variance/filtered_prompt_frac"] == 0.5


def test_compute_advantage_for_multi_trajectories_masks_zero_variance_groups():
    tensors = {
        "token_level_rewards": torch.tensor(
            [
                [1.0, 0.0],  # session 0 prefix of zero-var group
                [1.0, 0.0],  # session 0 final of zero-var group
                [1.0, 0.0],  # session 1 final of zero-var group
                [1.0, 0.0],  # mixed group final
                [0.0, 0.0],  # mixed group final
            ],
            dtype=torch.float32,
        ),
        "response_mask": torch.ones(5, 2, dtype=torch.long),
    }
    non_tensors = {
        "uid": np.array(["zero-var", "zero-var", "zero-var", "mixed", "mixed"], dtype=object),
    }
    data = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

    result = compute_advantage_for_multi_trajectories(
        data=data,
        batch_keys=["zero-var_0_0", "zero-var_0_1", "zero-var_1_0", "mixed_0_0", "mixed_1_0"],
        adv_estimator=AdvantageEstimator.GRPO,
    )

    assert torch.equal(result.batch["response_mask"][0], torch.zeros(2, dtype=torch.long))
    assert torch.equal(result.batch["response_mask"][1], torch.zeros(2, dtype=torch.long))
    assert torch.equal(result.batch["response_mask"][2], torch.zeros(2, dtype=torch.long))
    assert torch.equal(result.batch["response_mask"][3], torch.ones(2, dtype=torch.long))
    assert torch.equal(result.batch["response_mask"][4], torch.ones(2, dtype=torch.long))
    assert result.meta_info["zero_variance_metrics"]["training/zero_variance/filtered_prompt_count"] == 1.0
    assert result.meta_info["zero_variance_metrics"]["training/zero_variance/num_prompts"] == 2.0
    assert result.meta_info["zero_variance_metrics"]["training/zero_variance/filtered_prompt_frac"] == 0.5
