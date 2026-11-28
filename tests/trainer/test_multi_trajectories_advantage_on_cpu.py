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
import copy

import numpy as np
import pytest
import torch

from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.v1.utils import compute_advantage, compute_advantage_for_multi_trajectories

SESSION_ESTIMATORS = [
    AdvantageEstimator.GRPO,
    AdvantageEstimator.GRPO_VECTORIZED,
    AdvantageEstimator.GRPO_PASSK,
    AdvantageEstimator.RLOO,
    AdvantageEstimator.RLOO_VECTORIZED,
    AdvantageEstimator.GPG,
]


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


@pytest.mark.parametrize("adv_estimator", SESSION_ESTIMATORS)
def test_compute_advantage_for_single_trajectory(batch_data: DataProto, adv_estimator: AdvantageEstimator):
    config = {} if adv_estimator == AdvantageEstimator.GRPO_PASSK else None
    result = compute_advantage_for_multi_trajectories(
        data=copy.deepcopy(batch_data),
        batch_keys=[f"prompt_a_{i}_0" for i in range(len(batch_data))],
        adv_estimator=adv_estimator,
        config=config,
    )
    expected = compute_advantage(
        copy.deepcopy(batch_data),
        adv_estimator=adv_estimator,
        config=config,
    )
    assert torch.equal(result.batch["advantages"], expected.batch["advantages"])
    assert torch.equal(result.batch["returns"], expected.batch["returns"])


@pytest.mark.parametrize("adv_estimator", [*SESSION_ESTIMATORS, "grpo_vectorized"])
def test_compute_advantage_for_multi_trajectories(batch_data: DataProto, adv_estimator):
    config = {} if adv_estimator == AdvantageEstimator.GRPO_PASSK else None
    result = compute_advantage_for_multi_trajectories(
        data=copy.deepcopy(batch_data),
        batch_keys=["prompt_a_0_0", "prompt_a_0_1", "prompt_a_2_0", "prompt_a_2_1", "prompt_a_3_0", "prompt_a_4_0"],
        adv_estimator=adv_estimator,
        config=config,
    )
    expected = compute_advantage(
        copy.deepcopy(batch_data).select_idxs([1, 3, 4, 5]),
        adv_estimator=adv_estimator,
        config=config,
    )
    gather_row_indices = [0, 0, 1, 1, 2, 3]
    gather_col_indices = [0, 0, 1, 1, 0, 0]
    adv_expected = (
        expected.batch["advantages"][gather_row_indices, gather_col_indices].unsqueeze(-1)
        * result.batch["response_mask"]
    )
    assert torch.equal(result.batch["advantages"], adv_expected)
    assert torch.equal(result.batch["returns"], adv_expected)


@pytest.mark.parametrize("adv_estimator", SESSION_ESTIMATORS)
def test_multi_trajectory_winning_intermediate_turn_gets_positive_advantage(adv_estimator: AdvantageEstimator):
    """Intermediate turns of a winning session must inherit the positive final session advantage."""
    tensors = {
        "token_level_rewards": torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],  # session 0, turn 0 (intermediate)
                [0.0, 0.0, 0.0, 1.0],  # session 0, turn 1 (final, winning)
                [0.0, 0.0, 0.0, 0.0],  # session 1, turn 0 (intermediate)
                [0.0, 0.0, 0.0, 0.0],  # session 1, turn 1 (final, losing)
            ],
            dtype=torch.float32,
        ),
        "response_mask": torch.ones((4, 4), dtype=torch.long),
    }
    data = DataProto.from_dict(tensors=tensors, non_tensors={"uid": np.array(["prompt_a"] * 4, dtype=object)})
    keys = ["prompt_a_0_0", "prompt_a_0_1", "prompt_a_1_0", "prompt_a_1_1"]
    config = {} if adv_estimator == AdvantageEstimator.GRPO_PASSK else None

    result = compute_advantage_for_multi_trajectories(data, keys, adv_estimator=adv_estimator, config=config)
    adv = result.batch["advantages"]
    # Turn 0 of winning session 0 matches turn 1 and is strictly positive
    assert torch.equal(adv[0], adv[1])
    assert adv[0, 0].item() > 0.0
    # Turn 0 of losing session 1 matches turn 1 and is non-positive
    assert torch.equal(adv[2], adv[3])
    assert adv[2, 0].item() <= 0.0


def test_compute_advantage_forwards_norm_adv_by_std_in_grpo():
    """norm_adv_by_std_in_grpo=False (Dr.GRPO) must be honored by grpo_vectorized, grpo_passk, and gdpo."""
    base = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.tensor([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]], dtype=torch.float32),
            "response_mask": torch.ones((3, 2), dtype=torch.long),
        },
        non_tensors={"uid": np.array(["p"] * 3, dtype=object)},
    )

    grpo_unnorm = compute_advantage(copy.deepcopy(base), AdvantageEstimator.GRPO, norm_adv_by_std_in_grpo=False)
    vec_unnorm = compute_advantage(
        copy.deepcopy(base), AdvantageEstimator.GRPO_VECTORIZED, norm_adv_by_std_in_grpo=False
    )
    vec_norm = compute_advantage(copy.deepcopy(base), AdvantageEstimator.GRPO_VECTORIZED, norm_adv_by_std_in_grpo=True)
    assert torch.allclose(vec_unnorm.batch["advantages"], grpo_unnorm.batch["advantages"])
    assert torch.allclose(vec_unnorm.batch["advantages"][:, 0], torch.tensor([-2.0, 0.0, 2.0]))
    assert torch.allclose(vec_norm.batch["advantages"][:, 0], torch.tensor([-1.0, 0.0, 1.0]), atol=1e-5)

    passk_unnorm = compute_advantage(
        copy.deepcopy(base), AdvantageEstimator.GRPO_PASSK, norm_adv_by_std_in_grpo=False, config={}
    )
    passk_norm = compute_advantage(
        copy.deepcopy(base), AdvantageEstimator.GRPO_PASSK, norm_adv_by_std_in_grpo=True, config={}
    )
    assert torch.allclose(passk_unnorm.batch["advantages"][:, 0], torch.tensor([0.0, 0.0, 2.0]))
    assert torch.allclose(passk_norm.batch["advantages"][:, 0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)

    # GDPO with 2 groups of different variances: std-normalization changes relative group scaling
    gdpo_data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 10.0]], dtype=torch.float32),
            "response_mask": torch.ones((4, 2), dtype=torch.long),
        },
        non_tensors={"uid": np.array(["g1", "g1", "g2", "g2"], dtype=object)},
    )
    gdpo_norm = compute_advantage(copy.deepcopy(gdpo_data), AdvantageEstimator.GDPO, norm_adv_by_std_in_grpo=True)
    gdpo_unnorm = compute_advantage(copy.deepcopy(gdpo_data), AdvantageEstimator.GDPO, norm_adv_by_std_in_grpo=False)
    assert not torch.allclose(gdpo_norm.batch["advantages"], gdpo_unnorm.batch["advantages"])
