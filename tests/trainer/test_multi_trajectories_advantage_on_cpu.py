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
from verl.trainer.ppo import core_algos
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


@pytest.mark.parametrize(
    "adv_estimator",
    [
        AdvantageEstimator.GRPO_VECTORIZED,
        AdvantageEstimator.RLOO,
        AdvantageEstimator.RLOO_VECTORIZED,
        AdvantageEstimator.GPG,
    ],
)
def test_compute_advantage_for_multi_trajectories_supports_session_estimators(
    batch_data: DataProto, adv_estimator: AdvantageEstimator
):
    batch_keys = ["prompt_a_0_0", "prompt_a_0_1", "prompt_a_2_0", "prompt_a_2_1", "prompt_a_3_0", "prompt_a_4_0"]
    result = compute_advantage_for_multi_trajectories(
        data=batch_data,
        batch_keys=batch_keys,
        adv_estimator=adv_estimator,
    )
    expected = compute_advantage(
        batch_data.select_idxs([1, 3, 4, 5]),
        adv_estimator=adv_estimator,
    )
    gather_row_indices = [0, 0, 1, 1, 2, 3]
    gather_col_indices = [0, 0, 1, 1, 0, 0]
    adv_expected = (
        expected.batch["advantages"][gather_row_indices, gather_col_indices].unsqueeze(-1)
        * result.batch["response_mask"]
    )
    assert torch.equal(result.batch["advantages"], adv_expected)
    assert torch.equal(result.batch["returns"], adv_expected)


def test_compute_advantage_for_multi_trajectories_supports_grpo_passk(batch_data: DataProto):
    config = {"norm_adv_by_std_in_grpo": True}
    batch_keys = ["prompt_a_0_0", "prompt_a_0_1", "prompt_a_2_0", "prompt_a_2_1", "prompt_a_3_0", "prompt_a_4_0"]
    result = compute_advantage_for_multi_trajectories(
        data=batch_data,
        batch_keys=batch_keys,
        adv_estimator=AdvantageEstimator.GRPO_PASSK,
        config=config,
    )
    expected = compute_advantage(
        batch_data.select_idxs([1, 3, 4, 5]),
        adv_estimator=AdvantageEstimator.GRPO_PASSK,
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


def test_norm_adv_by_std_is_forwarded_to_grpo_vectorized(batch_data: DataProto):
    result = compute_advantage(
        batch_data,
        adv_estimator=AdvantageEstimator.GRPO_VECTORIZED,
        norm_adv_by_std_in_grpo=False,
    )
    expected_advantages, expected_returns = core_algos.compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=batch_data.batch["token_level_rewards"],
        response_mask=batch_data.batch["response_mask"],
        index=batch_data.non_tensor_batch["uid"],
        norm_adv_by_std_in_grpo=False,
    )
    assert torch.equal(result.batch["advantages"], expected_advantages)
    assert torch.equal(result.batch["returns"], expected_returns)


def test_norm_adv_by_std_is_forwarded_to_grpo_passk(batch_data: DataProto):
    config = {"norm_adv_by_std_in_grpo": True}
    result = compute_advantage(
        batch_data,
        adv_estimator=AdvantageEstimator.GRPO_PASSK,
        norm_adv_by_std_in_grpo=False,
        config=config,
    )
    expected_advantages, expected_returns = core_algos.compute_grpo_passk_outcome_advantage(
        token_level_rewards=batch_data.batch["token_level_rewards"],
        response_mask=batch_data.batch["response_mask"],
        index=batch_data.non_tensor_batch["uid"],
        norm_adv_by_std_in_grpo=False,
        config=config,
    )
    assert torch.equal(result.batch["advantages"], expected_advantages)
    assert torch.equal(result.batch["returns"], expected_returns)
    # Explicit raw Pass@k result for this fixture: only the winning response
    # (row 2, reward 400) receives r_max - r_second_max = 100 on its mask.
    raw_expected = torch.zeros_like(result.batch["advantages"])
    raw_expected[2, 0] = 100.0
    raw_expected[2, 2] = 100.0
    assert torch.equal(result.batch["advantages"], raw_expected)


def test_multi_trajectory_grpo_passk_skips_singleton_uid_groups():
    tensors = {
        "token_level_rewards": torch.tensor(
            [
                [4.0, 0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0, 0.0],
                [7.0, 0.0, 0.0, 0.0],
            ]
        ),
        "response_mask": torch.ones((3, 4), dtype=torch.long),
    }
    data = DataProto.from_dict(
        tensors=tensors,
        non_tensors={"uid": np.array(["p", "q", "q"], dtype=object)},
    )

    result = compute_advantage_for_multi_trajectories(
        data=data,
        batch_keys=["p_0_0", "q_0_0", "q_1_0"],
        adv_estimator=AdvantageEstimator.GRPO_PASSK,
        norm_adv_by_std_in_grpo=False,
        config={"norm_adv_by_std_in_grpo": False},
    )

    # p has one surviving session and cannot form Pass@k; it must not abort the
    # batch and receives zero advantage. q keeps its two sessions.
    assert result.batch["advantages"][0].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert result.batch["advantages"][1].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert result.batch["advantages"][2].tolist() == [2.0, 2.0, 2.0, 2.0]


def test_multi_trajectory_rloo_vectorized_matches_rloo_for_single_session():
    tensors = {
        "token_level_rewards": torch.tensor([[0.0, 0.0, 0.0, 4.0]]),
        "response_mask": torch.ones((1, 4), dtype=torch.long),
    }
    data = DataProto.from_dict(
        tensors=tensors,
        non_tensors={"uid": np.array(["p"], dtype=object)},
    )

    scalar = compute_advantage_for_multi_trajectories(
        data=data,
        batch_keys=["p_0_1"],
        adv_estimator=AdvantageEstimator.RLOO,
    )
    # Both calls mutate and return the same DataProto; snapshot the scalar result
    # before the vectorized call, otherwise the comparison is against itself.
    scalar_advantages = scalar.batch["advantages"].clone()
    vectorized = compute_advantage_for_multi_trajectories(
        data=data,
        batch_keys=["p_0_1"],
        adv_estimator=AdvantageEstimator.RLOO_VECTORIZED,
    )

    assert torch.equal(vectorized.batch["advantages"], scalar_advantages)
    assert scalar.batch["advantages"].tolist() == [[4.0, 4.0, 4.0, 4.0]]
