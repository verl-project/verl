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


@pytest.mark.parametrize(
    "adv_estimator",
    [
        AdvantageEstimator.GRPO,
        AdvantageEstimator.GRPO_VECTORIZED,
        AdvantageEstimator.RLOO,
        AdvantageEstimator.RLOO_VECTORIZED,
        AdvantageEstimator.GRPO_PASSK,
        AdvantageEstimator.GPG,
    ],
)
def test_compute_advantage_for_single_trajectory(batch_data: DataProto, adv_estimator: AdvantageEstimator):
    config = {"norm_adv_by_std_in_grpo": True}
    result = compute_advantage_for_multi_trajectories(
        data=batch_data,
        batch_keys=[f"prompt_a_{i}_0" for i in range(len(batch_data))],
        adv_estimator=adv_estimator,
        config=config,
    )
    expected = compute_advantage(
        batch_data,
        adv_estimator=adv_estimator,
        config=config,
    )
    torch.testing.assert_close(result.batch["advantages"], expected.batch["advantages"])
    torch.testing.assert_close(result.batch["returns"], expected.batch["returns"])


@pytest.mark.parametrize(
    "adv_estimator",
    [
        AdvantageEstimator.GRPO,
        AdvantageEstimator.GRPO_VECTORIZED,
        AdvantageEstimator.RLOO,
        AdvantageEstimator.RLOO_VECTORIZED,
        AdvantageEstimator.GRPO_PASSK,
        AdvantageEstimator.GPG,
    ],
)
def test_compute_advantage_for_multi_trajectories(batch_data: DataProto, adv_estimator: AdvantageEstimator):
    config = {"norm_adv_by_std_in_grpo": True}
    result = compute_advantage_for_multi_trajectories(
        data=batch_data,
        batch_keys=["prompt_a_0_0", "prompt_a_0_1", "prompt_a_2_0", "prompt_a_2_1", "prompt_a_3_0", "prompt_a_4_0"],
        adv_estimator=adv_estimator,
        config=config,
    )
    expected = compute_advantage(
        batch_data.select_idxs([1, 3, 4, 5]),
        adv_estimator=adv_estimator,
        config=config,
    )
    gather_row_indices = [0, 0, 1, 1, 2, 3]
    gather_col_indices = [0, 0, 1, 1, 0, 0]
    adv_expected = (
        expected.batch["advantages"][gather_row_indices, gather_col_indices].unsqueeze(-1)
        * result.batch["response_mask"]
    )
    torch.testing.assert_close(result.batch["advantages"], adv_expected)
    torch.testing.assert_close(result.batch["returns"], adv_expected)


@pytest.mark.parametrize("norm_adv_by_std_in_grpo", [True, False])
def test_multi_trajectories_grpo_and_rloo_vectorized_equivalence(batch_data: DataProto, norm_adv_by_std_in_grpo: bool):
    batch_keys = ["prompt_a_0_0", "prompt_a_0_1", "prompt_a_2_0", "prompt_a_2_1", "prompt_a_3_0", "prompt_a_4_0"]
    grpo_res = compute_advantage_for_multi_trajectories(
        data=batch_data,
        batch_keys=batch_keys,
        adv_estimator=AdvantageEstimator.GRPO,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
    grpo_vec_res = compute_advantage_for_multi_trajectories(
        data=batch_data,
        batch_keys=batch_keys,
        adv_estimator=AdvantageEstimator.GRPO_VECTORIZED,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
    torch.testing.assert_close(grpo_res.batch["advantages"], grpo_vec_res.batch["advantages"])
    torch.testing.assert_close(grpo_res.batch["returns"], grpo_vec_res.batch["returns"])

    rloo_res = compute_advantage_for_multi_trajectories(
        data=batch_data,
        batch_keys=batch_keys,
        adv_estimator=AdvantageEstimator.RLOO,
    )
    rloo_vec_res = compute_advantage_for_multi_trajectories(
        data=batch_data,
        batch_keys=batch_keys,
        adv_estimator=AdvantageEstimator.RLOO_VECTORIZED,
    )
    torch.testing.assert_close(rloo_res.batch["advantages"], rloo_vec_res.batch["advantages"])
    torch.testing.assert_close(rloo_res.batch["returns"], rloo_vec_res.batch["returns"])
