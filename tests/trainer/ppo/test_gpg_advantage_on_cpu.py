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

from verl.trainer.ppo.core_algos import compute_gpg_outcome_advantage


def test_gpg_singleton_group_returns_raw_score():
    token_level_rewards = torch.tensor([[0.0, 3.0, 0.0]], dtype=torch.float32)
    response_mask = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    index = np.array(["prompt-a"], dtype=object)

    advantages, returns = compute_gpg_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    raw_score = token_level_rewards.sum(dim=-1, keepdim=True) * response_mask
    torch.testing.assert_close(advantages, raw_score)
    torch.testing.assert_close(returns, advantages)


def test_gpg_applies_n_over_nonzero_scaling():
    token_level_rewards = torch.tensor([[4.0], [0.0]], dtype=torch.float32)
    response_mask = torch.ones_like(token_level_rewards)
    index = np.array(["prompt-a", "prompt-a"], dtype=object)

    advantages, _ = compute_gpg_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    expected = torch.tensor([[4.0], [-4.0]], dtype=torch.float32)
    torch.testing.assert_close(advantages, expected)
