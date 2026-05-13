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

import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.core_algos import FixedKLController
from verl.trainer.ppo.ray_trainer import apply_kl_penalty


def test_apply_kl_penalty_masks_invalid_logprob_nans_on_cpu():
    data = DataProto(
        batch=TensorDict(
            {
                "response_mask": torch.tensor([[1, 0]], dtype=torch.long),
                "token_level_scores": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                "old_log_probs": torch.tensor([[0.0, float("nan")]], dtype=torch.float32),
                "ref_log_prob": torch.tensor([[0.0, float("nan")]], dtype=torch.float32),
            },
            batch_size=1,
        )
    )

    data, metrics = apply_kl_penalty(data, kl_ctrl=FixedKLController(kl_coef=0.1), kl_penalty="kl")

    assert torch.isfinite(data.batch["token_level_rewards"]).all()
    assert data.batch["token_level_rewards"].tolist() == [[1.0, 0.0]]
    assert metrics["actor/reward_kl_penalty"] == 0.0
