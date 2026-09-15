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

import pytest
import torch

import verl.workers.utils.losses as losses_module
from verl.utils.tensordict_utils import get_tensordict
from verl.workers.config.actor import ActorConfig
from verl.workers.utils.losses import ppo_loss


def test_ppo_loss_applies_per_sample_loss_weight_to_policy_gradient(monkeypatch):
    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=2,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        loss_agg_mode="token-mean",
    )
    data = get_tensordict(
        {
            "response_mask": torch.ones(2, 2),
            "old_log_probs": torch.zeros(2, 2),
            "advantages": torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
            "loss_weight": torch.tensor([0.25, 0.5]),
        },
        non_tensor_dict={
            "dp_size": 1,
            "batch_num_tokens": None,
            "global_batch_size": None,
        },
    )

    monkeypatch.setattr(losses_module, "no_padding_2_padding", lambda tensor, _data: tensor)
    loss, _ = ppo_loss(config, {"log_probs": torch.zeros(2, 2)}, data)

    # ratio == 1, so the explicit weighted token-mean is
    # -(0.25 * 1 * 2 + 0.5 * 2 * 2) / 4 == -0.625.
    assert loss.item() == pytest.approx(-0.625)
