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

"""CPU coverage for selecting standard PPO instead of dual-clip PPO."""

import pytest
import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.workers.config.actor import ActorConfig, PolicyLossConfig


def _actor_config(*, clip_ratio_c) -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=clip_ratio_c,
        loss_agg_mode="token-mean",
        policy_loss=PolicyLossConfig(loss_mode="vanilla"),
    )


def test_vanilla_without_dual_clip_matches_standard_ppo():
    old_log_prob = torch.tensor([[0.0]])
    log_prob = torch.tensor([[torch.log(torch.tensor(10.0))]])
    advantages = torch.tensor([[-1.0]])
    response_mask = torch.ones_like(advantages)

    standard_loss, standard_metrics = compute_policy_loss_vanilla(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        "token-mean",
        _actor_config(clip_ratio_c=None),
    )
    dual_clip_loss, _ = compute_policy_loss_vanilla(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        "token-mean",
        _actor_config(clip_ratio_c=3.0),
    )

    assert standard_loss.item() == pytest.approx(10.0)
    assert dual_clip_loss.item() == pytest.approx(3.0)
    assert standard_metrics["actor/pg_clipfrac_lower"] == 0.0
