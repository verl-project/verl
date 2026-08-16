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

"""CPU coverage for dynamically selected policy losses."""

import math

import pytest
import torch

from verl.trainer.ppo.core_algos import (
    compute_policy_loss_cispo,
    compute_policy_loss_dro,
    get_policy_loss_fn,
)
from verl.workers.config.actor import ActorConfig, PolicyLossConfig


def _actor_config(*, loss_mode: str, dro_beta=None) -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        loss_agg_mode="token-mean",
        policy_loss=PolicyLossConfig(loss_mode=loss_mode, dro_beta=dro_beta),
    )


def test_dro_matches_direct_formula_and_requires_positive_beta():
    old_log_prob = torch.tensor([[-1.0, -0.7, -0.2]])
    log_prob = torch.tensor([[-0.8, -0.9, -0.1]], requires_grad=True)
    advantages = torch.tensor([[2.0, -1.5, 9.0]])
    response_mask = torch.tensor([[1.0, 1.0, 0.0]])
    beta = 0.05
    config = _actor_config(loss_mode="dro", dro_beta=beta)

    loss, _ = compute_policy_loss_dro(old_log_prob, log_prob, advantages, response_mask, "token-mean", config)
    log_ratio = log_prob - old_log_prob
    expected = (-(log_prob * advantages - 0.5 * beta * log_ratio.square()) * response_mask).sum()
    expected = expected / response_mask.sum()
    torch.testing.assert_close(loss, expected)

    with pytest.raises(ValueError, match="dro_beta"):
        compute_policy_loss_dro(
            old_log_prob,
            log_prob,
            advantages,
            response_mask,
            "token-mean",
            _actor_config(loss_mode="dro"),
        )


def test_cispo_clips_ratio_and_stops_ratio_gradient():
    old_log_prob = torch.zeros((1, 3), dtype=torch.float32)
    log_prob = torch.tensor(
        [[math.log(2.0), math.log(0.5), 0.25]],
        dtype=torch.float32,
        requires_grad=True,
    )
    advantages = torch.tensor([[1.0, -1.0, 5.0]], dtype=torch.float32)
    response_mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
    config = _actor_config(loss_mode="cispo")
    config.clip_ratio_low = 0.1
    config.clip_ratio_high = 0.2

    loss, metrics = compute_policy_loss_cispo(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
    )

    expected_loss = (-1.2 * math.log(2.0) + 0.9 * math.log(0.5)) / 2.0
    assert loss.item() == pytest.approx(expected_loss, abs=1e-6)

    loss.backward()
    expected_gradient = torch.tensor([[-0.6, 0.45, 0.0]])
    torch.testing.assert_close(log_prob.grad, expected_gradient)
    assert metrics["actor/pg_clipfrac"] == pytest.approx(1.0)
    assert metrics["actor/ppo_kl"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["actor/pg_clipfrac_lower"] == pytest.approx(0.0)


def test_cispo_is_registered():
    assert get_policy_loss_fn("cispo") is compute_policy_loss_cispo
