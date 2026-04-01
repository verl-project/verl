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
"""Diffusion-specific policy loss functions and KL penalties."""

from typing import Any, Optional

import torch
from omegaconf import DictConfig

from verl.trainer.ppo.core_algos import register_policy_loss
from verl.workers.config import ActorConfig


@register_policy_loss("flow_grpo")
def compute_policy_loss_flow_grpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for FlowGRPO.
    Adapted from
    https://github.com/yifan123/flow_grpo/blob/main/scripts/train_sd3_fast.py#L885
    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size,).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size,).
        response_mask (torch.Tensor):
            Not used currently.
        loss_agg_mode (str, optional):
            Not used currently.
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size,).
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_is_weights: `torch.Tensor, optional)`:
            Not used currently.
    """
    assert config is not None
    assert isinstance(config, ActorConfig)
    advantages = torch.clamp(
        advantages,
        -config.clip_ratio_high,
        config.clip_ratio_high,
    )
    log_ratio = log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    unclipped_loss = -advantages * ratio
    clipped_loss = -advantages * torch.clamp(
        ratio,
        1.0 - config.clip_ratio,
        1.0 + config.clip_ratio,
    )
    pg_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

    with torch.no_grad():
        ppo_kl = torch.mean(-log_ratio)
        pg_clipfrac = torch.mean((torch.abs(ratio - 1.0) > config.clip_ratio).float())
        pg_clipfrac_higher = torch.mean((ratio - 1.0 > config.clip_ratio).float())
        pg_clipfrac_lower = torch.mean((1.0 - ratio > config.clip_ratio).float())

    pg_metrics = {
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/pg_clipfrac_higher": pg_clipfrac_higher.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


def kl_penalty_image(
    prev_sample_mean: torch.Tensor, ref_prev_sample_mean: torch.Tensor, std_dev_t: torch.Tensor
) -> torch.Tensor:
    """Compute KL divergence given previous sample mean and reference previous sample mean (for images or videos).
    Args:
        prev_sample_mean: (torch.Tensor) shape is (bs, s, c)
        ref_prev_sample_mean: (torch.Tensor) shape is (bs, s, c)
        std_dev_t: (torch.Tensor) shape is (bs, 1, 1)
    """
    kl_loss = ((prev_sample_mean - ref_prev_sample_mean) ** 2).mean(dim=(1, 2), keepdim=True) / (2 * std_dev_t**2)
    return kl_loss.mean()
