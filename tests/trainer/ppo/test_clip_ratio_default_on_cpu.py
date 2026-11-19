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

"""`actor.clip_ratio` must be the effective clip range unless the asymmetric bounds are set."""

import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.workers.config.actor import ActorConfig
from verl.workers.config.distillation import DistillationLossConfig


def _actor_config(**kwargs) -> ActorConfig:
    """Build a valid actor config with the requested clipping settings."""
    return ActorConfig(
        strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1, loss_agg_mode="token-mean", **kwargs
    )


def _clipfrac(config: ActorConfig) -> float:
    """Measure the clipped fraction for a fixed positive-advantage PPO update."""
    # ratio = exp(0.4) = 1.49 with a positive advantage: clipped at 0.2 or 0.3, not at 0.5.
    old_log_prob = torch.zeros(1, 1)
    log_prob = torch.full((1, 1), 0.4)
    advantages = torch.ones(1, 1)
    response_mask = torch.ones(1, 1)
    _, metrics = compute_policy_loss_vanilla(old_log_prob, log_prob, advantages, response_mask, "token-mean", config)
    return metrics["actor/pg_clipfrac"]


def test_clip_ratio_alone_sets_the_clip_range():
    """A symmetric clip override changes the effective PPO clipping range."""
    assert _clipfrac(_actor_config()) == 1.0
    assert _clipfrac(_actor_config(clip_ratio=0.5)) == 0.0


def test_explicit_bounds_still_override_clip_ratio():
    """Explicit asymmetric bounds take precedence over the symmetric setting."""
    assert _clipfrac(_actor_config(clip_ratio=0.5, clip_ratio_low=0.1, clip_ratio_high=0.3)) == 1.0


def test_distillation_bounds_default_to_clip_ratio():
    """Distillation leaves both asymmetric bounds unset for the shared fallback."""
    config = DistillationLossConfig(clip_ratio=0.3)
    assert config.clip_ratio_low is None
    assert config.clip_ratio_high is None
