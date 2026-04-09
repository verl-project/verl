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

import os

import numpy as np
import pytest
import torch

from verl.trainer.diffusion import diffusion_algos


@pytest.mark.parametrize("norm_adv_by_std_in_grpo", [True, False])
@pytest.mark.parametrize("global_std", [True, False])
def test_flow_grpo_advantage_return(norm_adv_by_std_in_grpo: bool, global_std: bool) -> None:
    batch_size = 8
    steps = 10
    sample_level_rewards = torch.randn((batch_size, 1), dtype=torch.float32)
    response_mask = torch.ones((batch_size, steps), dtype=torch.int32)
    uid = np.array([f"uid-{idx}" for idx in range(batch_size)], dtype=object)

    advantages, returns = diffusion_algos.compute_flow_grpo_outcome_advantage(
        sample_level_rewards=sample_level_rewards,
        response_mask=response_mask,
        index=uid,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        global_std=global_std,
    )

    assert advantages.shape == returns.shape == (batch_size, steps)


def test_compute_policy_loss_flow_grpo() -> None:
    from hydra import compose, initialize_config_dir

    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.config.actor import FSDPActorConfig

    batch_size = 8
    steps = 10
    rollout_log_probs = torch.randn((batch_size, steps), dtype=torch.float32)
    current_log_probs = torch.randn((batch_size, steps), dtype=torch.float32)
    advantages = torch.randn((batch_size, steps), dtype=torch.float32)
    response_mask = torch.ones((batch_size, steps), dtype=torch.int32)

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor"), version_base=None):
        cfg = compose(
            config_name="dp_actor",
            overrides=[
                "strategy=fsdp",
                "clip_ratio=0.0001",
                "clip_ratio_high=5.0",
                "ppo_micro_batch_size_per_gpu=8",
            ],
        )
    actor_config: FSDPActorConfig = omega_conf_to_dataclass(cfg)

    for step in range(steps):
        pg_loss, pg_metrics = diffusion_algos.compute_policy_loss_flow_grpo(
            old_log_prob=rollout_log_probs[:, step],
            log_prob=current_log_probs[:, step],
            advantages=advantages[:, step],
            response_mask=response_mask[:, step],
            loss_agg_mode="token-mean",
            config=actor_config,
        )

        assert pg_loss.shape == ()
        assert isinstance(pg_loss.item(), float)
        assert "actor/ppo_kl" in pg_metrics
