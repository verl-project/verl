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

"""Validate the score-centering example's sparse actor mirror against real Hydra defaults."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from verl.utils.config import _validate_score_centering_config


@pytest.mark.parametrize("top_p, top_k", [(1.0, -1), (0.9, -1), (0.9, 64), (1.0, 256)])
def test_example_mirror_accepts_real_hydra_defaults_without_a_manual_centering_mode(top_p, top_k):
    config_dir = Path(__file__).resolve().parents[3] / "verl" / "trainer" / "config"
    # The example deliberately passes only the existing correction fields to the actor mirror.
    # New algorithm YAML defaults must not require another manually synchronized mode field.
    overrides = [
        "algorithm.rollout_correction.bypass_mode=true",
        "algorithm.rollout_correction.loss_type=reinforce",
        "algorithm.rollout_correction.rollout_is=token",
        "algorithm.rollout_correction.rollout_is_threshold=2.0",
        "algorithm.rollout_correction.score_centering=true",
        "actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode",
        "+actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=true",
        "+actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=reinforce",
        "+actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is=token",
        "+actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold=2.0",
        "+actor_rollout_ref.actor.policy_loss.rollout_correction.score_centering=true",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.topk_log_probs=128",
        "actor_rollout_ref.rollout.calculate_log_probs=true",
        f"actor_rollout_ref.rollout.top_p={top_p}",
        f"actor_rollout_ref.rollout.top_k={top_k}",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    _validate_score_centering_config(config)
