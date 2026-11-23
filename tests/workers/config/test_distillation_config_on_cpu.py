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
"""Check distillation loss compatibility and signed policy-gradient rewards."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.distillation.losses import distillation_loss
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.config.distillation import DistillationTeacherModelConfig
from verl.workers.config.rollout import RolloutConfig


@pytest.mark.parametrize("loss_mode", ["k1", "kl"])
@pytest.mark.parametrize("from_omegaconf", [False, True])
def test_k1_aliases_reject_direct_backpropagation(loss_mode, from_omegaconf):
    """Both estimator names must fail before training, including config conversion."""
    settings = {"loss_mode": loss_mode, "use_policy_gradient": False}
    with pytest.raises(ValueError, match="Directly backpropagating.*teacher log probabilities"):
        if from_omegaconf:
            omega_conf_to_dataclass(OmegaConf.create(settings), dataclass_type=DistillationLossConfig)
        else:
            DistillationLossConfig(**settings)


@pytest.mark.parametrize("loss_mode", ["abs", "mse", "k2", "k3", "low_var_kl"])
@pytest.mark.parametrize("from_omegaconf", [False, True])
def test_nonnegative_losses_reject_policy_gradient(loss_mode, from_omegaconf):
    """Reject non-negative single-token losses as policy-gradient rewards."""
    settings = {"loss_mode": loss_mode, "use_policy_gradient": True}
    with pytest.raises(ValueError, match="reverse-KL policy gradient"):
        if from_omegaconf:
            omega_conf_to_dataclass(OmegaConf.create(settings), dataclass_type=DistillationLossConfig)
        else:
            DistillationLossConfig(**settings)


def test_default_distillation_mode_matches_yaml():
    """Keep Python construction compatible with the existing YAML defaults."""
    yaml_config = OmegaConf.load(
        Path(__file__).resolve().parents[3] / "verl/trainer/config/distillation/distillation.yaml"
    )
    config = DistillationLossConfig()
    assert config.loss_mode == yaml_config.distillation_loss.loss_mode == "k3"
    assert config.use_policy_gradient is yaml_config.distillation_loss.use_policy_gradient is False


@pytest.mark.parametrize(
    "loss_mode,use_policy_gradient",
    [
        ("k1", True),
        ("kl", True),
        ("abs", False),
        ("mse", False),
        ("k2", False),
        ("k3", False),
        ("low_var_kl", False),
    ],
)
def test_supported_distillation_modes_remain_available(loss_mode, use_policy_gradient):
    """Allow signed k1 rewards and direct backpropagation of non-negative losses."""
    config = DistillationLossConfig(loss_mode=loss_mode, use_policy_gradient=use_policy_gradient)
    assert config.loss_settings.use_estimator
    assert not config.loss_settings.use_topk


@pytest.mark.parametrize("loss_mode", ["k1", "kl"])
@pytest.mark.parametrize("student_prob,teacher_prob", [(0.8, 0.2), (0.2, 0.8)], ids=["over", "under"])
def test_policy_gradient_k1_aliases_use_signed_log_ratio(loss_mode, student_prob, teacher_prob):
    """At ratio one, the policy loss must equal the signed student-teacher log ratio."""
    student_log_prob, teacher_log_prob = torch.tensor([student_prob, teacher_prob], dtype=torch.float64).log()
    padding = torch.zeros_like(student_log_prob)
    actor = ActorConfig(strategy="fsdp", rollout_n=1, ppo_mini_batch_size=1, ppo_micro_batch_size_per_gpu=1)
    config = DistillationConfig(
        enabled=True,
        n_gpus_per_node=1,
        nnodes=1,
        teacher_models={
            "teacher_model": DistillationTeacherModelConfig(
                model_path="synthetic-teacher",
                inference=RolloutConfig(name="vllm", tensor_model_parallel_size=1),
            )
        },
        distillation_loss=DistillationLossConfig(
            loss_mode=loss_mode,
            use_policy_gradient=True,
            use_task_rewards=False,
            loss_max_clamp=None,
            log_prob_min_clamp=None,
        ),
    )
    data = TensorDict(
        {
            "prompts": torch.tensor([[0]]),
            "responses": torch.tensor([[1]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.int64),
            "response_mask": torch.ones(1, 1, dtype=torch.bool),
            "old_log_probs": student_log_prob.reshape(1, 1),
            "teacher_logprobs": torch.stack([teacher_log_prob, padding]).unsqueeze(-1),
        },
        batch_size=[],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=1, global_batch_size=1)
    output = {"log_probs": torch.stack([student_log_prob, padding])}
    loss, _ = distillation_loss(actor, config, output, data)
    torch.testing.assert_close(loss, student_log_prob - teacher_log_prob)
