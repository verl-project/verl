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
"""Guard distillation estimator/mode compatibility and valid policy gradients."""

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


@pytest.mark.parametrize("loss_mode", ["k3", "low_var_kl"])
@pytest.mark.parametrize("from_omegaconf", [False, True])
def test_k3_aliases_reject_policy_gradient(loss_mode, from_omegaconf):
    """Reject detached k3 rewards through direct and OmegaConf construction."""
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
    "loss_mode,use_policy_gradient", [("k1", True), ("kl", True), ("k3", False), ("low_var_kl", False)]
)
def test_supported_distillation_modes_remain_available(loss_mode, use_policy_gradient):
    """The guard must preserve PG k1 aliases and supervised k3 aliases."""
    config = DistillationLossConfig(loss_mode=loss_mode, use_policy_gradient=use_policy_gradient)
    assert config.loss_settings.use_estimator
    assert not config.loss_settings.use_topk


@pytest.mark.parametrize("loss_mode", ["k1", "kl"])
@pytest.mark.parametrize(
    "student_probabilities,teacher_probabilities",
    [
        ([0.5, 0.5], [0.5, 0.5]),
        ([0.6, 0.4], [0.55, 0.45]),
        ([0.99, 0.01], [0.01, 0.99]),
        ([0.8, 0.2], [0.2, 0.8]),
        ([0.2, 0.8], [0.8, 0.2]),
        ([0.2, 0.3, 0.5], [0.5, 0.4, 0.1]),
    ],
    ids=["equal", "mild", "severe", "opposing", "swapped", "three-actions"],
)
def test_k1_alias_policy_gradient_matches_reverse_kl(loss_mode, student_probabilities, teacher_probabilities):
    """The exact expected API gradient must match autograd and finite differences of reverse KL."""
    student_probs = torch.tensor(student_probabilities, dtype=torch.float64)
    teacher_probs = torch.tensor(teacher_probabilities, dtype=torch.float64)
    logits = student_probs.log().requires_grad_()
    log_probs = logits.log_softmax(dim=-1)
    teacher_log_probs = teacher_probs.log()
    reference_kl = (log_probs.exp() * (log_probs - teacher_log_probs)).sum()
    reference_gradient = torch.autograd.grad(reference_kl, logits, retain_graph=True)[0]

    finite_difference = torch.empty_like(logits)
    for index in range(logits.numel()):
        delta = torch.zeros_like(logits)
        delta[index] = 1e-6
        plus = (logits.detach() + delta).log_softmax(dim=-1)
        minus = (logits.detach() - delta).log_softmax(dim=-1)
        plus_kl = (plus.exp() * (plus - teacher_log_probs)).sum()
        minus_kl = (minus.exp() * (minus - teacher_log_probs)).sum()
        finite_difference[index] = (plus_kl - minus_kl) / 2e-6
    torch.testing.assert_close(reference_gradient, finite_difference, rtol=1e-7, atol=1e-10)

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
    # Validated configuration only; no model, worker, distributed group, or GPU is created.
    expected_loss = logits.sum() * 0.0
    for action in range(student_probs.numel()):
        sampled_log_prob = log_probs[action]
        ratio = (sampled_log_prob - sampled_log_prob.detach()).exp()
        assert 1.0 - config.distillation_loss.clip_ratio_low < ratio.item()
        assert ratio.item() < 1.0 + config.distillation_loss.clip_ratio_high
        data = TensorDict(
            {
                "prompts": torch.tensor([[0]]),
                "responses": torch.tensor([[action]]),
                "attention_mask": torch.ones(1, 2, dtype=torch.int64),
                "response_mask": torch.ones(1, 1, dtype=torch.bool),
                "old_log_probs": sampled_log_prob.detach().reshape(1, 1),
                "teacher_logprobs": torch.stack([teacher_log_probs[action], teacher_log_probs.new_zeros(())]).reshape(
                    2, 1
                ),
            },
            batch_size=[],
        )
        tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=1, global_batch_size=1)
        output = {"log_probs": torch.stack([sampled_log_prob, sampled_log_prob.new_zeros(())])}
        loss, _ = distillation_loss(actor, config, output, data)
        # Sampling weights are fixed when taking the expectation of the SGD estimator.
        expected_loss = expected_loss + student_probs[action].detach() * loss

    actual_gradient = torch.autograd.grad(expected_loss, logits)[0]
    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=1e-10, atol=1e-12)
    updated_log_probs = (logits.detach() - 0.01 * actual_gradient).log_softmax(dim=-1)
    updated_kl = (updated_log_probs.exp() * (updated_log_probs - teacher_log_probs)).sum()
    if torch.equal(student_probs, teacher_probs):
        torch.testing.assert_close(actual_gradient, torch.zeros_like(actual_gradient), atol=1e-12, rtol=0)
        torch.testing.assert_close(updated_kl, reference_kl.detach(), atol=1e-12, rtol=0)
    else:
        assert updated_kl < reference_kl.detach()
