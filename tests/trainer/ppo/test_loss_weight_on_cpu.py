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


def _scalar(metric):
    (value,) = metric.values
    return float(value)


def _config(**overrides):
    base = dict(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=2,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        loss_agg_mode="token-mean",
    )
    base.update(overrides)
    return ActorConfig(**base)


def _data(**extra):
    tensors = {
        "response_mask": torch.ones(2, 2),
        "old_log_probs": torch.zeros(2, 2),
        # zero advantages: the pg term vanishes so entropy / KL can be read in isolation
        "advantages": torch.zeros(2, 2),
        "loss_weight": torch.tensor([0.25, 0.5]),
    }
    tensors.update(extra)
    return get_tensordict(
        tensors,
        non_tensor_dict={"dp_size": 1, "batch_num_tokens": None, "global_batch_size": None},
    )


def test_ppo_loss_applies_loss_weight_to_entropy_term(monkeypatch):
    """The entropy bonus is weighted per sample like the policy-gradient term."""
    monkeypatch.setattr(losses_module, "no_padding_2_padding", lambda tensor, _data: tensor)
    config = _config(entropy_coeff=1.0)
    entropy = torch.tensor([[1.0, 1.0], [3.0, 3.0]])

    _, metrics = ppo_loss(config, {"log_probs": torch.zeros(2, 2), "entropy": entropy}, _data())

    # weighted token-mean: (0.25 * 1 * 2 + 0.5 * 3 * 2) / 4 == 0.875 (unweighted would be 2.0)
    assert _scalar(metrics["actor/entropy_loss"]) == pytest.approx(0.875)


def test_ppo_loss_applies_loss_weight_to_kl_term(monkeypatch):
    """The KL penalty is weighted per sample like the policy-gradient term."""
    monkeypatch.setattr(losses_module, "no_padding_2_padding", lambda tensor, _data: tensor)
    config = _config(use_kl_loss=True, kl_loss_coef=1.0, kl_loss_type="kl")
    # kl_penalty("kl") == logprob - ref_logprob, so pick ref so the per-token kl is [1, 1] / [3, 3]
    data = _data(ref_log_prob=torch.tensor([[-1.0, -1.0], [-3.0, -3.0]]))

    _, metrics = ppo_loss(config, {"log_probs": torch.zeros(2, 2)}, data)

    assert _scalar(metrics["kl_loss"]) == pytest.approx(0.875)


def test_ppo_loss_neutral_weights_are_a_no_op_for_entropy_and_kl(monkeypatch):
    """With all-ones weights the entropy and KL terms match the unweighted computation exactly."""
    monkeypatch.setattr(losses_module, "no_padding_2_padding", lambda tensor, _data: tensor)
    config = _config(entropy_coeff=1.0, use_kl_loss=True, kl_loss_coef=1.0, kl_loss_type="kl")
    entropy = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
    ref = torch.tensor([[-1.0, -1.0], [-3.0, -3.0]])

    weighted = _data(ref_log_prob=ref, loss_weight=torch.ones(2))
    unweighted = _data(ref_log_prob=ref)
    del unweighted["loss_weight"]

    _, m_w = ppo_loss(config, {"log_probs": torch.zeros(2, 2), "entropy": entropy}, weighted)
    _, m_u = ppo_loss(config, {"log_probs": torch.zeros(2, 2), "entropy": entropy}, unweighted)

    assert _scalar(m_w["actor/entropy_loss"]) == _scalar(m_u["actor/entropy_loss"]) == pytest.approx(2.0)
    assert _scalar(m_w["kl_loss"]) == _scalar(m_u["kl_loss"]) == pytest.approx(2.0)
