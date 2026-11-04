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

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation import losses
from verl.trainer.distillation.fsdp.losses import compute_forward_kl_topk
from verl.trainer.ppo.padding_utils import SYNTHETIC_PADDING_SEQ_LEN, construct_minimal_padding_template
from verl.utils import tensordict_utils as tu
from verl.workers.config.distillation import DistillationLossConfig


@pytest.mark.parametrize("unused_value", [-1000.0, 1000.0])
def test_soft_target_gradient_uses_causal_response_positions_and_ignores_ppo(monkeypatch, unused_value):
    def forbid_ppo(*args, **kwargs):
        raise AssertionError("supervised distillation must not call PPO")

    monkeypatch.setattr(losses, "ppo_loss", forbid_ppo)
    teacher = torch.tensor([0.60, 0.30, 0.10], dtype=torch.float64)
    reference_logits = torch.tensor([0.2, -0.3, 0.7], dtype=torch.float64, requires_grad=True)
    reference = (teacher * (teacher.log() - reference_logits.log_softmax(-1))).sum()
    expected_grad = torch.autograd.grad(reference, reference_logits)[0]
    logits = reference_logits.detach().repeat(1, 7, 1)
    logits[0, [0, 2, 3, 6]] = torch.tensor([100.0, -100.0, 0.0], dtype=logits.dtype)
    logits.requires_grad_()
    offsets = torch.tensor([0, 3, 7])
    teacher_ids = torch.nested.nested_tensor_from_jagged(torch.arange(3).repeat(7, 1), offsets=offsets)
    teacher_logprobs = torch.nested.nested_tensor_from_jagged(teacher.log().repeat(7, 1), offsets=offsets)
    config = SimpleNamespace(
        distillation_loss=DistillationLossConfig(
            loss_mode="forward_kl_topk",
            topk=3,
            use_task_rewards=False,
            use_policy_gradient=False,
            log_prob_min_clamp=None,
            loss_max_clamp=None,
        )
    )
    raw = compute_forward_kl_topk(logits, teacher_logprobs, teacher_ids, config, "thd")
    model_output = {key: value.squeeze(0) for key, value in raw.items()}
    data = TensorDict(
        {
            "prompts": torch.nested.as_nested_tensor([torch.tensor([101, 102])] * 2, layout=torch.jagged),
            "responses": torch.nested.as_nested_tensor(
                [torch.tensor([21]), torch.tensor([22, 23])], layout=torch.jagged
            ),
            "response_mask": torch.nested.as_nested_tensor([torch.ones(1), torch.ones(2)], layout=torch.jagged),
            "old_log_probs": torch.full((2, 2), unused_value),
            "advantages": torch.full((2, 2), unused_value),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=3, global_batch_size=2)
    actor = SimpleNamespace(loss_agg_mode="token-mean", loss_scale_factor=None, global_batch_info={})
    actual, _ = losses.distillation_ppo_loss(actor, config, model_output=model_output, data=data)
    gradient = torch.autograd.grad(actual, logits)[0]
    torch.testing.assert_close(actual.double(), reference, atol=1e-6, rtol=1e-5)
    for index in (1, 4, 5):
        torch.testing.assert_close(gradient[0, index], expected_grad / 3, atol=1e-6, rtol=1e-5)
    assert torch.count_nonzero(gradient[0, [0, 2, 3, 6]]) == 0


def test_real_padding_builder_keeps_teacher_targets_aligned_and_masked():
    source = {
        "teacher_ids": torch.ones(6, 3, dtype=torch.long),
        "teacher_logprobs": torch.zeros(6, 3),
        "rollout_source": "teacher",
    }
    sample, tag = construct_minimal_padding_template(
        source, {"rollout_source": "teacher", "min_global_steps": None, "max_global_steps": None}, eos_token_id=2
    )
    assert sample["teacher_ids"].shape == sample["teacher_logprobs"].shape == (SYNTHETIC_PADDING_SEQ_LEN, 3)
    assert sample["loss_mask"].sum() == 0
    assert sample["rollout_source"] == tag["rollout_source"] == "teacher"
    assert tag["is_padding"] and tag["min_global_steps"] is None
