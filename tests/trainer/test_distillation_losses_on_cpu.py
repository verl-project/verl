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

import math
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import compute_distillation_loss_reverse_kl_estimator
from verl.trainer.ppo.core_algos import kl_penalty


def test_reverse_kl_estimator_clamps_negative_infinite_log_probs():
    student_log_probs = torch.tensor([[float("-inf"), -2.0]])
    teacher_log_probs = torch.tensor([[[float("-inf")], [float("-inf")]]])
    response_mask = torch.ones_like(student_log_probs, dtype=torch.bool)
    data = TensorDict(
        {"teacher_logprobs": teacher_log_probs, "response_mask": response_mask},
        batch_size=[1],
    )
    distillation_config = SimpleNamespace(distillation_loss=SimpleNamespace(loss_mode="k1", log_prob_min_clamp=-10.0))

    def identity_padding(tensor, _data):
        return tensor

    with patch("verl.trainer.distillation.losses.no_padding_2_padding", side_effect=identity_padding):
        losses, metrics = compute_distillation_loss_reverse_kl_estimator(
            config=None,
            distillation_config=distillation_config,
            model_output={"log_probs": student_log_probs},
            data=data,
        )

    expected = kl_penalty(
        logprob=student_log_probs.clamp_min(-10.0),
        ref_logprob=teacher_log_probs.squeeze(-1).clamp_min(-10.0),
        kl_penalty="k1",
    )
    assert torch.equal(losses, expected)
    assert torch.isfinite(losses).all()
    assert math.isfinite(metrics["distillation/abs_loss"].aggregate())
