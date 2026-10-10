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

import verl.trainer.distillation.losses as distillation_losses_module
from verl.utils.tensordict_utils import get_tensordict


def test_distillation_policy_gradient_applies_per_sample_loss_weight(monkeypatch):
    captured = {}

    def fake_distillation_loss_fn(**_kwargs):
        return torch.tensor([[1.0, 1.0], [2.0, 2.0]]), {}

    def fake_policy_loss_fn(**kwargs):
        captured["advantages"] = kwargs["advantages"]
        return torch.tensor(0.0), {}

    monkeypatch.setattr(distillation_losses_module, "get_distillation_loss_fn", lambda _name: fake_distillation_loss_fn)
    monkeypatch.setattr(distillation_losses_module, "get_policy_loss_fn", lambda _name: fake_policy_loss_fn)
    monkeypatch.setattr(distillation_losses_module, "no_padding_2_padding", lambda tensor, _data: tensor)

    config = SimpleNamespace(loss_agg_mode="token-mean", loss_scale_factor=None, global_batch_info={})
    loss_config = SimpleNamespace(
        loss_mode="fake",
        use_policy_gradient=True,
        policy_loss_mode="vanilla",
        loss_max_clamp=None,
        global_batch_info={},
    )
    distillation_config = SimpleNamespace(distillation_loss=loss_config)
    data = get_tensordict(
        {
            "response_mask": torch.ones(2, 2),
            "old_log_probs": torch.zeros(2, 2),
            "loss_weight": torch.tensor([0.25, 0.5]),
        },
        non_tensor_dict={
            "dp_size": 1,
            "batch_num_tokens": None,
            "global_batch_size": None,
        },
    )

    distillation_losses_module.distillation_loss(config, distillation_config, {"log_probs": torch.zeros(2, 2)}, data)

    assert "advantages" in captured
    torch.testing.assert_close(captured["advantages"], torch.tensor([[-0.25, -0.25], [-1.0, -1.0]]))


def test_distillation_policy_gradient_rejects_invalid_loss_weight(monkeypatch):
    def fake_distillation_loss_fn(**_kwargs):
        return torch.ones(2, 2), {}

    monkeypatch.setattr(distillation_losses_module, "get_distillation_loss_fn", lambda _name: fake_distillation_loss_fn)
    monkeypatch.setattr(distillation_losses_module, "no_padding_2_padding", lambda tensor, _data: tensor)

    config = SimpleNamespace(loss_agg_mode="token-mean", loss_scale_factor=None, global_batch_info={})
    loss_config = SimpleNamespace(
        loss_mode="fake",
        use_policy_gradient=True,
        policy_loss_mode="vanilla",
        loss_max_clamp=None,
        global_batch_info={},
    )
    distillation_config = SimpleNamespace(distillation_loss=loss_config)
    data = get_tensordict(
        {
            "response_mask": torch.ones(2, 2),
            "old_log_probs": torch.zeros(2, 2),
            # A negative weight would flip the gradient direction. Zero is *not* rejected
            # here: validate_loss_weights zeroes padding rows, so the apply step must
            # tolerate them.
            "loss_weight": torch.tensor([1.0, -0.5]),
        },
        non_tensor_dict={
            "dp_size": 1,
            "batch_num_tokens": None,
            "global_batch_size": None,
        },
    )

    with pytest.raises(ValueError, match="non-negative"):
        distillation_losses_module.distillation_loss(
            config, distillation_config, {"log_probs": torch.zeros(2, 2)}, data
        )
