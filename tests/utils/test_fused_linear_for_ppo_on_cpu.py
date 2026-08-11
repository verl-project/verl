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

import verl.utils.experimental.torch_functional as fused_ops


@pytest.mark.parametrize(
    "labels",
    [[1, 2, 3, 4], [1, -100, 3, -100], [-100, -100, -100, -100]],
    ids=["dense", "sparse", "empty"],
)
def test_fused_linear_for_ppo_honors_ignore_index(labels, monkeypatch):
    monkeypatch.setattr(fused_ops, "_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE", False)
    torch.manual_seed(0)

    hidden = torch.randn(1, 4, 8, requires_grad=True)
    weight = torch.randn(7, 8, requires_grad=True)
    labels = torch.tensor([labels])
    active = labels.ne(-100)

    log_probs, entropy = fused_ops.FusedLinearForPPO()(hidden, weight, labels, temperature=1.5)
    (log_probs.sum() + 0.3 * entropy.sum()).backward()

    ref_hidden = hidden.detach().clone().requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    ref_log_probs, ref_entropy = fused_ops.FusedLinearForPPOFunction.apply(
        ref_hidden, ref_weight, labels.masked_fill(~active, 0), 1.5, 512
    )
    ref_log_probs = ref_log_probs.masked_fill(~active, 0)
    ref_entropy = ref_entropy.masked_fill(~active, 0)
    (ref_log_probs.sum() + 0.3 * ref_entropy.sum()).backward()

    torch.testing.assert_close(log_probs, ref_log_probs)
    torch.testing.assert_close(entropy, ref_entropy)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad)
    torch.testing.assert_close(weight.grad, ref_weight.grad)
