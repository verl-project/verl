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

from verl.models.transformers.qwen3_vl import qwen3_vl_deepstack_process


class _ViewFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        return input_tensor[:]

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def _original_deepstack_process(hidden_states, visual_pos_masks, visual_embeds):
    visual_pos_masks = visual_pos_masks.to(hidden_states.device)
    visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
    local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
    hidden_states[visual_pos_masks, :] = local_this
    return hidden_states


def test_qwen3_vl_deepstack_process_handles_custom_autograd_views():
    base = torch.randn(4, 6, requires_grad=True)
    hidden_states = _ViewFunction.apply(base)
    visual_pos_masks = torch.tensor([True, False, True, False])
    visual_embeds = torch.randn(2, 6)

    with pytest.raises(RuntimeError, match="view.*modified inplace"):
        _original_deepstack_process(hidden_states, visual_pos_masks, visual_embeds)

    base = torch.randn(4, 6, requires_grad=True)
    hidden_states = _ViewFunction.apply(base)
    output = qwen3_vl_deepstack_process(None, hidden_states, visual_pos_masks, visual_embeds)

    expected = base.detach().clone()
    expected[visual_pos_masks] += visual_embeds
    torch.testing.assert_close(output, expected)

    output.square().sum().backward()
    torch.testing.assert_close(base.grad, 2 * output.detach())
