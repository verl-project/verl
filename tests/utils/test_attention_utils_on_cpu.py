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

import torch

from verl.utils.attention_utils import index_first_axis, pad_input, unpad_input


def test_index_first_axis_preserves_shape_and_gradient():
    hidden_states = torch.arange(48, dtype=torch.float32).reshape(8, 2, 3).requires_grad_()
    indices = torch.tensor([0, 3, 7])

    selected = index_first_axis(hidden_states, indices)

    torch.testing.assert_close(selected, hidden_states.detach()[indices])
    assert selected.shape == (3, 2, 3)

    selected.sum().backward()
    expected_grad = torch.zeros_like(hidden_states)
    expected_grad[indices] = 1
    torch.testing.assert_close(hidden_states.grad, expected_grad)


def test_pad_and_unpad_input_round_trip_with_unused_tokens():
    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3).requires_grad_()
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]])
    unused_mask = torch.tensor([[0, 0, 1, 0], [0, 1, 0, 0]])

    unpadded, indices, cu_seqlens, max_seqlen, used_seqlens = unpad_input(
        hidden_states,
        attention_mask,
        unused_mask,
    )
    padded = pad_input(unpadded, indices, batch=2, seqlen=4)

    selected_mask = (attention_mask + unused_mask).bool().unsqueeze(-1)
    expected = torch.where(selected_mask, hidden_states.detach(), torch.zeros_like(hidden_states))
    torch.testing.assert_close(padded, expected)
    torch.testing.assert_close(indices, torch.tensor([0, 1, 2, 4, 5, 6]))
    torch.testing.assert_close(cu_seqlens, torch.tensor([0, 3, 6], dtype=torch.int32))
    torch.testing.assert_close(used_seqlens, torch.tensor([2, 2], dtype=torch.int32))
    assert max_seqlen == 3

    padded.sum().backward()
    expected_grad = selected_mask.expand_as(hidden_states).to(hidden_states.dtype)
    torch.testing.assert_close(hidden_states.grad, expected_grad)
