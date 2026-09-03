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

from unittest.mock import patch

import pytest
import torch

from verl.models.mcore.model_forward import _build_full_loss_mask_nested
from verl.models.mcore.response_only_lm_head import (
    response_only_output_projection,
    restore_response_only_outputs,
    select_response_only_inputs,
)
from verl.models.mcore.util import preprocess_bshd_engine, preprocess_thd_engine


class _FakeOutputLayer(torch.nn.Module):
    def __init__(self, *, sequence_parallel=True, supports_disable_grad_reduce=True):
        super().__init__()
        self.sequence_parallel = sequence_parallel
        if supports_disable_grad_reduce:
            self.disable_grad_reduce = False
        self.tp_group = object()

    def forward(self, input_):
        return input_, None


class _FakeLinearOutputLayer(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.sequence_parallel = False
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, hidden_size))

    def forward(self, input_):
        return torch.nn.functional.linear(input_, self.weight), None


class _Model(torch.nn.Module):
    def __init__(self, output_layer):
        super().__init__()
        self.output_layer = output_layer

    def forward(self, hidden_states):
        logits, _ = self.output_layer(hidden_states)
        return logits.transpose(0, 1).contiguous()


@pytest.mark.parametrize("data_format", ["thd", "bshd"])
def test_response_mask_is_next_token_aligned_with_internal_tool_spans(data_format):
    padded_response_mask = torch.tensor(
        [
            [1, 1, 0, 1, 0],
            [1, 0, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    response_attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    if data_format == "thd":
        response_mask = torch.nested.as_nested_tensor(
            [padded_response_mask[0], padded_response_mask[1, :4]], layout=torch.jagged
        )
        response_attention_mask = None
        preprocess = preprocess_thd_engine
    else:
        response_mask = padded_response_mask
        preprocess = preprocess_bshd_engine
    full_mask = _build_full_loss_mask_nested(response_mask, [8, 8], response_attention_mask)

    with (
        patch("verl.models.mcore.util.mpu.get_context_parallel_world_size", return_value=1),
        patch("verl.models.mcore.util.mpu.get_context_parallel_rank", return_value=0),
        patch("verl.models.mcore.util.mpu.get_tensor_model_parallel_world_size", return_value=1),
    ):
        projection_mask = preprocess(full_mask, need_roll=True)[0]

    expected = torch.tensor(
        [
            [0, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    if data_format == "thd":
        expected = expected.reshape(1, -1)
    torch.testing.assert_close(projection_mask, expected)


def test_projection_hook_gathers_tp_sequence_before_selecting_hidden_states():
    output_layer = _FakeOutputLayer()
    model = _Model(output_layer)
    hidden = torch.arange(4, dtype=torch.float32).reshape(2, 1, 2)
    gathered = torch.cat((hidden, hidden + 10), dim=0)
    projection_mask = torch.tensor([[False, True, True, False]])

    with (
        patch(
            "megatron.core.tensor_parallel.gather_from_sequence_parallel_region",
            return_value=gathered,
        ) as gather,
        response_only_output_projection(model, projection_mask),
    ):
        projected, _ = output_layer(hidden)
        assert output_layer.sequence_parallel is False
        assert output_layer.disable_grad_reduce is True

    gather.assert_called_once_with(hidden, tensor_parallel_output_grad=True, group=output_layer.tp_group)
    expected = torch.stack((gathered[1, 0], gathered[2, 0])).reshape(2, 1, 2)
    torch.testing.assert_close(projected, expected)
    assert output_layer.sequence_parallel is True
    assert output_layer.disable_grad_reduce is False


def test_projection_hook_preserves_active_logits_and_gradients_without_sequence_parallelism():
    torch.manual_seed(7)
    projection_mask = torch.tensor([[False, True, True, False], [True, False, True, False]])
    dense_layer = _FakeLinearOutputLayer(hidden_size=3, vocab_size=5)
    sparse_layer = _FakeLinearOutputLayer(hidden_size=3, vocab_size=5)
    dense_model = _Model(dense_layer)
    sparse_model = _Model(sparse_layer)
    sparse_layer.load_state_dict(dense_layer.state_dict())
    dense_hidden = torch.randn(4, 2, 3, requires_grad=True)
    sparse_hidden = dense_hidden.detach().clone().requires_grad_()

    dense_active = dense_model(dense_hidden)[projection_mask]
    dense_active.square().sum().backward()

    with response_only_output_projection(sparse_model, projection_mask):
        sparse_logits = sparse_model(sparse_hidden)
    sparse_logits.square().sum().backward()

    torch.testing.assert_close(sparse_logits[0], dense_active)
    torch.testing.assert_close(sparse_hidden.grad, dense_hidden.grad)
    torch.testing.assert_close(sparse_layer.weight.grad, dense_layer.weight.grad)


def test_sequence_parallel_output_layer_must_support_disabling_grad_reduction():
    model = _Model(_FakeOutputLayer(supports_disable_grad_reduce=False))
    with (
        pytest.raises(RuntimeError, match="disable_grad_reduce"),
        response_only_output_projection(model, torch.ones(1, 2, dtype=torch.bool)),
    ):
        pass


def test_empty_local_response_mask_keeps_a_zero_gradient_path():
    projection_mask = torch.zeros(1, 4, dtype=torch.bool)
    model = _Model(_FakeLinearOutputLayer(hidden_size=3, vocab_size=5))
    hidden = torch.randn(4, 1, 3, requires_grad=True)
    with response_only_output_projection(model, projection_mask):
        logits = model(hidden)

    sparse_label, _, num_selected = select_response_only_inputs(
        torch.arange(4).reshape(1, 4), torch.ones(1, 4), projection_mask
    )
    assert num_selected == 0
    torch.testing.assert_close(sparse_label, torch.zeros(1, 1, dtype=torch.long))

    sparse_output = logits.sum(dim=-1)
    restored = restore_response_only_outputs({"log_probs": sparse_output}, projection_mask, num_selected)["log_probs"]
    restored.sum().backward()

    torch.testing.assert_close(restored, torch.zeros_like(restored))
    assert hidden.grad is not None and model.output_layer.weight.grad is not None
    torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden.grad))
    torch.testing.assert_close(model.output_layer.weight.grad, torch.zeros_like(model.output_layer.weight.grad))
