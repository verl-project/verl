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

from verl.models.mcore import model_forward_fused as mff
from verl.models.mcore.model_forward import _build_full_loss_mask_nested
from verl.models.mcore.response_only_lm_head import (
    response_only_output_projection,
    restore_response_only_outputs,
    select_response_only_inputs,
)
from verl.models.mcore.util import preprocess_thd_engine


class _OutputLayer(torch.nn.Module):
    def __init__(self, hidden_size=None, vocab_size=None, *, sequence_parallel=False):
        super().__init__()
        self.sequence_parallel = sequence_parallel
        self.disable_grad_reduce = False
        self.tp_group = object()
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, hidden_size)) if hidden_size is not None else None

    def forward(self, input_):
        output = input_ if self.weight is None else torch.nn.functional.linear(input_, self.weight)
        return output, None


class _Model(torch.nn.Module):
    def __init__(self, output_layer):
        super().__init__()
        self.output_layer = output_layer

    def forward(self, hidden_states):
        logits, _ = self.output_layer(hidden_states)
        return logits.transpose(0, 1).contiguous()


def test_response_mask_is_next_token_aligned():
    response_mask = torch.nested.as_nested_tensor(
        [torch.tensor([1, 1, 0, 1, 0]), torch.tensor([1, 0, 1, 1])], layout=torch.jagged
    )
    full_mask = _build_full_loss_mask_nested(response_mask, [8, 8], None)

    with (
        patch("verl.models.mcore.util.mpu.get_context_parallel_world_size", return_value=1),
        patch("verl.models.mcore.util.mpu.get_context_parallel_rank", return_value=0),
        patch("verl.models.mcore.util.mpu.get_tensor_model_parallel_world_size", return_value=1),
    ):
        projection_mask = preprocess_thd_engine(full_mask, need_roll=True)[0]

    expected = torch.tensor([[0, 0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 1, 0]], dtype=torch.bool).reshape(1, -1)
    torch.testing.assert_close(projection_mask, expected)


@pytest.mark.parametrize("empty", [False, True])
def test_unfused_projection_preserves_values_and_gradients(empty):
    torch.manual_seed(7)
    mask = torch.tensor([[False, True, True, False], [True, False, True, False]])
    if empty:
        mask.zero_()

    dense_layer = _OutputLayer(3, 5)
    sparse_layer = _OutputLayer(3, 5)
    sparse_layer.load_state_dict(dense_layer.state_dict())
    dense_model, sparse_model = _Model(dense_layer), _Model(sparse_layer)
    dense_hidden = torch.randn(4, 2, 3, requires_grad=True)
    sparse_hidden = dense_hidden.detach().clone().requires_grad_()

    dense_logits = dense_model(dense_hidden)
    dense_logits[mask].square().sum().backward()

    with response_only_output_projection(sparse_model, mask):
        sparse_logits = sparse_model(sparse_hidden)
    labels = torch.arange(mask.numel()).reshape_as(mask)
    sparse_labels, _, num_selected = select_response_only_inputs(labels, torch.ones_like(labels), mask)
    restored = restore_response_only_outputs({"values": sparse_logits.square().sum(-1)}, mask, num_selected)["values"]
    restored.sum().backward()

    assert num_selected == int(mask.sum())
    if num_selected:
        torch.testing.assert_close(sparse_logits[0], dense_logits[mask])
        torch.testing.assert_close(sparse_labels[0], labels[mask])
    torch.testing.assert_close(sparse_hidden.grad, dense_hidden.grad)
    torch.testing.assert_close(sparse_layer.weight.grad, dense_layer.weight.grad)


def test_unfused_sequence_parallel_gathers_before_selection():
    output_layer = _OutputLayer(sequence_parallel=True)
    model = _Model(output_layer)
    hidden = torch.arange(4, dtype=torch.float32).reshape(2, 1, 2)
    gathered = torch.cat((hidden, hidden + 10))
    mask = torch.tensor([[False, True, True, False]])

    with (
        patch(
            "megatron.core.tensor_parallel.gather_from_sequence_parallel_region",
            return_value=gathered,
        ) as gather,
        response_only_output_projection(model, mask),
    ):
        selected, _ = output_layer(hidden)
        assert not output_layer.sequence_parallel
        assert output_layer.disable_grad_reduce

    gather.assert_called_once_with(hidden, tensor_parallel_output_grad=True, group=output_layer.tp_group)
    torch.testing.assert_close(selected[:, 0], gathered[[1, 2], 0])
    assert output_layer.sequence_parallel
    assert not output_layer.disable_grad_reduce


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_fused_projection_matches_dense_values_and_gradients(sequence_parallel):
    torch.manual_seed(42)
    mask = torch.tensor([[False, True, False, True, True, False]])
    labels = torch.tensor([[0, 1, 2, 3, 4, 0]])
    initial_hidden = torch.randn(6, 1, 3)
    initial_weight = torch.randn(5, 3)
    coefficients = torch.randn(1, 6) * mask

    def run(projection_mask):
        hidden = initial_hidden.clone().requires_grad_()
        weight = initial_weight.clone().requires_grad_()
        projected_rows = []

        def kernel(h, w, y, temperature, reduction, group):
            projected_rows.append(h.shape[0])
            log_probs = (h.reshape(-1, 3) @ w.T / temperature).log_softmax(-1)
            return log_probs.gather(-1, y.reshape(-1, 1)).squeeze(-1), -(log_probs.exp() * log_probs).sum(-1)

        with (
            patch.object(mff, "linear_cross_entropy", side_effect=kernel),
            patch.object(mff.parallel_state, "get_tensor_model_parallel_group", return_value=object()),
            patch.object(mff, "gather_from_sequence_parallel_region", side_effect=lambda value: value) as gather,
            patch.object(mff, "copy_to_tensor_model_parallel_region", side_effect=lambda value, group: value) as copy,
        ):
            log_probs, entropy = mff._compute_fused_lm_head(
                hidden, weight, labels, 0.7, sequence_parallel, projection_mask
            )

        assert gather.called == sequence_parallel
        assert copy.called != sequence_parallel
        log_probs, entropy = log_probs.reshape_as(mask), entropy.reshape_as(mask)
        ((log_probs + 0.1 * entropy) * coefficients).sum().backward()
        return log_probs, entropy, hidden.grad, weight.grad, projected_rows

    sparse = run(mask)
    dense = run(None)
    torch.testing.assert_close(sparse[0][mask], dense[0][mask])
    torch.testing.assert_close(sparse[1][mask], dense[1][mask])
    torch.testing.assert_close(sparse[2], dense[2])
    torch.testing.assert_close(sparse[3], dense[3])
    assert sparse[4] == [int(mask.sum())]
    assert dense[4] == [mask.numel()]
    assert torch.count_nonzero(sparse[0][~mask]) == 0
    assert torch.count_nonzero(sparse[1][~mask]) == 0
