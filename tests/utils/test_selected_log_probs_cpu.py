# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import importlib.util
from pathlib import Path

import pytest
import torch


def load_operator():
    path = Path(__file__).resolve().parents[2] / "verl/utils/experimental/selected_log_probs.py"
    spec = importlib.util.spec_from_file_location("selected_log_probs_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.selected_log_probs


@pytest.mark.parametrize("chunk_size", [1, 4, 32])
@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.8])
def test_values_and_gradients_match_dense_with_repeated_candidates(chunk_size, temperature):
    operation = load_operator()
    torch.manual_seed(11)
    hidden = torch.randn(2, 7, 5, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(19, 5, dtype=torch.float64, requires_grad=True)
    ids = torch.randint(19, (2, 7, 4))
    ids[..., -1] = ids[..., 0]
    observed = operation(hidden, weights, ids, temperature=temperature, chunk_size=chunk_size)
    expected = (hidden @ weights.T / temperature).log_softmax(-1).gather(-1, ids)
    torch.testing.assert_close(observed, expected)
    upstream = torch.randn_like(expected)
    actual_grad = torch.autograd.grad((observed * upstream).sum(), (hidden, weights))
    reference_grad = torch.autograd.grad((expected * upstream).sum(), (hidden, weights))
    for actual, reference in zip(actual_grad, reference_grad, strict=False):
        torch.testing.assert_close(actual, reference)


def test_backward_does_not_retain_sequence_by_vocabulary_tensors():
    operation = load_operator()
    hidden = torch.randn(1, 23, 5, requires_grad=True)
    weights = torch.randn(101, 5, requires_grad=True)
    ids = torch.randint(101, (1, 23, 4))
    retained = []

    def pack(value):
        retained.append(tuple(value.shape))
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        operation(hidden, weights, ids, chunk_size=3).sum().backward()
    assert (23, 101) not in retained
    assert (1, 23, 101) not in retained
    assert set(retained) <= {(1, 23, 5), (101, 5), (1, 23, 4)}


def test_backward_recomputes_with_the_forward_autocast_dtype():
    operation = load_operator()
    torch.manual_seed(53)
    hidden = torch.randn(1, 7, 5, requires_grad=True)
    weights = torch.randn(19, 5, requires_grad=True)
    ids = torch.randint(19, (1, 7, 4))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        observed = operation(hidden, weights, ids, chunk_size=32)
        reference = (hidden @ weights.T).float().log_softmax(-1).gather(-1, ids)
    torch.testing.assert_close(observed, reference)
    actual = torch.autograd.grad(observed.sum(), (hidden, weights))
    expected = torch.autograd.grad(reference.sum(), (hidden, weights))
    for result, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, target, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("temperature,chunk_size", [(0, 4), (-1, 4), (1, 0)])
def test_invalid_limits_are_rejected(temperature, chunk_size):
    operation = load_operator()
    with pytest.raises(ValueError):
        operation(
            torch.randn(2, 3),
            torch.randn(5, 3),
            torch.zeros(2, 1, dtype=torch.long),
            temperature=temperature,
            chunk_size=chunk_size,
        )


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_candidates_shift_before_sequence_partition(rank):
    operation = load_operator()
    helper = operation.__globals__["prediction_candidates"]
    ids = torch.arange(35).reshape(7, 5)
    expected = torch.cat([ids.roll(-1, 0), torch.zeros(2, 5, dtype=torch.long)])[rank * 3 : (rank + 1) * 3]
    torch.testing.assert_close(helper(ids, pad_size=2, sp_rank=rank, sp_size=3), expected)
