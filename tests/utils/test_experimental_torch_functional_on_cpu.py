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

_MODULE_PATH = Path(__file__).resolve().parents[2] / "verl" / "utils" / "experimental" / "torch_functional.py"
_SPEC = importlib.util.spec_from_file_location("_experimental_torch_functional", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
experimental_F = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(experimental_F)


@pytest.mark.parametrize("hidden_shape", [(7, 5), (2, 7, 5)])
def test_fused_linear_for_ppo_chunked_fallback_matches_torch(monkeypatch, hidden_shape):
    monkeypatch.setattr(experimental_F, "_LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY", None)
    monkeypatch.setattr(experimental_F, "_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE", False)
    torch.manual_seed(42)

    temperature = 0.7
    vocab_size = 11
    hidden = torch.randn(hidden_shape, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_shape[-1], requires_grad=True)
    labels = torch.randint(vocab_size, hidden_shape[:-1], dtype=torch.int32)
    grad_log_probs = torch.randn(hidden_shape[:-1])
    grad_entropy = torch.randn(hidden_shape[:-1])

    log_probs, entropy = experimental_F.FusedLinearForPPO()(hidden, weight, labels, temperature)
    torch.autograd.backward((log_probs, entropy), (grad_log_probs, grad_entropy))

    expected_hidden = hidden.detach().clone().requires_grad_(True)
    expected_weight = weight.detach().clone().requires_grad_(True)
    logits = ((expected_hidden @ expected_weight.t()) / temperature).float()
    expected_log_probs = logits.log_softmax(dim=-1).gather(-1, labels.long().unsqueeze(-1)).squeeze(-1)
    probs = logits.softmax(dim=-1)
    expected_entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    torch.autograd.backward(
        (expected_log_probs, expected_entropy),
        (grad_log_probs, grad_entropy),
    )

    torch.testing.assert_close(log_probs, expected_log_probs)
    torch.testing.assert_close(entropy, expected_entropy)
    torch.testing.assert_close(hidden.grad, expected_hidden.grad)
    torch.testing.assert_close(weight.grad, expected_weight.grad)


def test_fused_linear_for_ppo_dispatches_to_liger(monkeypatch):
    calls = []

    class FakeLigerFusedLinearScaledCrossEntropyFunction:
        @staticmethod
        def apply(*args):
            calls.append(args)
            hidden_states = args[0]
            token_count = hidden_states.shape[0]
            nll = torch.arange(token_count, dtype=torch.float32)
            entropy = torch.arange(token_count, dtype=hidden_states.dtype) + 10
            return nll, entropy

    monkeypatch.setattr(
        experimental_F,
        "_LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY",
        FakeLigerFusedLinearScaledCrossEntropyFunction,
    )
    hidden = torch.randn(2, 3, 5)
    weight = torch.randn(7, 5)
    labels = torch.randint(7, (2, 3), dtype=torch.int32)

    log_probs, entropy = experimental_F.FusedLinearForPPO(impl_backend="liger")(hidden, weight, labels, temperature=0.8)

    assert len(calls) == 1
    liger_hidden, liger_weight, liger_labels, temperature, ignore_index, m_tiles, return_entropy = calls[0]
    assert liger_hidden.shape == (6, 5)
    assert liger_weight is weight
    assert liger_labels.shape == (6,)
    assert liger_labels.dtype == torch.int64
    assert temperature == 0.8
    assert ignore_index == -100
    assert m_tiles == 1
    assert return_entropy is True
    torch.testing.assert_close(log_probs, -torch.arange(6, dtype=torch.float32).reshape(2, 3))
    torch.testing.assert_close(entropy, (torch.arange(6, dtype=hidden.dtype) + 10).reshape(2, 3))


def test_fused_linear_for_ppo_torch_backend_does_not_dispatch_to_liger(monkeypatch):
    class UnexpectedLigerFusedLinearScaledCrossEntropyFunction:
        @staticmethod
        def apply(*args):
            raise AssertionError("The torch backend must not dispatch to Liger")

    monkeypatch.setattr(
        experimental_F,
        "_LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY",
        UnexpectedLigerFusedLinearScaledCrossEntropyFunction,
    )
    monkeypatch.setattr(experimental_F, "_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE", False)

    hidden = torch.randn(2, 3, 5)
    weight = torch.randn(7, 5)
    labels = torch.randint(7, (2, 3))

    log_probs, entropy = experimental_F.FusedLinearForPPO()(hidden, weight, labels)

    logits = (hidden @ weight.t()).float()
    expected_log_probs = logits.log_softmax(dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    probs = logits.softmax(dim=-1)
    expected_entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    torch.testing.assert_close(log_probs, expected_log_probs)
    torch.testing.assert_close(entropy, expected_entropy)


def test_fused_linear_for_ppo_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported FusedLinearForPPO backend"):
        experimental_F.FusedLinearForPPO(impl_backend="unknown")


def test_fused_linear_for_ppo_fallback_preserves_chunking(monkeypatch):
    monkeypatch.setattr(experimental_F, "_LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY", None)
    monkeypatch.setattr(experimental_F, "_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE", False)
    original_forward = experimental_F._fused_linear_for_ppo_fwd
    chunk_sizes = []

    def record_chunk_size(hidden_states, *args, **kwargs):
        chunk_sizes.append(hidden_states.shape[0])
        return original_forward(hidden_states, *args, **kwargs)

    monkeypatch.setattr(experimental_F, "_fused_linear_for_ppo_fwd", record_chunk_size)
    hidden = torch.randn(1, 7, 5)
    weight = torch.randn(11, 5)
    labels = torch.randint(11, (1, 7))

    experimental_F.FusedLinearForPPO(chunk_size=3)(hidden, weight, labels)

    assert chunk_sizes == [3, 3, 1]
