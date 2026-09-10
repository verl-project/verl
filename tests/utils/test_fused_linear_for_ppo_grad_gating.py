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
"""Regression tests for FusedLinearForPPO gradient gating.

``autograd.Function.forward`` runs with grad mode disabled, so
``hidden_states.flatten(0, 1)`` on a NON-contiguous 3-D input returns a *copy*
whose ``requires_grad`` is False (a contiguous input returns a view and keeps
the flag). That copy is what gets saved via ``ctx.save_for_backward``.

``backward`` used to gate the allocation of ``dhidden_states`` on the saved
tensor's ``requires_grad``, so for non-contiguous inputs it silently returned
``dhidden_states=None``: with the FSDP engine (which feeds exactly such a
``(1, total_nnz, hidden)`` tensor after remove-padding / SP gathers), every
parameter upstream of the LM head trained with ZERO gradient while lm_head
kept updating and all forward-side metrics (loss / entropy / reward) looked
normal — the only observable symptom was grad_norm dropping ~8x.

The fix gates on ``ctx.needs_input_grad``, which autograd records at
``apply()`` time and is immune to how the saved tensors were transformed.
These tests force the pure-PyTorch CE fallback so they run on CPU.
"""

import pytest
import torch

import verl.utils.experimental.torch_functional as tf
from verl.utils.experimental.torch_functional import FusedLinearForPPO


@pytest.fixture(autouse=True)
def _use_torch_fallback(monkeypatch):
    # flash-attn's triton cross entropy needs CUDA; the bug is independent of
    # the CE backend, so force the pure-PyTorch path to keep the test on CPU.
    monkeypatch.setattr(tf, "_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE", False)


def _make_hidden(leaf: torch.Tensor, contiguous: bool) -> torch.Tensor:
    """Non-leaf hidden with the same logical values, contiguous or not."""
    if contiguous:
        return leaf * 1.0
    # Materialize in (T, B, H) order, then view back as (B, T, H): logically
    # identical values, but is_contiguous() is False — the layout the FSDP
    # engine actually feeds. flatten(0, 1) on this input copies (and, under
    # the Function's no-grad forward, drops requires_grad).
    return (leaf * 1.0).permute(1, 0, 2).contiguous().permute(1, 0, 2)


@pytest.mark.parametrize("contiguous", [True, False], ids=["contiguous", "non_contiguous"])
def test_hidden_grad_flows_regardless_of_contiguity(contiguous):
    torch.manual_seed(0)
    B, T, H, V = 2, 24, 16, 48

    hidden_leaf = torch.randn(B, T, H, requires_grad=True)
    vocab_weights = torch.randn(V, H, requires_grad=True)
    input_ids = torch.randint(0, V, (B, T))

    hidden = _make_hidden(hidden_leaf, contiguous)
    assert hidden.is_contiguous() == contiguous

    log_probs, _entropy = FusedLinearForPPO()(hidden_states=hidden, vocab_weights=vocab_weights, input_ids=input_ids)
    log_probs.sum().backward()

    assert hidden_leaf.grad is not None, "trunk gradient was silently dropped"
    assert hidden_leaf.grad.abs().sum() > 0
    assert vocab_weights.grad is not None

    g_hidden_fused = hidden_leaf.grad.detach().clone()
    g_vocab_fused = vocab_weights.grad.detach().clone()
    hidden_leaf.grad = None
    vocab_weights.grad = None

    # Eager log-softmax reference over the identical graph
    ref_hidden = _make_hidden(hidden_leaf, contiguous)
    logits = (ref_hidden @ vocab_weights.t()).float()
    ref_log_probs = logits.log_softmax(dim=-1).gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    ref_log_probs.sum().backward()

    torch.testing.assert_close(g_hidden_fused, hidden_leaf.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(g_vocab_fused, vocab_weights.grad, rtol=1e-4, atol=1e-4)


def test_no_grad_inputs_return_none_grads():
    """needs_input_grad gating must not allocate grads that aren't needed."""
    torch.manual_seed(0)
    B, T, H, V = 1, 8, 16, 32
    hidden = torch.randn(B, T, H)  # requires_grad=False
    vocab_weights = torch.randn(V, H, requires_grad=True)
    input_ids = torch.randint(0, V, (B, T))

    log_probs, _ = FusedLinearForPPO()(hidden_states=hidden, vocab_weights=vocab_weights, input_ids=input_ids)
    log_probs.sum().backward()
    assert vocab_weights.grad is not None
    assert hidden.grad is None
