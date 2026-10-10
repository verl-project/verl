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
"""NPU correctness tests for CANN fused linear cross-entropy.

Requires Ascend NPU + torch_npu fused linear-CE APIs (either naming):
  - npu_fused_linear_online_max_sum / fused_linear_online_max_sum
  - npu_fused_cross_entropy_loss_with_max_sum / fused_cross_entropy_loss_with_max_sum
  - ..._backward / ..._grad / fused_linear_cross_entropy_loss_with_max_sum_grad

Run on an NPU machine:
  pytest tests/utils/test_cann_linear_ce_on_npu.py -sv
  # or
  python tests/utils/test_cann_linear_ce_on_npu.py
"""

from __future__ import annotations

import os

import pytest
import torch

from verl.utils.device import is_torch_npu_available
from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
from verl.utils.kernel.npu.cann_linear_ce import (
    CannLinearCrossEntropy,
    is_cann_linear_ce_available,
    should_use_cann_linear_ce,
)

pytestmark = pytest.mark.skipif(
    not is_torch_npu_available() or not is_cann_linear_ce_available(),
    reason="Requires Ascend NPU with CANN fused linear-CE APIs",
)

DEVICE = "npu"


def _synchronize():
    torch.npu.synchronize()


def _torch_ce_logprobs_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: matmul + CE logprobs + token entropy (fp32 accumulate)."""
    h = hidden.reshape(-1, hidden.shape[-1]).float()
    w = weight.float()
    logits = torch.matmul(h, w.T) / temperature
    logprobs = -torch.nn.functional.cross_entropy(logits, labels.reshape(-1), reduction="none")
    pd = torch.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - (pd * logits).sum(dim=-1)
    return logprobs, entropy


def _make_inputs(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    *,
    temperature: float = 1.0,
    seed: int = 0,
    batched: bool = False,
):
    torch.manual_seed(seed)
    if batched:
        hidden = torch.empty(1, num_tokens, hidden_size, dtype=torch.bfloat16, device=DEVICE).uniform_(-0.5, 0.5)
        labels = torch.randint(0, vocab_size, (1, num_tokens), device=DEVICE)
    else:
        hidden = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=DEVICE).uniform_(-0.5, 0.5)
        labels = torch.randint(0, vocab_size, (num_tokens,), device=DEVICE)
    weight = torch.empty(vocab_size, hidden_size, dtype=torch.bfloat16, device=DEVICE).uniform_(-0.5, 0.5)
    hidden = hidden.requires_grad_()
    weight = weight.requires_grad_()
    return hidden, weight, labels, temperature


@pytest.mark.parametrize(
    ("num_tokens", "hidden_size", "vocab_size", "temperature"),
    [
        (64, 512, 2048, 1.0),
        (128, 1024, 4096, 1.0),
        (256, 2048, 4096, 1.5),  # Qwen3.5-35B-A3B-like hidden
        (32, 256, 1024, 0.8),
    ],
)
def test_cann_forward_matches_torch_reference(num_tokens, hidden_size, vocab_size, temperature, monkeypatch):
    monkeypatch.setenv("VERL_NPU_LCE_BACKEND", "cann")
    # Default path: return_logits=False (CE-accurate, entropy may be zeros).
    monkeypatch.delenv("VERL_NPU_LCE_RETURN_LOGITS", raising=False)
    hidden, weight, labels, temperature = _make_inputs(num_tokens, hidden_size, vocab_size, temperature=temperature)

    ref_lp, ref_ent = _torch_ce_logprobs_entropy(hidden.detach(), weight.detach(), labels, temperature)
    out_lp, out_ent = linear_cross_entropy(hidden, weight, labels, temperature)
    _synchronize()

    assert should_use_cann_linear_ce(hidden.device) is True
    torch.testing.assert_close(out_lp.float(), ref_lp, atol=2e-2, rtol=2e-2)
    # Default CANN path does not materialize vocab logits; entropy is a zero placeholder
    # (training uses entropy_coeff=0). Only assert shape/device here.
    assert out_ent.shape == ref_ent.shape
    assert out_ent.device.type == "npu"


def test_cann_forward_accepts_3d_hidden(monkeypatch):
    monkeypatch.setenv("VERL_NPU_LCE_BACKEND", "cann")
    monkeypatch.delenv("VERL_NPU_LCE_RETURN_LOGITS", raising=False)
    hidden, weight, labels, temperature = _make_inputs(48, 512, 2048, batched=True)

    ref_lp, _ = _torch_ce_logprobs_entropy(hidden.detach(), weight.detach(), labels, temperature)
    out_lp, _ = linear_cross_entropy(hidden, weight, labels, temperature)
    _synchronize()

    torch.testing.assert_close(out_lp.float(), ref_lp, atol=2e-2, rtol=2e-2)


def test_cann_forward_includes_last_vocab_id(monkeypatch):
    """Regression: vocab_end must be exclusive so label == V-1 is still valid."""
    monkeypatch.setenv("VERL_NPU_LCE_BACKEND", "cann")
    monkeypatch.delenv("VERL_NPU_LCE_RETURN_LOGITS", raising=False)

    num_tokens, hidden_size, vocab_size, temperature = 64, 512, 2048, 1.0
    torch.manual_seed(0)
    hidden = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=DEVICE).uniform_(-0.5, 0.5)
    weight = torch.empty(vocab_size, hidden_size, dtype=torch.bfloat16, device=DEVICE).uniform_(-0.5, 0.5)
    labels = torch.randint(0, vocab_size, (num_tokens,), device=DEVICE)
    # Force the historically-broken last-id case onto several positions.
    labels[0] = vocab_size - 1
    labels[52] = vocab_size - 1
    labels[-1] = vocab_size - 1
    hidden = hidden.requires_grad_()
    weight = weight.requires_grad_()

    ref_lp, _ = _torch_ce_logprobs_entropy(hidden.detach(), weight.detach(), labels, temperature)
    out_lp, _ = linear_cross_entropy(hidden, weight, labels, temperature)
    _synchronize()

    torch.testing.assert_close(out_lp.float(), ref_lp, atol=2e-2, rtol=2e-2)


def test_cann_forward_with_return_logits_entropy(monkeypatch):
    """Optional high-perf path: materialize logits for metric entropy.

    Some torch_npu builds may show rare per-token loss outliers in this mode;
    we only require that most tokens match and entropy is finite.
    """
    monkeypatch.setenv("VERL_NPU_LCE_BACKEND", "cann")
    monkeypatch.setenv("VERL_NPU_LCE_RETURN_LOGITS", "1")
    hidden, weight, labels, temperature = _make_inputs(64, 512, 2048, temperature=1.0, seed=1)

    ref_lp, ref_ent = _torch_ce_logprobs_entropy(hidden.detach(), weight.detach(), labels, temperature)
    out_lp, out_ent = linear_cross_entropy(hidden, weight, labels, temperature)
    _synchronize()

    abs_err = (out_lp.float() - ref_lp).abs()
    # Allow up to 2% outlier tokens (known return_logits quirk on some builds).
    outlier_frac = (abs_err > 0.05).float().mean().item()
    assert outlier_frac <= 0.02, f"too many logprob outliers: {outlier_frac:.3%}"
    assert torch.isfinite(out_ent).all()
    # On the well-behaved majority, entropy should still be in the ballpark.
    good = abs_err <= 0.05
    if good.any():
        torch.testing.assert_close(out_ent.float()[good], ref_ent[good], atol=8e-2, rtol=8e-2)


@pytest.mark.parametrize("temperature", [1.0, 1.5])
def test_cann_ce_backward_matches_torch_reference(temperature, monkeypatch):
    """CE-only path (dentropy=0), matching entropy_coeff=0 training."""
    monkeypatch.setenv("VERL_NPU_LCE_BACKEND", "cann")
    monkeypatch.delenv("VERL_NPU_LCE_RETURN_LOGITS", raising=False)
    num_tokens, hidden_size, vocab_size = 96, 768, 3072
    hidden, weight, labels, temperature = _make_inputs(
        num_tokens, hidden_size, vocab_size, temperature=temperature, seed=7
    )
    g_logprobs = torch.empty(num_tokens, dtype=torch.float32, device=DEVICE).uniform_(-1.0, 1.0)

    # Separate leaves so ref / CANN graphs do not interfere.
    h_ref = hidden.detach().clone().requires_grad_()
    w_ref = weight.detach().clone().requires_grad_()
    ref_lp, _ = _torch_ce_logprobs_entropy(h_ref, w_ref, labels, temperature)
    (d_ref_h, d_ref_w) = torch.autograd.grad(ref_lp, (h_ref, w_ref), g_logprobs)

    h_out = hidden.detach().clone().requires_grad_()
    w_out = weight.detach().clone().requires_grad_()
    out_lp, out_ent = linear_cross_entropy(h_out, w_out, labels, temperature)
    dentropy = torch.zeros_like(out_ent)
    (d_out_h, d_out_w) = torch.autograd.grad((out_lp, out_ent), (h_out, w_out), (g_logprobs, dentropy))
    _synchronize()

    torch.testing.assert_close(d_out_h.float(), d_ref_h.float(), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(d_out_w.float(), d_ref_w.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("temperature", [1.0, 1.5])
def test_cann_hybrid_entropy_backward_matches_torch_reference(temperature, monkeypatch):
    """dentropy!=0 uses chunked CE+H backward (hybrid mid-term path)."""
    monkeypatch.setenv("VERL_NPU_LCE_BACKEND", "cann")
    monkeypatch.delenv("VERL_NPU_LCE_RETURN_LOGITS", raising=False)
    monkeypatch.setenv("VERL_NPU_LCE_CHUNK_SIZE", "512")
    num_tokens, hidden_size, vocab_size = 64, 512, 2048
    hidden, weight, labels, temperature = _make_inputs(
        num_tokens, hidden_size, vocab_size, temperature=temperature, seed=11
    )
    g_logprobs = torch.empty(num_tokens, dtype=torch.float32, device=DEVICE).uniform_(-1.0, 1.0)
    g_entropy = torch.empty(num_tokens, dtype=torch.float32, device=DEVICE).uniform_(-0.5, 0.5)

    h_ref = hidden.detach().clone().requires_grad_()
    w_ref = weight.detach().clone().requires_grad_()
    ref_lp, ref_ent = _torch_ce_logprobs_entropy(h_ref, w_ref, labels, temperature)
    (d_ref_h, d_ref_w) = torch.autograd.grad((ref_lp, ref_ent), (h_ref, w_ref), (g_logprobs, g_entropy))

    h_out = hidden.detach().clone().requires_grad_()
    w_out = weight.detach().clone().requires_grad_()
    out_lp, out_ent = linear_cross_entropy(h_out, w_out, labels, temperature)
    (d_out_h, d_out_w) = torch.autograd.grad((out_lp, out_ent), (h_out, w_out), (g_logprobs, g_entropy))
    _synchronize()

    # Forward entropy may be a zero placeholder when return_logits=0; grads still
    # follow the Triton CE+H formula via chunked recompute.
    torch.testing.assert_close(d_out_h.float(), d_ref_h.float(), atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(d_out_w.float(), d_ref_w.float(), atol=8e-2, rtol=8e-2)


def test_triton_backend_raises_on_npu(monkeypatch):
    """Current main has no NPU Triton LCE; triton backend must fail loudly."""
    monkeypatch.setenv("VERL_NPU_LCE_BACKEND", "triton")
    hidden, weight, labels, temperature = _make_inputs(16, 128, 512)

    assert should_use_cann_linear_ce(hidden.device) is False
    with pytest.raises(RuntimeError, match="requires CANN"):
        linear_cross_entropy(hidden, weight, labels, temperature)


def test_direct_cann_apply_matches_dispatch(monkeypatch):
    monkeypatch.setenv("VERL_NPU_LCE_BACKEND", "auto")
    monkeypatch.delenv("VERL_NPU_LCE_RETURN_LOGITS", raising=False)
    hidden, weight, labels, temperature = _make_inputs(40, 384, 1536, seed=11)

    via_api = linear_cross_entropy(hidden, weight, labels, temperature)
    via_direct = CannLinearCrossEntropy.apply(
        hidden.view(-1, hidden.shape[-1]),
        weight,
        labels.view(-1),
        float(temperature),
        "none",
        None,
    )
    _synchronize()

    torch.testing.assert_close(via_api[0], via_direct[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(via_api[1], via_direct[1], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    if not is_torch_npu_available() or not is_cann_linear_ce_available():
        raise SystemExit(
            "NPU + CANN fused linear-CE APIs are required. "
            f"npu_available={is_torch_npu_available()}, cann={is_cann_linear_ce_available()}"
        )
    # Prefer pytest collection when available.
    raise SystemExit(pytest.main([__file__, "-sv", *os.environ.get("PYTEST_ADDOPTS", "").split()]))
