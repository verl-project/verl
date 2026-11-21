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
"""Unit tests for fused_chunked_logprobs_and_entropy edge cases."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from verl.utils.torch_functional import fused_chunked_logprobs_and_entropy


def _reference_logprobs_and_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_fp64 = logits.double()
    if temperature is not None:
        if isinstance(temperature, torch.Tensor):
            t = temperature.double()
            if t.numel() == logits.shape[:-1].numel():
                t = t.reshape(*logits.shape[:-1], 1)
            elif logits.dim() >= 2 and t.numel() == logits.shape[0]:
                t = t.reshape(logits.shape[0], *([1] * (logits.dim() - 1)))
            logits_fp64 = logits_fp64 / t.clamp(min=1e-8)
        elif float(temperature) != 1.0:
            logits_fp64 = logits_fp64 / max(float(temperature), 1e-8)
    log_p = F.log_softmax(logits_fp64, dim=-1)
    p = log_p.exp()
    lp = torch.gather(log_p, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    ent = -(p * log_p).sum(dim=-1)
    return lp, ent


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk_size", [17, 64, 512])
@pytest.mark.parametrize("inplace_backward", [False, True])
def test_non_contiguous_response_slice_and_tail_chunks(
    dtype: torch.dtype,
    chunk_size: int,
    inplace_backward: bool,
):
    """Verify 3D non-contiguous response slices logits[:, -R-1:-1, :] and non-divisible chunk_size."""
    torch.manual_seed(42)
    bsz, full_seq_len, resp_len, vocab_size = 3, 29, 19, 257

    base_ref = torch.randn(bsz, full_seq_len, vocab_size, dtype=dtype, requires_grad=True)
    base_test = base_ref.detach().clone().requires_grad_(True)
    labels = torch.randint(0, vocab_size, (bsz, resp_len))
    w_lp = torch.randn(bsz, resp_len, dtype=torch.float32)
    w_ent = torch.randn(bsz, resp_len, dtype=torch.float32)

    slice_ref = base_ref[:, -resp_len - 1 : -1, :]
    slice_test = base_test[:, -resp_len - 1 : -1, :]
    assert not slice_test.is_contiguous()

    lp_ref, ent_ref = _reference_logprobs_and_entropy(slice_ref, labels, temperature=1.25)
    loss_ref = (lp_ref * w_lp.double() + 0.05 * ent_ref * w_ent.double()).mean()
    loss_ref.backward()

    lp_out, ent_out = fused_chunked_logprobs_and_entropy(
        slice_test,
        labels,
        temperature=1.25,
        chunk_size=chunk_size,
        compute_entropy=True,
        inplace_backward=inplace_backward,
    )
    loss_test = (lp_out * w_lp + 0.05 * ent_out * w_ent).mean()
    loss_test.backward()

    atol_fwd = 1e-5 if dtype == torch.float32 else 2e-2
    atol_bwd = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(lp_out, lp_ref.float(), atol=atol_fwd, rtol=1e-3)
    torch.testing.assert_close(ent_out, ent_ref.float(), atol=atol_fwd, rtol=1e-3)
    torch.testing.assert_close(base_test.grad, base_ref.grad, atol=atol_bwd, rtol=1e-2)
    assert torch.all(base_test.grad[:, : full_seq_len - resp_len - 1, :] == 0)
    assert torch.all(base_test.grad[:, -1:, :] == 0)


@pytest.mark.parametrize(
    "temp_kind",
    ["none", "unit_float", "low_scalar", "high_scalar", "per_token_rmpad", "per_sample_3d"],
)
def test_temperature_scaling_variants(temp_kind: str):
    """Verify scalar, unit, extreme low/high, and per-token/per-sample Tensor temperatures."""
    torch.manual_seed(108)
    if temp_kind == "per_sample_3d":
        shape = (4, 13, 193)
        labels = torch.randint(0, 193, (4, 13))
        temperature: float | torch.Tensor | None = torch.tensor([0.35, 1.0, 1.4, 2.1]).view(4, 1, 1)
    else:
        total_nnz, vocab_size = 73, 193
        shape = (total_nnz, vocab_size)
        labels = torch.randint(0, vocab_size, (total_nnz,))
        if temp_kind == "none":
            temperature = None
        elif temp_kind == "unit_float":
            temperature = 1.0
        elif temp_kind == "low_scalar":
            temperature = 0.25
        elif temp_kind == "high_scalar":
            temperature = 2.5
        else:
            temperature = torch.linspace(0.4, 1.9, total_nnz)

    logits_ref = torch.randn(*shape, dtype=torch.float32, requires_grad=True)
    logits_test = logits_ref.detach().clone().requires_grad_(True)
    g_lp = torch.randn(labels.shape, dtype=torch.float32)
    g_ent = torch.randn(labels.shape, dtype=torch.float32)

    lp_ref, ent_ref = _reference_logprobs_and_entropy(logits_ref, labels, temperature=temperature)
    (lp_ref * g_lp.double() - 0.02 * ent_ref * g_ent.double()).sum().backward()

    lp_out, ent_out = fused_chunked_logprobs_and_entropy(
        logits_test,
        labels,
        temperature=temperature,
        chunk_size=24,
        compute_entropy=True,
        inplace_backward=False,
    )
    (lp_out * g_lp - 0.02 * ent_out * g_ent).sum().backward()

    torch.testing.assert_close(lp_out, lp_ref.float(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ent_out, ent_ref.float(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(logits_test.grad, logits_ref.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "grad_mode",
    [
        "detached_entropy_logging_only",
        "zero_entropy_coeff_multiply",
        "compute_entropy_disabled",
        "entropy_loss_only",
    ],
)
def test_zero_or_partial_entropy_gradients(grad_mode: str):
    """Verify GRPO/DAPO common case where entropy is logged (grad=None/0) or disabled."""
    torch.manual_seed(2026)
    total_nnz, vocab_size = 65, 129
    logits_ref = torch.randn(total_nnz, vocab_size, dtype=torch.bfloat16, requires_grad=True)
    logits_test = logits_ref.detach().clone().requires_grad_(True)
    labels = torch.randint(0, vocab_size, (total_nnz,))
    advantages = torch.randn(total_nnz, dtype=torch.float32)

    lp_ref, ent_ref = _reference_logprobs_and_entropy(logits_ref, labels, temperature=1.1)
    compute_ent = grad_mode != "compute_entropy_disabled"
    lp_out, ent_out = fused_chunked_logprobs_and_entropy(
        logits_test,
        labels,
        temperature=1.1,
        chunk_size=16,
        compute_entropy=compute_ent,
        inplace_backward=False,
    )

    if grad_mode == "detached_entropy_logging_only":
        _ = ent_out.detach().mean()
        loss_ref = -(lp_ref * advantages.double()).mean()
        loss_test = -(lp_out * advantages).mean()
    elif grad_mode == "zero_entropy_coeff_multiply":
        loss_ref = -(lp_ref * advantages.double() + 0.0 * ent_ref).mean()
        loss_test = -(lp_out * advantages + 0.0 * ent_out).mean()
    elif grad_mode == "compute_entropy_disabled":
        assert ent_out is None
        loss_ref = -(lp_ref * advantages.double()).mean()
        loss_test = -(lp_out * advantages).mean()
    else:
        loss_ref = -ent_ref.mean()
        loss_test = -ent_out.mean()

    loss_ref.backward()
    loss_test.backward()

    torch.testing.assert_close(logits_test.grad, logits_ref.grad, atol=1e-2, rtol=1e-2)
