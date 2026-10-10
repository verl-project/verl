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
"""CPU unit tests for Ascend CANN fused linear-CE policy helpers."""

from __future__ import annotations

import pytest
import torch

from verl.utils.kernel.npu.cann_linear_ce import (
    _chunked_ce_entropy_backward,
    _resolve_vocab_range,
    should_use_cann_linear_ce,
)


def test_should_use_cann_linear_ce_non_npu_device(monkeypatch):
    monkeypatch.delenv("VERL_NPU_LCE_BACKEND", raising=False)
    monkeypatch.setattr(
        "verl.utils.kernel.npu.cann_linear_ce.is_cann_linear_ce_available",
        lambda: True,
    )
    assert should_use_cann_linear_ce(torch.device("cpu")) is False


def test_should_use_cann_linear_ce_triton_backend(monkeypatch):
    monkeypatch.setenv("VERL_NPU_LCE_BACKEND", "triton")
    monkeypatch.setattr(
        "verl.utils.kernel.npu.cann_linear_ce.is_cann_linear_ce_available",
        lambda: True,
    )

    class _NpuDev:
        type = "npu"

    assert should_use_cann_linear_ce(_NpuDev()) is False


def test_should_use_cann_linear_ce_auto_when_available(monkeypatch):
    monkeypatch.setenv("VERL_NPU_LCE_BACKEND", "auto")
    monkeypatch.setattr(
        "verl.utils.kernel.npu.cann_linear_ce.is_cann_linear_ce_available",
        lambda: True,
    )

    class _NpuDev:
        type = "npu"

    assert should_use_cann_linear_ce(_NpuDev()) is True


def test_should_use_cann_linear_ce_cann_backend_missing_apis(monkeypatch):
    monkeypatch.setenv("VERL_NPU_LCE_BACKEND", "cann")
    monkeypatch.setattr(
        "verl.utils.kernel.npu.cann_linear_ce.is_cann_linear_ce_available",
        lambda: False,
    )

    class _NpuDev:
        type = "npu"

    with pytest.raises(RuntimeError, match="VERL_NPU_LCE_BACKEND=cann"):
        should_use_cann_linear_ce(_NpuDev())


@pytest.mark.parametrize("temperature", [1.0, 1.5])
@pytest.mark.parametrize("chunk_size", [32, 128, 2048])
def test_chunked_ce_entropy_backward_matches_torch(temperature, chunk_size):
    """Hybrid CE+H path must match autograd through matmul+CE+Shannon H."""
    torch.manual_seed(0)
    num_tokens, hidden_size, vocab_size = 24, 64, 96
    hidden = torch.empty(num_tokens, hidden_size, dtype=torch.float32).uniform_(-0.5, 0.5)
    weight = torch.empty(vocab_size, hidden_size, dtype=torch.float32).uniform_(-0.5, 0.5)
    labels = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int64)
    g_logprobs = torch.empty(num_tokens, dtype=torch.float32).uniform_(-1.0, 1.0)
    g_entropy = torch.empty(num_tokens, dtype=torch.float32).uniform_(-0.5, 0.5)

    h_ref = hidden.detach().clone().requires_grad_()
    w_ref = weight.detach().clone().requires_grad_()
    logits = torch.matmul(h_ref, w_ref.T) / temperature
    logprobs = -torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    pd = torch.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - (pd * logits).sum(dim=-1)
    (d_ref_h, d_ref_w) = torch.autograd.grad((logprobs, entropy), (h_ref, w_ref), (g_logprobs, g_entropy))

    with torch.no_grad():
        logits_det = torch.matmul(hidden, weight.T) / temperature
        maximum = logits_det.max(dim=-1).values
        accumulate = torch.exp(logits_det - maximum[:, None]).sum(dim=-1)
        hidden_scaled = hidden * (1.0 / temperature)
        d_h, d_w = _chunked_ce_entropy_backward(
            g_logprobs,
            g_entropy,
            hidden_scaled,
            weight,
            labels,
            maximum,
            accumulate,
            temperature,
            "none",
            vocab_start=0,
            chunk_size=chunk_size,
        )

    torch.testing.assert_close(d_h, d_ref_h, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(d_w, d_ref_w, atol=1e-4, rtol=1e-4)


def test_chunked_ce_entropy_backward_ce_only_matches_torch():
    """dentropy=0 should still match CE-only autograd."""
    torch.manual_seed(1)
    num_tokens, hidden_size, vocab_size = 16, 48, 64
    temperature = 1.0
    hidden = torch.empty(num_tokens, hidden_size, dtype=torch.float32).uniform_(-0.5, 0.5)
    weight = torch.empty(vocab_size, hidden_size, dtype=torch.float32).uniform_(-0.5, 0.5)
    labels = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int64)
    g_logprobs = torch.empty(num_tokens, dtype=torch.float32).uniform_(-1.0, 1.0)

    h_ref = hidden.detach().clone().requires_grad_()
    w_ref = weight.detach().clone().requires_grad_()
    logits = torch.matmul(h_ref, w_ref.T) / temperature
    logprobs = -torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    (d_ref_h, d_ref_w) = torch.autograd.grad(logprobs, (h_ref, w_ref), g_logprobs)

    with torch.no_grad():
        logits_det = torch.matmul(hidden, weight.T) / temperature
        maximum = logits_det.max(dim=-1).values
        accumulate = torch.exp(logits_det - maximum[:, None]).sum(dim=-1)
        d_h, d_w = _chunked_ce_entropy_backward(
            g_logprobs,
            torch.zeros_like(g_logprobs),
            hidden,
            weight,
            labels,
            maximum,
            accumulate,
            temperature,
            "none",
            vocab_start=0,
            chunk_size=40,
        )

    torch.testing.assert_close(d_h, d_ref_h, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(d_w, d_ref_w, atol=1e-4, rtol=1e-4)


def test_resolve_vocab_range_is_half_open():
    """Ascend/Megatron use [start, end); end == start + local_vocab."""
    weight = torch.empty(2048, 512)
    start, end = _resolve_vocab_range(weight, None)
    assert (start, end) == (0, 2048)
    # Last valid id is end - 1.
    assert end - 1 == weight.shape[0] - 1


def test_resolve_vocab_range_with_tp(monkeypatch):
    weight = torch.empty(1024, 256)

    class _FakeGroup:
        pass

    group = _FakeGroup()
    monkeypatch.setattr(
        "verl.utils.kernel.npu.cann_linear_ce.dist.get_rank",
        lambda pg: 2 if pg is group else 0,
    )
    start, end = _resolve_vocab_range(weight, group)
    assert (start, end) == (2048, 3072)
