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
    CannLinearCrossEntropy,
    _chunked_ce_entropy_backward,
    _chunked_entropy_stats,
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
@pytest.mark.parametrize("token_chunk_size", [7, 256])
def test_chunked_ce_entropy_backward_matches_torch(temperature, chunk_size, token_chunk_size):
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
            token_chunk_size=token_chunk_size,
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


@pytest.mark.parametrize("temperature", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("offset", [0.0, 10000.0])
def test_entropy_stats_fp64_finite_difference(temperature, offset):
    """Check stable tiled statistics against autograd and finite differences."""
    torch.manual_seed(31)
    h = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    w = torch.randn(11, 4, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        h[:, 0] = 1
        w[:, 0] = offset
    labels = torch.tensor([0, 10, 3, 8, 1])
    glp = torch.tensor([0.3, 0.0, -0.2, 0.7, 0.1], dtype=torch.float64)
    gh = torch.tensor([-0.1, 0.0, 0.4, 0.2, -0.3], dtype=torch.float64)

    def tiled(a, b):
        m, s, q = _chunked_entropy_stats(a / temperature, b, chunk_size=4, token_chunk_size=2)
        ent = s.log() - q / s
        selected = ((a / temperature) * b[labels]).sum(-1)
        lp = (selected - m) - s.log()
        return (lp * glp + ent * gh).sum()

    logits = (h / temperature) @ w.T
    lp = logits.log_softmax(-1)
    entropy = -(lp.exp() * lp).sum(-1)
    reference = (lp[torch.arange(5), labels] * glp + entropy * gh).sum()
    torch.testing.assert_close(tiled(h, w), reference, atol=1e-8, rtol=1e-6)
    expected = torch.autograd.grad(reference, (h, w))
    actual = torch.autograd.grad(tiled(h, w), (h, w))
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-6)
    for tensor, grad in zip((h, w), actual, strict=True):
        for index in [(0, 1), (tensor.shape[0] - 1, 3)]:
            step = 1e-4
            with torch.no_grad():
                tensor[index] += step
                plus = tiled(h, w)
                tensor[index] -= 2 * step
                minus = tiled(h, w)
                tensor[index] += step
            torch.testing.assert_close((plus - minus) / (2 * step), grad[index], atol=1e-7, rtol=1e-5)


def test_entropy_stats_preserve_fp32_under_autocast():
    """An enclosing model autocast must not downcast the entropy matmul."""
    torch.manual_seed(23)
    hidden = torch.randn(9, 8)
    weight = torch.randn(19, 8)
    expected = _chunked_entropy_stats(hidden, weight, chunk_size=5, token_chunk_size=4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = _chunked_entropy_stats(hidden, weight, chunk_size=5, token_chunk_size=4)
    for value, reference in zip(actual, expected, strict=True):
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, reference, atol=1e-6, rtol=1e-6)


def test_entropy_stats_extreme_logits():
    """Large common offsets and saturated distributions retain finite entropy."""
    hidden = torch.eye(3, dtype=torch.float64)
    weight = torch.tensor([[10000, 1000, -1000], [10000, -1000, 1000], [10000, 0, 0]], dtype=torch.float64)
    _, s, q = _chunked_entropy_stats(hidden, weight, chunk_size=2, token_chunk_size=2)
    logp = (hidden @ weight.T).log_softmax(-1)
    torch.testing.assert_close(s.log() - q / s, -(logp.exp() * logp).sum(-1), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("entropy_grad", [0.0, 0.7])
@pytest.mark.parametrize("softmax_result", ["none", "empty", "present"])
def test_cann_forward_backward_entropy_contract(monkeypatch, reduction, entropy_grad, softmax_result):
    """An absent CANN softmax must not turn a differentiable entropy into zero."""
    from verl.utils.kernel.npu import cann_linear_ce as module

    torch.manual_seed(47)
    h = (torch.randn(5, 4) * 0.2).to(torch.bfloat16).requires_grad_()
    w = (torch.randn(11, 4) * 0.2).to(torch.bfloat16).requires_grad_()
    labels = torch.tensor([0, 10, 1, 4, 9])
    temperature = 1.5
    scaled = (h.detach() / temperature).float()
    logits = scaled @ w.detach().float().T
    lp_ref = logits.log_softmax(-1)
    ent_ref = -(lp_ref.exp() * lp_ref).sum(-1)
    calls = []

    def online(hidden, weight, target, start, end, return_logits):
        z = hidden.float() @ weight.float().T
        m = z.max(-1).values
        s = (z - m[:, None]).exp().sum(-1)
        selected = z[torch.arange(5), target.long()] - m
        return m, s, selected, torch.empty(0), target, None

    def ce(m, s, selected, **kwargs):
        softmax = None
        if softmax_result == "empty":
            softmax = torch.empty(0)
        elif softmax_result == "present":
            softmax = lp_ref.exp()
        return s.log() - selected, softmax

    def backward(g, hidden, weight, mask, target, smoothing, m, s, softmax):
        calls.append("cann")
        p = (hidden.float() @ weight.float().T - m[:, None]).exp() / s[:, None]
        p[torch.arange(5), target.long()] -= 1
        dz = g[:, None] * p
        return dz @ weight.float(), dz.T @ hidden.float()

    monkeypatch.setattr(module, "_resolve_cann_apis", lambda: (online, ce, backward))
    monkeypatch.delenv("VERL_NPU_LCE_RETURN_LOGITS", raising=False)
    monkeypatch.setenv("VERL_NPU_LCE_CHUNK_SIZE", "4")
    lp, entropy = CannLinearCrossEntropy.apply(h, w, labels, temperature, reduction)
    expected_lp = lp_ref[torch.arange(5), labels]
    if reduction != "none":
        expected_lp = getattr(expected_lp, reduction)()
    torch.testing.assert_close(lp, expected_lp)
    torch.testing.assert_close(entropy, ent_ref, atol=1e-6, rtol=1e-6)
    assert torch.all(entropy > 0)
    mask = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
    glp = mask if reduction == "none" else torch.tensor(0.3)
    gent = mask * entropy_grad
    dh, dw = torch.autograd.grad((lp, entropy), (h, w), (glp, gent))
    hr = scaled.double().requires_grad_()
    wr = w.detach().double().requires_grad_()
    ref_lp = (hr @ wr.T).log_softmax(-1)
    ref_ent = -(ref_lp.exp() * ref_lp).sum(-1)
    selected = ref_lp[torch.arange(5), labels]
    if reduction != "none":
        selected = getattr(selected, reduction)()
    rh, rw = torch.autograd.grad((selected, ref_ent), (hr, wr), (glp.double(), gent.double()))
    torch.testing.assert_close(dh.float(), (rh / temperature).float(), atol=5e-4, rtol=1e-2)
    torch.testing.assert_close(dw.float(), rw.float(), atol=5e-4, rtol=1e-2)
    assert calls == (["cann"] if entropy_grad == 0 else [])


@pytest.mark.parametrize("offset", [0.0, 32.0, 128.0])
def test_entropy_gradient_is_independent_of_cann_ce_rounding(monkeypatch, offset):
    """Entropy must use its own normalization when CANN rounds CE logits."""
    from verl.utils.kernel.npu import cann_linear_ce as module

    torch.manual_seed(42)
    h = (torch.randn(5, 8) * 0.2).bfloat16()
    w = (torch.randn(11, 8) * 0.2).bfloat16()
    h[:, 0] = 1
    w[:, 0] = offset
    h.requires_grad_()
    w.requires_grad_()
    labels = torch.tensor([0, 10, 3, 1, 7])

    def online(hidden, weight, target, start, end, return_logits):
        z = (hidden.float() @ weight.float().T).bfloat16().float()
        maximum = z.max(-1).values
        partition = (z - maximum[:, None]).exp().sum(-1)
        selected = z[torch.arange(5), target.long()] - maximum
        return maximum, partition, selected, torch.empty(0), target, None

    def ce(maximum, partition, selected, **kwargs):
        return partition.log() - selected, None

    def backward(*args):
        raise AssertionError("Entropy-only backward must use the hybrid path")

    monkeypatch.setattr(module, "_resolve_cann_apis", lambda: (online, ce, backward))
    monkeypatch.delenv("VERL_NPU_LCE_RETURN_LOGITS", raising=False)
    monkeypatch.setenv("VERL_NPU_LCE_CHUNK_SIZE", "4")
    lp, entropy = CannLinearCrossEntropy.apply(h, w, labels)
    g = torch.linspace(-0.2, 0.3, 5)
    actual = torch.autograd.grad((lp, entropy), (h, w), (torch.zeros_like(lp), g))
    hr = h.detach().double().requires_grad_()
    wr = w.detach().double().requires_grad_()
    logp = (hr @ wr.T).log_softmax(-1)
    expected_entropy = -(logp.exp() * logp).sum(-1)
    expected = torch.autograd.grad(expected_entropy, (hr, wr), g.double())
    torch.testing.assert_close(entropy.double(), expected_entropy, atol=1e-5, rtol=1e-5)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value.double(), reference, atol=5e-4, rtol=1e-2)
