# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import gc

import pytest
import torch

import verl.utils.kernel.linear_topk_log_probs as linear_topk_log_probs_module
from verl.utils.kernel import topk_log_probs_kernels
from verl.utils.kernel.linear_topk_log_probs import linear_topk_log_probs


def _reference(hidden, weight, topk_ids, temperature):
    logits = torch.mm(hidden.reshape(-1, hidden.shape[-1]).float(), weight.float().T)
    log_probs = torch.log_softmax(logits / temperature, dim=-1)
    selected = torch.gather(log_probs, dim=-1, index=topk_ids.reshape(logits.shape[0], -1).long())
    return selected.view(topk_ids.shape)


@pytest.mark.parametrize(
    ("num_tokens", "vocab_size", "expected"),
    [
        (1, 1, 128),
        (32, 32768, 256),
        (33, 32768, 512),
        (1024, 32768, 512),
        (1025, 32768, 1024),
        (2048, 129, 256),
        (32, 151936, 1280),
    ],
)
def test_select_forward_vocab_per_split(num_tokens, vocab_size, expected):
    assert topk_log_probs_kernels._select_forward_vocab_per_split(num_tokens, vocab_size) == expected


@pytest.mark.parametrize(
    ("num_tokens", "vocab_size", "element_size", "expected"),
    [
        (1, 1, 2, 1),
        (32, 32768, 2, 32768),
        (2048, 32768, 2, 8192),
        (4096, 32768, 2, 4096),
        (8192, 32768, 2, 2048),
        (2048, 32768, 4, 4096),
        (1 << 20, 32768, 4, 128),
        (32, 1025, 2, 1025),
        (4096, 5000, 2, 5000),
    ],
)
def test_select_backward_vocab_per_split(num_tokens, vocab_size, element_size, expected):
    actual = topk_log_probs_kernels._select_backward_vocab_per_split(num_tokens, vocab_size, element_size)
    assert actual == expected
    allocated_bytes = num_tokens * actual * element_size
    minimum_bytes = num_tokens * min(128, vocab_size) * element_size
    if actual < vocab_size and minimum_bytes <= topk_log_probs_kernels._TARGET_BACKWARD_DLOGITS_BYTES:
        assert allocated_bytes <= topk_log_probs_kernels._TARGET_BACKWARD_DLOGITS_BYTES
    if minimum_bytes <= topk_log_probs_kernels._MAX_BACKWARD_DLOGITS_BYTES:
        assert allocated_bytes <= topk_log_probs_kernels._MAX_BACKWARD_DLOGITS_BYTES


def _assert_forward_backward_close(
    hidden,
    weight,
    topk_ids,
    temperature,
    *,
    forward_atol,
    forward_rtol,
    backward_atol,
    backward_rtol,
    chunk_size=None,
):
    actual = linear_topk_log_probs(hidden, weight, topk_ids, temperature, chunk_size=chunk_size)
    expected = _reference(hidden, weight, topk_ids, temperature)
    assert actual.shape == topk_ids.shape
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=forward_atol, rtol=forward_rtol)

    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (hidden, weight), upstream, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (hidden, weight), upstream)
    assert actual_grads[0].dtype == hidden.dtype
    assert actual_grads[1].dtype == weight.dtype
    torch.testing.assert_close(actual_grads[0], expected_grads[0], atol=backward_atol, rtol=backward_rtol)
    torch.testing.assert_close(actual_grads[1], expected_grads[1], atol=backward_atol, rtol=backward_rtol)


@pytest.mark.parametrize(
    ("hidden_shape", "ids_shape", "vocab_size", "temperature", "chunk_size"),
    [
        ((1, 1), (1, 1), 1, 0.1, 1),
        ((7, 16), (7, 3), 31, 0.73, 2),
        ((2, 5, 16), (2, 5, 4), 31, 4.0, 3),
        ((5, 33), (5, 9), 1025, 1.0, 1),
    ],
)
def test_linear_topk_log_probs_forward_and_backward_matches_reference(
    hidden_shape, ids_shape, vocab_size, temperature, chunk_size
):
    torch.manual_seed(1234)
    hidden = torch.randn(hidden_shape, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_shape[-1], dtype=torch.float32, requires_grad=True)
    topk_ids = torch.randint(0, weight.shape[0], ids_shape)
    # Exercise duplicate IDs: their upstream gradients must accumulate.
    if topk_ids.shape[-1] > 1:
        topk_ids[..., -1] = topk_ids[..., 0]

    _assert_forward_backward_close(
        hidden,
        weight,
        topk_ids,
        temperature,
        forward_atol=2e-5,
        forward_rtol=2e-5,
        backward_atol=4e-5,
        backward_rtol=4e-5,
        chunk_size=chunk_size,
    )


def test_linear_topk_log_probs_cpu_is_stable_for_large_logits():
    torch.manual_seed(2345)
    hidden = (torch.randn(4, 17) * 50).requires_grad_()
    weight = (torch.randn(67, 17) * 50).requires_grad_()
    ids = torch.tensor([[0, 66, 33], [1, 1, 65], [2, 40, 3], [66, 0, 66]])
    actual = linear_topk_log_probs(hidden, weight, ids, 0.25, chunk_size=2)
    expected = _reference(hidden, weight, ids, 0.25)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-6)


def test_linear_topk_log_probs_accepts_noncontiguous_inputs():
    torch.manual_seed(3456)
    hidden = torch.randn(19, 6).T.requires_grad_()
    weight = torch.randn(19, 43).T.requires_grad_()
    ids = torch.randint(0, 43, (6, 10))[:, ::2]
    assert not hidden.is_contiguous()
    assert not weight.is_contiguous()
    assert not ids.is_contiguous()
    _assert_forward_backward_close(
        hidden,
        weight,
        ids,
        1.7,
        forward_atol=2e-5,
        forward_rtol=2e-5,
        backward_atol=4e-5,
        backward_rtol=4e-5,
        chunk_size=2,
    )


def test_linear_topk_log_probs_accepts_int32_ids():
    torch.manual_seed(3567)
    hidden = torch.randn(3, 8, requires_grad=True)
    weight = torch.randn(17, 8, requires_grad=True)
    ids = torch.tensor([[0, 16], [8, 8], [3, 12]], dtype=torch.int32)
    actual = linear_topk_log_probs(hidden, weight, ids, 1.0)
    expected = _reference(hidden, weight, ids, 1.0)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"temperature": 0.0}, ValueError, "temperature must be positive"),
        ({"temperature": -1.0}, ValueError, "temperature must be positive"),
        ({"temperature": 1}, TypeError, "temperature must be a float"),
        ({"chunk_size": 0}, ValueError, "chunk_size must be positive"),
    ],
)
def test_linear_topk_log_probs_rejects_invalid_temperature(kwargs, error, match):
    with pytest.raises(error, match=match):
        linear_topk_log_probs(torch.randn(2, 4), torch.randn(7, 4), torch.ones(2, 1, dtype=torch.long), **kwargs)


@pytest.mark.parametrize(
    ("hidden", "weight", "ids", "error", "match"),
    [
        (torch.empty(0, 4), torch.randn(7, 4), torch.empty(0, 1, dtype=torch.long), ValueError, "at least one token"),
        (torch.randn(2, 4), torch.empty(0, 4), torch.zeros(2, 1, dtype=torch.long), ValueError, "vocabulary row"),
        (
            torch.randn(2, 4),
            torch.randn(7, 4),
            torch.empty(2, 0, dtype=torch.long),
            ValueError,
            "at least one selected",
        ),
        (
            torch.ones(2, 4, dtype=torch.long),
            torch.ones(7, 4, dtype=torch.long),
            torch.zeros(2, 1, dtype=torch.long),
            TypeError,
            "floating-point",
        ),
        (
            torch.randn(2, 4, dtype=torch.float32),
            torch.randn(7, 4, dtype=torch.float64),
            torch.zeros(2, 1, dtype=torch.long),
            ValueError,
            "same dtype",
        ),
        (torch.randn(2, 4), torch.randn(7, 5), torch.zeros(2, 1, dtype=torch.long), ValueError, "hidden size mismatch"),
        (torch.randn(2, 4), torch.randn(7, 4), torch.zeros(3, 1, dtype=torch.long), ValueError, "token count mismatch"),
        (torch.randn(2, 4), torch.randn(7, 4), torch.zeros(2, 1), TypeError, "int32 or int64"),
        (torch.randn(2, 4), torch.randn(7, 4), torch.full((2, 1), 7), ValueError, "must be in"),
        (torch.randn(2, 4), torch.randn(7, 4), torch.full((2, 1), -1), ValueError, "must be in"),
    ],
)
def test_linear_topk_log_probs_rejects_invalid_inputs(hidden, weight, ids, error, match):
    with pytest.raises(error, match=match):
        linear_topk_log_probs(hidden, weight, ids, 1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_linear_topk_log_probs_bfloat16_cuda_matches_reference(monkeypatch):
    torch.manual_seed(4321)
    hidden = torch.randn(29, 130, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(1057, 130, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    topk_ids = torch.randint(0, weight.shape[0], (29, 16), device="cuda")
    temperature = 1.2
    assert topk_log_probs_kernels.can_use_triton(hidden, topk_ids.shape[-1])

    def fail_fallback(*_args, **_kwargs):
        raise AssertionError("CUDA test must execute the Triton path")

    monkeypatch.setattr(linear_topk_log_probs_module, "_linear_fp32", fail_fallback)

    _assert_forward_backward_close(
        hidden,
        weight,
        topk_ids,
        temperature,
        forward_atol=2e-2,
        forward_rtol=2e-3,
        backward_atol=6e-2,
        backward_rtol=8e-3,
        chunk_size=7,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_linear_topk_log_probs_cuda_does_not_call_tensor_item(monkeypatch):
    """CUDA input validation must not introduce a GPU-to-CPU synchronization."""
    torch.manual_seed(4432)
    hidden = torch.randn(5, 32, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(129, 32, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, 129, (5, 8), device="cuda")
    original_item = torch.Tensor.item

    def reject_cuda_item(tensor, *args, **kwargs):
        if tensor.is_cuda:
            raise AssertionError("linear_topk_log_probs called Tensor.item() on a CUDA tensor")
        return original_item(tensor, *args, **kwargs)

    with monkeypatch.context() as patch_context:
        patch_context.setattr(torch.Tensor, "item", reject_cuda_item)
        output = linear_topk_log_probs(hidden, weight, ids, 1.0)
        torch.cuda.synchronize()

    expected = _reference(hidden, weight, ids, 1.0)
    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("dtype", "num_tokens", "hidden_size", "vocab_size", "topk", "temperature"),
    [
        (torch.float32, 1, 31, 1, 1, 0.5),
        (torch.float32, 33, 65, 1025, 7, 2.0),
        (torch.float16, 9, 127, 513, 8, 0.9),
        (torch.bfloat16, 37, 130, 2051, 16, 1.2),
        (torch.bfloat16, 8, 256, 257, 32, 0.73),
        (torch.bfloat16, 1025, 32, 2051, 8, 1.0),
        (torch.bfloat16, 33, 65, 8201, 16, 1.1),
        (torch.bfloat16, 4096, 17, 5000, 16, 0.8),
    ],
)
def test_linear_topk_log_probs_triton_shape_matrix(
    monkeypatch, dtype, num_tokens, hidden_size, vocab_size, topk, temperature
):
    torch.manual_seed(4567 + num_tokens + vocab_size)
    hidden = torch.randn(num_tokens, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    topk_ids = torch.randint(0, vocab_size, (num_tokens, topk), device="cuda")
    boundary_ids = [0, vocab_size - 1]
    boundary_ids += [
        value
        for value in (127, 128, 255, 256, 511, 512, 1023, 1024, 2047, 2048, 4095, 4096, 8191, 8192)
        if value < vocab_size
    ]
    topk_ids[0, : min(topk, len(boundary_ids))] = torch.tensor(boundary_ids[:topk], device="cuda")
    if topk > 1:
        topk_ids[:, -1] = topk_ids[:, 0]

    def fail_fallback(*_args, **_kwargs):
        raise AssertionError("shape-matrix case must execute the Triton path")

    monkeypatch.setattr(linear_topk_log_probs_module, "_linear_fp32", fail_fallback)
    if dtype == torch.float32:
        tolerances = dict(forward_atol=8e-4, forward_rtol=8e-4, backward_atol=2e-3, backward_rtol=2e-3)
    else:
        tolerances = dict(forward_atol=3e-2, forward_rtol=3e-3, backward_atol=8e-2, backward_rtol=1e-2)
    _assert_forward_backward_close(hidden, weight, topk_ids, temperature, **tolerances)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_linear_topk_log_probs_triton_supports_maximum_topk(monkeypatch):
    torch.manual_seed(5670)
    hidden = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(129, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ids = torch.arange(128, device="cuda").expand(2, -1).contiguous()
    assert topk_log_probs_kernels.can_use_triton(hidden, 128)

    def fail_fallback(*_args, **_kwargs):
        raise AssertionError("topk=128 must execute the Triton path")

    monkeypatch.setattr(linear_topk_log_probs_module, "_linear_fp32", fail_fallback)
    _assert_forward_backward_close(
        hidden,
        weight,
        ids,
        1.0,
        forward_atol=2e-2,
        forward_rtol=2e-3,
        backward_atol=8e-2,
        backward_rtol=1e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_linear_topk_log_probs_topk_above_triton_limit_uses_fallback(monkeypatch):
    torch.manual_seed(5671)
    hidden = torch.randn(3, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(257, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ids = torch.randint(0, 257, (3, 129), device="cuda")
    assert not topk_log_probs_kernels.can_use_triton(hidden, 129)

    def fail_triton(*_args, **_kwargs):
        raise AssertionError("topk=129 must use the bounded PyTorch fallback")

    monkeypatch.setattr(topk_log_probs_kernels, "topk_log_probs_forward", fail_triton)
    _assert_forward_backward_close(
        hidden,
        weight,
        ids,
        1.3,
        forward_atol=2e-2,
        forward_rtol=2e-3,
        backward_atol=6e-2,
        backward_rtol=8e-3,
        chunk_size=1,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_linear_topk_log_probs_forward_uses_less_peak_memory_than_full_logits():
    torch.manual_seed(5678)
    hidden = torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(32768, 256, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.randint(0, weight.shape[0], (1024, 16), device="cuda")

    def measure(callable_):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = callable_()
        torch.cuda.synchronize()
        peak_delta = torch.cuda.max_memory_allocated() - baseline
        del result
        return peak_delta

    fused_peak = measure(lambda: linear_topk_log_probs(hidden, weight, topk_ids, 1.0, chunk_size=32))
    reference_peak = measure(lambda: _reference(hidden, weight, topk_ids, 1.0))
    print(f"Triton peak={fused_peak / 2**20:.1f} MiB, full-logits peak={reference_peak / 2**20:.1f} MiB")

    assert fused_peak < reference_peak * 0.5, (
        f"expected chunked operator to use less than half the full-logits peak; "
        f"chunked={fused_peak / 2**20:.1f} MiB, full={reference_peak / 2**20:.1f} MiB"
    )
