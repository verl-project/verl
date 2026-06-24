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
"""Unit tests for :func:`verl.workers.teacher.utils.chunked_gather_logprobs`.

The chunked gather is the inner kernel of the FSDP teacher worker — it computes
``log_softmax(logits).gather(idx)`` without materializing the full
``log_softmax`` tensor. These tests run on CPU with random tensors and do not
depend on Ray, FSDP, or CUDA.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch
import torch.nn.functional as F

from verl.workers.teacher.utils import chunked_gather_logprobs


def _reference_gather(logits: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
    """One-shot reference: full log_softmax then gather. fp32 throughout."""
    return torch.gather(F.log_softmax(logits.float(), dim=-1), dim=-1, index=topk_ids).to(logits.dtype)


@pytest.mark.parametrize("chunk_size", [1, 7, 64, 1_000_000])
def test_chunked_gather_matches_oneshot(chunk_size):
    """chunk_size sweep — including a value that exceeds the row count — agrees
    with the one-shot reference up to fp32 precision."""
    torch.manual_seed(0)
    bsz, seq, vocab, k = 2, 5, 13, 3
    logits = torch.randn(bsz, seq, vocab)
    topk_ids = torch.randint(0, vocab, (bsz, seq, k))

    out = chunked_gather_logprobs(logits, topk_ids, chunk_size=chunk_size)
    expected = _reference_gather(logits, topk_ids)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_chunked_gather_preserves_shape_and_dtype():
    """Output shape == leading shape of logits + (K,), dtype matches logits."""
    logits = torch.randn(3, 4, 17, dtype=torch.float32)
    topk_ids = torch.randint(0, 17, (3, 4, 5), dtype=torch.int64)
    out = chunked_gather_logprobs(logits, topk_ids, chunk_size=8)
    assert out.shape == (3, 4, 5)
    assert out.dtype == torch.float32


def test_chunked_gather_works_with_2d_input():
    """2D inputs (rmpad layout): logits ``(N, V)`` and topk_ids ``(N, K)``."""
    torch.manual_seed(1)
    N, V, K = 11, 9, 4
    logits = torch.randn(N, V)
    topk_ids = torch.randint(0, V, (N, K))
    out = chunked_gather_logprobs(logits, topk_ids, chunk_size=3)
    expected = _reference_gather(logits, topk_ids)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_chunked_gather_rejects_mismatched_leading_shape():
    logits = torch.randn(2, 3, 5)
    topk_ids = torch.randint(0, 5, (2, 4, 2))  # seq dim mismatch
    with pytest.raises(ValueError, match="Leading shapes"):
        chunked_gather_logprobs(logits, topk_ids)


def test_chunked_gather_rejects_nonpositive_chunk_size():
    logits = torch.randn(1, 1, 4)
    topk_ids = torch.zeros(1, 1, 2, dtype=torch.int64)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunked_gather_logprobs(logits, topk_ids, chunk_size=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_chunked_gather_handles_extreme_logits_in_fp32(dtype):
    """logsumexp in fp32 keeps ``log_softmax`` numerically stable even when raw
    logits span large magnitudes (regression for naive ``log(softmax(x))``)."""
    torch.manual_seed(2)
    logits = torch.randn(8, 32, dtype=dtype) * 50.0  # magnitude ~ 50
    topk_ids = torch.randint(0, 32, (8, 4))
    out = chunked_gather_logprobs(logits, topk_ids, chunk_size=4)
    assert torch.isfinite(out).all()
    expected = _reference_gather(logits, topk_ids)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
