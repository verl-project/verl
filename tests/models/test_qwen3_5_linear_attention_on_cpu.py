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

"""CPU unit tests for Qwen3.5 linear attention packed chunk gated delta rule (Issue #8629).

Ensures that when FLA is not installed (e.g. Pure PyTorch or Ascend NPU fallbacks),
packed sequences do not leak recurrent state across sample boundaries by taking
the per-sample split path instead of silently dropping cu_seqlens into **kwargs.
"""

from unittest.mock import MagicMock

import pytest
import torch

from verl.models.transformers.qwen3_5 import (
    _call_accepts_kwarg,
    _packed_chunk_gated_delta_rule,
)


def test_call_accepts_kwarg():
    def explicit_cu_seqlens(q, k, v, cu_seqlens=None, cu_seqlens_cpu=None, **kwargs):
        pass

    def generic_kwargs_only(q, k, v, **kwargs):
        pass

    def no_kwargs(q, k, v):
        pass

    assert _call_accepts_kwarg(explicit_cu_seqlens, "cu_seqlens") is True
    assert _call_accepts_kwarg(explicit_cu_seqlens, "cu_seqlens_cpu") is True
    assert _call_accepts_kwarg(explicit_cu_seqlens, "nonexistent") is False

    # Functions with **kwargs should NOT claim to accept cu_seqlens if not in explicit parameters
    assert _call_accepts_kwarg(generic_kwargs_only, "cu_seqlens") is False
    assert _call_accepts_kwarg(generic_kwargs_only, "cu_seqlens_cpu") is False
    assert _call_accepts_kwarg(no_kwargs, "cu_seqlens") is False


def test_packed_chunk_gated_delta_rule_splits_samples_without_fla():
    """Verify that when chunk_gated_delta_rule does not explicitly declare cu_seqlens,

    _packed_chunk_gated_delta_rule evaluates each packed sample independently,
    preventing recurrent state leakage across sample boundaries.
    """
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            torch_chunk_gated_delta_rule,
        )
    except ImportError:
        pytest.skip("transformers Qwen3.5 model implementation is not available")

    torch.manual_seed(42)
    batch_size, num_heads, head_dim = 1, 2, 8
    L1, L2 = 4, 6
    total_len = L1 + L2
    cu_seqlens = torch.tensor([0, L1, total_len], dtype=torch.long)

    q = torch.randn(batch_size, total_len, num_heads, head_dim)
    k = torch.randn(batch_size, total_len, num_heads, head_dim)
    v = torch.randn(batch_size, total_len, num_heads, head_dim)
    g = torch.randn(batch_size, total_len, num_heads)
    beta = torch.randn(batch_size, total_len, num_heads)

    # Reference output: evaluate each sample individually
    kw = {
        "initial_state": None,
        "output_final_state": False,
        "use_qk_l2norm_in_kernel": True,
    }
    out_sample1, _ = torch_chunk_gated_delta_rule(q[:, :L1], k[:, :L1], v[:, :L1], g=g[:, :L1], beta=beta[:, :L1], **kw)
    out_sample2, _ = torch_chunk_gated_delta_rule(q[:, L1:], k[:, L1:], v[:, L1:], g=g[:, L1:], beta=beta[:, L1:], **kw)

    out_packed, _ = _packed_chunk_gated_delta_rule(torch_chunk_gated_delta_rule, q, k, v, g, beta, cu_seqlens, None)

    assert torch.allclose(out_packed[:, :L1], out_sample1, atol=1e-5), "Sample 1 output diverged"
    assert torch.allclose(out_packed[:, L1:], out_sample2, atol=1e-5), (
        "Sample 2 output diverged (recurrent state leaked across samples)"
    )


def test_packed_chunk_gated_delta_rule_calls_fla_kernel_directly_when_supported():
    """Verify that when the kernel explicitly declares cu_seqlens, it is called directly."""
    mock_kernel = MagicMock()

    def mock_fn(q, k, v, cu_seqlens=None, **kwargs):
        return mock_kernel(q, k, v, cu_seqlens=cu_seqlens, **kwargs)

    mock_kernel.return_value = (torch.zeros(1, 10, 2, 8), None)

    q = torch.zeros(1, 10, 2, 8)
    k = torch.zeros(1, 10, 2, 8)
    v = torch.zeros(1, 10, 2, 8)
    g = torch.zeros(1, 10, 2)
    beta = torch.zeros(1, 10, 2)
    cu_seqlens = torch.tensor([0, 4, 10], dtype=torch.long)

    out, _ = _packed_chunk_gated_delta_rule(mock_fn, q, k, v, g, beta, cu_seqlens, None)

    assert mock_kernel.call_count == 1
    call_kwargs = mock_kernel.call_args[1]
    assert "cu_seqlens" in call_kwargs
    assert torch.equal(call_kwargs["cu_seqlens"], cu_seqlens)
