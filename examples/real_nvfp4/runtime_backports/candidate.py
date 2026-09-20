# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Production patch input for TE 2.18 row-scaled grouped-GEMM batching.

patch_te.py verifies this file's hash and extracts batched_row_scaled_gemm into
TE's installed GEMM module. Unsupported layouts use the original implementation;
quantizers, scale definitions, rounding points and backward remain unchanged.
"""

from contextlib import contextmanager

import torch
from transformer_engine.pytorch.cpp_extensions import gemm
from transformer_engine.pytorch.module import grouped_linear
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4TensorStorage

ORIGINAL = gemm.general_grouped_gemm
COUNTERS = {"batched": 0, "fallback": 0}


def batched_row_scaled_gemm(
    A,
    B,
    out,
    quantization_params,
    out_dtype,
    layout="TN",
    m_splits=None,
    gelu=False,
    grad=False,
    accumulate=False,
    bias=None,
    use_bias=False,
    use_split_accumulator=False,
    D_dtype=None,
    single_output=False,
):
    supported = (
        layout == "TN"
        and not (gelu or grad or accumulate or use_bias)
        and D_dtype is None
        and single_output
        and len(out) == 1
        and m_splits is not None
        and len(A) == len(B) == len(m_splits) == len(quantization_params)
        and len(A) > 0
        and all(q is None for q in quantization_params)
        and all(isinstance(t, NVFP4TensorStorage) and not t._row_scaled_nvfp4 for t in A)
        and all(isinstance(t, NVFP4TensorStorage) and t._row_scaled_nvfp4 for t in B)
        and out[0].ndim == 2
        and out[0].is_contiguous()
        and out[0].numel() > 0
    )
    if not supported:
        COUNTERS["fallback"] += 1
        return ORIGINAL(
            A,
            B,
            out,
            quantization_params,
            out_dtype,
            layout,
            m_splits,
            gelu,
            grad,
            accumulate,
            bias,
            use_bias,
            use_split_accumulator,
            D_dtype,
            single_output,
        )

    # A and B retain their original quantized bytes/block scales. Like TE's
    # scalar fallback, use amax=1 aliases for GEMM and apply global scales in FP32.
    # One shared constant is immutable throughout this synchronous call; no cache
    # or state can become stale across optimizer updates or CUDA graph replay.
    one = torch.ones(1, dtype=torch.float32, device=out[0].device)
    aliases_a, aliases_b, activation_amaxes, weight_amaxes = [], [], [], []
    for a, b, rows in zip(A, B, m_splits, strict=True):
        assert a._amax_rowwise.dtype == b._amax_rowwise.dtype == torch.float32
        assert a._amax_rowwise.numel() == 1 and b._amax_rowwise.numel() == rows
        ma, mb = a.get_metadata(), b.get_metadata()
        ma.update(amax_rowwise=one, row_scaled_nvfp4=False)
        mb.update(amax_rowwise=one, row_scaled_nvfp4=False)
        aliases_a.append(NVFP4TensorStorage(**ma))
        aliases_b.append(NVFP4TensorStorage(**mb))
        activation_amaxes.append(b._amax_rowwise.reshape(-1))
        weight_amaxes.append(a._amax_rowwise.reshape(1).expand(rows))
    scales = (torch.cat(activation_amaxes) * torch.cat(weight_amaxes)).reshape(-1, 1)
    assert scales.shape[0] == out[0].shape[0]
    fp32_out = torch.empty_like(out[0], dtype=torch.float32)
    _, bias_result, gelu_result = ORIGINAL(
        aliases_a,
        aliases_b,
        [fp32_out],
        quantization_params,
        torch.float32,
        layout=layout,
        m_splits=m_splits,
        single_output=True,
        use_split_accumulator=use_split_accumulator,
    )
    fp32_out.mul_(scales)
    # Same single FP32->output rounding as TE's per-expert path.
    out[0].copy_(fp32_out)
    COUNTERS["batched"] += 1
    return out[0], bias_result, gelu_result


@contextmanager
def installed():
    assert gemm.general_grouped_gemm is ORIGINAL
    assert grouped_linear.general_grouped_gemm is ORIGINAL
    gemm.general_grouped_gemm = batched_row_scaled_gemm
    grouped_linear.general_grouped_gemm = batched_row_scaled_gemm
    try:
        yield
    finally:
        gemm.general_grouped_gemm = ORIGINAL
        grouped_linear.general_grouped_gemm = ORIGINAL
