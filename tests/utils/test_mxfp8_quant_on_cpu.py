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
"""Engine-neutral MXFP8 quantization core shared by the SGLang and vLLM rollout paths."""

import math
import sys
import types

import torch

from verl.utils.mxfp8_quant import MXFP8_GROUP_SIZE, mxfp8_quantize


def _install_fake_te(monkeypatch, recorded):
    class FakeQuantized:
        def __init__(self, data):
            rows, k = data.shape
            recorded["quantized_shape"] = (rows, k)
            self._rowwise_data = data.to(torch.uint8)
            scale_rows = math.ceil(rows / 128) * 128
            scale_cols = math.ceil((k // MXFP8_GROUP_SIZE) / 4) * 4
            self._rowwise_scale_inv = (
                torch.arange(scale_rows * scale_cols, dtype=torch.float32)
                .reshape(scale_rows, scale_cols)
                .to(torch.uint8)
            )

    class FakeMXFP8Quantizer:
        def __init__(self, fp8_dtype, rowwise, columnwise):
            recorded["rowwise"] = rowwise
            recorded["columnwise"] = columnwise

        def quantize(self, tensor):
            return FakeQuantized(tensor)

    te = types.ModuleType("transformer_engine")
    te_pytorch = types.ModuleType("transformer_engine.pytorch")
    te_constants = types.ModuleType("transformer_engine.pytorch.constants")
    te_pytorch.MXFP8Quantizer = FakeMXFP8Quantizer
    te_constants.TE_DType = {torch.float8_e4m3fn: "kFloat8E4M3"}
    te.pytorch = te_pytorch
    te_pytorch.constants = te_constants
    monkeypatch.setitem(sys.modules, "transformer_engine", te)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", te_pytorch)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.constants", te_constants)


def test_sglang_module_reexports_shared_core():
    from verl.utils.sglang import sglang_mxfp8_utils

    assert sglang_mxfp8_utils.mxfp8_quantize is mxfp8_quantize
    assert sglang_mxfp8_utils.MXFP8_GROUP_SIZE == MXFP8_GROUP_SIZE


def test_mxfp8_quantize_3d_expert_weight(monkeypatch):
    # vLLM fused-MoE expert weights are 3D [num_experts, n, k]; the shared core
    # must flatten leading dims, quantize, and restore the scale shape.
    recorded = {}
    _install_fake_te(monkeypatch, recorded)

    weight = torch.randn(4, 16, 64, dtype=torch.bfloat16)
    qweight, scale = mxfp8_quantize(weight)

    assert recorded["quantized_shape"] == (64, 64)
    assert qweight.shape == (4, 16, 64)
    assert qweight.dtype == torch.float8_e4m3fn
    assert scale.shape == (4, 16, 64 // MXFP8_GROUP_SIZE)
    assert scale.dtype == torch.uint8
