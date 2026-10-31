# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The fused Triton fast path for FP8_BLOCK weight export must produce the same compressed-tensors state dict as the
eager reference path (``weight`` fp8 e4m3 + ``weight_scale`` fp32 [M/128, N/128], "multiply to dequantize")."""

import pytest
import torch

pytest.importorskip("compressed_tensors")

from verl.utils.qat.compressed_tensors import (  # noqa: E402
    _compress_packed_experts_fp8_block,
    _compress_weight,
    _compress_weight_eager,
    _fp8_block_fast_path_applies,
    _packed_experts_fp8_block_applies,
    _scheme_for_preset,
    _unpack_experts,
)

requires_fast_path = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not __import__("verl.utils.kernel.fp8_kernel", fromlist=["x"]).is_triton_available(),
    reason="needs CUDA + Triton",
)

FP8_SCHEME = _scheme_for_preset("FP8_BLOCK", ["Linear"])


def _random_weight(shape, device="cuda", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(shape, generator=g, dtype=torch.float32)
    # a few outliers and a few tiny rows so that scales span several binades
    w[0, :] *= 50.0
    w[-1, :] *= 1e-3
    return w.to(torch.bfloat16).to(device)


def _fp8_e4m3_ulp(v: torch.Tensor) -> torch.Tensor:
    """Spacing of representable e4m3 values around |v|: 2^(e-3) for normals (|v| >= 2^-6), 2^-9 for subnormals."""
    m = v.abs()
    e = torch.floor(torch.log2(m.clamp_min(2.0**-6)))
    return torch.where(m < 2.0**-6, torch.full_like(m, 2.0**-9), torch.exp2(e - 3))


def _assert_same_compressed(fast: dict, eager: dict):
    assert set(fast) == set(eager) == {"weight", "weight_scale"}
    assert fast["weight"].dtype == eager["weight"].dtype == torch.float8_e4m3fn
    assert fast["weight"].shape == eager["weight"].shape
    assert fast["weight_scale"].dtype == torch.float32
    assert fast["weight_scale"].shape == eager["weight_scale"].shape
    torch.testing.assert_close(fast["weight_scale"], eager["weight_scale"].to(torch.float32), rtol=1e-6, atol=0)
    a, b = fast["weight"].float(), eager["weight"].float()
    # Both round-to-nearest-even the same fp32 scale; the Triton kernel multiplies by the reciprocal scale while the
    # eager path divides, so a tiny fraction of elements on exact rounding boundaries may land one fp8 ulp apart
    # (including 0 vs the smallest subnormal). Never more than one ulp, never more than 0.1% of the elements.
    diff = (a - b).abs()
    ulp = torch.maximum(_fp8_e4m3_ulp(a), _fp8_e4m3_ulp(b))
    frac = (diff > 0).float().mean().item()
    max_ulps = (diff / ulp).max().item()
    assert frac <= 1e-3, f"{frac:.2e} of elements differ (max {max_ulps:.2f} ulp)"
    assert max_ulps <= 1.001, f"fast and eager fp8 values differ by {max_ulps:.2f} ulp ({frac:.2e} of elements differ)"


@requires_fast_path
@pytest.mark.parametrize("shape", [(768, 2048), (2048, 768), (2048, 2048), (1000, 2048), (768, 2000), (64, 64)])
def test_fast_path_matches_eager(shape):
    w = _random_weight(shape)
    assert _fp8_block_fast_path_applies(w, FP8_SCHEME)
    fast = _compress_weight(w, FP8_SCHEME)
    eager = _compress_weight_eager(w, FP8_SCHEME)
    _assert_same_compressed(fast, eager)


@requires_fast_path
def test_fast_path_disabled_by_env(monkeypatch):
    w = _random_weight((256, 256))
    monkeypatch.setenv("VERL_FP8_EXPORT_FAST_PATH", "0")
    assert not _fp8_block_fast_path_applies(w, FP8_SCHEME)
    monkeypatch.setenv("VERL_FP8_EXPORT_FAST_PATH", "1")
    assert _fp8_block_fast_path_applies(w, FP8_SCHEME)


@requires_fast_path
def test_fast_path_only_for_fp8_block_on_cuda():
    w = _random_weight((256, 256))
    assert not _fp8_block_fast_path_applies(w.cpu(), FP8_SCHEME)
    assert not _fp8_block_fast_path_applies(w.to(torch.float8_e4m3fn), FP8_SCHEME)  # already low precision
    assert not _fp8_block_fast_path_applies(w[0], FP8_SCHEME)  # 1-D
    mx = _scheme_for_preset("MXFP8", ["Linear"])
    assert not _fp8_block_fast_path_applies(w, mx)


@requires_fast_path
@pytest.mark.parametrize("proj,shape", [("gate_up_proj", (4, 2 * 768, 2048)), ("down_proj", (4, 2048, 768))])
def test_packed_experts_batched_matches_per_expert(proj, shape):
    name = f"model.layers.0.mlp.experts.{proj}"
    packed = _random_weight(shape[1:], seed=1).unsqueeze(0).repeat(shape[0], 1, 1)
    for e in range(shape[0]):  # make experts distinct
        packed[e] = packed[e] * (0.5 + e)
    assert _packed_experts_fp8_block_applies(name, packed, FP8_SCHEME)
    batched = dict(_compress_packed_experts_fp8_block(name, packed, FP8_SCHEME))
    for weight_name, weight in _unpack_experts(name, packed):
        eager = _compress_weight_eager(weight, FP8_SCHEME)
        prefix = weight_name[: -len("weight")]
        fast = {"weight": batched[prefix + "weight"], "weight_scale": batched[prefix + "weight_scale"]}
        _assert_same_compressed(fast, eager)
    assert len(batched) == shape[0] * (4 if proj == "gate_up_proj" else 2)


@requires_fast_path
def test_packed_experts_fallback_when_blocks_would_straddle_experts():
    # 200 rows per expert is not a multiple of 128: must fall back to the per-expert path
    packed = _random_weight((200, 256)).unsqueeze(0).repeat(2, 1, 1)
    assert not _packed_experts_fp8_block_applies("m.experts.down_proj", packed, FP8_SCHEME)
