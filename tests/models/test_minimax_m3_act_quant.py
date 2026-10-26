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
"""MiniMax M3 routed-expert activation QAT vs. the deployed vLLM / FlashInfer quantizers."""

import zlib

import pytest
import torch

from verl.models.transformers.minimax_m3.act_quant import (
    NVFP4ActQuantLinear,
    e2m1_round,
    fc1_quantize,
    fc1_quantize_kernel,
    fc2_quantize,
    fc2_quantize_kernel,
)

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _decode(packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Packed E2M1 (low nibble first) and E4M3 linear block scales -> code * sf (FP32)."""
    rows = packed.shape[0]
    lut = _E2M1.to(packed.device)
    codes = torch.stack((lut[(packed & 0xF).long()], lut[(packed >> 4).long()]), -1).view(rows, -1)
    cols = codes.shape[1]
    sf = sf.reshape(rows, -1)[:, : cols // 16].float()
    return (codes.view(rows, cols // 16, 16) * sf[..., None]).view(rows, cols)


def _inputs(kind: str, rows: int, cols: int, device="cpu") -> torch.Tensor:
    g = torch.Generator().manual_seed(zlib.crc32(kind.encode()))
    x = torch.randn(rows, cols, generator=g)
    if kind == "outliers":
        x[:, ::97] *= 300.0
    elif kind == "tiny":
        x *= 1e-6
    elif kind == "zero-row":
        x[1] = 0.0
    elif kind == "mixed-scale-blocks":
        x[:, : cols // 2] *= 1e-4  # far below the row max: E4M3 block scales underflow
    return x.to(device=device, dtype=torch.bfloat16)


KINDS = ["normal", "outliers", "tiny", "zero-row", "mixed-scale-blocks"]


def _served_per_token_quantize(x: torch.Tensor):
    """The rollout's per-token NVFP4 quantizer: vLLM ``quantize_nvfp4_per_token_input`` (FlashInfer)."""
    try:
        from vllm.model_executor.layers.quantization.utils.flashinfer_utils import quantize_nvfp4_per_token_input
    except ImportError:
        flashinfer = pytest.importorskip("flashinfer")
        return flashinfer.nvfp4_quantize(
            x, 1.0 / 2688.0, sfLayout=flashinfer.SfLayout.layout_linear, per_token_activation=True
        )
    return quantize_nvfp4_per_token_input(x)


class TestE2M1Round:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.25, 0.0),
            (0.2501, 0.5),
            (0.75, 1.0),
            (1.25, 1.0),
            (1.2501, 1.5),
            (1.75, 2.0),
            (2.5, 2.0),
            (2.5001, 3.0),
            (3.5, 4.0),
            (5.0, 4.0),
            (5.0001, 6.0),
            (100.0, 6.0),
            (-0.3, -0.5),
            (-3.5, -4.0),
        ],
        ids=[
            "tie-0",
            "above-0.25",
            "tie-up-1",
            "tie-down-1",
            "above-1.25",
            "tie-2",
            "tie-down-2",
            "above-2.5",
            "tie-4",
            "tie-down-4",
            "above-5",
            "saturate",
            "negative",
            "negative-tie",
        ],
    )
    def test_round_nearest_even_saturating(self, value, expected):
        assert e2m1_round(torch.tensor([value])).item() == expected


class TestFC1VsFlashInfer:
    @cuda
    @pytest.mark.parametrize("kind", KINDS, ids=KINDS)
    def test_vs_per_token_kernel(self, kind):
        x = _inputs(kind, 64, 6144, "cuda")
        packed, sf, token_scale = _served_per_token_quantize(x)
        ref = _decode(packed.view(torch.uint8).reshape(x.shape[0], -1), sf.view(torch.float8_e4m3fn))
        token_scale = token_scale.float().reshape(-1)
        unit_k, scale_k = fc1_quantize_kernel(x)
        assert torch.equal(unit_k.float(), ref) and torch.equal(scale_k, token_scale)
        unit, scale = fc1_quantize(x)
        assert torch.equal(scale, token_scale)
        # The torch path uses IEEE division where the kernel uses rcp.approx:
        # rare last-ulp differences flip a code; the default "kernel" backend avoids them.
        mismatch = (unit.float() != ref).float().mean().item()
        print(f"fc1 torch-vs-kernel mismatch fraction [{kind}] = {mismatch:.3e}")
        assert mismatch < 1e-3


class TestFC2VsFlashInfer:
    @cuda
    @pytest.mark.parametrize("kind", KINDS, ids=KINDS)
    @pytest.mark.parametrize("shards", [1, 4], ids=["tp1", "tp4"])
    def test_vs_flashinfer_per_token_kernel(self, kind, shards):
        a = _inputs(kind, 64, 3072, "cuda")
        units, scales = fc2_quantize(a, shards)
        for part, unit, scale in zip(a.chunk(shards, dim=-1), units, scales, strict=True):
            packed, sf, token_scale = _served_per_token_quantize(part.contiguous())
            ref = _decode(packed.view(torch.uint8).reshape(part.shape[0], -1), sf.view(torch.float8_e4m3fn))
            assert torch.equal(scale, token_scale.float().reshape(-1)), (scale[:4], token_scale.reshape(-1)[:4])
            mismatch = (unit.float() != ref).float().mean().item()
            print(f"fc2 torch-vs-kernel mismatch fraction [{kind}, {shards}] = {mismatch:.3e}")
            assert mismatch < 1e-3
        units_k, scales_k = fc2_quantize_kernel(a, shards)
        for part, unit, scale in zip(a.chunk(shards, dim=-1), units_k, scales_k, strict=True):
            packed, sf, token_scale = _served_per_token_quantize(part.contiguous())
            ref = _decode(packed.view(torch.uint8).reshape(part.shape[0], -1), sf.view(torch.float8_e4m3fn))
            assert torch.equal(unit.float(), ref) and torch.equal(scale, token_scale.float().reshape(-1))

    def test_shard_scales_are_independent(self):
        a = torch.ones(1, 64, dtype=torch.bfloat16)
        a[0, :16] = 100.0
        units, scales = fc2_quantize(a, 4)
        assert [round(s.item() * 2688, 3) for s in scales] == [100.0, 1.0, 1.0, 1.0]
        for u, s, p in zip(units, scales, a.chunk(4, -1), strict=True):
            torch.testing.assert_close(u.float() * s[:, None], p.float(), rtol=1e-6, atol=0)


class TestActQuantLinear:
    @pytest.mark.parametrize("mode,shards", [("none", 1), ("fc1", 1), ("fc2", 4)], ids=["none", "fc1", "fc2-tp4"])
    def test_forward_is_scaled_qdq_and_backward_is_ste(self, mode, shards):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(5, 64, generator=g).to(torch.bfloat16).requires_grad_(True)
        w = torch.randn(8, 64, generator=g).to(torch.bfloat16)
        gs = torch.tensor(3.0)
        y = NVFP4ActQuantLinear.apply(x, w, gs, mode, shards, "torch")
        assert y.dtype == torch.bfloat16
        if mode == "none":
            expected = x.detach().float() @ w.float().t()
        elif mode == "fc1":
            unit, scale = fc1_quantize(x.detach())
            expected = (unit.float() @ w.float().t()) * scale[:, None]
        else:
            units, scales = fc2_quantize(x.detach(), shards)
            expected = sum(
                (u.float() @ wp.float().t()) * s[:, None]
                for u, wp, s in zip(units, w.chunk(shards, 1), scales, strict=True)
            )
        torch.testing.assert_close(y.float(), (expected * (1.0 / 3.0)).bfloat16().float(), rtol=2**-8, atol=0)
        # The accumulator is scaled in FP32 and rounded once.
        unrounded = NVFP4ActQuantLinear.apply(x, w, gs, mode, shards, "torch", True)
        assert unrounded.dtype == torch.float32
        assert torch.equal(unrounded.bfloat16(), y)
        grad = torch.randn(5, 8, generator=g).to(torch.bfloat16)
        y.backward(grad)
        # Straight-through: the gradient of x @ W^T / gs, from an FP64 autograd reference.
        x64 = x.detach().double().requires_grad_(True)
        ((x64 @ w.double().t()) / 3.0).backward(grad.double())
        assert x.grad.dtype == torch.bfloat16
        torch.testing.assert_close(x.grad.double(), x64.grad, rtol=2**-8, atol=1e-4)

    def test_fp32_upstream_gradient_is_not_rounded(self):
        # Two output channels with opposite weights: the input gradient is the
        # difference of the upstream gradients, which BF16 rounding would erase.
        x = torch.ones(1, 16, dtype=torch.bfloat16, requires_grad=True)
        w = torch.zeros(2, 16, dtype=torch.bfloat16)
        w[0, 0], w[1, 0] = 1.0, -1.0
        y = NVFP4ActQuantLinear.apply(x, w, torch.tensor(1.0), "none", 1, "torch", True)
        y.backward(torch.tensor([[1.0, 1.0 + 1.0 / 512]]))
        assert x.grad[0, 0].item() == -1.0 / 512
        assert not x.grad[0, 1:].any()

    @pytest.mark.parametrize(
        "mode,backend,device",
        [("fc1", "torch", "cpu"), ("fc2", "torch", "cpu"), ("fc1", "kernel", "cuda"), ("fc2", "kernel", "cuda")],
        ids=["fc1-torch", "fc2-torch", "fc1-kernel", "fc2-kernel"],
    )
    def test_zero_and_empty_rows(self, mode, backend, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        w = torch.ones(4, 32, dtype=torch.bfloat16, device=device)
        gs = torch.tensor(1.0, device=device)
        assert not NVFP4ActQuantLinear.apply(
            torch.zeros(2, 32, dtype=torch.bfloat16, device=device), w, gs, mode, 2, backend
        ).any()
        assert NVFP4ActQuantLinear.apply(
            torch.zeros(0, 32, dtype=torch.bfloat16, device=device), w, gs, mode, 2, backend
        ).shape == (0, 4)


class TestExpertQAT:
    def test_expert_composes_fc1_swiglu_fc2(self):
        from verl.models.transformers.minimax_m3.act_quant import _mm_fp32
        from verl.models.transformers.minimax_m3.modeling_minimax_m3 import MiniMaxM3Args, MiniMaxM3Expert

        args = MiniMaxM3Args(
            vocab_size=8, hidden_size=32, intermediate_size=64, dense_intermediate_size=64,
            shared_intermediate_size=64, num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
            head_dim=8, rms_norm_eps=1e-6, rope_theta=1e4, rotary_dim=4, swiglu_alpha=1.702, swiglu_limit=7.0,
            num_local_experts=1, num_experts_per_tok=1, n_shared_experts=0, routed_scaling_factor=1.0,
            moe_layer_freq=[1], sparse_attention_config={}, expert_activation_qat=True,
            expert_qat_tp_shards=2, expert_qat_backend="torch",
        )  # fmt: skip
        expert = MiniMaxM3Expert(args)
        g = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for lin, gs in ((expert.w1, 2.0), (expert.w2, 4.0), (expert.w3, 3.0)):
                lin.weight.copy_(torch.randn(lin.weight.shape, generator=g).to(torch.bfloat16))
                lin.weight_global_scale.fill_(gs)
        expert.requires_grad_(False)
        x = torch.randn(5, 32, generator=g).to(torch.bfloat16)
        # Explicit precision flow (only the GEMM primitive is shared with the model):
        # FC1 accumulators stay FP32 into the activation, the activation is rounded
        # to BF16 (the FC2 input), the FC2 accumulator is rounded once.
        u1, r1 = fc1_quantize(x)
        gate = _mm_fp32(u1, expert.w1.weight) * r1[:, None] * (1.0 / torch.tensor(2.0))
        up = _mm_fp32(u1, expert.w3.weight) * r1[:, None] * (1.0 / torch.tensor(3.0))
        gate, up = gate.clamp(max=7.0), up.clamp(min=-7.0, max=7.0)
        act = (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).bfloat16()
        units, scales = fc2_quantize(act, 2)
        acc = sum(
            _mm_fp32(u, w) * s[:, None] for u, w, s in zip(units, expert.w2.weight.chunk(2, 1), scales, strict=True)
        )
        out = expert(x)
        assert out.dtype == torch.bfloat16
        assert torch.equal(out, (acc * (1.0 / torch.tensor(4.0))).bfloat16())
