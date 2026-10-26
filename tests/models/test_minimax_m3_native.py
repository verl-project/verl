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
"""Key logic of the verl-native MiniMax M3 model: dequantization, block
selection, routing, activation, segment parsing and the adapter-name contract."""

import math

import pytest
import torch

from verl.models.transformers.minimax_m3.act_quant import _mm_fp32
from verl.models.transformers.minimax_m3.checkpoint import dequant_mxfp8, dequant_nvfp4_partial
from verl.models.transformers.minimax_m3.modeling_minimax_m3 import (
    MiniMaxM3Args,
    MiniMaxM3Attention,
    MiniMaxM3Expert,
    MiniMaxM3ExpertLinear,
    MiniMaxM3ForCausalLM,
    MiniMaxM3MoE,
    _linear,
    _read_text_config,
    _rounded,
    _segments_from_inputs,
    mxfp8_qdq,
    qk_norm_rope,
    rope_cos_sin,
    select_blocks,
    swigluoai,
)


def _tiny_args(**overrides) -> MiniMaxM3Args:
    values = dict(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=4,
        dense_intermediate_size=8,
        shared_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rotary_dim=4,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        num_local_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=0,
        routed_scaling_factor=2.0,
        moe_layer_freq=[1],
        sparse_attention_config={},
    )
    values.update(overrides)
    return MiniMaxM3Args(**values)


class TestDequantMXFP8:
    @pytest.mark.parametrize(
        "value,e8m0,expected",
        [
            (1.5, 127, 1.5),
            (-3.0, 130, -24.0),
            (448.0, 120, 3.5),
            (0.015625, 100, 0.015625 * 2.0**-27),
        ],
        ids=["unit-scale", "scale-2^3", "fp8-max-scale-2^-7", "small-scale-2^-27"],
    )
    def test_value_times_power_of_two(self, value, e8m0, expected):
        weight = torch.full((2, 32), value).to(torch.float8_e4m3fn)
        scale = torch.full((2, 1), e8m0, dtype=torch.uint8)
        out = dequant_mxfp8(weight, scale)
        assert out.dtype == torch.bfloat16
        assert torch.equal(out.float(), torch.full((2, 32), expected))

    def test_scale_applies_per_32_columns(self):
        weight = torch.ones(1, 64).to(torch.float8_e4m3fn)
        scale = torch.tensor([[127, 128]], dtype=torch.uint8)
        out = dequant_mxfp8(weight, scale).float()
        assert torch.equal(out[0, :32], torch.ones(32)) and torch.equal(out[0, 32:], torch.full((32,), 2.0))

    def test_nan_scale_rejected(self):
        with pytest.raises(ValueError, match="NaN"):
            dequant_mxfp8(torch.ones(1, 32).to(torch.float8_e4m3fn), torch.full((1, 1), 255, dtype=torch.uint8))


class TestDequantNVFP4:
    @pytest.mark.parametrize(
        "byte,scale,expected",
        [
            (0x21, 1.0, (0.5, 1.0)),
            (0x7F, 1.0, (-6.0, 6.0)),
            (0x8A, 2.0, (-2.0, -0.0)),
            (0x35, 0.5, (1.5, 0.75)),
        ],
        ids=["low-nibble-first", "max-magnitudes", "negative-and-neg-zero", "fractional-block-scale"],
    )
    def test_e2m1_times_block_scale(self, byte, scale, expected):
        packed = torch.full((1, 8), byte, dtype=torch.uint8)  # 16 values = one block
        block_scale = torch.tensor([[scale]]).to(torch.float8_e4m3fn)
        out = dequant_nvfp4_partial(packed, block_scale).float()
        assert torch.equal(out[0], torch.tensor(expected * 8))

    def test_global_scale_divides_gemm_output(self):
        layer = MiniMaxM3ExpertLinear(2, 1).requires_grad_(False)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[6.0, 1.5]], dtype=torch.bfloat16))
            layer.weight_global_scale.fill_(3.0)
        x = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
        assert torch.equal(layer(x).float(), torch.tensor([[3.0]]))  # (6 + 3) / 3


def _brute_force_selection(q, k, block, topk, local, init):
    """Per-(head, token) selection straight from the rule, with Python loops."""
    seq, heads, _ = q.shape
    nb = math.ceil(seq / block)
    out = torch.zeros(heads, seq, nb, dtype=torch.bool)
    for h in range(heads):
        for t in range(seq):
            valid = list(range(t // block + 1))
            if nb <= topk:
                chosen = set(valid)
            else:
                forced = {b for b in valid if b >= t // block + 1 - local} | {b for b in valid if b < init}
                rest = [b for b in valid if b not in forced]
                score = {
                    b: max(float(q[t, h] @ k[s]) for s in range(b * block, min((b + 1) * block, t + 1))) for b in rest
                }
                chosen = forced | set(sorted(rest, key=lambda b: -score[b])[: max(0, topk - len(forced))])
            for b in chosen:
                out[h, t, b] = True
    return out


class TestSelectBlocks:
    @pytest.mark.parametrize(
        "seq,block,topk,local,init,seed",
        [
            (6, 2, 3, 1, 0, 0),
            (7, 2, 4, 1, 0, 1),
            (13, 2, 3, 1, 0, 2),
            (16, 4, 2, 1, 0, 3),
            (17, 2, 4, 2, 1, 4),
            (31, 3, 4, 1, 0, 5),
        ],
        ids=[
            "dense-equal-topk",
            "dense-below-topk",
            "sparse-odd-tail",
            "sparse-topk2",
            "local2-init1",
            "sparse-block3",
        ],
    )
    def test_matches_rule(self, seq, block, topk, local, init, seed):
        g = torch.Generator().manual_seed(seed)
        q = torch.randn(seq, 2, 4, generator=g)
        k = torch.randn(seq, 4, generator=g)
        got = select_blocks(q, k, block, topk, local, init)
        assert torch.equal(got, _brute_force_selection(q, k, block, topk, local, init))

    def test_forced_local_block_beats_scores(self):
        seq, block = 12, 2
        q = torch.ones(seq, 1, 1)
        k = torch.full((seq, 1), 5.0)
        k[10:] = -100.0  # current block of t = 10, 11 scores lowest
        sel = select_blocks(q, k, block, topk=2, local_blocks=1, init_blocks=0)
        assert sel[0, 11, 5] and sel[0, 10, 5]
        assert int(sel[0, 11].sum()) == 2

    def test_future_blocks_never_selected(self):
        seq, block = 20, 2
        g = torch.Generator().manual_seed(0)
        sel = select_blocks(torch.randn(seq, 1, 3, generator=g), torch.randn(seq, 3, generator=g), block, 3, 1, 0)
        blocks = torch.arange(sel.shape[-1])
        future = blocks[None, :] > (torch.arange(seq) // block)[:, None]
        assert not (sel[0] & future).any()

    def test_score_ignores_keys_after_query_in_current_block(self):
        # t = 4 sits at the start of block 2; key 5 (same block, future) must not lift block 2
        # above block 0 for t = 4 -- only the forced local rule may select it.
        q = torch.ones(6, 1, 1)
        k = torch.tensor([[3.0], [0.0], [1.0], [1.0], [-9.0], [99.0]])
        sel = select_blocks(q, k, block_size=2, topk=2, local_blocks=0, init_blocks=0)
        # t = 5 sees key 5 -> block 2 wins; t = 4 does not -> blocks 0 and 1 win.
        assert sel[0, 5].tolist() == [True, False, True]
        assert sel[0, 4].tolist() == [True, True, False]


class TestRouter:
    @pytest.mark.parametrize(
        "bias,expected_ids",
        [
            ([0.0, 0.0, 0.0, 0.0], [3, 2]),
            ([0.0, 5.0, 0.0, 0.0], [1, 3]),
        ],
        ids=["no-bias", "bias-flips-selection"],
    )
    def test_select_biased_weight_unbiased(self, bias, expected_ids):
        moe = MiniMaxM3MoE(_tiny_args())
        with torch.no_grad():
            moe.gate.weight.copy_(torch.eye(4, 8))
            moe.e_score_correction_bias.copy_(torch.tensor(bias))
        x = torch.zeros(1, 8, dtype=torch.bfloat16)
        x[0, :4] = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        weights, ids = moe.route(x)
        assert ids[0].tolist() == expected_ids
        scores = torch.sigmoid(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
        sel = scores[expected_ids]
        assert weights.dtype == torch.float32
        torch.testing.assert_close(weights[0], sel / sel.sum() * 2.0, rtol=0, atol=1e-7)


class TestSwigluOAI:
    @pytest.mark.parametrize(
        "gate,up,expected_gate,expected_up",
        [
            (1.0, 2.0, 1.0, 2.0),
            (10.0, 0.0, 7.0, 0.0),
            (-10.0, 0.0, -10.0, 0.0),
            (0.5, 10.0, 0.5, 7.0),
            (0.5, -10.0, 0.5, -7.0),
        ],
        ids=["in-range", "gate-clamped-above", "gate-not-clamped-below", "up-clamped-above", "up-clamped-below"],
    )
    def test_clamp_then_glu(self, gate, up, expected_gate, expected_up):
        out = swigluoai(torch.tensor([gate]), torch.tensor([up]), alpha=1.702, beta=1.0, limit=7.0)
        g = torch.tensor([expected_gate])
        torch.testing.assert_close(out, g * torch.sigmoid(1.702 * g) * (expected_up + 1.0))


class TestQKNormRope:
    def test_gemma_norm_then_partial_neox_rotation(self):
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])  # [T=1, H=1, D=6], rotary_dim=4
        w = torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, -0.5])
        angle = torch.tensor([[0.3, 0.7]])
        out = qk_norm_rope(x, w, 1e-6, angle.cos(), angle.sin(), rotary_dim=4)
        y = x[0, 0] / torch.sqrt((x[0, 0] ** 2).mean() + 1e-6) * (1 + w)
        c, s = angle[0].cos(), angle[0].sin()
        expected = torch.stack(
            [
                y[0] * c[0] - y[2] * s[0],
                y[1] * c[1] - y[3] * s[1],
                y[2] * c[0] + y[0] * s[0],
                y[3] * c[1] + y[1] * s[1],
                y[4],
                y[5],
            ]
        )
        torch.testing.assert_close(out[0, 0], expected)


class TestSegments:
    @pytest.mark.parametrize(
        "batch,seq,mask,cu,pos,expected",
        [
            (1, 10, None, [0, 4, 10], None, [[(0, 4), (4, 10)]]),
            (1, 12, None, [0, 4, 10, 12], None, [[(0, 4), (4, 10), (10, 12)]]),
            (2, 4, [[1, 1, 1, 0], [1, 1, 1, 1]], None, None, [[(0, 3)], [(0, 4)]]),
            (2, 3, [[0, 0, 0], [1, 0, 0]], None, None, [[], [(0, 1)]]),
            (1, 7, None, None, [[0, 1, 2, 0, 1, 0, 1]], [[(0, 3), (3, 5), (5, 7)]]),
            (2, 5, None, None, None, [[(0, 5)], [(0, 5)]]),
        ],
        ids=["cu-seqlens", "cu-seqlens-pad-tail", "right-padded-mask", "empty-row", "position-resets", "no-info"],
    )
    def test_parse(self, batch, seq, mask, cu, pos, expected):
        mask = None if mask is None else torch.tensor(mask)
        cu = None if cu is None else torch.tensor(cu)
        pos = None if pos is None else torch.tensor(pos)
        assert _segments_from_inputs(batch, seq, mask, cu, pos) == expected

    @pytest.mark.parametrize(
        "batch,seq,mask,cu",
        [
            (1, 4, [[0, 1, 1, 1]], None),
            (2, 4, None, [0, 2, 4]),
            (1, 4, None, [0, 3, 2, 4]),
        ],
        ids=["left-padded-mask", "cu-seqlens-batch2", "cu-seqlens-decreasing"],
    )
    def test_rejects(self, batch, seq, mask, cu):
        with pytest.raises(ValueError):
            _segments_from_inputs(
                batch, seq, None if mask is None else torch.tensor(mask), None if cu is None else torch.tensor(cu)
            )


_P = "base_model.model.language_model.model.layers.3."


class TestLoraExportContract:
    @pytest.mark.parametrize(
        "name",
        [
            _P + "self_attn.q_proj.lora_A.weight",
            _P + "self_attn.o_proj.lora_B.weight",
            "base_model.model.language_model.model.layers.0.mlp.down_proj.lora_A.weight",
            _P + "block_sparse_moe.shared_experts.gate_proj.lora_B.weight",
        ],
        ids=["q", "o", "dense-down", "shared-gate"],
    )
    def test_accepts(self, name):
        pair = name.replace(".lora_A.", ".lora_B.") if ".lora_A." in name else name.replace(".lora_B.", ".lora_A.")
        MiniMaxM3ForCausalLM.verl_validate_lora_export([name, pair], None)

    @pytest.mark.parametrize(
        "name",
        [
            _P + "self_attn.index_q_proj.lora_A.weight",
            "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight",
            _P + "self_attn.q_proj.lora_A.default.weight",
            _P + "mlp.experts.0.gate_proj.lora_A.weight",
            _P + "block_sparse_moe.gate.lora_A.weight",
            _P + "self_attn._checkpoint_wrapped_module.q_proj.lora_A.weight",
            "base_model.model.language_model.model.layers.03.self_attn.q_proj.lora_A.weight",
            _P + "block_sparse_moe.experts.00.w1.lora_A.weight",
        ],
        ids=[
            "index-proj",
            "missing-language_model",
            "peft-adapter-name",
            "hf-expert-naming",
            "router",
            "wrapper-fragment",
            "layer-leading-zero",
            "expert-leading-zero",
        ],
    )
    def test_rejects(self, name):
        pair = name.replace(".lora_A.", ".lora_B.")
        with pytest.raises(ValueError, match="schema"):
            MiniMaxM3ForCausalLM.verl_validate_lora_export([name, pair], None)

    @pytest.mark.parametrize(
        "names,match",
        [
            ([_P + "self_attn.q_proj.lora_A.weight"], "counterpart"),
            ([_P + "self_attn.q_proj.lora_B.weight"], "counterpart"),
            ([_P + "self_attn.q_proj.lora_A.weight"] * 2 + [_P + "self_attn.q_proj.lora_B.weight"], "Duplicate"),
        ],
        ids=["A-without-B", "B-without-A", "duplicate-name"],
    )
    def test_rejects_incomplete_modules(self, names, match):
        with pytest.raises(ValueError, match=match):
            MiniMaxM3ForCausalLM.verl_validate_lora_export(names, None)

    @staticmethod
    def _expert_names(layer: int, expert_ids) -> list[str]:
        prefix = f"base_model.model.language_model.model.layers.{layer}."
        return [
            prefix + f"block_sparse_moe.experts.{e}.{w}.lora_{ab}.weight"
            for e in expert_ids
            for w in ("w1", "w2", "w3")
            for ab in "AB"
        ]

    def test_accepts_full_expert_set(self):
        args = _tiny_args(num_hidden_layers=4, moe_layer_freq=[0, 1, 1, 1])
        MiniMaxM3ForCausalLM.verl_validate_lora_export(self._expert_names(3, range(4)), args)

    @pytest.mark.parametrize(
        "names",
        [
            _expert_names(3, range(4))[:-2],
            _expert_names(3, [1, 2, 3, 4]),
            _expert_names(4, range(4)),
            _expert_names(0, range(4)),
            ["base_model.model.language_model.model.layers.3.mlp.down_proj.lora_A.weight"],
            ["base_model.model.language_model.model.layers.0.block_sparse_moe.shared_experts.up_proj.lora_A.weight"],
        ],
        ids=[
            "missing-one-projection",
            "same-count-wrong-expert-ids",
            "layer-out-of-range",
            "experts-in-dense-layer",
            "dense-mlp-in-moe-layer",
            "shared-expert-in-dense-layer",
        ],
    )
    def test_rejects_with_layout(self, names):
        args = _tiny_args(num_hidden_layers=4, moe_layer_freq=[0, 1, 1, 1])
        names = sorted(set(names) | {n.replace(".lora_A.", ".lora_B.") for n in names})
        with pytest.raises(ValueError, match="schema|missing or extra"):
            MiniMaxM3ForCausalLM.verl_validate_lora_export(names, args)


def _randomize(module: torch.nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for tensor in list(module.parameters()) + list(module.buffers()):
            if tensor.dim() == 0:
                tensor.fill_(2.0)
            else:
                tensor.copy_(torch.randn(tensor.shape, generator=g).to(tensor.dtype))


class TestExpertPrecision:
    def test_fc1_accumulator_feeds_activation_unrounded(self):
        args = _tiny_args(hidden_size=16, intermediate_size=32)
        expert = MiniMaxM3Expert(args).requires_grad_(False)
        _randomize(expert, 0)
        x = torch.randn(64, 16, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
        # Only the GEMM primitive is shared with the model; the rounding points are explicit.
        gate = _mm_fp32(x, expert.w1.weight) / 2.0
        up = _mm_fp32(x, expert.w3.weight) / 2.0
        gate, up = gate.clamp(max=7.0), up.clamp(min=-7.0, max=7.0)
        act = (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).bfloat16()
        expected = (_mm_fp32(act, expert.w2.weight) / 2.0).bfloat16()
        out = expert(x)
        assert torch.equal(out, expected)
        # Rounding the FC1 outputs to BF16 first (the old behaviour) is a different function.
        rounded = (gate.bfloat16().float() * torch.sigmoid(1.702 * gate.bfloat16().float())) * (
            up.bfloat16().float() + 1.0
        )
        assert not torch.equal(out, (_mm_fp32(rounded.bfloat16(), expert.w2.weight) / 2.0).bfloat16())

    def test_fc2_adapter_delta_is_added_before_the_single_rounding(self):
        args = _tiny_args(hidden_size=16, intermediate_size=32)
        expert = MiniMaxM3Expert(args).requires_grad_(False)
        _randomize(expert, 4)
        x = torch.randn(64, 16, generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
        base = expert.w2
        hidden = swigluoai(expert.w1(x), expert.w3(x), *expert.act, out_dtype=x.dtype)
        product = base(hidden)
        delta = product * 2.0**-8

        class WithDelta(torch.nn.Module):
            def forward(self, value):
                return base(value) + delta

        expert.w2 = WithDelta()
        out = expert(x)
        assert out.dtype == torch.bfloat16
        assert torch.equal(out, (product + delta).bfloat16())
        assert not torch.equal(out, product.bfloat16() + delta.bfloat16())

    def test_routed_sum_is_rounded_before_shared_add(self):
        args = _tiny_args(n_shared_experts=1)
        moe = MiniMaxM3MoE(args).requires_grad_(False)
        _randomize(moe, 2)
        x = torch.randn(48, 8, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
        weights, ids = moe.route(x)
        routed = torch.zeros(x.shape, dtype=torch.float32)
        for expert_id, expert in enumerate(moe.experts):  # ascending expert id per token
            hit = ids == expert_id
            tok = hit.any(-1).nonzero().flatten()
            routed[tok] += expert(x[tok]).float() * weights[hit][:, None]
        shared = moe.shared_experts(x)
        assert shared.dtype == torch.bfloat16
        out = moe(x)
        assert torch.equal(out, routed.bfloat16() + shared)
        # Adding the shared output inside the FP32 sum (the old behaviour) differs.
        assert not torch.equal(out, (routed + shared.float()).bfloat16())

    def test_trainable_base_weight_is_rejected(self):
        layer = MiniMaxM3ExpertLinear(2, 1)
        with pytest.raises(RuntimeError, match="frozen"):
            layer(torch.ones(1, 2, dtype=torch.bfloat16))


class TestProjectionPrecision:
    def test_adapter_delta_is_added_before_the_single_rounding(self):
        layer = _linear(16, 16).requires_grad_(False)
        _randomize(layer, 7)
        x = torch.randn(64, 16, generator=torch.Generator().manual_seed(8)).to(torch.bfloat16)
        product = layer(x)
        assert product.dtype == torch.float32
        assert torch.equal(product, torch.nn.functional.linear(x.float(), layer.weight.float()))
        # A delta of 0.4% of each element is about one BF16 ulp or less.
        delta = product * 2.0**-8

        class WithDelta(torch.nn.Module):
            def forward(self, value):
                return layer(value) + delta

        out = _rounded(WithDelta(), x)
        assert out.dtype == torch.bfloat16
        assert torch.equal(out, (product + delta).bfloat16())
        # Adding the delta to the rounded product (the old behaviour) is a different function.
        assert not torch.equal(out, product.bfloat16() + delta.bfloat16())

    def test_frozen_weight_backward_is_the_bf16_product(self):
        layer = _linear(16, 12).requires_grad_(False)
        _randomize(layer, 9)
        x = torch.randn(5, 16, generator=torch.Generator().manual_seed(10)).to(torch.bfloat16).requires_grad_()
        grad = torch.randn(5, 12, generator=torch.Generator().manual_seed(11))
        layer(x).backward(grad)
        assert torch.equal(x.grad, grad.bfloat16() @ layer.weight)


class TestLinearActivationQat:
    def test_qdq_is_e4m3_on_a_power_of_two_block_scale(self):
        x = torch.zeros(1, 64)
        x[0, :4] = torch.tensor([448.0, 1.0, 3.3, -0.07])  # block scale 1: E4M3 grid directly
        x[0, 32:36] = torch.tensor([900.0, 1.0, 5.0, -3.0])  # block scale 4
        out = mxfp8_qdq(x)
        assert out[0, :4].tolist() == [448.0, 1.0, 3.25, -0.0703125]
        assert out[0, 32:36].tolist() == [896.0, 1.0, 5.0, -3.0]

    def test_base_product_reads_the_quantized_input_and_backward_is_straight_through(self):
        layer = _linear(64, 8, act_quant=True).requires_grad_(False)
        _randomize(layer, 12)
        x = torch.randn(5, 64, generator=torch.Generator().manual_seed(13)).to(torch.bfloat16).requires_grad_()
        out = layer(x)
        assert torch.equal(out, torch.nn.functional.linear(mxfp8_qdq(x.detach().float()), layer.weight.float()))
        assert not torch.equal(out, torch.nn.functional.linear(x.detach().float(), layer.weight.float()))
        grad = torch.randn(5, 8, generator=torch.Generator().manual_seed(14))
        out.backward(grad)
        assert torch.equal(x.grad, grad.bfloat16() @ layer.weight)

    def test_rejects_an_input_dim_that_is_not_a_block_multiple(self):
        with pytest.raises(ValueError, match="divisible"):
            _linear(48, 8, act_quant=True)


class TestRouterReplay:
    @staticmethod
    def _moe():
        moe = MiniMaxM3MoE(_tiny_args())
        with torch.no_grad():
            moe.gate.weight.copy_(torch.eye(4, 8))
            moe.e_score_correction_bias.zero_()
        x = torch.zeros(2, 8, dtype=torch.bfloat16)
        x[:, :4] = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        return moe, x

    def test_replays_recorded_rows_and_keeps_native_for_unrecorded(self):
        moe, x = self._moe()
        # Row 0: the rollout chose experts {0, 1}; row 1: all-zero padding (no recorded choice).
        weights, ids = moe.route(x, torch.tensor([[0, 1], [0, 0]]))
        assert ids.tolist() == [[0, 1], [3, 2]]
        scores = torch.sigmoid(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
        # Weights come from this model's own scores over the replayed selection.
        torch.testing.assert_close(weights[0], scores[[0, 1]] / scores[[0, 1]].sum() * 2.0, rtol=0, atol=1e-7)
        torch.testing.assert_close(weights[1], scores[[3, 2]] / scores[[3, 2]].sum() * 2.0, rtol=0, atol=1e-7)

    @pytest.mark.parametrize(
        "replay", [torch.tensor([[0, 1, 2], [0, 1, 2]]), torch.tensor([[0, 4], [0, 1]])], ids=["wrong-top-k", "bad-id"]
    )
    def test_rejects_mismatched_routing(self, replay):
        moe, x = self._moe()
        with pytest.raises(ValueError, match="routed_experts"):
            moe.route(x, replay)

    def test_forward_indexes_by_absolute_layer(self):
        from transformers import PretrainedConfig

        # Layer 0 is dense, layer 1 is MoE: the model must read column 1.
        args = _tiny_args(num_hidden_layers=2, moe_layer_freq=[0, 1], sparse_attention_config=_SPARSE)
        model = MiniMaxM3ForCausalLM(PretrainedConfig(), args=args)
        _randomize(model, 4)
        model.requires_grad_(False)
        ids = torch.arange(6)[None] % 32
        native = model.language_model.model.layers[1].block_sparse_moe
        seen = []
        route = native.route
        native.route = lambda x, replay_ids=None: seen.append(replay_ids) or route(x, replay_ids)
        routed = torch.zeros(1, 6, 2, 2, dtype=torch.long)
        routed[0, :, 0] = torch.tensor([1, 0])
        routed[0, :, 1] = torch.tensor([2, 3])
        replayed = model(ids, routed_experts=routed).logits
        assert seen[0].tolist() == [[2, 3]] * 6
        assert torch.isfinite(replayed).all()
        with pytest.raises(ValueError, match="routed_experts"):
            model(ids, routed_experts=routed[:, :, :1])


class TestSharedOuterExpertLora:
    """Outer factors (gate A, up A, down B) are one tensor per MoE layer; the export stays per expert."""

    @staticmethod
    def _peft_model(shared: bool, seed: int = 11):
        from peft import LoraConfig, get_peft_model
        from transformers import PretrainedConfig

        model = MiniMaxM3ForCausalLM(
            PretrainedConfig(),
            args=_tiny_args(num_hidden_layers=2, moe_layer_freq=[1, 1], sparse_attention_config=_SPARSE),
        )
        _randomize(model, 4)
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["w1", "w2", "w3"]))
        if shared:
            MiniMaxM3ForCausalLM.verl_share_expert_outer_loras(model)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if ".lora_" in name:
                    param.copy_(torch.randn(param.shape, generator=generator) * 0.05)
        return model

    @staticmethod
    def _role(name: str) -> str:
        proj, factor = name.split(".")[-4], name.split(".")[-3]
        return "outer" if (proj, factor) in {("w1", "lora_A"), ("w3", "lora_A"), ("w2", "lora_B")} else "inner"

    def test_parameters_are_shared_but_export_names_are_per_expert(self):
        import re

        plain, shared = self._peft_model(False), self._peft_model(True)
        assert set(shared.state_dict()) == set(plain.state_dict())
        expert = [n for n, _ in shared.named_parameters() if ".experts." in n and ".lora_" in n]
        outer = [n for n in expert if self._role(n) == "outer"]
        inner = [n for n in expert if self._role(n) == "inner"]
        # 2 MoE layers: 3 shared outer factors per layer, 3 inner factors for each of 4 experts.
        assert (len(outer), len(inner)) == (2 * 3, 2 * 3 * 4)
        assert all(".experts.0." in n for n in outer)
        state = shared.state_dict()
        for name in state:
            if ".experts." in name and ".lora_" in name and self._role(name) == "outer":
                first = re.sub(r"\.experts\.\d+\.", ".experts.0.", name)
                assert state[name].data_ptr() == state[first].data_ptr()

    def test_expanded_export_reproduces_the_shared_model(self):
        shared, plain = self._peft_model(True), self._peft_model(False, seed=12)
        plain.load_state_dict(shared.state_dict())
        ids = torch.arange(6).reshape(1, 6) % 32
        assert torch.equal(shared(ids).logits, plain(ids).logits)
        shared.load_state_dict(plain.state_dict())
        assert torch.equal(shared(ids).logits, plain(ids).logits)

    def test_shared_factor_gradient_is_the_sum_over_experts(self):
        shared, plain = self._peft_model(True), self._peft_model(False, seed=12)
        plain.load_state_dict(shared.state_dict())
        ids = torch.arange(6).reshape(1, 6) % 32
        for model in (shared, plain):
            model(ids).logits.float().square().sum().backward()
        grads = dict(plain.named_parameters())
        for name, param in shared.named_parameters():
            if ".experts.0." in name and ".lora_" in name and self._role(name) == "outer":
                summed = sum(grads[name.replace(".experts.0.", f".experts.{i}.")].grad for i in range(4))
                assert param.grad.abs().sum() > 0
                torch.testing.assert_close(param.grad, summed, rtol=1e-4, atol=1e-6)

    def test_requires_an_adapter_on_every_expert_projection(self):
        from peft import LoraConfig, get_peft_model
        from transformers import PretrainedConfig

        model = MiniMaxM3ForCausalLM(PretrainedConfig(), args=_tiny_args(sparse_attention_config=_SPARSE))
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["w1", "w2"]))
        with pytest.raises(ValueError, match="w3"):
            MiniMaxM3ForCausalLM.verl_share_expert_outer_loras(model)


class TestForwardGuards:
    @staticmethod
    def _model():
        from transformers import PretrainedConfig

        model = MiniMaxM3ForCausalLM(PretrainedConfig(), args=_tiny_args(sparse_attention_config=_SPARSE))
        _randomize(model, 4)
        return model.requires_grad_(False)

    def test_labels_rejected(self):
        ids = torch.zeros(1, 4, dtype=torch.long)
        with pytest.raises(ValueError, match="labels"):
            self._model()(ids, labels=ids)

    @pytest.mark.parametrize(
        "positions,ok",
        [([[0, 1, 2, 0, 1, 2]], True), ([[0, 1, 2, 3, 4, 5]], False), ([[1, 2, 3, 0, 1, 2]], False)],
        ids=["restart-per-sequence", "global-arange", "offset-start"],
    )
    def test_positions_must_be_segment_local(self, positions, ok):
        model = self._model()
        ids = torch.zeros(1, 6, dtype=torch.long)
        kwargs = dict(position_ids=torch.tensor(positions), cu_seqlens_cpu=torch.tensor([0, 3, 6]))
        if ok:
            assert model(ids, **kwargs).logits.shape == (1, 6, 32)
        else:
            with pytest.raises(ValueError, match="position_ids"):
                model(ids, **kwargs)


_SPARSE = {
    "sparse_num_index_heads": 2,
    "sparse_index_dim": 8,
    "sparse_block_size": 4,
    "sparse_topk_blocks": 2,
    "sparse_local_block": 1,
    "sparse_init_block": 1,
}


class TestEmptyRankGradients:
    def test_attention_projections_get_zero_gradients_without_segments(self):
        attn = MiniMaxM3Attention(_tiny_args(sparse_attention_config=_SPARSE))
        _randomize(attn, 5)
        x = torch.randn(6, 8, generator=torch.Generator().manual_seed(6)).to(torch.bfloat16)
        cos, sin = rope_cos_sin(torch.zeros(6, dtype=torch.long), 4, 10000.0)
        out = attn(x, cos, sin, segments=[])
        assert not out.float().any()
        out.float().sum().backward()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            grad = getattr(attn, name).weight.grad
            assert grad is not None and not grad.float().any(), name


class TestConfigSource:
    @staticmethod
    def _write(tmp_path, text):
        import json

        (tmp_path / "config.json").write_text(json.dumps({"text_config": text}))

    def test_requires_local_config_json(self, tmp_path):
        from types import SimpleNamespace

        with pytest.raises(FileNotFoundError):
            _read_text_config(SimpleNamespace(_name_or_path=str(tmp_path), text_config={}))

    @pytest.mark.parametrize(
        "live,ok",
        [
            ({"rope_theta": 1e7, "expert_activation_qat": True}, True),
            ({"rope_theta": 10000000, "moe_layer_freq": (0, 1), "swiglu_beta": 1.0}, True),
            ({"rope_theta": 1e4}, False),
            ({"num_hidden_layers": 4}, False),
            ({"swiglu_beta": 2.0}, False),
        ],
        ids=[
            "training-switch-only",
            "normalized-types-and-default",
            "rope-theta-override",
            "layer-count-override",
            "override-of-key-omitted-by-config-json",
        ],
    )
    def test_in_memory_override_of_checkpoint_key(self, tmp_path, live, ok):
        from types import SimpleNamespace

        raw = {"rope_theta": 1e7, "num_hidden_layers": 60, "moe_layer_freq": [0, 1]}
        self._write(tmp_path, raw)
        config = SimpleNamespace(_name_or_path=str(tmp_path), text_config=live)
        if ok:
            assert _read_text_config(config) == raw
        else:
            with pytest.raises(ValueError, match="overrides"):
                _read_text_config(config)
