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
"""CPU regression tests for packed Qwen3.5 linear attention without FLA kernels."""

import functools
import sys
import types

import pytest
import torch
from transformers.integrations.hub_kernels import use_kernel_func_from_hub_with_fallback
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
    causal_conv1d_fn,
    torch_chunk_gated_delta_rule,
)

from verl.models.transformers.qwen3_5 import (
    _call_accepts_kwarg,
    _packed_chunk_gated_delta_rule,
    qwen3_5_decoder_layer_forward,
    qwen3_5_gated_delta_net_forward,
)


def _tiny_qwen35_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"],
    )


def test_call_accepts_kwarg_rejects_var_keyword_only():
    """Only explicitly declared parameters should match, never bare `**kwargs`."""

    def fallback(query, key, value, **kwargs):
        return query, None

    def explicit(query, key, value, cu_seqlens=None, cu_seqlens_cpu=None, cp_context=None):
        return query, None

    assert not _call_accepts_kwarg(fallback, "cu_seqlens")
    assert not _call_accepts_kwarg(fallback, "cp_context")
    assert not _call_accepts_kwarg(torch_chunk_gated_delta_rule, "cu_seqlens")
    assert not _call_accepts_kwarg(torch_chunk_gated_delta_rule, "cp_context")
    assert not _call_accepts_kwarg(causal_conv1d_fn, "seq_idx")

    assert _call_accepts_kwarg(explicit, "cu_seqlens")
    assert _call_accepts_kwarg(explicit, "cu_seqlens_cpu")
    assert _call_accepts_kwarg(explicit, "cp_context")
    assert not _call_accepts_kwarg(explicit, "seq_idx")


def test_call_accepts_kwarg_unwraps_hub_kernel_wrapper(monkeypatch):
    """Inspect the resolved implementation behind transformers hub kernel decorators."""
    fake_pkg = types.ModuleType("_test_fake_fla")
    fake_ops = types.ModuleType("_test_fake_fla.ops")
    fake_gdr = types.ModuleType("_test_fake_fla.ops.gated_delta_rule")

    def fake_fla_rule(q, k, v, g=None, beta=None, cu_seqlens=None, cu_seqlens_cpu=None, cp_context=None):
        return q, None

    fake_pkg.ops = fake_ops
    fake_ops.gated_delta_rule = fake_gdr
    fake_gdr.chunk_gated_delta_rule = fake_fla_rule
    monkeypatch.setitem(sys.modules, "_test_fake_fla", fake_pkg)
    monkeypatch.setitem(sys.modules, "_test_fake_fla.ops", fake_ops)
    monkeypatch.setitem(sys.modules, "_test_fake_fla.ops.gated_delta_rule", fake_gdr)

    @use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "_test_fake_fla")
    def wrapped_with_kernel(query, key, value, **kwargs):
        return query, None

    @functools.wraps(wrapped_with_kernel)
    def double_wrapped(*args, **kwargs):
        return wrapped_with_kernel(*args, **kwargs)

    @use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "_test_missing_fla_pkg")
    def wrapped_fallback(query, key, value, **kwargs):
        return query, None

    assert _call_accepts_kwarg(wrapped_with_kernel, "cu_seqlens")
    assert _call_accepts_kwarg(wrapped_with_kernel, "cu_seqlens_cpu")
    assert _call_accepts_kwarg(wrapped_with_kernel, "cp_context")
    assert _call_accepts_kwarg(double_wrapped, "cu_seqlens")

    assert not _call_accepts_kwarg(wrapped_fallback, "cu_seqlens")
    assert not _call_accepts_kwarg(wrapped_fallback, "cp_context")


def test_packed_chunk_gated_delta_rule_splits_without_fla():
    """Packed calls without FLA must match per-sample forward and backward exactly."""
    torch.manual_seed(42)
    lengths = [5, 7, 4]
    total = sum(lengths)
    cu_seqlens = torch.tensor([0, 5, 12, 16], dtype=torch.long)

    q_raw = torch.randn(1, total, 4, 8)
    k_raw = torch.randn(1, total, 4, 8)
    v_raw = torch.randn(1, total, 4, 8)
    g_raw = -torch.rand(1, total, 4) - 0.1
    beta_raw = torch.rand(1, total, 4)

    q_p = q_raw.clone().requires_grad_(True)
    k_p = k_raw.clone().requires_grad_(True)
    v_p = v_raw.clone().requires_grad_(True)
    g_p = g_raw.clone().requires_grad_(True)
    b_p = beta_raw.clone().requires_grad_(True)

    packed_out, state = _packed_chunk_gated_delta_rule(
        torch_chunk_gated_delta_rule,
        q_p,
        k_p,
        v_p,
        g_p,
        b_p,
        cu_seqlens,
        cu_seqlens,
    )
    assert state is None
    weights = torch.randn_like(packed_out)
    (packed_out * weights).sum().backward()

    ref_outs = []
    ref_grads = [[], [], [], [], []]
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True):
        qi = q_raw[:, start:end].clone().requires_grad_(True)
        ki = k_raw[:, start:end].clone().requires_grad_(True)
        vi = v_raw[:, start:end].clone().requires_grad_(True)
        gi = g_raw[:, start:end].clone().requires_grad_(True)
        bi = beta_raw[:, start:end].clone().requires_grad_(True)
        out_i, _ = torch_chunk_gated_delta_rule(
            qi,
            ki,
            vi,
            g=gi,
            beta=bi,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        ref_outs.append(out_i)
        (out_i * weights[:, start:end]).sum().backward()
        for idx, t in enumerate((qi, ki, vi, gi, bi)):
            ref_grads[idx].append(t.grad)

    ref_out = torch.cat(ref_outs, dim=1)
    torch.testing.assert_close(packed_out, ref_out, atol=1e-6, rtol=1e-6)
    for actual, ref_parts in zip((q_p.grad, k_p.grad, v_p.grad, g_p.grad, b_p.grad), ref_grads, strict=True):
        torch.testing.assert_close(actual, torch.cat(ref_parts, dim=1), atol=1e-6, rtol=1e-6)


def test_packed_chunk_gated_delta_rule_uses_explicit_varlen_kernel():
    """Callables with explicit cu_seqlens must receive the packed call directly without splitting."""
    calls = []

    def varlen_kernel(
        query,
        key,
        value,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
    ):
        calls.append((query.shape[1], cu_seqlens, cu_seqlens_cpu))
        return query, None

    q = torch.randn(1, 9, 4, 8)
    k = torch.randn(1, 9, 4, 8)
    v = torch.randn(1, 9, 4, 8)
    g = torch.randn(1, 9, 4)
    beta = torch.rand(1, 9, 4)
    cu = torch.tensor([0, 4, 9], dtype=torch.long)

    out, _ = _packed_chunk_gated_delta_rule(varlen_kernel, q, k, v, g, beta, cu, cu)
    assert len(calls) == 1
    assert calls[0][0] == 9
    assert calls[0][1] is cu
    assert calls[0][2] is cu
    assert torch.equal(out, q)


def test_packed_chunk_gated_delta_rule_rejects_cp_context_without_support():
    """Context-parallel packed calls without cp_context support must fail loudly."""
    q = torch.randn(1, 6, 4, 8)
    cu = torch.tensor([0, 3, 6], dtype=torch.long)

    with pytest.raises(NotImplementedError, match="cp_context"):
        _packed_chunk_gated_delta_rule(
            torch_chunk_gated_delta_rule,
            q,
            q,
            q,
            torch.zeros(1, 6, 4),
            torch.ones(1, 6, 4),
            cu,
            cu,
            cp_context=object(),
        )


def test_qwen3_5_gated_delta_net_packed_matches_unpacked_on_cpu(monkeypatch):
    """End-to-end packed Qwen3_5DecoderLayer forward and backward match separate sample runs."""
    monkeypatch.setattr(Qwen3_5DecoderLayer, "forward", qwen3_5_decoder_layer_forward)
    monkeypatch.setattr(Qwen3_5GatedDeltaNet, "forward", qwen3_5_gated_delta_net_forward)

    torch.manual_seed(7)
    config = _tiny_qwen35_config()
    layer = Qwen3_5DecoderLayer(config, 0)
    layer.eval()

    lengths = [5, 7, 4]
    total = sum(lengths)
    cu_seqlens = torch.tensor([0, 5, 12, 16], dtype=torch.long)
    cu_seqlens_cpu = cu_seqlens.clone()
    pos_emb = (torch.empty(0), torch.empty(0))

    raw_hidden = torch.randn(1, total, config.hidden_size)
    packed_hidden = raw_hidden.clone().requires_grad_(True)
    packed_out = layer(
        packed_hidden,
        position_embeddings=pos_emb,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    loss_weights = torch.randn_like(packed_out)
    (packed_out * loss_weights).sum().backward()
    packed_input_grad = packed_hidden.grad.detach().clone()
    packed_param_grads = {name: p.grad.detach().clone() for name, p in layer.named_parameters() if p.grad is not None}

    layer.zero_grad(set_to_none=True)
    ref_outs = []
    ref_input_grads = []
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True):
        sample = raw_hidden[:, start:end].clone().requires_grad_(True)
        out_i = layer(sample, position_embeddings=pos_emb)
        ref_outs.append(out_i)
        (out_i * loss_weights[:, start:end]).sum().backward()
        ref_input_grads.append(sample.grad.detach())

    ref_out = torch.cat(ref_outs, dim=1)
    ref_input_grad = torch.cat(ref_input_grads, dim=1)

    torch.testing.assert_close(packed_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(packed_input_grad, ref_input_grad, atol=1e-5, rtol=1e-5)
    for name, param in layer.named_parameters():
        if param.grad is not None:
            torch.testing.assert_close(packed_param_grads[name], param.grad, atol=1e-5, rtol=1e-5)
