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

import sys
import types
from types import SimpleNamespace

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _install_fake_gdn_dependencies(monkeypatch, conv_backend="fla"):
    """Install fake megatron/fla modules so the GDN patch can run on CPU.

    ``conv_backend`` selects which convolution path the patch will exercise:
    - "fla": expose ``fla.modules.convolution.causal_conv1d`` (preferred path).
    - "ccfn": omit the fla conv and expose ``causal_conv1d.causal_conv1d_fn``
      so the varlen ``seq_idx`` fallback path is taken instead.
    """
    calls = {}

    class FakeGatedDeltaNet:
        pass

    def torch_chunk_gated_delta_rule(query, key, value, **kwargs):
        calls["torch_chunk"] = kwargs
        return value, None

    def chunk_gated_delta_rule(query, key, value, **kwargs):
        calls["chunk"] = kwargs
        return value, None

    def fla_causal_conv1d(x, weight, bias, activation, initial_state, output_final_state, cu_seqlens):
        calls["conv"] = {
            "x_shape": tuple(x.shape),
            "cu_seqlens": cu_seqlens,
            "activation": activation,
            "initial_state": initial_state,
            "output_final_state": output_final_state,
        }
        return x, None

    def causal_conv1d_fn(x, weight, bias, activation, seq_idx):
        calls["ccfn"] = {
            "x_shape": tuple(x.shape),
            "seq_idx": seq_idx,
            "activation": activation,
        }
        return x

    modules = {
        "megatron": types.ModuleType("megatron"),
        "megatron.core": types.ModuleType("megatron.core"),
        "megatron.core.ssm": types.ModuleType("megatron.core.ssm"),
        "megatron.core.ssm.gated_delta_net": types.ModuleType("megatron.core.ssm.gated_delta_net"),
        "megatron.core.utils": types.ModuleType("megatron.core.utils"),
        "fla": types.ModuleType("fla"),
        "fla.modules": types.ModuleType("fla.modules"),
        "fla.modules.convolution": types.ModuleType("fla.modules.convolution"),
        "fla.modules.l2norm": types.ModuleType("fla.modules.l2norm"),
        "fla.ops": types.ModuleType("fla.ops"),
        "fla.ops.gated_delta_rule": types.ModuleType("fla.ops.gated_delta_rule"),
    }

    modules["megatron.core.ssm.gated_delta_net"].GatedDeltaNet = FakeGatedDeltaNet
    modules["megatron.core.ssm.gated_delta_net"].torch_chunk_gated_delta_rule = torch_chunk_gated_delta_rule
    modules["megatron.core.utils"].deprecate_inference_params = lambda inference_context, inference_params: (
        inference_context if inference_context is not None else inference_params
    )
    modules["megatron.core.utils"].nvtx_range_push = lambda suffix: None
    modules["megatron.core.utils"].nvtx_range_pop = lambda suffix: None
    modules["fla.modules.l2norm"].l2norm = lambda x: x
    modules["fla.ops.gated_delta_rule"].chunk_gated_delta_rule = chunk_gated_delta_rule

    if conv_backend == "fla":
        modules["fla.modules.convolution"].causal_conv1d = fla_causal_conv1d
    elif conv_backend == "ccfn":
        # No fla conv: patch falls back to causal_conv1d_fn (seq_idx path).
        modules["causal_conv1d"] = types.ModuleType("causal_conv1d")
        modules["causal_conv1d"].causal_conv1d_fn = causal_conv1d_fn
    else:
        raise ValueError(conv_backend)

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return FakeGatedDeltaNet, calls


class _FakeLinear:
    def __call__(self, hidden_states):
        full_seq_len = hidden_states.shape[0] * 2
        batch = hidden_states.shape[1]
        return torch.arange(full_seq_len * batch * 10, dtype=torch.float32).reshape(full_seq_len, batch, 10), None


class _FakeOutProj:
    def __call__(self, norm_out):
        return norm_out, None


class _FakeConv1d:
    def __init__(self):
        self.weight = torch.ones(6, 1, 3)
        self.bias = torch.zeros(6)

    def __call__(self, qkv):
        return qkv


class _FakeGdnInstance:
    def __init__(self):
        self.sp_size = 2
        self.config = SimpleNamespace(sequence_parallel=True, deterministic_mode=False)
        self.in_proj = _FakeLinear()
        self.out_proj = _FakeOutProj()
        self.conv1d = _FakeConv1d()
        self.activation = "silu"
        self.act_fn = lambda x: x
        self.qk_dim = 2
        self.v_dim = 2
        self.tp_size = 1
        self.num_value_heads = 1
        self.num_key_heads = 1
        self.value_head_dim = 2
        self.key_head_dim = 2
        self.use_qk_l2norm = True
        self.A_log = torch.zeros(1)
        self.dt_bias = torch.zeros(1)

    def _apply_gated_norm(self, core_attn_out, gate):
        assert core_attn_out.shape == gate.shape
        return core_attn_out


def test_gdn_patch_passes_cu_seqlens_to_fla_varlen_paths(monkeypatch):
    if torch is None:
        pytest.skip("torch is not installed")

    # Import before installing fake modules: importing verl.models.mcore runs its
    # package __init__, which needs the real megatron.core (e.g. ModelParallelConfig).
    # The patch's own megatron/fla imports are lazy, so the fakes still apply at call time.
    from verl.models.mcore.patch import apply_patch_megatron_gated_delta_net

    fake_gdn_cls, calls = _install_fake_gdn_dependencies(monkeypatch)

    apply_patch_megatron_gated_delta_net()

    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)
    packed_seq_params = SimpleNamespace(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_q_padded=None,
        seq_idx=None,
    )

    hidden_states = torch.zeros(2, 1, 4)
    out, bias = fake_gdn_cls.forward(
        _FakeGdnInstance(),
        hidden_states,
        attention_mask=None,
        packed_seq_params=packed_seq_params,
    )

    assert bias is None
    assert tuple(out.shape) == (4, 1, 2)
    assert calls["conv"]["cu_seqlens"] is cu_seqlens
    assert calls["conv"]["x_shape"] == (1, 4, 6)
    assert calls["chunk"]["cu_seqlens"] is cu_seqlens


def test_gdn_patch_builds_seq_idx_for_causal_conv1d_fn(monkeypatch):
    if torch is None:
        pytest.skip("torch is not installed")

    from verl.models.mcore.patch import apply_patch_megatron_gated_delta_net

    # No fla conv available -> patch falls back to causal_conv1d_fn (seq_idx path).
    fake_gdn_cls, calls = _install_fake_gdn_dependencies(monkeypatch, conv_backend="ccfn")

    apply_patch_megatron_gated_delta_net()

    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)
    packed_seq_params = SimpleNamespace(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_q_padded=None,
        seq_idx=None,
    )

    hidden_states = torch.zeros(2, 1, 4)  # sp_size=2 -> full seq len 4
    out, bias = fake_gdn_cls.forward(
        _FakeGdnInstance(),
        hidden_states,
        attention_mask=None,
        packed_seq_params=packed_seq_params,
    )

    assert bias is None
    assert tuple(out.shape) == (4, 1, 2)
    # causal_conv1d_fn receives the channel-first (b, d, s) layout.
    assert calls["ccfn"]["x_shape"] == (1, 6, 4)
    # seq_idx is derived from cu_seqlens: two sequences of length 2 -> [[0,0,1,1]].
    seq_idx = calls["ccfn"]["seq_idx"]
    assert seq_idx is not None
    assert seq_idx.dtype == torch.int32
    assert seq_idx.tolist() == [[0, 0, 1, 1]]
    # cu_seqlens still flows into the varlen gated-delta-rule kernel.
    assert calls["chunk"]["cu_seqlens"] is cu_seqlens


def test_gdn_patch_rejects_packed_deterministic_mode(monkeypatch):
    if torch is None:
        pytest.skip("torch is not installed")

    from verl.models.mcore.patch import apply_patch_megatron_gated_delta_net

    fake_gdn_cls, _ = _install_fake_gdn_dependencies(monkeypatch)

    apply_patch_megatron_gated_delta_net()

    instance = _FakeGdnInstance()
    instance.config.deterministic_mode = True
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)
    packed_seq_params = SimpleNamespace(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_q_padded=None,
        seq_idx=None,
    )

    try:
        fake_gdn_cls.forward(instance, torch.zeros(2, 1, 4), attention_mask=None, packed_seq_params=packed_seq_params)
    except NotImplementedError as exc:
        assert "deterministic mode" in str(exc)
    else:
        raise AssertionError("GDN packed deterministic mode should be rejected")
