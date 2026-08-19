# Copyright 2026 Bytedance Ltd. and/or its affiliates.
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

"""CPU tests for the phase-2 MoE-LoRA adapter key split in the automodel engine.

Phase-2 LoRA weight sync sends Automodel's fused 3D ``GroupedExpertsLoRA``
adapter params (``lora_gate_and_up_A/B`` fused w1+w3, ``lora_down_A/B`` = w2) to
vLLM ``add_lora``, which expects per-expert
``<prefix>.{i}.{gate_proj,up_proj,down_proj}.lora_{A,B}.weight`` 2D tensors.
These tests pin that mapping (and the merge path).
"""

import torch
import torch.nn as nn

from verl.workers.engine.automodel.utils import (
    _MoELoRASpec,
    _PackedExpertSpec,
    collect_automodel_lora_param_maps,
    merged_dense_lora_weight,
    merged_packed_expert_base,
    split_moe_lora_adapter,
    split_packed_expert,
)


def _spec(n_inter=2048, gated=True):
    return _MoELoRASpec(prefix="model.layers.1.mlp.experts", moe_inter_dim=n_inter, is_gated=gated)


# ---------------------------------------------------------------------------
# Merge-path fixtures: real GroupedExpertsLoRA / LinearLoRA with initialized
# params (nemo_automodel leaves new params via torch.empty, so we reinit).
# ---------------------------------------------------------------------------


def _moe_cfg(n_experts=2, expert_dim=8, moe_inter_dim=16):
    from nemo_automodel.components.moe.config import MoEConfig

    return MoEConfig(
        n_routed_experts=n_experts,
        n_shared_experts=0,
        n_activated_experts=1,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="sigmoid",
        route_scale=1.0,
        dim=expert_dim,
        inter_dim=moe_inter_dim,
        moe_inter_dim=moe_inter_dim,
        norm_topk_prob=True,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        apply_router_weight_after_down=False,
        activation_alpha=1.702,
        activation_limit=7.0,
        swiglu_limit=0.0,
        softmax_before_topk=False,
        router_weights_fp32=False,
        router_weight_uses_score_correction_bias=False,
        shared_expert_gate=False,
        shared_expert_inter_dim=None,
        shared_expert_activation="swiglu",
        force_e_score_correction_bias=False,
        moe_latent_size=None,
        enable_routing_replay=False,
    )


def _grouped_experts_lora(n_experts=2, expert_dim=8, moe_inter_dim=16, lora_dim=4, alpha=8):
    """A real GroupedExpertsLoRA with initialized (non-NaN) params."""
    from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA
    from nemo_automodel.components.moe.experts import GroupedExperts

    m = GroupedExpertsLoRA(
        GroupedExperts(_moe_cfg(n_experts, expert_dim, moe_inter_dim)), lora_dim=lora_dim, alpha=alpha
    )
    # nemo_automodel allocates params via torch.empty; reinit everything to avoid NaNs.
    for p in m.parameters():
        nn.init.normal_(p, std=0.1)
    return m


def test_split_gate_and_up_a_emits_w1_and_w3_a():
    """lora_gate_and_up_A [n, expert_dim, lora_dim] -> gate_proj.lora_A (w1) and
    up_proj.lora_A (w3), identical (shared input factor)."""
    n_exp, expert_dim, lora_dim = 3, 5120, 32
    t = torch.randn(n_exp, expert_dim, lora_dim)
    out = list(split_moe_lora_adapter(_spec(), "lora_gate_and_up_A", t))

    keys = [k for k, _ in out]
    assert keys == [
        "model.layers.1.mlp.experts.0.gate_proj.lora_A.weight",
        "model.layers.1.mlp.experts.0.up_proj.lora_A.weight",
        "model.layers.1.mlp.experts.1.gate_proj.lora_A.weight",
        "model.layers.1.mlp.experts.1.up_proj.lora_A.weight",
        "model.layers.1.mlp.experts.2.gate_proj.lora_A.weight",
        "model.layers.1.mlp.experts.2.up_proj.lora_A.weight",
    ]
    # w1 A and w3 A are [in, rank] and identical (shared input).
    for i in range(n_exp):
        w1a = dict(out)[f"model.layers.1.mlp.experts.{i}.gate_proj.lora_A.weight"]
        w3a = dict(out)[f"model.layers.1.mlp.experts.{i}.up_proj.lora_A.weight"]
        assert w1a.shape == (expert_dim, lora_dim)
        assert w3a.shape == (expert_dim, lora_dim)
        torch.testing.assert_close(w1a, t[i])
        torch.testing.assert_close(w3a, t[i])
        # w3 A must be a clone, not a view (add_lora may move/keep it).
        assert w3a.data_ptr() != w1a.data_ptr()


def test_split_gate_and_up_b_emits_w1_and_w3_b_halves():
    """lora_gate_and_up_B [n, lora_dim, 2*moe_inter_dim] -> gate_proj.lora_B (w1,
    first half) and up_proj.lora_B (w3, second half), each [rank, moe_inter_dim]."""
    n_exp, lora_dim, moe_inter_dim = 3, 32, 2048
    t = torch.randn(n_exp, lora_dim, 2 * moe_inter_dim)
    out = dict(split_moe_lora_adapter(_spec(n_inter=moe_inter_dim), "lora_gate_and_up_B", t))

    for i in range(n_exp):
        w1b = out[f"model.layers.1.mlp.experts.{i}.gate_proj.lora_B.weight"]
        w3b = out[f"model.layers.1.mlp.experts.{i}.up_proj.lora_B.weight"]
        assert w1b.shape == (lora_dim, moe_inter_dim)
        assert w3b.shape == (lora_dim, moe_inter_dim)
        torch.testing.assert_close(w1b, t[i, :, :moe_inter_dim])
        torch.testing.assert_close(w3b, t[i, :, moe_inter_dim:])


def test_split_down_a_emits_w2_a():
    """lora_down_A [n, moe_inter_dim, lora_dim] -> down_proj.lora_A (w2 A)
    [moe_inter_dim, lora_dim]."""
    n_exp, moe_inter_dim, lora_dim = 2, 2048, 32
    t = torch.randn(n_exp, moe_inter_dim, lora_dim)
    out = dict(split_moe_lora_adapter(_spec(n_inter=moe_inter_dim), "lora_down_A", t))

    for i in range(n_exp):
        w2a = out[f"model.layers.1.mlp.experts.{i}.down_proj.lora_A.weight"]
        assert w2a.shape == (moe_inter_dim, lora_dim)
        torch.testing.assert_close(w2a, t[i])


def test_split_down_b_emits_w2_b():
    """lora_down_B [n, lora_dim, expert_dim] -> down_proj.lora_B (w2 B)
    [lora_dim, expert_dim]."""
    n_exp, lora_dim, expert_dim = 2, 32, 5120
    t = torch.randn(n_exp, lora_dim, expert_dim)
    out = dict(split_moe_lora_adapter(_spec(), "lora_down_B", t))

    for i in range(n_exp):
        w2b = out[f"model.layers.1.mlp.experts.{i}.down_proj.lora_B.weight"]
        assert w2b.shape == (lora_dim, expert_dim)
        torch.testing.assert_close(w2b, t[i])


def test_split_non_gated_skips_w3():
    """For non-gated MoE w3 is unused (pack_moe aliases w3 to w1); up_proj keys
    are skipped."""
    n_exp, expert_dim, lora_dim = 2, 5120, 32
    # gate_and_up_B is [n, lora_dim, moe_inter_dim] (no doubled up_proj_dim) when
    # non-gated; w3 must not be emitted.
    moe_inter_dim = 2048
    a = torch.randn(n_exp, expert_dim, lora_dim)
    b = torch.randn(n_exp, lora_dim, moe_inter_dim)

    out_a = list(split_moe_lora_adapter(_spec(n_inter=moe_inter_dim, gated=False), "lora_gate_and_up_A", a))
    out_b = list(split_moe_lora_adapter(_spec(n_inter=moe_inter_dim, gated=False), "lora_gate_and_up_B", b))

    assert all("up_proj" not in k for k, _ in out_a)
    assert all("up_proj" not in k for k, _ in out_b)
    assert any(k.endswith("gate_proj.lora_A.weight") for k, _ in out_a)
    assert any(k.endswith("gate_proj.lora_B.weight") for k, _ in out_b)


def test_split_unknown_attr_raises():
    t = torch.randn(1, 2, 3)
    try:
        list(split_moe_lora_adapter(_spec(), "lora_unknown", t))
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown adapter attr")


def test_to_vllm_peft_dict_match_all_linear_passes_none():
    """match_all_linear=True -> target_modules=None (vLLM treats None as 'all';
    an empty list means 'nothing')."""
    from verl.workers.engine.automodel.utils import to_vllm_peft_dict

    class _Cfg:
        def to_dict(self):
            return {"dim": 32, "alpha": 64, "match_all_linear": True, "use_dora": False}

    d = to_vllm_peft_dict(_Cfg())
    assert d["target_modules"] is None
    assert d["r"] == 32
    assert d["lora_alpha"] == 64
    assert d["use_dora"] is False


def test_to_vllm_peft_dict_explicit_target_modules_passes_list():
    """Explicit target_modules -> pass as a list (no match_all_linear)."""
    from verl.workers.engine.automodel.utils import to_vllm_peft_dict

    class _Cfg:
        def to_dict(self):
            return {"dim": 8, "alpha": 16, "match_all_linear": False, "use_dora": True}

    d = to_vllm_peft_dict(_Cfg())
    assert d["target_modules"] == []
    assert d["use_dora"] is True


def test_collect_moe_lora_prefixes_finds_grouped_experts_lora():
    """collect_automodel_lora_param_maps maps all four adapter attrs of every
    GroupedExpertsLoRA module to a spec with the right moe_inter_dim/is_gated."""
    from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA

    from verl.workers.engine.automodel.utils import collect_automodel_lora_param_maps

    class _Cfg:
        # GroupedExpertsLoRA._init_adapter reads obj.config.moe_inter_dim and
        # obj.is_gated; we only exercise collect_automodel_lora_param_maps (which
        # reads module.config.moe_inter_dim and module.is_gated), so set both.
        moe_inter_dim = 16

    # Bypass the heavy GroupedExperts.__init__ (allocates real params, needs a
    # full MoEConfig) — the collector only inspects isinstance + module.config /
    # module.is_gated. Still init nn.Module so named_modules works.
    lora_experts = GroupedExpertsLoRA.__new__(GroupedExpertsLoRA)
    torch.nn.Module.__init__(lora_experts)
    lora_experts.config = _Cfg()
    lora_experts.is_gated = True

    class _Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = lora_experts

    _, prefixes, _, moe_modules, _ = collect_automodel_lora_param_maps(_Holder())
    # state_dict keys for the holder are "experts.<attr>"
    expected = {
        "experts.lora_gate_and_up_A",
        "experts.lora_gate_and_up_B",
        "experts.lora_down_A",
        "experts.lora_down_B",
    }
    assert set(prefixes) == expected
    spec = prefixes["experts.lora_gate_and_up_A"]
    assert spec.prefix == "experts"
    assert spec.moe_inter_dim == 16
    assert spec.is_gated is True
    # The merge path needs the module ref to fold adapters.
    assert set(moe_modules) == {"experts"}
    assert moe_modules["experts"] is lora_experts


def test_collect_moe_lora_prefixes_normalizes_checkpoint_wrapper():
    """The ``._checkpoint_wrapped_module.`` segment from activation checkpointing
    is normalized out so prefixes match state_dict() keys."""
    from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA

    from verl.workers.engine.automodel.utils import collect_automodel_lora_param_maps

    class _Cfg:
        moe_inter_dim = 16

    lora_experts = GroupedExpertsLoRA.__new__(GroupedExpertsLoRA)
    torch.nn.Module.__init__(lora_experts)
    lora_experts.config = _Cfg()
    lora_experts.is_gated = True

    class _ACWrapper(torch.nn.Module):
        """Mimics activation-checkpointing wrapping: the real module sits under
        ``_checkpoint_wrapped_module`` nested inside a real parent (as it is in
        practice — e.g. ``model.layers.1._checkpoint_wrapped_module.experts``)."""

        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Module()
            self.layers._checkpoint_wrapped_module = torch.nn.Module()
            self.layers._checkpoint_wrapped_module.experts = lora_experts

    _, prefixes, _, _, _ = collect_automodel_lora_param_maps(_ACWrapper())
    # The ``._checkpoint_wrapped_module.`` segment is normalized out so the prefix
    # matches the state_dict() key the rest of the pipeline emits.
    assert "layers._checkpoint_wrapped_module.experts.lora_gate_and_up_A" not in prefixes
    assert "layers.experts.lora_gate_and_up_A" in prefixes
    assert prefixes["layers.experts.lora_gate_and_up_A"].prefix == "layers.experts"


# ---------------------------------------------------------------------------
# Merge path (model.lora.merge=true): fold adapters into base weights.
# ---------------------------------------------------------------------------


def test_merged_dense_lora_weight():
    """LinearLoRA merge = weight + scale*(lora_B @ lora_A)."""
    from nemo_automodel.components._peft.lora import LinearLoRA

    m = LinearLoRA(nn.Linear(8, 16, bias=False), dim=4, alpha=8)
    for p in (m.weight, m.lora_A.weight, m.lora_B.weight):
        nn.init.normal_(p, std=0.1)
    out = merged_dense_lora_weight(m)
    expected = m.weight + m.scale * (m.lora_B.weight @ m.lora_A.weight)
    torch.testing.assert_close(out, expected)
    assert out.shape == (16, 8)
    # No aliasing of module storage (on-the-fly merge, not in-place).
    assert out.data_ptr() != m.weight.data_ptr()


def test_merged_packed_expert_base_gate_up():
    """gate_and_up_projs merge: per-expert base + scale*(A@B), shape [in, 2*mid]."""
    n, expert_dim, mid, rank = 2, 8, 16, 4
    m = _grouped_experts_lora(n, expert_dim, mid, rank, alpha=8)
    out = merged_packed_expert_base(m, "gate_and_up_projs")
    assert out.shape == m.gate_and_up_projs.shape  # [n, in, 2*mid]
    for i in range(n):
        exp = m.gate_and_up_projs[i] + m.scale * (m.lora_gate_and_up_A[i] @ m.lora_gate_and_up_B[i])
        torch.testing.assert_close(out[i], exp)


def test_merged_packed_expert_base_down():
    """down_projs merge: per-expert base + scale*(A@B), shape [mid, expert_dim]."""
    n, expert_dim, mid, rank = 2, 8, 16, 4
    m = _grouped_experts_lora(n, expert_dim, mid, rank, alpha=8)
    out = merged_packed_expert_base(m, "down_projs")
    assert out.shape == m.down_projs.shape  # [n, mid, expert_dim]
    for i in range(n):
        exp = m.down_projs[i] + m.scale * (m.lora_down_A[i] @ m.lora_down_B[i])
        torch.testing.assert_close(out[i], exp)


def test_merged_packed_expert_base_rejects_unknown_attr():
    m = _grouped_experts_lora()
    try:
        merged_packed_expert_base(m, "nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown attr")


def test_merge_splits_to_per_expert_keys():
    """The merge generator yields per-expert gate/up/down_proj.weight 2D keys
    (split_packed_expert on the merged 3D), each == split(base+delta)."""
    n, expert_dim, mid, rank = 2, 8, 16, 4
    m = _grouped_experts_lora(n, expert_dim, mid, rank, alpha=8)
    spec_gu = _PackedExpertSpec(prefix="experts", packed_attr="gate_and_up_projs", splits=(("gate_proj", "up_proj"),))
    spec_dn = _PackedExpertSpec(prefix="experts", packed_attr="down_projs", splits=(("down_proj",),))
    merged_gu = merged_packed_expert_base(m, "gate_and_up_projs")
    merged_dn = merged_packed_expert_base(m, "down_projs")

    for i in range(n):
        keys_gu = [k for k, _ in split_packed_expert(spec_gu, merged_gu, i)]
        assert keys_gu == ["gate_proj.weight", "up_proj.weight"]
        vals = dict(split_packed_expert(spec_gu, merged_gu, i))
        # split_packed_expert transposes to [out, in]; merged base chunk is [in, out].
        gate_chunk = merged_gu[i].chunk(2, dim=-1)[0].t().contiguous()
        torch.testing.assert_close(vals["gate_proj.weight"], gate_chunk)

    for i in range(n):
        keys_dn = [k for k, _ in split_packed_expert(spec_dn, merged_dn, i)]
        assert keys_dn == ["down_proj.weight"]


def test_merge_yields_peft_config_none():
    """Merge branch returns peft_config=None (rollout has enable_lora disabled)."""
    # The contract is in get_per_tensor_param; here we assert the helper that
    # builds the merged generator is wired (no peft_config returned). A full
    # engine test needs GPU, so we check the generator yields only base keys.
    n, expert_dim, mid, rank = 1, 8, 16, 4
    m = _grouped_experts_lora(n, expert_dim, mid, rank, alpha=8)

    class _Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = m

        def state_dict(self):
            return {f"experts.{k}": v for k, v in m.state_dict().items()}

    holder = _Holder()
    _, _, _, moe_modules, dense_modules = collect_automodel_lora_param_maps(holder)
    packed = {
        "experts.gate_and_up_projs": _PackedExpertSpec("experts", "gate_and_up_projs", (("gate_proj", "up_proj"),)),
        "experts.down_projs": _PackedExpertSpec("experts", "down_projs", (("down_proj",),)),
    }
    params = {k: v for k, v in holder.state_dict().items() if "_extra_state" not in k}

    # Inline the merge generator body (mirrors _merged_lora_param_generator).
    out = []
    for name, param in params.items():
        if "lora_" in name:
            continue
        spec = packed.get(name)
        if spec is not None and moe_modules.get(spec.prefix) is not None:
            merged = merged_packed_expert_base(moe_modules[spec.prefix], spec.packed_attr)
            for eid in range(merged.size(0)):
                for sub_name, sub_tensor in split_packed_expert(spec, merged, eid):
                    out.append(f"{spec.prefix}.{eid}.{sub_name}")
            continue
        out.append(name)
    # All adapter keys dropped; only base + per-expert split keys remain.
    assert not any("lora_" in k for k in out)
    assert "experts.0.gate_proj.weight" in out
    assert "experts.0.down_proj.weight" in out


def test_merge_numerical_equivalence():
    """Folded-merge forward == online-LoRA forward for a 1-expert module
    (gate_up path, no routing): x @ merged_gate_up then act then @ merged_down
    matches x @ base_gate_up + LoRA delta path, within tolerance."""
    n, expert_dim, mid, rank = 1, 8, 16, 4
    m = _grouped_experts_lora(n, expert_dim, mid, rank, alpha=8)
    m.eval()
    merged_gu = merged_packed_expert_base(m, "gate_and_up_projs")[0]  # [in, 2*mid]
    merged_dn = merged_packed_expert_base(m, "down_projs")[0]  # [mid, expert_dim]

    x = torch.randn(3, expert_dim, dtype=m.gate_and_up_projs.dtype)
    # Merged (folded) path.
    gu = x @ merged_gu  # [3, 2*mid]
    gate, up = gu.chunk(2, dim=-1)
    act = torch.nn.functional.silu(gate) * up  # swiglu
    y_merged = act @ merged_dn  # [3, expert_dim]

    # Online-LoRA path (mirrors GroupedExpertsLoRA._forward_loop math).
    gu_online = x @ m.gate_and_up_projs[0] + (x @ m.lora_gate_and_up_A[0] @ m.lora_gate_and_up_B[0]) * m.scale
    gate_o, up_o = gu_online.chunk(2, dim=-1)
    act_o = torch.nn.functional.silu(gate_o) * up_o
    y_online = act_o @ m.down_projs[0] + (act_o @ m.lora_down_A[0] @ m.lora_down_B[0]) * m.scale

    # Folded merge and online LoRA use different op orders (merge folds the
    # delta into one matmul; online adds two), so bf16 rounding diverges ~1e-3.
    torch.testing.assert_close(y_merged, y_online, atol=2e-3, rtol=2e-2)
