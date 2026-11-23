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
"""CPU unit tests for resident multi-LoRA adapters on one shared base model."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file
from tensordict import NonTensorStack, TensorDict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

from verl.model_merger.base_model_merger import BaseModelMerger, ModelMergerConfig
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.fsdp_utils import (
    _merge_or_unmerge_lora_,
    apply_fsdp2,
    collect_lora_params,
    get_fsdp_wrap_policy,
    layered_summon_lora_params,
)
from verl.utils.lora_utils import (
    MultiAdapterLRScheduler,
    adapter_context,
    bind_optimizer_to_adapter,
    build_adapter_param_groups,
    build_lora_config_from_spec,
    collect_adapter_submodules,
    extract_adapter_name_from_param_key,
    get_max_lora_rank,
    get_max_loras,
    get_peft_config,
    list_peft_adapters,
    param_belongs_to_adapter,
    resolve_adapter_name,
    resolve_global_adapter_schedule,
    select_adapter_parameters,
    split_batch_by_adapter,
    strip_adapter_name_from_param_key,
    validate_adapter_name,
)
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
from verl.workers.config.optimizer import FSDPOptimizerConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
from verl.workers.engine_workers import TrainingWorker


class _TinyDecoderLayer(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x) + self.v_proj(x)


class _TinyCausalModel(nn.Module):
    _no_split_modules = ["_TinyDecoderLayer"]

    def __init__(self, num_layers: int = 2, dim: int = 16):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False, model_type="tiny")
        self.layers = nn.ModuleList([_TinyDecoderLayer(dim=dim) for _ in range(num_layers)])

    def enable_input_require_grads(self):
        pass

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {}

    def forward(self, input_ids: torch.Tensor, **_kwargs) -> torch.Tensor:
        x = input_ids
        for layer in self.layers:
            x = layer(x)
        return x


def _make_multi_adapter_peft_model(
    adapters: dict[str, tuple[int, int]] | None = None,
    dim: int = 16,
    dtype: torch.dtype = torch.float32,
):
    if adapters is None:
        adapters = {"tenant_a": (4, 8), "tenant_b": (8, 32)}
    base = _TinyCausalModel(num_layers=2, dim=dim).to(dtype=dtype)
    items = list(adapters.items())
    first_name, (first_r, first_alpha) = items[0]
    pm = get_peft_model(
        base,
        LoraConfig(
            r=first_r,
            lora_alpha=first_alpha,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        ),
        adapter_name=first_name,
    )
    for name, (rank, alpha) in items[1:]:
        pm.add_adapter(
            name,
            LoraConfig(
                r=rank,
                lora_alpha=alpha,
                target_modules=["q_proj", "v_proj"],
                task_type=TaskType.CAUSAL_LM,
            ),
        )
    # Give each adapter distinct non-zero weights (PEFT initializes lora_B to zero).
    with torch.no_grad():
        for idx, (name, _) in enumerate(items, start=1):
            for p_idx, p in enumerate(select_adapter_parameters(pm, name)):
                p.data = p.data.to(dtype=dtype)
                p.data.fill_(float(idx * 10 + p_idx) * 0.01)
    pm.set_adapter(first_name)
    return pm


# ---------------------------------------------------------------------------
# 1. Adapter resolution, parameter key parsing, and adapter_context
# ---------------------------------------------------------------------------


def test_adapter_resolution_and_key_helpers():
    pm = _make_multi_adapter_peft_model({"tenant_a": (4, 8), "tenant_b": (8, 32)})

    assert list_peft_adapters(pm) == ["tenant_a", "tenant_b"]
    assert resolve_adapter_name(pm) == "tenant_a"
    assert resolve_adapter_name(pm, "tenant_b") == "tenant_b"
    assert get_peft_config(pm, "tenant_b").r == 8

    pm.set_adapter("tenant_b")
    assert resolve_adapter_name(pm) == "tenant_b"

    with pytest.raises(KeyError, match="missing_adapter"):
        resolve_adapter_name(pm, "missing_adapter")

    with pytest.raises(ValueError, match="PEFT adapters"):
        resolve_adapter_name(_TinyCausalModel())

    # Key extraction & stripping across linear, ParameterDict, modules_to_save, and colliding names
    assert extract_adapter_name_from_param_key("base_model.model.layers.0.q_proj.lora_A.tenant_a.weight") == "tenant_a"
    assert extract_adapter_name_from_param_key("base_model.model.layers.0.experts.lora_A.tenant_b") == "tenant_b"
    assert (
        extract_adapter_name_from_param_key("base_model.model.embed_tokens.modules_to_save.tenant_a.weight")
        == "tenant_a"
    )
    assert extract_adapter_name_from_param_key("base_model.model.layers.0.q_proj.lora_A.weight") is None
    assert extract_adapter_name_from_param_key("base_model.model.layers.0.q_proj.base_layer.weight") is None

    # Adapter named "layers" collides with base path, must only strip after lora_A
    colliding = "base_model.model.layers.0.q_proj.lora_A.layers.weight"
    assert extract_adapter_name_from_param_key(colliding, known_adapters=["layers", "other"]) == "layers"
    assert param_belongs_to_adapter(colliding, "layers", all_adapter_names=["layers", "other"]) is True
    assert param_belongs_to_adapter(colliding, "other", all_adapter_names=["layers", "other"]) is False
    assert strip_adapter_name_from_param_key(colliding, "layers") == "base_model.model.layers.0.q_proj.lora_A.weight"
    assert (
        strip_adapter_name_from_param_key("base_model.model.layers.0.experts.lora_B.tenant_b", "tenant_b")
        == "base_model.model.layers.0.experts.lora_B"
    )


def test_adapter_context_restores_on_exit_and_error():
    pm = _make_multi_adapter_peft_model()
    assert pm.active_adapter == "tenant_a"

    with adapter_context(pm, "tenant_b") as active:
        assert active == "tenant_b"
        assert pm.active_adapter == "tenant_b"
        assert all(p.requires_grad for p in select_adapter_parameters(pm, "tenant_b"))
        assert all(not p.requires_grad for p in select_adapter_parameters(pm, "tenant_a"))
    assert pm.active_adapter == "tenant_a"

    with pytest.raises(RuntimeError, match="boom"):
        with adapter_context(pm, "tenant_b"):
            assert pm.active_adapter == "tenant_b"
            raise RuntimeError("boom")
    assert pm.active_adapter == "tenant_a"


def test_select_adapter_parameters_includes_inactive_adapters():
    pm = _make_multi_adapter_peft_model()
    pm.set_adapter("tenant_a")
    params_a = select_adapter_parameters(pm, "tenant_a")
    params_b = select_adapter_parameters(pm, "tenant_b")
    assert len(params_a) == 8
    assert len(params_b) == 8
    assert {id(p) for p in params_a}.isdisjoint({id(p) for p in params_b})
    assert all(p.requires_grad for p in params_a)
    assert all(not p.requires_grad for p in params_b)


def test_lora_config_rank_and_max_loras_helpers():
    cfg = SimpleNamespace(
        lora_rank=4,
        lora_alpha=16,
        target_modules=["q_proj"],
        target_parameters=None,
        exclude_modules=None,
        lora={},
        lora_adapters={"a": {"r": 4, "lora_alpha": 8}, "b": {"rank": 16, "alpha": 32}},
    )
    assert get_max_lora_rank(cfg) == 16
    assert get_max_loras(cfg) == 2

    lora_b = build_lora_config_from_spec(cfg.lora_adapters["b"], cfg, adapter_name="b")
    assert lora_b.r == 16
    assert lora_b.lora_alpha == 32
    assert lora_b.task_type == TaskType.CAUSAL_LM

    cfg_override = SimpleNamespace(lora_rank=8, lora={"max_loras": 4}, lora_adapters={"a": {"r": 8}})
    assert get_max_loras(cfg_override) == 4

    cfg_bad = SimpleNamespace(lora_rank=8, lora={"max_loras": 0}, lora_adapters={})
    with pytest.raises(ValueError, match="max_loras"):
        get_max_loras(cfg_bad)


# ---------------------------------------------------------------------------
# 2. Per-adapter optimizer rebind, gradient isolation, & LR scheduler
# ---------------------------------------------------------------------------


def test_per_adapter_optimizer_rebind_isolates_grads_and_weights():
    pm = _make_multi_adapter_peft_model()
    param_groups = build_adapter_param_groups(
        pm,
        adapter_specs={"tenant_a": {"lr": 1e-2}, "tenant_b": {"lr": 1e-3}},
    )
    opt = torch.optim.AdamW(param_groups, lr=1e-4, weight_decay=0.1)

    x = torch.randn(4, 16)
    with adapter_context(pm, "tenant_a"):
        pm(x).pow(2).mean().backward()
    with adapter_context(pm, "tenant_b"):
        pm(x).pow(2).mean().backward()

    params_a = select_adapter_parameters(pm, "tenant_a")
    params_b = select_adapter_parameters(pm, "tenant_b")
    base_params = [p for n, p in pm.named_parameters() if "lora_" not in n]

    a_before = [p.detach().clone() for p in params_a]
    b_before = [p.detach().clone() for p in params_b]
    b_grads_before = [p.grad.detach().clone() for p in params_b]
    base_before = [p.detach().clone() for p in base_params]

    # Step ONLY tenant_a
    with bind_optimizer_to_adapter(opt, "tenant_a"):
        opt.step()

    # tenant_a moved and owns AdamW state
    assert any(not torch.equal(p, prev) for p, prev in zip(params_a, a_before, strict=True))
    assert all(p in opt.state for p in params_a)

    # tenant_b weights, accumulated grads, and optimizer state are completely untouched
    for p, prev, g_prev in zip(params_b, b_before, b_grads_before, strict=True):
        assert torch.equal(p, prev)
        assert p.grad is not None and torch.equal(p.grad, g_prev)
        assert p not in opt.state

    # Base weights untouched
    for p, prev in zip(base_params, base_before, strict=True):
        assert torch.equal(p, prev)

    # zero_grad ONLY tenant_a; tenant_b's accumulated gradients survive
    with bind_optimizer_to_adapter(opt, "tenant_a"):
        opt.zero_grad()
    assert all(p.grad is None for p in params_a)
    for p, g_prev in zip(params_b, b_grads_before, strict=True):
        assert p.grad is not None and torch.equal(p.grad, g_prev)

    # Full optimizer param_groups restored after context exit
    assert len(opt.param_groups) == 2


def test_multi_adapter_lr_scheduler_steps_independently_and_checkpoints():
    pm = _make_multi_adapter_peft_model()
    param_groups = build_adapter_param_groups(
        pm,
        adapter_specs={"tenant_a": {"lr": 1e-2}, "tenant_b": {"lr": 2e-3}},
    )
    opt = torch.optim.AdamW(param_groups, lr=1e-4)

    sched = MultiAdapterLRScheduler(
        optimizer=opt,
        scheduler_factory=lambda o: get_cosine_schedule_with_warmup(
            optimizer=o,
            num_warmup_steps=2,
            num_training_steps=10,
            min_lr_ratio=0.1,
        ),
    )

    b_lr_init = sched.get_last_lr("tenant_b")[0]
    for _ in range(5):
        sched.step("tenant_a")

    # tenant_b LR has not moved while tenant_a progressed through warmup + cosine decay
    assert sched.get_last_lr("tenant_b")[0] == pytest.approx(b_lr_init)
    assert sched.get_last_lr("tenant_a")[0] != pytest.approx(1e-2)

    state = sched.state_dict()
    opt2 = torch.optim.AdamW(
        build_adapter_param_groups(pm, adapter_specs={"tenant_a": {"lr": 1e-2}, "tenant_b": {"lr": 2e-3}}),
        lr=1e-4,
    )
    sched2 = MultiAdapterLRScheduler(
        optimizer=opt2,
        scheduler_factory=lambda o: get_cosine_schedule_with_warmup(
            optimizer=o,
            num_warmup_steps=2,
            num_training_steps=10,
            min_lr_ratio=0.1,
        ),
    )
    sched2.load_state_dict(state)
    assert sched2.get_last_lr("tenant_a") == pytest.approx(sched.get_last_lr("tenant_a"))
    assert sched2.get_last_lr("tenant_b") == pytest.approx(sched.get_last_lr("tenant_b"))


# ---------------------------------------------------------------------------
# 3. Mixed-adapter batch splitting & cross-rank canonical scheduling
# ---------------------------------------------------------------------------


def test_split_batch_by_adapter_tensordict_and_dict():
    td = TensorDict(
        {
            "input_ids": torch.arange(12).reshape(4, 3),
            "adapter_name": NonTensorStack("tenant_b", "tenant_a", "tenant_b", "tenant_a"),
        },
        batch_size=[4],
    )
    subs, idxs = split_batch_by_adapter(td, allowed_adapters=["tenant_a", "tenant_b"])
    assert list(subs.keys()) == ["tenant_b", "tenant_a"]
    assert idxs["tenant_b"] == [0, 2]
    assert idxs["tenant_a"] == [1, 3]
    assert torch.equal(subs["tenant_b"]["input_ids"], td["input_ids"][[0, 2]])
    assert torch.equal(subs["tenant_a"]["input_ids"], td["input_ids"][[1, 3]])

    # Dict batch with numpy array + scalar metadata
    d = {
        "x": torch.arange(6).reshape(3, 2),
        "meta": "shared_tag",
        "adapter_name": np.array(["a", "b", "a"]),
    }
    d_subs, d_idxs = split_batch_by_adapter(d)
    assert d_idxs == {"a": [0, 2], "b": [1]}
    assert d_subs["a"]["meta"] == "shared_tag"
    assert torch.equal(d_subs["b"]["x"], d["x"][[1]])

    # Missing key falls back to default_adapter
    d_default, _ = split_batch_by_adapter({"x": torch.zeros(2, 4)}, default_adapter="tenant_a")
    assert list(d_default.keys()) == ["tenant_a"]

    with pytest.raises(KeyError, match="unknown"):
        split_batch_by_adapter(
            {"x": torch.zeros(2, 4), "adapter_name": ["tenant_a", "unknown"]},
            allowed_adapters=["tenant_a"],
        )


def _gloo_schedule_worker(rank: int, world_size: int, rdzv_file: str):
    dist.init_process_group("gloo", init_method=f"file://{rdzv_file}", rank=rank, world_size=world_size)
    try:
        local = ["tenant_b"] if rank == 0 else ["tenant_a"]
        schedule = resolve_global_adapter_schedule(
            local_adapters=local,
            resident_adapters=["tenant_a", "tenant_b", "tenant_c"],
        )
        assert schedule == ["tenant_a", "tenant_b"]
    finally:
        dist.destroy_process_group()


def test_resolve_global_adapter_schedule_two_rank_gloo(tmp_path):
    rdzv_file = str(tmp_path / "rdzv_schedule")
    mp.spawn(_gloo_schedule_worker, args=(2, rdzv_file), nprocs=2, join=True)


# ---------------------------------------------------------------------------
# 4. Per-adapter FSDP2 units, zero inactive all-gather, & layered summon
# ---------------------------------------------------------------------------


def _gloo_fsdp2_multi_lora_worker(rank: int, world_size: int, rdzv_file: str):
    dist.init_process_group("gloo", init_method=f"file://{rdzv_file}", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(42)
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
        pm = _make_multi_adapter_peft_model({"tenant_a": (4, 8), "tenant_b": (8, 32)}, dim=16)

        # Capture ground-truth weights before FSDP2 sharding
        expected_a = {
            k: v.detach().clone() for k, v in collect_lora_params(pm, False, True, adapter_name="tenant_a").items()
        }
        expected_b = {
            k: v.detach().clone() for k, v in collect_lora_params(pm, False, True, adapter_name="tenant_b").items()
        }

        apply_fsdp2(pm, {"mesh": mesh}, {})

        # Verify each adapter's submodules per decoder layer are wrapped as independent FSDPModules
        for layer in pm.base_model.model.layers:
            assert isinstance(layer, FSDPModule)
            for name in ("tenant_a", "tenant_b"):
                subs = collect_adapter_submodules(layer, name)
                assert len(subs) == 4  # q_proj/v_proj x lora_A/lora_B
                assert all(isinstance(s, FSDPModule) for s in subs)

        # Instrument FSDPParamGroup.unshard directly to prove inactive adapter FSDP2 units
        # are never unsharded/all-gathered (even independent of forward execution).
        adapter_param_groups: dict[int, str] = {}
        for name in ("tenant_a", "tenant_b"):
            for sub in collect_adapter_submodules(pm, name):
                state = getattr(sub, "_get_fsdp_state", lambda: None)()
                pg = getattr(state, "_fsdp_param_group", None)
                if pg is not None:
                    adapter_param_groups[id(pg)] = name

        unshard_hits: dict[str, int] = {"tenant_a": 0, "tenant_b": 0}
        orig_unshard = FSDPParamGroup.unshard

        def _counting_unshard(self_pg, *args, **kwargs):
            label = adapter_param_groups.get(id(self_pg))
            if label is not None:
                unshard_hits[label] += 1
            return orig_unshard(self_pg, *args, **kwargs)

        FSDPParamGroup.unshard = _counting_unshard
        try:
            torch.manual_seed(123)
            x = torch.randn(2, 16)

            # Build unsharded reference with identical seed/weights to check FSDP2 gradient parity
            torch.manual_seed(42)
            ref_pm = _make_multi_adapter_peft_model({"tenant_a": (4, 8), "tenant_b": (8, 32)}, dim=16)

            with adapter_context(pm, "tenant_a"):
                pm(x).pow(2).mean().backward()
            assert unshard_hits["tenant_a"] > 0
            assert unshard_hits["tenant_b"] == 0
            assert all(p.grad is not None for p in select_adapter_parameters(pm, "tenant_a"))
            assert all(p.grad is None for p in select_adapter_parameters(pm, "tenant_b"))

            with adapter_context(ref_pm, "tenant_a"):
                ref_pm(x).pow(2).mean().backward()
            for p_fsdp, p_ref in zip(
                select_adapter_parameters(pm, "tenant_a"),
                select_adapter_parameters(ref_pm, "tenant_a"),
                strict=True,
            ):
                got_grad = p_fsdp.grad.full_tensor() if hasattr(p_fsdp.grad, "full_tensor") else p_fsdp.grad
                torch.testing.assert_close(got_grad, p_ref.grad, rtol=1e-5, atol=1e-5)

            unshard_hits["tenant_a"] = 0
            unshard_hits["tenant_b"] = 0
            with adapter_context(pm, "tenant_b"):
                pm(x).pow(2).mean().backward()
            assert unshard_hits["tenant_b"] > 0
            assert unshard_hits["tenant_a"] == 0
            assert all(p.grad is not None for p in select_adapter_parameters(pm, "tenant_b"))
        finally:
            FSDPParamGroup.unshard = orig_unshard

        # Verify layered_summon_lora_params and collect_lora_params extract exact per-adapter weights
        pm.set_adapter("tenant_b")
        got_active = collect_lora_params(pm, layered_summon=True, base_sync_done=True)
        got_a_layered = layered_summon_lora_params(pm, adapter_name="tenant_a")
        got_b_full = collect_lora_params(pm, layered_summon=False, base_sync_done=True, adapter_name="tenant_b")

        assert set(got_active.keys()) == set(expected_b.keys())
        assert set(got_a_layered.keys()) == set(expected_a.keys())
        assert set(got_b_full.keys()) == set(expected_b.keys())

        for k, exp in expected_a.items():
            torch.testing.assert_close(got_a_layered[k], exp)
        for k, exp in expected_b.items():
            torch.testing.assert_close(got_active[k], exp)
            torch.testing.assert_close(got_b_full[k], exp)
    finally:
        dist.destroy_process_group()


def test_fsdp2_per_adapter_wrapping_and_layered_summon_two_rank_gloo(tmp_path):
    rdzv_file = str(tmp_path / "rdzv_fsdp2_multi_lora")
    mp.spawn(_gloo_fsdp2_multi_lora_worker, args=(2, rdzv_file), nprocs=2, join=True)


def test_fsdp1_wrap_policy_includes_inactive_resident_adapters():
    pm = _make_multi_adapter_peft_model()
    pm.set_adapter("tenant_a")
    policy = get_fsdp_wrap_policy(pm, config={"transformer_layer_cls_to_wrap": ["_TinyDecoderLayer"]}, is_lora=True)
    assert policy is not None

    # Inactive tenant_b leaf Linear modules must still be matched by the lambda wrap policy
    for sub in collect_adapter_submodules(pm, "tenant_b"):
        assert policy(sub, recurse=False, nonwrapped_numel=0) is True
    # Frozen base linear layers must NOT be matched by the lambda leaf policy
    base_q_proj = pm.base_model.model.layers[0].q_proj.base_layer
    assert policy(base_q_proj, recurse=False, nonwrapped_numel=0) is False


def test_selective_adapter_merge_unmerge():
    pm = _make_multi_adapter_peft_model()
    x = torch.randn(2, 16)
    with adapter_context(pm, "tenant_b"):
        expected_b_out = pm(x).detach()
    with pm.disable_adapter():
        base_out = pm(x).detach()

    # Even while tenant_a is active, merging adapter_names=["tenant_b"] merges only tenant_b
    pm.set_adapter("tenant_a")
    _merge_or_unmerge_lora_(pm, merge=True, adapter_names=["tenant_b"])
    merged_out = x
    for layer in pm.base_model.model.layers:
        merged_out = layer.q_proj.base_layer(merged_out) + layer.v_proj.base_layer(merged_out)
    torch.testing.assert_close(merged_out.detach(), expected_b_out, rtol=1e-4, atol=1e-4)

    _merge_or_unmerge_lora_(pm, merge=False)
    restored_out = x
    for layer in pm.base_model.model.layers:
        restored_out = layer.q_proj.base_layer(restored_out) + layer.v_proj.base_layer(restored_out)
    torch.testing.assert_close(restored_out.detach(), base_out, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# 5. FSDPEngine multi-adapter build, isolated step, and per-adapter export
# ---------------------------------------------------------------------------


def test_fsdp_engine_multi_lora_build_step_and_export(monkeypatch):
    engine = object.__new__(FSDPEngine)
    engine.rank = 0
    engine._is_lora = True
    engine._is_offload_param = False
    engine._is_offload_optimizer = False
    engine._uses_fsdp2_cpu_offload_policy = False
    engine.scaler = None
    engine.model_config = SimpleNamespace(
        lora_rank=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        target_parameters=None,
        exclude_modules=None,
        lora_adapter_path=None,
        lora_adapters={
            "tenant_a": {"r": 4, "lora_alpha": 8, "lr": 1e-2},
            "tenant_b": {"r": 8, "lora_alpha": 32, "lr": 5e-3},
        },
        lora={"merge": False},
        use_shm=False,
        hf_config=SimpleNamespace(model_type="tiny"),
    )
    engine.optimizer_config = FSDPOptimizerConfig(
        lr=1e-3,
        clip_grad=1.0,
        lr_scheduler_type="cosine",
        total_training_steps=10,
        lr_warmup_steps=2,
        min_lr_ratio=0.1,
    )

    # Base model in bfloat16 -> BOTH active and inactive resident adapters must be cast to bfloat16
    base = _TinyCausalModel(num_layers=2, dim=16).to(dtype=torch.bfloat16)
    peft_mod = engine._build_lora_module(base)
    engine.module = peft_mod

    assert engine.list_adapters() == ["tenant_a", "tenant_b"]
    assert engine.get_active_adapter() == "tenant_a"
    for name in ("tenant_a", "tenant_b"):
        assert all(p.dtype == torch.bfloat16 for p in select_adapter_parameters(peft_mod, name))

    # Build per-adapter optimizer + scheduler on float32 copy for exact step verification
    engine.module = _make_multi_adapter_peft_model({"tenant_a": (4, 8), "tenant_b": (8, 32)})
    engine.optimizer = engine._build_optimizer(engine.module)
    engine.lr_scheduler = engine._build_lr_scheduler(engine.optimizer)
    assert isinstance(engine.lr_scheduler, MultiAdapterLRScheduler)

    # Accumulate grads for both adapters, inject NaN into tenant_a only, verify tenant_b is untouched
    x = torch.randn(2, 16)
    with engine.adapter_context("tenant_a"):
        engine.module(x).pow(2).mean().backward()
    with engine.adapter_context("tenant_b"):
        engine.module(x).pow(2).mean().backward()

    params_a = select_adapter_parameters(engine.module, "tenant_a")
    params_b = select_adapter_parameters(engine.module, "tenant_b")
    b_grads_saved = [p.grad.detach().clone() for p in params_b]

    params_a[0].grad.fill_(float("nan"))
    with engine.adapter_context("tenant_a"):
        gn_nan = engine.optimizer_step()
    assert not np.isfinite(gn_nan)
    assert all(p.grad is None for p in params_a)
    for p, saved in zip(params_b, b_grads_saved, strict=True):
        assert p.grad is not None and torch.equal(p.grad, saved)

    # Normal step on tenant_b advances only tenant_b
    with engine.adapter_context("tenant_b"):
        gn_b = engine.optimizer_step()
        lr_b_after = engine.lr_scheduler_step()
    assert np.isfinite(gn_b) and gn_b > 0
    assert lr_b_after == pytest.approx(engine.lr_scheduler.get_last_lr("tenant_b")[0])

    # Export tenant_b explicitly via get_per_tensor_param without needing GPU
    monkeypatch.setattr("verl.workers.engine.fsdp.transformer_impl.log_gpu_memory_usage", lambda *_a, **_kw: None)
    monkeypatch.setattr("verl.workers.engine.fsdp.transformer_impl.load_fsdp_model_to_gpu", lambda _m: None)
    monkeypatch.setattr("verl.workers.engine.fsdp.transformer_impl.offload_fsdp_model_to_cpu", lambda _m: None)
    params_iter, peft_cfg_dict = engine.get_per_tensor_param(base_sync_done=True, adapter_name="tenant_b")
    exported = dict(params_iter)
    assert peft_cfg_dict["r"] == 8
    assert peft_cfg_dict["lora_alpha"] == 32
    assert any(tuple(t.shape) == (8, 16) for t in exported.values())


# ---------------------------------------------------------------------------
# 6. Checkpoint metadata & model merger for single/multi-adapter checkpoints
# ---------------------------------------------------------------------------


def test_checkpoint_manager_lora_train_meta_single_and_multi(monkeypatch):
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist, "barrier", lambda: None)

    # Single "default" adapter keeps exact legacy JSON schema
    single_pm = _make_multi_adapter_peft_model({"default": (8, 16)})
    mgr_single = FSDPCheckpointManager(model=single_pm, optimizer=None, lr_scheduler=None, processing_class=None)
    assert mgr_single._get_lora_train_meta(single_pm) == {"r": 8, "lora_alpha": 16, "task_type": "CAUSAL_LM"}

    # Multi-adapter records per-adapter r/alpha/task_type plus active_adapter
    multi_pm = _make_multi_adapter_peft_model({"tenant_a": (4, 8), "tenant_b": (8, 32)})
    multi_pm.set_adapter("tenant_b")
    mgr_multi = FSDPCheckpointManager(model=multi_pm, optimizer=None, lr_scheduler=None, processing_class=None)
    meta = mgr_multi._get_lora_train_meta(multi_pm)
    assert meta["r"] == 8
    assert meta["lora_alpha"] == 32
    assert meta["active_adapter"] == "tenant_b"
    assert meta["adapters"] == {
        "tenant_a": {"r": 4, "lora_alpha": 8, "task_type": "CAUSAL_LM"},
        "tenant_b": {"r": 8, "lora_alpha": 32, "task_type": "CAUSAL_LM"},
    }


class _DummyMerger(BaseModelMerger):
    def __init__(self, local_dir: str, target_dir: str):
        self.config = ModelMergerConfig(
            operation="merge",
            backend="fsdp",
            local_dir=local_dir,
            target_dir=target_dir,
        )

    def merge_and_save(self):
        pass

    def cleanup(self):
        pass


def test_model_merger_saves_non_default_and_multi_adapter_checkpoints(tmp_path):
    # Case A: single non-default adapter ("tenant_a")
    src_single = tmp_path / "ckpt_single"
    dst_single = tmp_path / "merged_single"
    src_single.mkdir()
    with open(src_single / "lora_train_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "r": 4,
                "lora_alpha": 12,
                "task_type": "CAUSAL_LM",
                "active_adapter": "tenant_a",
                "adapters": {"tenant_a": {"r": 4, "lora_alpha": 12, "task_type": "CAUSAL_LM"}},
            },
            f,
        )

    pm_single = _make_multi_adapter_peft_model({"tenant_a": (4, 12)})
    sd_single = {k: v.detach().clone() for k, v in pm_single.state_dict().items()}
    merger_single = _DummyMerger(str(src_single), str(dst_single))
    out_single = merger_single.save_lora_adapter(sd_single)
    assert out_single == str(dst_single / "lora_adapter")

    with open(dst_single / "lora_adapter" / "adapter_config.json", encoding="utf-8") as f:
        cfg_single = json.load(f)
    assert cfg_single["r"] == 4
    assert cfg_single["lora_alpha"] == 12
    assert set(cfg_single["target_modules"]) == {"q_proj", "v_proj"}
    saved_tensors = load_file(str(dst_single / "lora_adapter" / "adapter_model.safetensors"))
    assert all(".tenant_a." not in k for k in saved_tensors)
    assert all("lora_" not in k for k in sd_single)
    assert "layers.0.q_proj.weight" in sd_single

    # Case B: multi-adapter checkpoint ("tenant_a" + "tenant_b")
    src_multi = tmp_path / "ckpt_multi"
    dst_multi = tmp_path / "merged_multi"
    src_multi.mkdir()
    with open(src_multi / "lora_train_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "r": 4,
                "lora_alpha": 8,
                "task_type": "CAUSAL_LM",
                "active_adapter": "tenant_a",
                "adapters": {
                    "tenant_a": {"r": 4, "lora_alpha": 8, "task_type": "CAUSAL_LM"},
                    "tenant_b": {"r": 8, "lora_alpha": 32, "task_type": "CAUSAL_LM"},
                },
            },
            f,
        )

    pm_multi = _make_multi_adapter_peft_model({"tenant_a": (4, 8), "tenant_b": (8, 32)})
    sd_multi = {k: v.detach().clone() for k, v in pm_multi.state_dict().items()}
    merger_multi = _DummyMerger(str(src_multi), str(dst_multi))
    out_multi = merger_multi.save_lora_adapter(sd_multi)
    assert out_multi == str(dst_multi / "lora_adapters")

    for name, exp_r, exp_alpha in (("tenant_a", 4, 8), ("tenant_b", 8, 32)):
        adapter_dir = dst_multi / "lora_adapters" / name
        with open(adapter_dir / "adapter_config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg["r"] == exp_r
        assert cfg["lora_alpha"] == exp_alpha
        assert set(cfg["target_modules"]) == {"q_proj", "v_proj"}
        tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))
        assert len(tensors) == 8
        assert all(f".{name}." not in k for k in tensors)
        assert any(min(t.shape) == exp_r for t in tensors.values())

    assert all("lora_" not in k for k in sd_multi)
    assert "layers.0.q_proj.weight" in sd_multi


# ---------------------------------------------------------------------------
# 7. Codex-review regression tests: modules_to_save reload, path-only rank,
#    single-adapter overrides, bias/root-embedding export, per-adapter scalers,
#    FSDP1 clipping isolation, adapter name validation, and worker guardrails.
# ---------------------------------------------------------------------------


class _TinyModelWithHead(nn.Module):
    _no_split_modules = ["_TinyDecoderLayer"]

    def __init__(self, dim: int = 16):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=True, model_type="tiny")
        self.embed = nn.Embedding(32, dim)
        self.layers = nn.ModuleList([_TinyDecoderLayer(dim=dim)])
        self.lm_head = nn.Linear(dim, 8, bias=True)

    def enable_input_require_grads(self):
        pass

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {}

    def forward(self, input_ids: torch.Tensor, **_kwargs) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def test_model_merger_modules_to_save_peft_reload_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist, "barrier", lambda: None)

    torch.manual_seed(7)
    base = _TinyModelWithHead(dim=16)
    base_state_snapshot = {k: v.detach().clone() for k, v in base.state_dict().items()}

    pm = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            modules_to_save=["lm_head"],
            task_type=TaskType.CAUSAL_LM,
        ),
        adapter_name="tenant_a",
    )
    pm.add_adapter(
        "tenant_b",
        LoraConfig(
            r=8,
            lora_alpha=24,
            target_modules=["q_proj", "v_proj"],
            modules_to_save=["lm_head"],
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    with torch.no_grad():
        for idx, name in enumerate(("tenant_a", "tenant_b"), start=1):
            for p_idx, p in enumerate(select_adapter_parameters(pm, name)):
                p.data.fill_(float(idx * 5 + p_idx) * 0.02)

    tokens = torch.tensor([[1, 2, 3]])
    expected_outs = {}
    for name in ("tenant_a", "tenant_b"):
        with adapter_context(pm, name):
            expected_outs[name] = pm(tokens).detach()

    src = tmp_path / "ckpt_mts"
    dst = tmp_path / "merged_mts"
    src.mkdir()
    mgr = FSDPCheckpointManager(model=pm, optimizer=None, lr_scheduler=None, processing_class=None)
    meta = mgr._get_lora_train_meta(pm)
    assert meta["adapters"]["tenant_a"]["modules_to_save"] == ["lm_head"]
    assert meta["adapters"]["tenant_b"]["modules_to_save"] == ["lm_head"]
    with open(src / "lora_train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    sd = {k: v.detach().clone() for k, v in pm.state_dict().items()}
    merger = _DummyMerger(str(src), str(dst))
    out_root = merger.save_lora_adapter(sd)
    assert out_root == str(dst / "lora_adapters")
    assert all("modules_to_save" not in k and "original_module" not in k for k in sd)
    assert "lm_head.weight" in sd

    # Reload EACH exported adapter via PeftModel.from_pretrained onto a fresh base model
    for name in ("tenant_a", "tenant_b"):
        fresh_base = _TinyModelWithHead(dim=16)
        fresh_base.load_state_dict(base_state_snapshot)
        reloaded = PeftModel.from_pretrained(fresh_base, str(dst / "lora_adapters" / name))
        reloaded_out = reloaded(tokens).detach()
        torch.testing.assert_close(reloaded_out, expected_outs[name], rtol=1e-5, atol=1e-5)


def test_get_max_lora_rank_reads_adapter_path_and_validates(tmp_path):
    adapter_dir = tmp_path / "pretrained_adapter"
    adapter_dir.mkdir()
    with open(adapter_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump({"r": 16, "lora_alpha": 32}, f)

    # Path-only adapter config with lora_rank=0 must still report rank=16
    cfg_path_only = SimpleNamespace(
        lora_rank=0,
        lora={},
        lora_adapter_path=None,
        lora_adapters={"a": {"adapter_path": str(adapter_dir)}},
    )
    assert get_max_lora_rank(cfg_path_only) == 16

    # Explicit smaller rank must not underestimate loaded adapter rank
    cfg_under = SimpleNamespace(
        lora_rank=4,
        lora={},
        lora_adapter_path=None,
        lora_adapters={"a": {"r": 4, "adapter_path": str(adapter_dir)}},
    )
    assert get_max_lora_rank(cfg_under) == 16

    # Unreadable path without explicit rank must fail fast instead of silently returning 0
    cfg_missing = SimpleNamespace(
        lora_rank=0,
        lora={},
        lora_adapter_path=None,
        lora_adapters={"a": {"adapter_path": str(tmp_path / "nonexistent")}},
    )
    with pytest.raises(ValueError, match="Cannot determine LoRA rank"):
        get_max_lora_rank(cfg_missing)


def test_adapter_name_edge_cases_and_safety(tmp_path):
    # Adapter literally named "weight" is parsed cleanly both with and without known_adapters
    key_weight = "base_model.model.layers.0.q_proj.lora_A.weight.weight"
    assert extract_adapter_name_from_param_key(key_weight) == "weight"
    assert extract_adapter_name_from_param_key(key_weight, known_adapters=["weight"]) == "weight"
    assert strip_adapter_name_from_param_key(key_weight, "weight") == "base_model.model.layers.0.q_proj.lora_A.weight"

    for bad_name in ("", ".", "..", "../escape", "a/b", "a\\b", "a.b"):
        with pytest.raises(ValueError):
            validate_adapter_name(bad_name)

    # Merger rejects path-separator adapter names before writing directories
    src = tmp_path / "src_bad"
    dst = tmp_path / "dst_bad"
    src.mkdir()
    with open(src / "lora_train_meta.json", "w", encoding="utf-8") as f:
        json.dump({"adapters": {"bad/escape": {"r": 4, "lora_alpha": 8}}}, f)
    merger = _DummyMerger(str(src), str(dst))
    with pytest.raises(ValueError, match="Invalid adapter name"):
        merger.save_lora_adapter({"base_model.model.q_proj.lora_A.bad/escape.weight": torch.zeros(4, 8)})


def test_single_resident_adapter_honors_optimizer_overrides_and_rejects_shared_bias():
    engine = object.__new__(FSDPEngine)
    engine.rank = 0
    engine._is_lora = True
    engine._adapter_scalers = {}
    engine.scaler = None
    engine.model_config = SimpleNamespace(
        lora_rank=0,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        target_parameters=None,
        exclude_modules=None,
        lora_adapter_path=None,
        lora_adapters={"tenant_a": {"r": 4, "lora_alpha": 8, "lr": 7e-3, "weight_decay": 0.05}},
        lora={},
        use_shm=False,
    )
    engine.optimizer_config = FSDPOptimizerConfig(lr=1e-4, weight_decay=0.01)

    pm = engine._build_lora_module(_TinyCausalModel(num_layers=1, dim=16))
    opt = engine._build_optimizer(pm)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["adapter_name"] == "tenant_a"
    assert opt.param_groups[0]["lr"] == pytest.approx(7e-3)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.05)

    # Multi-adapter with shared trainable bias must be rejected
    engine.model_config.lora_adapters = {
        "tenant_a": {"r": 4, "lora_alpha": 8, "bias": "all"},
        "tenant_b": {"r": 4, "lora_alpha": 8},
    }
    with pytest.raises(ValueError, match="shared trainable biases"):
        engine._build_lora_module(_TinyCausalModel(num_layers=1, dim=16))


def test_per_adapter_grad_scaler_isolation():
    class _FakeScaler:
        def __init__(self, init_scale: float):
            self.scale_val = init_scale
            self.unscaled_calls = 0
            self.step_calls = 0
            self.update_calls = 0
            self._overflow = False

        def scale(self, loss):
            return loss * self.scale_val

        def unscale_(self, optimizer):
            self.unscaled_calls += 1
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.div_(self.scale_val)
                        if not torch.isfinite(p.grad).all():
                            self._overflow = True

        def step(self, optimizer):
            self.step_calls += 1
            if not self._overflow:
                optimizer.step()

        def update(self):
            self.update_calls += 1
            if self._overflow:
                self.scale_val *= 0.5
                self._overflow = False

    engine = object.__new__(FSDPEngine)
    engine.module = _make_multi_adapter_peft_model({"tenant_a": (4, 8), "tenant_b": (4, 8)})
    engine.optimizer_config = FSDPOptimizerConfig(lr=1e-2, clip_grad=1e6)
    engine.optimizer = torch.optim.AdamW(build_adapter_param_groups(engine.module), lr=1e-2)
    engine.scaler = None
    scaler_a = _FakeScaler(init_scale=64.0)
    scaler_b = _FakeScaler(init_scale=64.0)
    engine._adapter_scalers = {"tenant_a": scaler_a, "tenant_b": scaler_b}

    x = torch.randn(2, 16)
    # Accumulate scaled gradients for both A and B at scale=64
    with engine.adapter_context("tenant_a"):
        engine._get_scaler().scale(engine.module(x).pow(2).mean()).backward()
    with engine.adapter_context("tenant_b"):
        engine._get_scaler().scale(engine.module(x).pow(2).mean()).backward()

    # Record B's expected true (unscaled) grad before A overflows
    params_b = select_adapter_parameters(engine.module, "tenant_b")
    expected_b_true_grads = [p.grad.detach().clone() / 64.0 for p in params_b]

    # Inject Inf into A -> A's scaler halves to 32, B's scaler MUST stay at 64
    select_adapter_parameters(engine.module, "tenant_a")[0].grad.fill_(float("inf"))
    with engine.adapter_context("tenant_a"):
        engine.optimizer_step()
    assert scaler_a.scale_val == pytest.approx(32.0)
    assert scaler_b.scale_val == pytest.approx(64.0)
    assert scaler_b.update_calls == 0

    # Stepping B unscales using B's untouched scale=64
    with engine.adapter_context("tenant_b"):
        engine.optimizer_step()
    assert scaler_b.step_calls == 1
    assert scaler_b.scale_val == pytest.approx(64.0)
    for p, exp_g in zip(params_b, expected_b_true_grads, strict=True):
        torch.testing.assert_close(p.grad, exp_g, rtol=1e-5, atol=1e-5)


def test_layered_summon_preserves_root_embedding_adapter_and_single_adapter_bias(tmp_path):
    # Verify root-owned tied embedding lora_embedding_A/B and single-adapter trainable bias
    # are collected by layered_summon_lora_params under FSDP2.
    rdzv_file = str(tmp_path / "rdzv_root_embed")
    dist.init_process_group("gloo", init_method=f"file://{rdzv_file}", rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        base = _TinyModelWithHead(dim=16)
        pm = get_peft_model(
            base,
            LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["q_proj", "embed"],
                bias="all",
                task_type=TaskType.CAUSAL_LM,
            ),
        )
        apply_fsdp2(pm, {"mesh": mesh}, {})
        layered = layered_summon_lora_params(pm)
        assert any("lora_embedding_A" in k for k in layered), f"Missing embedding LoRA in {list(layered.keys())}"
        assert any(k.endswith(".bias") for k in layered), f"Missing trainable bias in {list(layered.keys())}"
    finally:
        dist.destroy_process_group()


def _gloo_fsdp1_clip_and_schedule_mismatch_worker(rank: int, world_size: int, rdzv_file: str):
    dist.init_process_group("gloo", init_method=f"file://{rdzv_file}", rank=rank, world_size=world_size)
    try:
        # 1. Verify schedule mismatch across ranks raises ValueError
        resident = ["tenant_a", "tenant_b"] if rank == 0 else ["tenant_b", "tenant_a"]
        with pytest.raises(ValueError, match="registration order differs"):
            resolve_global_adapter_schedule(local_adapters=["tenant_a"], resident_adapters=resident)

        # 2. Verify _clip_fsdp1_adapter_grad_norm_ reduces sharded L2 norm across ranks
        #    and leaves other adapter's gradients untouched.
        p_a = nn.Parameter(torch.ones(2))
        p_b = nn.Parameter(torch.ones(2))
        p_a.grad = torch.full((2,), 3.0 if rank == 0 else 4.0)  # global L2 = sqrt(2*9 + 2*16) = sqrt(50)
        p_b.grad = torch.full((2,), 100.0)

        fake_fsdp = SimpleNamespace(
            parameters=lambda: iter([p_a, p_b]),
            process_group=None,
            sharding_strategy=None,
        )
        norm_a = FSDPEngine._clip_fsdp1_adapter_grad_norm_(fake_fsdp, [p_a], max_norm=1.0)
        assert norm_a.item() == pytest.approx(50.0**0.5, rel=1e-5)
        # p_b.grad must be completely untouched
        assert torch.equal(p_b.grad, torch.full((2,), 100.0))
    finally:
        dist.destroy_process_group()


def test_fsdp1_clip_isolation_and_schedule_order_validation_two_rank_gloo(tmp_path):
    rdzv_file = str(tmp_path / "rdzv_fsdp1_clip")
    mp.spawn(_gloo_fsdp1_clip_and_schedule_mismatch_worker, args=(2, rdzv_file), nprocs=2, join=True)


def test_training_worker_single_adapter_validation():
    td_scalar = TensorDict({"x": torch.zeros(2, 4)}, batch_size=[2])
    assert TrainingWorker._validate_single_adapter_request(td_scalar, "tenant_a") == "tenant_a"
    assert TrainingWorker._validate_single_adapter_request(td_scalar, None) is None

    with pytest.raises(ValueError, match="non-empty string"):
        TrainingWorker._validate_single_adapter_request(td_scalar, ["tenant_a", "tenant_b"])

    td_mixed = TensorDict(
        {"x": torch.zeros(2, 4), "adapter_name": NonTensorStack("tenant_a", "tenant_b")},
        batch_size=[2],
    )
    with pytest.raises(ValueError, match="Mixed-adapter batch"):
        TrainingWorker._validate_single_adapter_request(td_mixed, None)

    td_homogeneous = TensorDict(
        {"x": torch.zeros(2, 4), "adapter_name": NonTensorStack("tenant_b", "tenant_b")},
        batch_size=[2],
    )
    assert TrainingWorker._validate_single_adapter_request(td_homogeneous, None) == "tenant_b"
    with pytest.raises(ValueError, match="Conflicting adapter selection"):
        TrainingWorker._validate_single_adapter_request(td_homogeneous, "tenant_a")


def test_optimizer_and_scheduler_checkpoint_round_trip():
    pm = _make_multi_adapter_peft_model({"tenant_a": (4, 8), "tenant_b": (8, 32)})
    specs = {"tenant_a": {"lr": 1e-2}, "tenant_b": {"lr": 2e-3}}
    opt = torch.optim.AdamW(build_adapter_param_groups(pm, adapter_specs=specs), lr=1e-4)
    sched = MultiAdapterLRScheduler(
        optimizer=opt,
        scheduler_factory=lambda o: get_cosine_schedule_with_warmup(
            optimizer=o, num_warmup_steps=1, num_training_steps=6, min_lr_ratio=0.1
        ),
    )

    x = torch.randn(2, 16)
    for _ in range(3):
        with adapter_context(pm, "tenant_a"), bind_optimizer_to_adapter(opt, "tenant_a"):
            opt.zero_grad()
            pm(x).pow(2).mean().backward()
            opt.step()
        sched.step("tenant_a")

    with adapter_context(pm, "tenant_b"), bind_optimizer_to_adapter(opt, "tenant_b"):
        opt.zero_grad()
        pm(x).pow(2).mean().backward()
        opt.step()
    sched.step("tenant_b")

    opt_sd = opt.state_dict()
    sched_sd = sched.state_dict()

    # Reload into a fresh model + optimizer + scheduler
    pm2 = _make_multi_adapter_peft_model({"tenant_a": (4, 8), "tenant_b": (8, 32)})
    opt2 = torch.optim.AdamW(build_adapter_param_groups(pm2, adapter_specs=specs), lr=1e-4)
    sched2 = MultiAdapterLRScheduler(
        optimizer=opt2,
        scheduler_factory=lambda o: get_cosine_schedule_with_warmup(
            optimizer=o, num_warmup_steps=1, num_training_steps=6, min_lr_ratio=0.1
        ),
    )
    opt2.load_state_dict(opt_sd)
    sched2.load_state_dict(sched_sd)

    # Check actual param_group LRs and AdamW step counters/moments match per adapter
    assert opt2.param_groups[0]["adapter_name"] == "tenant_a"
    assert opt2.param_groups[1]["adapter_name"] == "tenant_b"
    assert opt2.param_groups[0]["lr"] == pytest.approx(opt.param_groups[0]["lr"])
    assert opt2.param_groups[1]["lr"] == pytest.approx(opt.param_groups[1]["lr"])
    p_a_first = select_adapter_parameters(pm2, "tenant_a")[0]
    p_b_first = select_adapter_parameters(pm2, "tenant_b")[0]
    assert opt2.state[p_a_first]["step"].item() == 3
    assert opt2.state[p_b_first]["step"].item() == 1
