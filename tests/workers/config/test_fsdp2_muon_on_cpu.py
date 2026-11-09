# Copyright 2026 Individual Contributor: yue-zeng-yue
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
"""Config/grouping tests; dependency stubs are explicitly isolated from real tests."""

import copy
import importlib
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_module

from verl.workers.config.optimizer import FSDPOptimizerConfig, build_optimizer


class TinyModel(torch.nn.Module):
    def __init__(self, tied=False):
        super().__init__()
        self.embed = torch.nn.Embedding(7, 4)
        self.proj = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.lm_head = torch.nn.Linear(4, 7, bias=False)
        if tied:
            self.lm_head.weight = self.embed.weight
        self.frozen = torch.nn.Linear(4, 4)
        self.frozen.requires_grad_(False)
        self.special_matrix = torch.nn.Parameter(torch.ones(2, 2))

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, x):
        return self.lm_head(self.norm(self.proj(self.embed(x))))


def context(**kwargs):
    return SimpleNamespace(strategy="fsdp2", **kwargs)


def build(model, **kwargs):
    return build_optimizer(
        model.parameters(), FSDPOptimizerConfig(optimizer="Muon", **kwargs), model=model, engine_config=context()
    )


def test_default_and_hydra_overrides():
    assert FSDPOptimizerConfig().optimizer == "AdamW"
    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.strategy=fsdp2",
                "actor_rollout_ref.actor.optim.optimizer=Muon",
                "actor_rollout_ref.actor.optim.muon_momentum=0.8",
                "actor_rollout_ref.actor.optim.muon_nesterov=true",
            ],
        )
    assert cfg.actor_rollout_ref.actor.optim.muon_momentum == 0.8
    assert cfg.actor_rollout_ref.actor.optim.muon_nesterov is True


@pytest.mark.parametrize("tied", [False, True])
def test_semantic_grouping_complete_deterministic(tied):
    from verl.utils.fsdp2_muon import group_fsdp2_muon_parameters

    model = TinyModel(tied=tied)
    # Shared projection used under a second name must be updated once.
    model.alias = model.proj
    groups = group_fsdp2_muon_parameters(model)
    names = {g["algorithm"]: g["param_names"] for g in groups}
    assert names["muon"] == ["alias.weight"]  # lexicographically first alias
    assert {"embed.weight", "norm.weight", "norm.bias", "alias.bias", "special_matrix"} <= set(names["adamw"])
    assert ("lm_head.weight" not in names["adamw"]) == tied
    params = [p for g in groups for p in g["params"]]
    assert len(params) == len({id(p) for p in params})
    assert {id(p) for p in params} == {id(p) for p in model.parameters() if p.requires_grad}
    assert [g["param_names"] for g in groups] == [g["param_names"] for g in group_fsdp2_muon_parameters(model)]
    # A matrix shared between embedding and projection always gets AdamW.
    model.proj.weight = model.embed.weight
    assert all(
        model.embed.weight is not p
        for g in group_fsdp2_muon_parameters(model)
        if g["algorithm"] == "muon"
        for p in g["params"]
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(muon_momentum=1.0),
        dict(muon_epsilon=0),
        dict(muon_adjust_lr="unknown"),
        dict(override_optimizer_config={"flatten": True}),
        dict(optimizer_impl="megatron"),
    ],
)
def test_invalid_muon_options(options):
    with pytest.raises(ValueError):
        build(TinyModel(), **options)


@pytest.mark.parametrize(
    "engine,message",
    [
        (SimpleNamespace(strategy="fsdp"), "native fsdp2"),
        (context(param_offload=True), "CPU offload"),
        (context(optimizer_offload=True), "CPU offload"),
        (context(offload_policy=True), "CPU offload"),
        (context(ulysses_sequence_parallel_size=2), "ulysses_sequence_parallel_size"),
        (context(mixed_precision={"param_dtype": "fp16"}), "FP32/BF16"),
    ],
)
def test_unsupported_engine(engine, message):
    model = TinyModel()
    with pytest.raises(ValueError, match=message):
        build_optimizer(model.parameters(), FSDPOptimizerConfig(optimizer="Muon"), model=model, engine_config=engine)


def test_dependency_is_optional(monkeypatch):
    real_import = importlib.import_module

    def missing(name, *args, **kwargs):
        if name == "dion":
            raise ImportError("simulated absent optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", missing)
    model = TinyModel()
    optimizer = build_optimizer(model.parameters(), FSDPOptimizerConfig())
    assert type(optimizer) is torch.optim.AdamW
    with pytest.raises(ImportError, match="requirements-fsdp2-muon"):
        build(model)


def test_stub_constructor_receives_options(monkeypatch):
    calls = {}

    class StubMuon(torch.optim.Optimizer):
        def __init__(self, params, **kwargs):
            calls.update(kwargs)
            super().__init__(params, {})

    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, *a, **kw: SimpleNamespace(Muon=StubMuon) if name == "dion" else real_import(name, *a, **kw),
    )
    opt = build(TinyModel(), lr=0.02, betas=(0.7, 0.8), muon_momentum=0.8, muon_nesterov=True)
    assert calls["mu"] == 0.8 and calls["nesterov"] is True
    assert calls["lr"] == 0.02 and calls["betas"] == (0.7, 0.8)
    assert calls["use_polar_express"] is False and calls["distributed_mesh"] is None
    assert {g["algorithm"] for g in opt.param_groups} == {"muon", "adamw"}


def test_real_dependency_update_scheduler_restore():
    dion = pytest.importorskip("dion", reason="real optional Dion dependency missing")
    model = TinyModel()
    opt = build(model, lr=0.01)
    assert isinstance(opt, dion.Muon)
    from verl.utils.torch_functional import get_constant_schedule_with_warmup

    scheduler = get_constant_schedule_with_warmup(opt, num_warmup_steps=2)
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    for _ in range(3):
        opt.zero_grad()
        model(torch.tensor([[1, 2, 3]])).square().mean().backward()
        opt.step()
        scheduler.step()
    for name in ["proj.weight", "embed.weight", "norm.weight", "lm_head.weight"]:
        assert not torch.equal(original[name], dict(model.named_parameters())[name])
    for state in opt.state.values():
        assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v, torch.Tensor))
    restored = TinyModel()
    restored.load_state_dict(model.state_dict())
    restored_opt = build(restored, lr=0.01)
    restored_sched = get_constant_schedule_with_warmup(restored_opt, num_warmup_steps=2)
    restored_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    restored_sched.load_state_dict(copy.deepcopy(scheduler.state_dict()))
    for m, o, s in [(model, opt, scheduler), (restored, restored_opt, restored_sched)]:
        o.zero_grad()
        m(torch.tensor([[2, 3, 4]])).square().mean().backward()
        o.step()
        s.step()
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert scheduler.state_dict() == restored_sched.state_dict()


def test_real_zero_and_missing_gradients():
    pytest.importorskip("dion", reason="real optional Dion dependency missing")
    model = TinyModel()
    opt = build(model)
    original = model.proj.weight.detach().clone()
    opt.step()  # all None: no decay or update
    torch.testing.assert_close(model.proj.weight, original, rtol=0, atol=0)
    model.proj.weight.grad = torch.zeros_like(model.proj.weight)
    opt.step()
    torch.testing.assert_close(model.proj.weight, original * (1 - 1e-3 * 0.01))
    assert torch.isfinite(model.proj.weight).all()


def test_hybrid_attention_special_projections_use_adamw():
    from verl.utils.fsdp2_muon import group_fsdp2_muon_parameters

    model = TinyModel()
    model.linear_attn = torch.nn.Module()
    model.linear_attn.in_proj_a = torch.nn.Linear(4, 4)
    groups = group_fsdp2_muon_parameters(model)
    names = {g["algorithm"]: g["param_names"] for g in groups}
    assert "linear_attn.in_proj_a.weight" in names["adamw"]
    assert "proj.weight" in names["muon"]


def test_engine_entry_and_default_adamw_equivalence():
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    model = TinyModel()
    other = copy.deepcopy(model)
    config = FSDPOptimizerConfig(lr=0.01, betas=(0.8, 0.9), weight_decay=0.02)
    engine = SimpleNamespace(optimizer_config=config, engine_config=context(), model_config=None)
    opt = FSDPEngine._build_optimizer(engine, model)
    reference = torch.optim.AdamW(other.parameters(), lr=0.01, betas=(0.8, 0.9), weight_decay=0.02)
    for m, o in [(model, opt), (other, reference)]:
        m(torch.tensor([[1, 2]])).square().mean().backward()
        o.step()
    for p, q in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    dion = pytest.importorskip("dion", reason="real optional Dion dependency missing")
    engine.optimizer_config = FSDPOptimizerConfig(optimizer="Muon")
    assert isinstance(FSDPEngine._build_optimizer(engine, model), dion.Muon)


@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("scale", [None, "spectral_norm", "rms_norm"])
def test_real_unsharded_reference_formula(nesterov, scale):
    dion = pytest.importorskip("dion", reason="real optional Dion dependency missing")
    model = TinyModel()
    reference = copy.deepcopy(model)
    opt = build(model, muon_nesterov=nesterov, muon_adjust_lr=scale)
    # Independent expected semantic grouping, not the production grouping helper.
    ref_opt = dion.Muon(
        [
            {"params": [reference.proj.weight], "algorithm": "muon"},
            {
                "params": [p for name, p in reference.named_parameters() if p.requires_grad and name != "proj.weight"],
                "algorithm": "adamw",
            },
        ],
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        mu=0.95,
        epsilon=1e-8,
        nesterov=nesterov,
        adjust_lr=scale,
        use_polar_express=False,
    )
    for step in range(5):
        for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
            if not p.requires_grad:
                continue
            grad = torch.sin(torch.arange(p.numel()).reshape(p.shape) + step).float()
            p.grad = grad.clone()
            q.grad = grad.clone()
        opt.step()
        ref_opt.step()
        for p, q in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            for key, value in opt.state[p].items():
                torch.testing.assert_close(value, ref_opt.state[q][key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "model_config",
    [
        SimpleNamespace(lora_rank=8),
        SimpleNamespace(lora_rank=0, lora_adapter_path="adapter"),
        SimpleNamespace(hf_config=SimpleNamespace(num_experts=4)),
    ],
)
def test_reject_lora_and_moe(model_config):
    from verl.utils.fsdp2_muon import validate_fsdp2_muon_config

    with pytest.raises(ValueError, match="LoRA|MoE"):
        validate_fsdp2_muon_config(FSDPOptimizerConfig(optimizer="Muon"), context(), model_config)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_reject_low_precision_stored_parameters(dtype):
    with pytest.raises(ValueError, match="FP32"):
        build(TinyModel().to(dtype))


def test_missing_module_metadata_and_partial_parameter_list_rejected():
    model = TinyModel()
    config = FSDPOptimizerConfig(optimizer="Muon")
    with pytest.raises(ValueError, match="module metadata"):
        build_optimizer(model.parameters(), config)
    with pytest.raises(ValueError, match="exactly"):
        build_optimizer([model.proj.weight], config, model=model, engine_config=context())


def test_real_closure_and_sparse_gradients_rejected_before_update():
    pytest.importorskip("dion", reason="real optional Dion dependency missing")
    model = TinyModel()
    opt = build(model)
    with pytest.raises(ValueError, match="closures"):
        opt.step(lambda: torch.tensor(1.0))
    model.proj.weight.grad = torch.eye(4).to_sparse()
    with pytest.raises(ValueError, match="dense gradients"):
        opt.step()
    assert all(g["step"] == 0 for g in opt.param_groups)


def test_attention_output_is_not_confused_with_lm_head():
    from verl.utils.fsdp2_muon import group_fsdp2_muon_parameters

    model = TinyModel()
    model.attention = torch.nn.Module()
    model.attention.output = torch.nn.Module()
    model.attention.output.dense = torch.nn.Linear(4, 4)
    names = {g["algorithm"]: g["param_names"] for g in group_fsdp2_muon_parameters(model)}
    assert "attention.output.dense.weight" in names["muon"]


@pytest.mark.parametrize("optimizer_name", ["AdamW", "Muon"])
def test_engine_scheduler_returns_host_scalar(optimizer_name):
    """Ray's CPU driver must receive a host metric even when Dion keeps LR on device."""
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    if optimizer_name == "Muon":
        pytest.importorskip("dion", reason="real optional Dion dependency missing")
    config = FSDPOptimizerConfig(optimizer=optimizer_name, lr=1e-3, lr_warmup_steps=2, total_training_steps=30)
    engine = SimpleNamespace(optimizer_config=config, engine_config=context(), model_config=None, rank=0)
    optimizer = FSDPEngine._build_optimizer(engine, TinyModel())
    engine.lr_scheduler = FSDPEngine._build_lr_scheduler(engine, optimizer)
    for expected in (5e-4, 1e-3):
        optimizer.step()
        metric = FSDPEngine.lr_scheduler_step(engine)
        assert type(metric) is float
        assert metric == pytest.approx(expected)
    if optimizer_name == "Muon":
        # The conversion is only for reporting; preserve upstream optimizer state.
        assert isinstance(optimizer.param_groups[0]["lr"], torch.Tensor)
    assert engine.lr_scheduler.last_epoch == 2
