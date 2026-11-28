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
"""Tests for ``optim.optimizer_in_backward`` (fused optimizer step, verl#144).

Both paths are exercised: the default ``zero_grad -> backward -> optimizer_step`` path must
be untouched (``test_disabled_by_default`` and the fact that every equivalence test compares
against it), and the opt-in path must reach the same parameters and grad norm.

Covered here:
* numeric equivalence with the standard path, single and multi micro-batch (gradient
  accumulation), with and without grad clipping;
* per-parameter gradient release (the memory win) in hook mode;
* non-finite grad-norm skip, including the atomic whole-step skip in hook mode and the
  fail-closed refusal when a non-finite gradient only shows up after a partial update;
* optimizer ``state_dict`` interchangeability with the standard path;
* parameter-sharding semantics on FSDP1 and FSDP2 (2 ranks, CPU gloo), including the
  FSDP1 fallback to per-parameter release after the backward pass (FSDP1 gradients bypass
  the autograd accumulator, so ``post_accumulate_grad_hook`` never fires there);

Out of scope for this change (already implemented elsewhere): activation offloading
(``model.enable_activation_offload`` / ``verl/utils/activation_offload.py``) and
``fsdp_config.reshard_after_forward`` / ``enable_full_shard``.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, fully_shard
from torch.distributed.tensor import DTensor

from verl.utils.optimizer_in_backward import (
    OptimizerInBackwardWrapper,
    create_optim_in_bwd_wrapper,
    supports_step_in_hooks,
)
from verl.workers.config import FSDPOptimizerConfig
from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

HIDDEN = 4


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _build_model(seed: int = 2026):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(HIDDEN, 8), torch.nn.GELU(), torch.nn.Linear(8, 2))


def _build_batches(num_micro_batches: int = 3):
    generator = torch.Generator().manual_seed(7)
    return [
        (torch.randn(4, HIDDEN, generator=generator), torch.randn(4, 2, generator=generator))
        for _ in range(num_micro_batches)
    ]


def _loss(model, inputs, targets):
    return torch.nn.functional.mse_loss(model(inputs), targets, reduction="sum")


def _build_optimizer(model, clip_grad):
    return torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.01), FSDPOptimizerConfig(
        lr=0.05, clip_grad=clip_grad
    )


def _standard_step(model, batches, clip_grad):
    """The unchanged reference path: zero_grad -> backward -> clip + optimizer_step."""
    optimizer, _ = _build_optimizer(model, clip_grad)
    optimizer.zero_grad()
    for inputs, targets in batches:
        _loss(model, inputs, targets).backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad) if clip_grad is not None else None
    if clipped is not None:
        grad_norm = clipped
    optimizer.step()
    return optimizer, grad_norm.item()


def _in_backward_step(model, batches, clip_grad, *, hook_mode):
    """The opt-in path, driven exactly like FSDPEngine.forward_backward_batch drives it."""
    optimizer, _ = _build_optimizer(model, clip_grad)
    wrapper = create_optim_in_bwd_wrapper(optimizer, step_in_hooks=hook_mode)
    wrapper.zero_grad()
    for index, (inputs, targets) in enumerate(batches):
        wrapper.begin_backward(stepping=index == len(batches) - 1)
        _loss(model, inputs, targets).backward()
    if clip_grad is None:
        grad_norm = wrapper.finalize()
    else:
        grad_norm = wrapper.finalize(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad).item())
    return wrapper, grad_norm


# --------------------------------------------------------------------------------------
# wrapper semantics
# --------------------------------------------------------------------------------------
def test_disabled_by_default():
    config = FSDPOptimizerConfig()
    assert config.optimizer_in_backward is False
    assert config.clip_grad == 1.0


def test_build_optimizer_wraps_only_when_enabled():
    model = _build_model()
    engine = object.__new__(FSDPEngine)
    engine.engine_config = SimpleNamespace(strategy="fsdp")
    engine.scaler = None
    engine.device_mesh = None
    engine.optimizer_config = FSDPOptimizerConfig()

    assert not isinstance(engine._build_optimizer(model), OptimizerInBackwardWrapper)

    engine.optimizer_config = FSDPOptimizerConfig(optimizer_in_backward=True)
    assert isinstance(engine._build_optimizer(model), OptimizerInBackwardWrapper)


def test_build_optimizer_rejects_fp16_scaler_and_fsdp_turbo():
    model = _build_model()
    engine = object.__new__(FSDPEngine)
    engine.engine_config = SimpleNamespace(strategy="fsdp")
    engine.device_mesh = None
    engine.scaler = None
    engine.optimizer_config = FSDPOptimizerConfig(optimizer_in_backward=True)

    engine.scaler = object()
    with pytest.raises(ValueError, match="loss scaling"):
        engine._build_optimizer(model)

    engine.scaler = None
    engine.engine_config = SimpleNamespace(strategy="fsdp_turbo")
    with pytest.raises(ValueError, match="fsdp_turbo"):
        engine._build_optimizer(model)


def test_hook_mode_matches_standard_path_without_clipping():
    reference_model = _build_model()
    _, reference_grad_norm = _standard_step(reference_model, _build_batches(), clip_grad=None)

    model = _build_model()
    _, grad_norm = _in_backward_step(model, _build_batches(), clip_grad=None, hook_mode=True)

    for reference, actual in zip(reference_model.parameters(), model.parameters(), strict=True):
        assert torch.equal(reference, actual)
    assert grad_norm == pytest.approx(reference_grad_norm, rel=1e-6)


def test_hook_mode_matches_standard_path_with_no_op_clip():
    """``clip_grad=1e9`` never scales the gradients, so it isolates the fusion itself."""
    reference_model = _build_model()
    _, reference_grad_norm = _standard_step(reference_model, _build_batches(), clip_grad=1e9)

    model = _build_model()
    _, grad_norm = _in_backward_step(model, _build_batches(), clip_grad=None, hook_mode=True)

    for reference, actual in zip(reference_model.parameters(), model.parameters(), strict=True):
        assert torch.equal(reference, actual)
    assert grad_norm == pytest.approx(reference_grad_norm, rel=1e-6)


def test_clipped_mode_matches_standard_path_exactly():
    """With a clip configured the global norm is only known after backward, so the step is
    taken right after the last backward; it must reproduce the standard path bit for bit."""
    reference_model = _build_model()
    _, reference_grad_norm = _standard_step(reference_model, _build_batches(), clip_grad=0.05)

    model = _build_model()
    _, grad_norm = _in_backward_step(model, _build_batches(), clip_grad=0.05, hook_mode=False)

    for reference, actual in zip(reference_model.parameters(), model.parameters(), strict=True):
        assert torch.equal(reference, actual)
    assert grad_norm == pytest.approx(reference_grad_norm, rel=1e-6)


def test_hook_mode_releases_gradients_immediately():
    """The memory win: nothing is left holding the mini-batch's gradients afterwards."""
    model = _build_model()
    optimizer, _ = _build_optimizer(model, clip_grad=None)
    wrapper = create_optim_in_bwd_wrapper(optimizer, step_in_hooks=True)
    wrapper.zero_grad()

    batches = _build_batches()
    for index, (inputs, targets) in enumerate(batches):
        wrapper.begin_backward(stepping=index == len(batches) - 1)
        _loss(model, inputs, targets).backward()
        if index < len(batches) - 1:
            # Accumulation micro-batches must keep their gradients for the final step.
            assert all(parameter.grad is not None for parameter in model.parameters())

    assert all(parameter.grad is None for parameter in model.parameters())
    wrapper.finalize()


def test_zero_grad_disarms_pending_hook_stepping():
    """A failed backward must not leave the hooks armed for the next micro-batch."""
    model = _build_model()
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer, _ = _build_optimizer(model, clip_grad=None)
    wrapper = create_optim_in_bwd_wrapper(optimizer, step_in_hooks=True)

    wrapper.begin_backward(stepping=True)
    wrapper.zero_grad()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
        wrapper._on_grad_ready(parameter)

    for before, after in zip(initial, model.parameters(), strict=True):
        assert torch.equal(before, after)


def test_accumulates_every_micro_batch_before_stepping():
    """Gradient accumulation must sum every micro-batch before the single update."""
    reference_model = _build_model()
    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=0.05, weight_decay=0.01)
    reference_optimizer.zero_grad()
    batches = _build_batches()
    for inputs, targets in batches:
        _loss(reference_model, inputs, targets).backward()
    reference_grads = [parameter.grad.clone() for parameter in reference_model.parameters()]

    model = _build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.01)
    wrapper = create_optim_in_bwd_wrapper(optimizer, step_in_hooks=False)
    wrapper.zero_grad()
    for index, (inputs, targets) in enumerate(batches):
        wrapper.begin_backward(stepping=index == len(batches) - 1)
        _loss(model, inputs, targets).backward()

    for reference, actual in zip(reference_grads, model.parameters(), strict=True):
        assert torch.equal(reference, actual.grad)


def test_non_finite_grad_norm_skips_the_update():
    model = _build_model()
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer, _ = _build_optimizer(model, clip_grad=0.5)
    wrapper = create_optim_in_bwd_wrapper(optimizer, step_in_hooks=False)
    wrapper.zero_grad()

    batches = _build_batches()
    for index, (inputs, targets) in enumerate(batches):
        wrapper.begin_backward(stepping=index == len(batches) - 1)
        _loss(model, inputs, targets).backward()
    model[0].weight.grad.fill_(float("nan"))
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    assert not torch.isfinite(grad_norm)

    wrapper.finalize(grad_norm.item(), step=False)

    for before, after in zip(initial, model.parameters(), strict=True):
        assert torch.equal(before, after)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_hook_mode_skips_the_whole_update_when_the_first_grad_is_non_finite():
    """An exploding loss makes every gradient non-finite, so the first hook handles it
    before any parameter is stepped and the mini-batch is skipped atomically."""
    model = _build_model()
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer, _ = _build_optimizer(model, clip_grad=None)
    wrapper = create_optim_in_bwd_wrapper(optimizer, step_in_hooks=True)
    wrapper.zero_grad()
    wrapper.begin_backward(stepping=True)

    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, float("nan")) if index == 0 else torch.zeros_like(parameter)
        wrapper._on_grad_ready(parameter)

    grad_norm = wrapper.finalize()

    assert not torch.tensor(grad_norm).isfinite()
    for before, after in zip(initial, model.parameters(), strict=True):
        assert torch.equal(before, after)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_hook_mode_steps_leftover_accumulated_grads_and_counts_their_norm():
    """A parameter that only received gradients in earlier micro-batches fires no hook in the
    last one, but must still be stepped and contribute to the reported grad norm."""
    model = _build_model()
    optimizer, _ = _build_optimizer(model, clip_grad=None)
    wrapper = create_optim_in_bwd_wrapper(optimizer, step_in_hooks=True)
    wrapper.zero_grad()
    wrapper.begin_backward(stepping=True)

    parameters = list(model.parameters())
    parameters[0].grad = torch.ones_like(parameters[0])
    parameters[-1].grad = torch.full_like(parameters[-1], 2.0)
    expected_norm = (
        sum(parameter.grad.double().pow(2).sum() for parameter in parameters if parameter.grad is not None)
        .sqrt()
        .item()
    )
    # Only the last parameter is reached by the final backward, so only it fires a hook.
    wrapper._on_grad_ready(parameters[-1])

    grad_norm = wrapper.finalize()

    assert grad_norm == pytest.approx(expected_norm, rel=1e-6)
    assert all(parameter.grad is None for parameter in parameters)


def test_hook_mode_skips_the_update_when_only_a_leftover_grad_is_non_finite():
    model = _build_model()
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer, _ = _build_optimizer(model, clip_grad=None)
    wrapper = create_optim_in_bwd_wrapper(optimizer, step_in_hooks=True)
    wrapper.zero_grad()
    wrapper.begin_backward(stepping=True)

    # No hook fires for this gradient, so only finalize can see it.
    model[0].weight.grad = torch.full_like(model[0].weight, float("nan"))

    grad_norm = wrapper.finalize()

    assert not torch.tensor(grad_norm).isfinite()
    for before, after in zip(initial, model.parameters(), strict=True):
        assert torch.equal(before, after)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_hook_mode_refuses_a_partial_update_when_a_later_grad_is_non_finite():
    """Parameters stepped in the backward pass cannot be rolled back, so a late non-finite
    gradient fails closed instead of leaving a half-updated model behind."""
    model = _build_model()
    optimizer, _ = _build_optimizer(model, clip_grad=None)
    wrapper = create_optim_in_bwd_wrapper(optimizer, step_in_hooks=True)
    wrapper.zero_grad()
    wrapper.begin_backward(stepping=True)

    parameters = list(model.parameters())
    parameters[0].grad = torch.zeros_like(parameters[0])
    wrapper._on_grad_ready(parameters[0])
    parameters[-1].grad = torch.full_like(parameters[-1], float("inf"))
    wrapper._on_grad_ready(parameters[-1])

    with pytest.raises(RuntimeError, match="cannot be rolled back"):
        wrapper.finalize()


def test_finalize_measures_the_grad_norm_when_the_caller_does_not():
    """FSDP1 has no grad hooks, so the no-clip fallback asks the wrapper for the norm."""
    model = _build_model()
    optimizer, _ = _build_optimizer(model, clip_grad=None)
    wrapper = create_optim_in_bwd_wrapper(optimizer, step_in_hooks=False)
    wrapper.zero_grad()

    batches = _build_batches()
    for index, (inputs, targets) in enumerate(batches):
        wrapper.begin_backward(stepping=index == len(batches) - 1)
        _loss(model, inputs, targets).backward()

    expected = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
    grad_norm = wrapper.finalize()

    assert grad_norm == pytest.approx(expected, rel=1e-6)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_supports_step_in_hooks_on_plain_modules():
    assert supports_step_in_hooks(_build_model()) is True


def test_state_dict_is_interchangeable_with_the_standard_optimizer():
    reference_model = _build_model()
    reference_optimizer, _ = _standard_step(reference_model, _build_batches(), clip_grad=None)

    model = _build_model()
    wrapper, _ = _in_backward_step(model, _build_batches(), clip_grad=None, hook_mode=True)

    reference_state = reference_optimizer.state_dict()
    state = wrapper.state_dict()
    assert sorted(state["state"]) == sorted(reference_state["state"])
    assert len(state["param_groups"]) == len(reference_state["param_groups"])
    for index in reference_state["state"]:
        for key, value in reference_state["state"][index].items():
            assert torch.equal(value, state["state"][index][key])

    fresh_model = _build_model()
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=0.05, weight_decay=0.01)
    fresh_optimizer.load_state_dict(state)
    assert sorted(fresh_optimizer.state_dict()["state"]) == sorted(reference_state["state"])


def test_lr_scheduler_updates_are_picked_up():
    """The wrapper snapshots the optimizer's groups before every step, so scheduler-set LRs apply."""
    model = _build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5**step)
    wrapper = create_optim_in_bwd_wrapper(optimizer, step_in_hooks=True)
    inputs, targets = _build_batches(1)[0]

    def step_and_measure_update():
        before = [parameter.detach().clone() for parameter in model.parameters()]
        wrapper.zero_grad()
        wrapper.begin_backward(stepping=True)
        _loss(model, inputs, targets).backward()
        wrapper.finalize()
        return max((after - prior).abs().max().item() for prior, after in zip(before, model.parameters(), strict=True))

    first_update = step_and_measure_update()
    assert first_update > 0

    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    second_update = step_and_measure_update()
    # AdamW's first-step update magnitude is ~lr, so halving the LR halves the update.
    assert second_update == pytest.approx(0.5 * first_update, rel=0.1)


# --------------------------------------------------------------------------------------
# engine integration (forward_backward_batch orchestration)
# --------------------------------------------------------------------------------------
class _StubEngine(FSDPEngine):
    """Runs the real ``forward_backward_batch`` loop on a tiny unsharded model."""

    def __init__(self, model, optimizer, optimizer_config):
        self.module = model
        self.optimizer = optimizer
        self.optimizer_config = optimizer_config
        self.engine_config = SimpleNamespace(use_no_sync_for_gradient_accumulation=False)
        self.ulysses_sequence_parallel_size = 1
        self.scaler = None
        self._fused_grad_norm = None

    def forward_step(self, micro_batch, loss_function, forward_only):
        return _loss(self.module, micro_batch["x"], micro_batch["y"]), {"metrics": {}}

    def get_data_parallel_group(self):
        return None

    def get_data_parallel_size(self):
        return 1


def _run_engine_step(monkeypatch, optimizer_in_backward, clip_grad, batches, *, non_finite=False):
    model = _build_model()
    optimizer_config = FSDPOptimizerConfig(optimizer_in_backward=optimizer_in_backward, clip_grad=clip_grad)
    engine = _StubEngine(model, None, optimizer_config)
    engine.optimizer = engine._build_optimizer(model)

    micro_batches = [TensorDict({"x": inputs, "y": targets}, batch_size=[len(inputs)]) for inputs, targets in batches]

    def forward_step(micro_batch, loss_function, forward_only):
        loss = _loss(engine.module, micro_batch["x"], micro_batch["y"])
        # A non-finite loss replicates the failure mode the skip-path exists for.
        if non_finite and micro_batch is micro_batches[-1]:
            loss = loss * float("nan")
        return loss, {"metrics": {}}

    engine.forward_step = forward_step

    monkeypatch.setattr(transformer_impl, "prepare_micro_batches", lambda **_: (micro_batches, None))
    monkeypatch.setattr(transformer_impl, "postprocess_batch_func", lambda output_lst, indices, data: {"metrics": {}})
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)

    data = TensorDict({"loss_mask": torch.ones(8)}, batch_size=[8])
    engine.optimizer_zero_grad()
    engine.forward_backward_batch(data, loss_function=lambda: None)
    grad_norm = engine.optimizer_step()
    return [parameter.detach().clone() for parameter in model.parameters()], grad_norm


def test_engine_fused_matches_standard_with_clipping(monkeypatch):
    reference_params, reference_grad_norm = _run_engine_step(
        monkeypatch, optimizer_in_backward=False, clip_grad=0.05, batches=_build_batches()
    )
    fused_params, fused_grad_norm = _run_engine_step(
        monkeypatch, optimizer_in_backward=True, clip_grad=0.05, batches=_build_batches()
    )

    for reference, actual in zip(reference_params, fused_params, strict=True):
        assert torch.equal(reference, actual)
    assert fused_grad_norm == pytest.approx(reference_grad_norm, rel=1e-6)


def test_engine_fused_matches_standard_without_clipping(monkeypatch):
    reference_params, reference_grad_norm = _run_engine_step(
        monkeypatch, optimizer_in_backward=False, clip_grad=1e9, batches=_build_batches()
    )
    fused_params, fused_grad_norm = _run_engine_step(
        monkeypatch, optimizer_in_backward=True, clip_grad=None, batches=_build_batches()
    )

    for reference, actual in zip(reference_params, fused_params, strict=True):
        assert torch.equal(reference, actual)
    assert fused_grad_norm == pytest.approx(reference_grad_norm, rel=1e-6)


@pytest.mark.parametrize("clip_grad", [None, 0.05])
def test_engine_non_finite_grad_skips_the_update(monkeypatch, clip_grad):
    initial = [parameter.detach().clone() for parameter in _build_model().parameters()]
    params, grad_norm = _run_engine_step(
        monkeypatch,
        optimizer_in_backward=True,
        clip_grad=clip_grad,
        batches=_build_batches(),
        non_finite=True,
    )
    assert not torch.tensor(grad_norm).isfinite()
    for before, after in zip(initial, params, strict=True):
        assert torch.equal(before, after)


def test_engine_step_requires_forward_backward(monkeypatch):
    model = _build_model()
    optimizer_config = FSDPOptimizerConfig(optimizer_in_backward=True, clip_grad=None)
    engine = _StubEngine(model, None, optimizer_config)
    engine.optimizer = engine._build_optimizer(model)
    with pytest.raises(RuntimeError, match="forward_backward_batch"):
        engine.optimizer_step()


# --------------------------------------------------------------------------------------
# FSDP1 / FSDP2 parameter sharding
# --------------------------------------------------------------------------------------
def _clip_grad_norm(model, strategy, max_norm):
    if strategy == "fsdp":
        return model.clip_grad_norm_(max_norm).item()
    from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_

    grad_norm = fsdp2_clip_grad_norm_(model.parameters(), max_norm=max_norm)
    return (grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm).item()


def _full_params(model, strategy):
    if strategy == "fsdp":
        with FSDP.summon_full_params(model):
            return [parameter.detach().clone() for parameter in model.parameters()]
    return [parameter.full_tensor().detach().clone() for parameter in model.parameters()]


def _build_sharded_model(strategy, mesh):
    model = _build_model()
    if strategy == "fsdp":
        return FSDP(model, device_id=torch.device("cpu"), sharding_strategy=ShardingStrategy.FULL_SHARD)
    fully_shard(model, mesh=mesh)
    return model


def _sharded_step(model, strategy, batches, *, optimizer_in_backward, hook_mode, clip_grad, group):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.01)
    if optimizer_in_backward:
        wrapper = create_optim_in_bwd_wrapper(optimizer, step_in_hooks=hook_mode, grad_norm_group=group)
        wrapper.zero_grad()
        for index, (inputs, targets) in enumerate(batches):
            wrapper.begin_backward(stepping=index == len(batches) - 1)
            _loss(model, inputs, targets).backward()
        if clip_grad is None:
            grad_norm = wrapper.finalize()
        else:
            grad_norm = wrapper.finalize(_clip_grad_norm(model, strategy, clip_grad))
        return _full_params(model, strategy), grad_norm

    optimizer.zero_grad()
    for inputs, targets in batches:
        _loss(model, inputs, targets).backward()
    grad_norm = _clip_grad_norm(model, strategy, clip_grad if clip_grad is not None else float("inf"))
    optimizer.step()
    return _full_params(model, strategy), grad_norm


def _sharded_worker(rank, world_size, rendezvous_file):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        import verl.utils.fsdp_utils as fsdp_utils

        fsdp_utils.get_device_id = lambda: "cpu"
        transformer_impl.get_device_id = lambda: "cpu"
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=["fsdp"])
        group = mesh.get_group(mesh_dim="fsdp")
        batches = _build_batches(2)

        for strategy in ("fsdp", "fsdp2"):
            # FSDP1 cannot step from grad hooks; the engine must fall back for it.
            hooks_supported = supports_step_in_hooks(_build_sharded_model(strategy, mesh))
            assert hooks_supported is (strategy == "fsdp2"), (strategy, hooks_supported)

            engine = object.__new__(FSDPEngine)
            engine.engine_config = SimpleNamespace(strategy="fsdp")
            engine.device_mesh = mesh
            engine.scaler = None
            engine.optimizer_config = FSDPOptimizerConfig(clip_grad=None)
            probe_model = _build_sharded_model(strategy, mesh)
            probe_optimizer = torch.optim.AdamW(probe_model.parameters(), lr=0.05)
            wrapped = engine._wrap_optimizer_in_backward(probe_optimizer, probe_model)
            assert wrapped.step_in_hooks is hooks_supported, (strategy, wrapped.step_in_hooks)

            for clip_grad, hook_mode in ((None, True), (0.05, False)):
                hook_mode = hook_mode and hooks_supported
                reference_params, reference_grad_norm = _sharded_step(
                    _build_sharded_model(strategy, mesh),
                    strategy,
                    batches,
                    optimizer_in_backward=False,
                    hook_mode=False,
                    clip_grad=clip_grad,
                    group=group,
                )
                fused_params, fused_grad_norm = _sharded_step(
                    _build_sharded_model(strategy, mesh),
                    strategy,
                    batches,
                    optimizer_in_backward=True,
                    hook_mode=hook_mode,
                    clip_grad=clip_grad,
                    group=group,
                )

                assert fused_grad_norm == pytest.approx(reference_grad_norm, rel=1e-5), (
                    strategy,
                    clip_grad,
                    fused_grad_norm,
                    reference_grad_norm,
                )
                for reference, actual in zip(reference_params, fused_params, strict=True):
                    torch.testing.assert_close(reference, actual, rtol=1e-5, atol=1e-6)
    finally:
        dist.destroy_process_group()


def test_sharded_parameters_match_standard_path(tmp_path):
    world_size = 2
    mp.spawn(
        _sharded_worker,
        args=(world_size, str(tmp_path / "optimizer_in_backward_rdzv")),
        nprocs=world_size,
        join=True,
    )
