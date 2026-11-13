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

import builtins
from types import SimpleNamespace

import pytest
import torch

from verl.workers.engine_workers import TrainingWorker, _set_lr_internal


def test_client_lr_preserves_lora_plus_updates_across_scheduler_steps():
    a = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    b = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    opt_a = torch.optim.SGD([{"params": [a], "max_lr": 0.001}], lr=0.001)
    opt_b = torch.optim.SGD([{"params": [b], "max_lr": 0.016}], lr=0.016)
    # Megatron's chained/mixed-precision wrappers expose the same groups at
    # multiple levels. Updating them twice must not compound the multiplier.
    wrapped_b = SimpleNamespace(optimizer=opt_b, param_groups=opt_b.param_groups)
    optimizer = SimpleNamespace(
        chained_optimizers=[opt_a, wrapped_b], param_groups=opt_a.param_groups + opt_b.param_groups
    )

    def step():
        opt_a.step()
        opt_b.step()
        return None

    def scheduler_step():
        for group in optimizer.param_groups:
            group["lr"] = group["max_lr"]
        return 0.001

    worker = SimpleNamespace(
        engine=SimpleNamespace(
            optimizer=optimizer,
            optimizer_step=step,
            lr_scheduler_step=scheduler_step,
            optimizer_zero_grad=lambda: (opt_a.zero_grad(), opt_b.zero_grad()),
            is_mp_src_rank_with_outputs=lambda: True,
        ),
        optimizer_config=SimpleNamespace(lr=0.001),
        _client_lr=None,
    )
    for requested in (0.002, 0.003):
        TrainingWorker.set_lr(worker, requested)
        a.grad = torch.ones_like(a)
        b.grad = torch.ones_like(b)
        metrics = TrainingWorker.optim_step_only(worker)
        assert metrics["lr"] == requested
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([requested, 16 * requested])

    assert a.item() == pytest.approx(-0.005)
    assert b.item() == pytest.approx(-0.080)


def test_plain_torch_lr_override_does_not_import_megatron(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.001)
    real_import = builtins.__import__

    def without_megatron(name, *args, **kwargs):
        if name == "megatron" or name.startswith("megatron."):
            raise ModuleNotFoundError("Megatron is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_megatron)
    _set_lr_internal(optimizer, 0.002, 0.001)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert parameter.item() == pytest.approx(0.998)


def test_real_megatron_chained_optimizer_preserves_effective_config_overrides():
    chained_optimizer = pytest.importorskip("megatron.core.optimizer").ChainedOptimizer
    parameters = [torch.nn.Parameter(torch.tensor(1.0)) for _ in range(2)]
    config = SimpleNamespace(lr=0.002)
    children = []
    for parameter, maximum in zip(parameters, (0.002, 0.032), strict=True):
        optimizer = torch.optim.SGD([{"params": [parameter], "max_lr": maximum}], lr=maximum)
        children.append(SimpleNamespace(optimizer=optimizer, param_groups=optimizer.param_groups, config=config))
    optimizer = chained_optimizer(children)
    # Effective Megatron config overrides the worker's outer base LR.
    _set_lr_internal(optimizer, 0.003, 0.001)
    for parameter, child in zip(parameters, children, strict=True):
        parameter.grad = torch.ones_like(parameter)
        child.optimizer.step()
    assert [parameter.item() for parameter in parameters] == pytest.approx([0.997, 0.952])


def test_torch_scheduler_group_ratios_and_tensor_lr_are_preserved():
    parameters = [torch.nn.Parameter(torch.tensor(1.0)) for _ in range(2)]
    optimizer = torch.optim.SGD(
        [{"params": [parameters[0]], "lr": torch.tensor(0.001)}, {"params": [parameters[1]], "lr": 0.016}]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    lr_tensor = optimizer.param_groups[0]["lr"]
    for _ in range(2):
        _set_lr_internal(optimizer, 0.002, 0.001)
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] is lr_tensor
    assert [parameter.item() for parameter in parameters] == pytest.approx([0.996, 0.936])
