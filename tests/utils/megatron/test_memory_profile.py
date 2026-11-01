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

"""CPU-only checks for the CUDA ledger's traversal and missing-data semantics."""

import gc
import importlib.util
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class CudaLedgerTensor(torch.Tensor):
    """CPU storage with CUDA classification, so these checks never initialize CUDA."""

    @property
    def is_cuda(self):
        return True


class OptimizerChain:
    def __init__(self, *children):
        self.chained_optimizers = list(children)

    @property
    def optimizer(self):
        raise AssertionError("ChainedOptimizer has more than one optimizer")

    @property
    def state(self):
        raise AssertionError("Read child state instead of the chain proxy")


def tensor(count=4):
    return torch.empty(count, dtype=torch.float32).as_subclass(CudaLedgerTensor)


def model(*parameters):
    return SimpleNamespace(parameters=lambda: iter(parameters))


@pytest.fixture
def profile(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "verl/utils/megatron/memory_profile.py"
    spec = importlib.util.spec_from_file_location("memory_profile_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("DSV41_MEM_PROFILE", "1")
    monkeypatch.delenv("DSV41_MEM_PARTS", raising=False)
    monkeypatch.delenv("DSV41_MEM_SNAPSHOT", raising=False)
    monkeypatch.delenv("DSV41_MEM_HISTORY", raising=False)
    monkeypatch.setattr(module, "_ENABLED", True, raising=False)
    monkeypatch.setattr(module, "_rank_zero", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 1000)
    monkeypatch.setattr(module, "phase", lambda _: None, raising=False)
    monkeypatch.setattr(module, "_GB", 1)
    return module


def log_parts(profile, model_value, optimizer):
    callback = getattr(profile, "log_parts", None) or profile.step_end
    callback(model_value, optimizer)


def test_nested_chain_counts_each_state_once_and_keeps_other_keys(profile):
    master = tensor()
    state = {
        object(): {
            "exp_avg": tensor(),
            "exp_avg_sq": tensor(),
            "master_param": master,
            "step": tensor(1),
            "momentum_buffer": {"value": tensor(2)},
            "cpu_metadata": torch.empty(5),
        }
    }
    child = SimpleNamespace(state=state, shard_fp32_from_float16_groups=[[master]])
    empty = SimpleNamespace(state={})
    chain = OptimizerChain(OptimizerChain(child, empty), child)
    totals, reason = profile._sum_optimizer_states(chain)
    assert reason is None
    assert totals == {"optim_exp_avg": 16, "optim_exp_avg_sq": 16, "optim_master": 16, "optim_other": 12}


def test_distributed_master_group_and_forwarded_state(profile):
    raw = SimpleNamespace(state={object(): {"exp_avg": tensor(), "exp_avg_sq": tensor()}})
    child = SimpleNamespace(optimizer=raw, state=raw.state, shard_fp32_from_float16_groups=[[tensor()]])
    totals, reason = profile._sum_optimizer_states(OptimizerChain(child))
    assert reason is None
    assert totals == {"optim_exp_avg": 16, "optim_exp_avg_sq": 16, "optim_master": 16, "optim_other": 0}


def test_lora_frozen_weights_do_not_require_main_grad_and_residual_is_printed(profile, capsys):
    frozen, adapter = tensor(100), tensor()
    adapter.requires_grad_(True)
    adapter.main_grad = tensor()
    child = SimpleNamespace(
        state={object(): {"exp_avg": tensor(), "exp_avg_sq": tensor(), "master_param": tensor(), "extra": tensor(2)}}
    )
    log_parts(profile, model(frozen, adapter), OptimizerChain(child))
    output = capsys.readouterr().out
    assert "main_grads=16." in output
    assert "optim_other=8." in output
    assert "other=512." in output
    assert "n/a" not in output


@pytest.mark.parametrize("gradient_name", ["grad", "decoupled_grad"])
def test_main_grads_fall_back_to_all_optimizer_shards(profile, gradient_name):
    parameters = [tensor(), tensor(2)]
    for parameter in parameters:
        setattr(parameter, gradient_name, tensor(parameter.numel()))
    child_a = SimpleNamespace(param_groups=[{"params": [parameters[0]]}])
    child_b = SimpleNamespace(param_groups=[{"params": [parameters[1]]}])
    adapter = tensor().requires_grad_(True)
    total, reason = profile._sum_main_grads(model(adapter), OptimizerChain(child_a, child_b))
    assert (total, reason) == (24, None)


def test_unavailable_components_keep_residual_unavailable(profile, capsys):
    adapter = tensor().requires_grad_(True)
    child = SimpleNamespace(state={object(): {"exp_avg": tensor(), "exp_avg_sq": tensor(), "master_param": tensor()}})
    log_parts(profile, model(adapter), OptimizerChain(child))
    output = capsys.readouterr().out
    assert "main_grads=n/a" in output
    assert "other=n/a" in output
    assert "main gradients unavailable" in output


def test_unreadable_child_does_not_return_partial_state(profile):
    chain = OptimizerChain(SimpleNamespace(state={object(): {"exp_avg": tensor()}}), object())
    totals, reason = profile._sum_optimizer_states(chain)
    assert totals is None
    assert "state" in reason


@pytest.mark.parametrize("parts_enabled,rank_zero", [(False, True), (True, False)])
def test_disabled_or_nonzero_rank_does_not_walk_state(profile, monkeypatch, parts_enabled, rank_zero):
    monkeypatch.setenv("DSV41_MEM_PARTS", "1" if parts_enabled else "0")
    monkeypatch.setattr(profile, "_rank_zero", lambda: rank_zero)

    def fail(*args):
        raise AssertionError("Ledger traversal must be gated")

    monkeypatch.setattr(profile, "_iter_parameters", fail)
    monkeypatch.setattr(profile, "_iter_optimizers", fail)
    log_parts(profile, object(), object())


def test_accounting_does_not_retain_state_tensors(profile):
    state_tensor = tensor()
    reference = weakref.ref(state_tensor)
    child = SimpleNamespace(state={object(): {"exp_avg": state_tensor}})
    profile._sum_optimizer_states(OptimizerChain(child))
    del child, state_tensor
    gc.collect()
    assert reference() is None


def test_memory_history_records_before_step_and_dumps_then_disables(profile, tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(profile, "_history_rank_zero", lambda: True)
    monkeypatch.setattr(
        torch.cuda.memory,
        "_record_memory_history",
        lambda **kwargs: events.append(("record", kwargs)),
        raising=False,
    )
    monkeypatch.setattr(torch.cuda.memory, "_dump_snapshot", lambda path: events.append(("dump", path)), raising=False)

    path = tmp_path / "step.pickle"
    profile.start_memory_history(str(path))
    profile.finish_memory_history(str(path))

    assert events == [
        ("record", {"max_entries": profile._MEMORY_HISTORY_MAX_ENTRIES}),
        ("dump", str(path)),
        ("record", {"enabled": None}),
    ]


def test_memory_history_is_gated_to_rank_zero(profile, monkeypatch, tmp_path):
    monkeypatch.setattr(profile, "_history_rank_zero", lambda: False)
    monkeypatch.setattr(
        torch.cuda.memory,
        "_record_memory_history",
        lambda **_: (_ for _ in ()).throw(AssertionError()),
        raising=False,
    )
    monkeypatch.setattr(
        torch.cuda.memory,
        "_dump_snapshot",
        lambda _: (_ for _ in ()).throw(AssertionError()),
        raising=False,
    )

    profile.start_memory_history(str(tmp_path / "step.pickle"))
    profile.finish_memory_history(str(tmp_path / "step.pickle"))
