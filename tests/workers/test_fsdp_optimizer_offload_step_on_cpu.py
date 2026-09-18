# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace

import pytest
import torch

from verl.workers.engine import base
from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def _make_engine(monkeypatch, *, enabled=False, param_offload=False, foreach=False):
    engine = object.__new__(FSDPEngine)
    engine.module = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        engine.module.weight.fill_(0.5)
    engine.optimizer = torch.optim.AdamW(engine.module.parameters(), lr=0.01, foreach=foreach)
    engine.optimizer_config = SimpleNamespace(clip_grad=1.0)
    engine.engine_config = SimpleNamespace(optimizer_offload_step=enabled)
    engine._is_offload_param = param_offload
    engine._is_offload_optimizer = True
    engine._qat_enabled = False
    engine.scaler = None
    engine.mode = None
    engine.ulysses_parallel_group = None
    events = []
    engine.to = lambda **kwargs: events.append(("transfer", kwargs))
    step = engine.optimizer.step

    def record_step():
        events.append(("step",))
        return step()

    engine.optimizer.step = record_step
    monkeypatch.setattr(base, "get_device_name", lambda: "cuda")
    monkeypatch.setattr(transformer_impl, "get_device_name", lambda: "cuda")
    monkeypatch.setattr(transformer_impl, "get_ulysses_sequence_parallel_group", lambda: None)
    monkeypatch.setattr(transformer_impl, "set_ulysses_sequence_parallel_group", lambda _: None)
    return engine, events


def _optimizer_transfers(events):
    return [event[1]["device"] for event in events if event[0] == "transfer" and event[1]["optimizer"]]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("param_offload", [False, True])
def test_optimizer_residency_and_parameter_transfers(monkeypatch, enabled, param_offload):
    engine, events = _make_engine(monkeypatch, enabled=enabled, param_offload=param_offload)
    with engine.train_mode():
        assert _optimizer_transfers(events) == ([] if enabled else ["cuda"])
        assert any(e[0] == "transfer" and e[1]["model"] for e in events) == param_offload
        engine.module(torch.ones(1, 2)).sum().backward()
        before_step = len(events)
        engine.optimizer_step()
        step_events = events[before_step:]
        assert [e[0] for e in step_events] == (["transfer", "step", "transfer"] if enabled else ["step"])
        assert _optimizer_transfers(step_events) == (["cuda", "cpu"] if enabled else [])
    assert _optimizer_transfers(events)[-1] == "cpu"
    assert engine._optimizer_offload_step is False


@pytest.mark.parametrize("enabled", [False, True])
def test_manual_offload_control_is_preserved(monkeypatch, enabled):
    engine, events = _make_engine(monkeypatch, enabled=enabled, param_offload=True)
    with engine.train_mode(disable_auto_offload=True):
        engine.module(torch.ones(1, 2)).sum().backward()
        engine.optimizer_step()
    assert events == [("step",)]


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_nonfinite_gradients_skip_update_without_state_transfer(monkeypatch, value):
    engine, events = _make_engine(monkeypatch, enabled=True)
    initial = engine.module.weight.detach().clone()
    with engine.train_mode():
        engine.module.weight.grad = torch.full_like(engine.module.weight, value)
        engine.optimizer_step()
        assert events == []
        assert not engine.optimizer.state
        torch.testing.assert_close(engine.module.weight, initial, rtol=0, atol=0)
    assert engine.module.weight.grad is None


def test_state_offloaded_when_optimizer_step_raises(monkeypatch):
    engine, events = _make_engine(monkeypatch, enabled=True)

    def fail_step():
        events.append(("step",))
        raise RuntimeError("update failed")

    engine.optimizer.step = fail_step
    with pytest.raises(RuntimeError, match="update failed"):
        with engine.train_mode():
            engine.module(torch.ones(1, 2)).sum().backward()
            engine.optimizer_step()
    assert [e[0] for e in events[:3]] == ["transfer", "step", "transfer"]
    assert _optimizer_transfers(events)[:2] == ["cuda", "cpu"]
    assert engine._optimizer_offload_step is False
    assert engine.module.weight.grad is None


@pytest.mark.parametrize("unsupported", ["optimizer", "scaler"])
def test_unsupported_opt_in_fails_before_loading(monkeypatch, unsupported):
    engine, events = _make_engine(monkeypatch, enabled=True)
    if unsupported == "optimizer":
        engine.optimizer = torch.optim.SGD(engine.module.parameters(), lr=0.1)
    else:
        engine.scaler = object()
    with pytest.raises(ValueError, match="ordinary AdamW"):
        with engine.train_mode():
            pytest.fail("unsupported configuration entered training")
    assert events == []
    assert engine.mode is None


def test_default_context_does_not_restrict_other_optimizers(monkeypatch):
    engine, _ = _make_engine(monkeypatch)
    engine.optimizer = torch.optim.SGD(engine.module.parameters(), lr=0.1)
    with engine.train_mode():
        engine.module(torch.ones(1, 2)).sum().backward()
        engine.optimizer_step()
    assert torch.all(engine.module.weight < 0.5)


@pytest.mark.parametrize("foreach", [False, True])
def test_scheduling_does_not_change_adamw_updates(monkeypatch, foreach):
    baseline, _ = _make_engine(monkeypatch, foreach=foreach)
    optimized, _ = _make_engine(monkeypatch, enabled=True, foreach=foreach)
    for engine in (baseline, optimized):
        with engine.train_mode():
            for _ in range(3):
                engine.optimizer_zero_grad()
                engine.module(torch.ones(1, 2)).square().sum().backward()
                engine.optimizer_step()
    torch.testing.assert_close(baseline.module.weight, optimized.module.weight, rtol=0, atol=0)
    a = baseline.optimizer.state[baseline.module.weight]
    b = optimized.optimizer.state[optimized.module.weight]
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_step_offload_context_restores_previous_state(monkeypatch):
    engine, _ = _make_engine(monkeypatch, enabled=True)
    engine._optimizer_offload_step = True
    with engine.train_mode(disable_auto_offload=True):
        assert engine._optimizer_offload_step is True
    assert engine._optimizer_offload_step is True


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("param_offload", [False, True])
def test_nested_worker_context_preserves_outer_policy(monkeypatch, enabled, param_offload):
    engine, events = _make_engine(monkeypatch, enabled=enabled, param_offload=param_offload)
    with engine.train_mode():
        for _ in range(3):
            before_inner = len(events)
            # TrainingWorker.train_mini_batch delegates to train_batch this way.
            with engine.train_mode(disable_auto_offload=True):
                engine.module(torch.ones(1, 2)).sum().backward()
                engine.optimizer_step()
            inner_events = events[before_inner:]
            assert _optimizer_transfers(inner_events) == (["cuda", "cpu"] if enabled else [])
            assert engine._optimizer_offload_step is enabled
            assert not any(e[0] == "transfer" and e[1]["model"] for e in inner_events)
    assert engine._optimizer_offload_step is False
