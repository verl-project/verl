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
"""Async worker stages must remain inside their profiling context across awaits."""

import asyncio
import importlib
import inspect
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from verl.utils.profiler.profile import DistProfiler


@pytest.fixture(params=["torch", "npu", "nsys"])
def backend(request, monkeypatch, tmp_path):
    """Exercise real backend decorators with mocked collectors and markers."""
    events = []
    collector = MagicMock()
    collector.start.side_effect = lambda: events.append("start")
    collector.step.side_effect = lambda: events.append("step")
    collector.stop.side_effect = lambda: events.append("stop")
    if request.param == "torch":
        from verl.utils.profiler import torch_profile as module

        cls = module.Profiler
        factory = "get_torch_profiler"

        @contextmanager
        def record_function(name):
            events.append("range_start")
            try:
                yield
            finally:
                events.append("range_end")

        monkeypatch.setattr(module.torch.profiler, "record_function", record_function)
    else:
        pytest.importorskip("torch_npu" if request.param == "npu" else "nvtx")
        module = importlib.import_module(
            "verl.utils.profiler." + ("mstx_profile" if request.param == "npu" else "nvtx_profile")
        )
        cls = module.NPUProfiler if request.param == "npu" else module.NsightSystemsProfiler
        factory = "get_npu_profiler" if request.param == "npu" else None
        monkeypatch.setattr(module, "mark_start_range", lambda **kw: events.append("range_start") or 1)
        monkeypatch.setattr(module, "mark_end_range", lambda token: events.append("range_end"))
        if request.param == "nsys":
            platform = SimpleNamespace(profiler_start=collector.start, profiler_stop=collector.stop)
            monkeypatch.setattr(module, "get_platform", lambda: platform)
    if factory:
        monkeypatch.setattr(module, factory, MagicMock(return_value=collector))
    profiler = cls.__new__(cls)
    for key, value in dict(
        discrete=True,
        contents=["cpu"],
        save_path=str(tmp_path),
        save_file_prefix="actor",
        rank=0,
        _profile_step=2,
        profile_contents=["cpu"],
        profile_level="level0",
        profile_save_path=str(tmp_path),
        analysis=False,
        _schedule_kwargs=None,
    ).items():
        setattr(profiler, key, value)
    return profiler, events, collector, request.param


def _worker(impl, gates=(True, True, True)):
    return SimpleNamespace(
        profiler=SimpleNamespace(
            rank=0,
            _impl=impl,
            check_enable=lambda: gates[0],
            check_this_step=lambda: gates[1],
            check_this_rank=lambda: gates[2],
        )
    )


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("outcome", ["return", "error", "cancel"])
def test_async_stage_keeps_collector_open_across_await(backend, discrete, outcome):
    impl, events, collector, tool = backend
    impl.discrete = discrete

    @DistProfiler.annotate(role="update_weights")
    async def update_weights(self, value):
        events.append("body")
        assert events[-2] == "range_start"
        await asyncio.sleep(0)
        events.append("awaited")
        assert "range_end" not in events
        if outcome == "error":
            raise RuntimeError("weight transfer failed")
        if outcome == "cancel":
            raise asyncio.CancelledError
        return value

    assert inspect.iscoroutinefunction(update_weights)
    coroutine = update_weights(_worker(impl), 42)
    assert events == []  # Merely creating a coroutine must not start/stop collection.
    if outcome == "return":
        assert asyncio.run(coroutine) == 42
    else:
        expected = RuntimeError if outcome == "error" else asyncio.CancelledError
        with pytest.raises(expected):
            asyncio.run(coroutine)
    assert events.count("body") == 1
    assert events.count("awaited") == 1
    assert events.index("range_end") > events.index("awaited")
    if discrete:
        assert events[0] == "start"
        assert events[-1] == "stop"
        collector.start.assert_called_once()
        collector.stop.assert_called_once()
        if tool == "npu":
            assert events[-2:] == ["range_end", "stop"]
            collector.step.assert_not_called()
            assert impl.profile_npu is None
            assert type(impl)._active_prof is None
    else:
        collector.start.assert_not_called()
        collector.stop.assert_not_called()


@pytest.mark.parametrize("gates", [(False, True, True), (True, False, True), (True, True, False)])
def test_async_stage_respects_enable_step_and_rank(backend, gates):
    impl, events, collector, _ = backend

    @DistProfiler.annotate(role="update_weights")
    async def update_weights(self):
        events.append("body")
        await asyncio.sleep(0)
        return 42

    assert asyncio.run(update_weights(_worker(impl, gates))) == 42
    assert events == ["body"]
    collector.start.assert_not_called()


@pytest.mark.parametrize("outcome", ["return", "error"])
@pytest.mark.parametrize("has_profiler", [False, True])
def test_async_dispatcher_without_profiler_or_with_failed_setup(outcome, has_profiler):
    class BadBackend:
        def annotate(self, **kwargs):
            raise RuntimeError("setup failed")

    worker = _worker(BadBackend()) if has_profiler else SimpleNamespace()
    calls = []

    @DistProfiler.annotate(role="update_weights")
    async def update_weights(self):
        calls.append(1)
        await asyncio.sleep(0)
        if outcome == "error":
            raise ValueError("worker failed")
        return 42

    if outcome == "return":
        assert asyncio.run(update_weights(worker)) == 42
    else:
        with pytest.raises(ValueError, match="worker failed"):
            asyncio.run(update_weights(worker))
    assert calls == [1]


def test_worker_weight_update_uses_async_profiler(backend):
    from verl.workers.engine_workers import ActorRolloutRefWorker

    impl, events, collector, _ = backend
    worker = _worker(impl)
    engine = MagicMock()
    engine.get_per_tensor_param.return_value = (iter(()), None)
    engine.is_param_offload_enabled = False
    worker.actor = SimpleNamespace(engine=engine)

    async def send_weights(weights, global_steps):
        events.append("send")
        await asyncio.sleep(0)
        assert "stop" not in events
        assert global_steps == 2
        return {"sent": 1}

    worker.checkpoint_engine = SimpleNamespace(send_weights=send_weights)
    assert inspect.iscoroutinefunction(ActorRolloutRefWorker.update_weights)
    result = asyncio.run(ActorRolloutRefWorker.update_weights(worker, global_steps=2, mode="broadcast"))
    assert result == {"sent": 1}
    assert events.index("send") < events.index("stop")
    collector.stop.assert_called_once()


@pytest.mark.parametrize("outcome", ["return", "error"])
def test_sync_stage_preserves_results_and_closes_on_error(backend, outcome):
    impl, events, collector, _ = backend

    @DistProfiler.annotate(role="update_weights")
    def update_weights(self):
        events.append("body")
        if outcome == "error":
            raise RuntimeError("weight transfer failed")
        return 42

    assert not inspect.iscoroutinefunction(update_weights)
    if outcome == "return":
        assert update_weights(_worker(impl)) == 42
    else:
        with pytest.raises(RuntimeError, match="weight transfer failed"):
            update_weights(_worker(impl))
    assert events.count("body") == 1
    assert events[-1] == "stop"
    collector.stop.assert_called_once()


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("outcome", ["return", "error", "cancel"])
def test_npu_schedule_and_async_stage_cleanup(monkeypatch, outcome, scheduled):
    """Mini-batch schedules survive awaits; unscheduled weight transfers collect in full."""
    pytest.importorskip("torch_npu")
    from verl.utils.profiler import mstx_profile as module
    from verl.utils.profiler.config import NPUToolConfig, ProfilerConfig, TorchProfilerScheduleConfig

    collector = MagicMock()
    factory = MagicMock(return_value=collector)
    monkeypatch.setattr(module, "get_npu_profiler", factory)
    monkeypatch.setattr(module, "mark_start_range", lambda **kw: 1)
    monkeypatch.setattr(module, "mark_end_range", lambda token: None)
    monkeypatch.setattr(module.NPUProfiler, "_active_prof", None)
    config = ProfilerConfig(enable=True, all_ranks=True, tool="npu")
    tool_config = NPUToolConfig(discrete=True, schedule=TorchProfilerScheduleConfig(wait=0, warmup=0, active=2))
    outer = DistProfiler(rank=0, config=config, tool_config=tool_config)
    inner = DistProfiler(rank=0, config=config, tool_config=tool_config)
    outer.start()
    worker = SimpleNamespace(profiler=outer)

    @DistProfiler.annotate(role="actor_update" if scheduled else "update_weights", scheduled=scheduled)
    async def stage(self):
        assert module.NPUProfiler._active_prof is collector
        if scheduled:
            inner.step()
        await asyncio.sleep(0)
        collector.stop.assert_not_called()
        assert module.NPUProfiler._active_prof is collector
        if scheduled:
            inner.step()
        if outcome == "error":
            raise RuntimeError("stage failed")
        if outcome == "cancel":
            raise asyncio.CancelledError
        return 42

    if outcome == "return":
        assert asyncio.run(stage(worker)) == 42
    else:
        with pytest.raises(RuntimeError if outcome == "error" else asyncio.CancelledError):
            asyncio.run(stage(worker))
    assert factory.call_args.kwargs["schedule"] == (outer._impl._schedule_kwargs if scheduled else None)
    assert collector.step.call_count == (2 if scheduled else 0)
    collector.stop.assert_called_once()
    assert outer._impl.profile_npu is None
    assert module.NPUProfiler._active_prof is None
    outer.stop()
