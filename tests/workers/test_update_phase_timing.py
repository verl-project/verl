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

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.metric.utils import promote_update_phase_metrics, reduce_metrics
from verl.workers.engine.base import BaseEngine
from verl.workers.engine_workers import TrainingWorker


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        value = self.now
        self.now += 1.0
        return value


class _RecordingEngine(BaseEngine):
    def __init__(self, report_metrics):
        self.report_metrics = report_metrics
        self.engine_config = SimpleNamespace(enable_update_phase_timing=True)
        self.events = []

    @property
    def is_param_offload_enabled(self):
        return False

    @property
    def is_optimizer_offload_enabled(self):
        return False

    def is_mp_src_rank_with_outputs(self):
        return self.report_metrics

    def optimizer_zero_grad(self):
        self.events.append("zero_grad")

    def optimizer_step(self):
        self.events.append("optimizer_step")
        return 0.25

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        self.events.append("forward_backward")
        return {"metrics": {}, "loss": [0.0], "model_output": {}}


def test_train_batch_times_forward_and_optimizer_separately(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr("verl.workers.engine.base.time.perf_counter", clock)
    monkeypatch.setattr("verl.workers.engine.base._synchronize_for_phase_timing", lambda: None)
    engine = _RecordingEngine(report_metrics=True)

    outputs = engine.train_batch(TensorDict({}, batch_size=[]), loss_function=None)

    # perf_counter advances once per read: zero_grad is 1s, forward-backward is 1s, optimizer.step is 1s.
    assert engine.events == ["zero_grad", "forward_backward", "optimizer_step"]
    assert outputs["metrics"]["timing_s/forward_backward"] == 1.0
    assert outputs["metrics"]["timing_s/optimizer"] == 2.0
    assert outputs["metrics"]["grad_norm"] == 0.25


def test_non_reporting_rank_does_not_publish_phase_timers(monkeypatch):
    monkeypatch.setattr("verl.workers.engine.base.time.perf_counter", _Clock())
    monkeypatch.setattr("verl.workers.engine.base._synchronize_for_phase_timing", lambda: None)
    engine = _RecordingEngine(report_metrics=False)

    outputs = engine.train_batch(TensorDict({}, batch_size=[]), loss_function=None)

    assert "timing_s/forward_backward" not in outputs["metrics"]
    assert "timing_s/optimizer" not in outputs["metrics"]


class _Flops:
    def estimate_flops(self, tokens, delta_time, images_seqlens=None):
        # Achieved FLOPs/s scales as 1/time, so MFU isolates the denominator.
        return 120.0 / delta_time, 10.0


def test_postprocess_reports_forward_backward_mfu_without_optimizer(monkeypatch):
    class _Device:
        def max_memory_allocated(self):
            return 0

        def max_memory_reserved(self):
            return 0

    monkeypatch.setattr("verl.workers.engine_workers.get_torch_device", lambda: _Device())
    monkeypatch.setattr("verl.workers.engine_workers.psutil.virtual_memory", lambda: SimpleNamespace(used=0))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    worker = SimpleNamespace(
        engine=SimpleNamespace(get_data_parallel_group=lambda: None),
        flops_counter=_Flops(),
        device_name="cpu",
    )
    raw = {
        "metrics": {"timing_s/forward_backward": 2.0, "timing_s/optimizer": 1.0},
        "loss": [1.0],
        "model_output": {},
    }

    result = TrainingWorker._postprocess_output(
        worker,
        raw,
        global_token_num=[8, 8],
        delta_time=4.0,
        forward_only=False,
        images_seqlens=None,
    )
    metrics = tu.get(result, "metrics")

    # 120/4/10/4 = 0.75 for the whole update; forward-backward drops the 2s of optimizer work.
    assert metrics["mfu"] == 0.75
    assert metrics["mfu_forward_backward"] == 1.5
    assert metrics["timing_s/forward_backward"] == 2.0
    assert metrics["timing_s/optimizer"] == 1.0


def test_promote_update_phase_metrics_keeps_whole_update_mfu():
    metrics = {
        "actor/mfu": [0.05, 0.07],
        "actor/mfu_forward_backward": [0.08, 0.10],
        "actor/timing_s/forward_backward": [10.0, 12.0],
        "actor/timing_s/optimizer": [3.0, 5.0],
    }
    metrics["perf/mfu/actor"] = metrics.pop("actor/mfu")
    promote_update_phase_metrics(metrics, "actor")
    reduced = reduce_metrics(metrics)

    assert reduced["perf/mfu/actor"] == pytest.approx(0.06)
    assert reduced["perf/mfu/actor_forward_backward"] == pytest.approx(0.09)
    assert reduced["timing_s/actor_forward_backward_mean"] == pytest.approx(11.0)
    assert reduced["timing_s/actor_optimizer_mean"] == pytest.approx(4.0)
    assert "actor/mfu_forward_backward" not in reduced


def test_default_off_adds_no_synchronization_or_phase_metrics(monkeypatch):
    def unexpected():
        raise AssertionError("default path must not synchronize or read the phase clock")

    monkeypatch.setattr("verl.workers.engine.base._synchronize_for_phase_timing", unexpected)
    monkeypatch.setattr("verl.workers.engine.base.time.perf_counter", unexpected)
    engine = _RecordingEngine(report_metrics=True)
    engine.engine_config.enable_update_phase_timing = False
    outputs = engine.train_batch(TensorDict({}, batch_size=[]), loss_function=None)
    assert engine.events == ["zero_grad", "forward_backward", "optimizer_step"]
    assert outputs["metrics"] == {"grad_norm": 0.25}


def test_colocated_transition_timers_preserve_call_order(monkeypatch):
    from contextlib import contextmanager

    from verl.trainer.ppo.v1 import trainer_colocate_async

    events = []

    @contextmanager
    def timer(name, timing, color):
        events.append(("begin", name))
        yield
        events.append(("end", name))

    monkeypatch.setattr(trainer_colocate_async, "marked_timer", timer)
    manager = SimpleNamespace(
        abort_replicas=lambda: events.append("abort"),
        sleep_replicas=lambda: events.append("sleep"),
        update_weights=lambda step: events.append(("sync", step)),
        resume_generation_replicas=lambda: events.append("resume"),
    )
    trainer = SimpleNamespace(checkpoint_manager=manager, timing_raw={}, global_steps=7, curr_step_profile=False)
    trainer_colocate_async.PPOTrainerColocateAsync.on_sample_end(trainer)
    trainer_colocate_async.PPOTrainerColocateAsync.on_step_end(trainer)
    assert events == [
        ("begin", "rollout_abort"),
        "abort",
        ("end", "rollout_abort"),
        ("begin", "rollout_sleep"),
        "sleep",
        ("end", "rollout_sleep"),
        ("begin", "update_weights"),
        ("begin", "weight_sync"),
        ("sync", 7),
        ("end", "weight_sync"),
        ("begin", "rollout_resume"),
        "resume",
        ("end", "rollout_resume"),
        ("end", "update_weights"),
    ]


def test_phase_rank_means_survive_metadata_only_worker_collection(monkeypatch):
    from verl.protocol import BatchData
    from verl.workers import engine_workers

    device = SimpleNamespace(max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0)
    monkeypatch.setattr(engine_workers, "get_torch_device", lambda: device)
    monkeypatch.setattr(engine_workers.psutil, "virtual_memory", lambda: SimpleNamespace(used=0))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    seen = []

    def gather(data, group):
        assert group == "dp"
        if not data:
            return {}
        seen.append(data)
        assert set(data) == {"timing_s/forward_backward", "timing_s/optimizer", "mfu_forward_backward"}
        return {
            "timing_s/forward_backward": [2.0, 6.0],
            "timing_s/optimizer": [1.0, 3.0],
            "mfu_forward_backward": [3.0, 1.0],
        }

    monkeypatch.setattr(engine_workers, "allgather_dict_into_dict", gather)
    worker = SimpleNamespace(
        engine=SimpleNamespace(get_data_parallel_group=lambda: "dp"), flops_counter=_Flops(), device_name="cpu"
    )
    outputs = []
    for forward, optimizer in [(2.0, 1.0), (6.0, 3.0)]:
        raw = {
            "metrics": {"timing_s/forward_backward": forward, "timing_s/optimizer": optimizer},
            "loss": [1.0],
            "model_output": {},
        }
        outputs.append(
            TrainingWorker._postprocess_output(
                worker, raw, global_token_num=[8, 8], delta_time=8.0, forward_only=False, images_seqlens=None
            )
        )
    # The real collector drops later metadata-only TensorDicts. All reporting
    # ranks must therefore carry the same already-aggregated phase observations.
    collected = BatchData(outputs).concat()
    collected_metrics = tu.get(collected, "metrics")
    assert collected_metrics["timing_s/forward_backward"] == 4.0
    assert collected_metrics["timing_s/optimizer"] == 2.0
    assert collected_metrics["mfu_forward_backward"] == 2.0
    assert len(seen) == 2
    assert seen[0]["timing_s/forward_backward"] == 2.0
    assert seen[1]["timing_s/forward_backward"] == 6.0
    promoted = {"actor/" + key: [value] for key, value in collected_metrics.items()}
    promote_update_phase_metrics(promoted, "actor")
    reduced = reduce_metrics(promoted)
    assert reduced["timing_s/actor_forward_backward_mean"] == 4.0
    assert reduced["timing_s/actor_optimizer_mean"] == 2.0
    assert reduced["perf/mfu/actor_forward_backward"] == 2.0


def test_default_worker_path_adds_no_phase_collective(monkeypatch):
    from verl.workers import engine_workers

    device = SimpleNamespace(max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0)
    monkeypatch.setattr(engine_workers, "get_torch_device", lambda: device)
    monkeypatch.setattr(engine_workers.psutil, "virtual_memory", lambda: SimpleNamespace(used=0))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    calls = []

    def gather(data, group):
        calls.append(data.copy())
        return {}

    monkeypatch.setattr(engine_workers, "allgather_dict_into_dict", gather)
    worker = SimpleNamespace(
        engine=SimpleNamespace(get_data_parallel_group=lambda: "dp"), flops_counter=_Flops(), device_name="cpu"
    )
    raw = {"metrics": {}, "loss": [1.0], "model_output": {}}
    output = TrainingWorker._postprocess_output(
        worker, raw, global_token_num=[8, 8], delta_time=8.0, forward_only=False, images_seqlens=None
    )
    assert calls == [{}]  # Only the existing ordinary-metric collective.
    assert "mfu_forward_backward" not in tu.get(output, "metrics")
    assert "timing_s/forward_backward" not in tu.get(output, "metrics")


def test_worker_reports_executed_updates_for_cycle_phase_weighting():
    from contextlib import nullcontext
    from unittest.mock import MagicMock

    data = TensorDict({"sample": torch.arange(8)}, batch_size=[8])
    tu.assign_non_tensor(data, num_mini_batch=4, epochs=2)
    worker = SimpleNamespace(
        engine=SimpleNamespace(
            get_data_parallel_size=lambda: 1,
            get_data_parallel_rank=lambda: 0,
            train_mode=lambda **kwargs: nullcontext(),
            is_mp_src_rank_with_outputs=lambda: True,
        ),
        profiler=MagicMock(),
        train_batch=lambda batch: tu.get_tensordict({}, {"metrics": {}}),
    )
    output = TrainingWorker.train_mini_batch(worker, data)
    assert tu.get(output, "metrics")["mini_batches_executed"] == [8]
    assert worker.profiler.step.call_count == 8
