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
"""CPU-only production-path tests for V1 GPU stall diagnostics lifecycle wiring."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from omegaconf import OmegaConf


class _FakeKVBatchMeta:
    def __init__(self, partition_id, keys, tags):
        self.partition_id = partition_id
        self.keys = keys
        self.tags = tags


if "transfer_queue" not in sys.modules:
    # The unit-test environment may omit this optional runtime dependency. The
    # V1 modules only need this name at import time; every exercised operation
    # is replaced by a local fake below.
    _transfer_queue = ModuleType("transfer_queue")
    _transfer_queue.KVBatchMeta = _FakeKVBatchMeta
    _transfer_queue.__version__ = "0.0.0"
    sys.modules["transfer_queue"] = _transfer_queue

from verl.trainer.ppo import ray_trainer  # noqa: E402
from verl.trainer.ppo.v1 import trainer_base  # noqa: E402
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, get_trainer_cls  # noqa: E402
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync  # noqa: E402
from verl.trainer.ppo.v1.trainer_separate_async import HybridEngineMode, PPOTrainerSeparateAsync  # noqa: E402
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync  # noqa: E402
from verl.trainer.ppo.v1.utils import MetricsAggregator  # noqa: E402


class _FakeDiagnostics:
    def __init__(self, events, *, begin_errors=(), end_errors=(), close_error=False, include_collision=False):
        self.events = events
        self.begin_errors = set(begin_errors)
        self.end_errors = set(end_errors)
        self.close_error = close_error
        self.include_collision = include_collision

    def begin_phase(self, phase):
        self.events.append(f"begin:{phase}")
        if phase in self.begin_errors:
            raise RuntimeError(f"begin {phase}")

    def end_phase(self, phase):
        self.events.append(f"end:{phase}")
        if phase in self.end_errors:
            raise RuntimeError(f"end {phase}")
        updates = {
            f"gpu_stall/{phase}/node_0/sample_count": 2.0,
            f"gpu_stall/{phase}/node_0/utilization_mean": 50.0,
        }
        if self.include_collision:
            updates["existing"] = 999.0
        return updates

    def close(self):
        self.events.append("close")
        if self.close_error:
            raise RuntimeError("close")


class _FakeTracking:
    def __init__(self):
        self.logs = []

    def log(self, data, step):
        self.logs.append((dict(data), step))


class _FakeCheckpointManager:
    def __init__(self, events, name):
        self.events = events
        self.name = name

    def update_weights(self, *args):
        self.events.append(f"{self.name}:update_weights")

    def resume_generation_replicas(self):
        self.events.append(f"{self.name}:resume_generation")

    def abort_replicas(self):
        self.events.append(f"{self.name}:abort")

    def sleep_replicas(self):
        self.events.append(f"{self.name}:sleep")


class _FakeProgress:
    def update(self, _count):
        return None

    def close(self):
        return None


class _Batch:
    def __init__(self):
        self.extra_info = {}
        self.keys = []
        self.partition_id = "train"
        self.tags = []


def _config(*, mode="sync", enable=True, val_before_train=False, val_only=False):
    return OmegaConf.create(
        {
            "trainer": {
                "project_name": "test",
                "experiment_name": "v1-gpu-stall",
                "logger": [],
                "val_before_train": val_before_train,
                "val_only": val_only,
                "total_epochs": 1,
                "save_freq": -1,
                "test_freq": -1,
                "rollout_data_dir": None,
                "critic_warmup": 0,
                "v1": {
                    "trainer_mode": mode,
                    "colocate_async": {"num_warmup_batches": 0},
                    "separate_async": {"num_warmup_batches": 0},
                },
            },
            "global_profiler": {
                "steps": None,
                "gpu_stall_diagnostics": {
                    "enable": enable,
                    "sample_interval_s": 0.25,
                    "zero_utilization_threshold": 1.0,
                },
            },
            "skip": {"rollout_tq": {"enable": True}},
            "actor_rollout_ref": {"rollout": {"temperature": 1.0}},
        }
    )


def _make_fit_trainer(
    monkeypatch, trainer_cls, *, enable=True, val_before_train=False, val_only=False, step_error=None
):
    events = []
    diagnostics_events = []
    diagnostics = _FakeDiagnostics(diagnostics_events, include_collision=True)
    tracking = _FakeTracking()
    starts = []
    trainer = object.__new__(trainer_cls)
    trainer.config = _config(
        mode={
            PPOTrainerSync: "sync",
            PPOTrainerColocateAsync: "colocate_async",
            PPOTrainerSeparateAsync: "separate_async",
        }[trainer_cls],
        enable=enable,
        val_before_train=val_before_train,
        val_only=val_only,
    )
    trainer.global_steps = 0
    trainer.steps_per_epoch = 1
    trainer.total_training_steps = 2
    trainer.checkpoint_manager = _FakeCheckpointManager(events, "checkpoint")
    if trainer_cls is PPOTrainerSeparateAsync:
        trainer.standalone_checkpoint_manager = _FakeCheckpointManager(events, "standalone")

    def step(metrics, _timing_raw):
        events.append("step")
        metrics["existing"] = 1.0
        if step_error is not None:
            raise step_error
        return _Batch()

    trainer.step = step
    trainer._reissue_inflight_prompts = lambda: 0
    trainer._start_profiling = lambda: events.append("profile:start")
    trainer._stop_profiling = lambda: events.append("profile:stop")
    trainer._compute_metrics = lambda _batch, metrics, *_args, **_kwargs: metrics.setdefault("computed", 1.0)
    trainer._shutdown_dump_executor = lambda: events.append("shutdown")
    trainer._save_checkpoint = lambda: events.append("checkpoint:save")
    trainer._log_rollout_data = lambda *_args: events.append("rollout:log")
    trainer._validate = lambda: {"validation/reward": 1.0}

    monkeypatch.setattr(trainer_base, "Tracking", lambda **_kwargs: tracking)
    monkeypatch.setattr(trainer_base, "ValidationGenerationsLogger", lambda **_kwargs: object())
    monkeypatch.setattr(trainer_base, "tqdm", lambda **_kwargs: _FakeProgress())
    monkeypatch.setattr(trainer_base, "pprint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trainer_base.SkipManager, "init", staticmethod(lambda _config: None))
    monkeypatch.setattr(trainer_base.SkipManager, "set_step", staticmethod(lambda _step: None))
    monkeypatch.setattr(trainer_base.tq, "kv_clear", lambda **_kwargs: None, raising=False)

    def start(config):
        starts.append(config)
        return diagnostics

    monkeypatch.setattr(trainer_base, "start_gpu_stall_diagnostics", start)
    return trainer, diagnostics, starts, tracking, events, diagnostics_events


def test_main_ppo_selects_v1_or_legacy_task_runner_without_starting_ray(monkeypatch):
    from verl.trainer import main_ppo
    from verl.trainer.main_ppo_v0 import TaskRunner as LegacyTaskRunner

    selected = []
    monkeypatch.setattr(main_ppo, "auto_set_device", lambda _config: None)
    monkeypatch.setattr(main_ppo, "validate_config", lambda **_kwargs: None)
    monkeypatch.setattr(main_ppo, "need_reference_policy", lambda _config: False)
    monkeypatch.setattr(main_ppo, "need_critic", lambda _config: False)
    monkeypatch.setattr(
        main_ppo, "run_ppo", lambda config, task_runner_class: selected.append((config, task_runner_class))
    )

    main_ppo.main.__wrapped__(OmegaConf.create({"trainer": {"use_v1": True}}))
    main_ppo.main.__wrapped__(OmegaConf.create({"trainer": {"use_v1": False}}))

    assert selected[0][1] is main_ppo.TaskRunnerV1
    assert selected[1][1] is LegacyTaskRunner
    assert get_trainer_cls("sync") is PPOTrainerSync
    assert get_trainer_cls("colocate_async") is PPOTrainerColocateAsync
    assert get_trainer_cls("separate_async") is PPOTrainerSeparateAsync


def test_legacy_val_only_path_still_starts_and_closes_diagnostics(monkeypatch):
    events = []
    diagnostics_events = []
    diagnostics = _FakeDiagnostics(diagnostics_events)
    starts = []
    tracking = _FakeTracking()
    trainer = object.__new__(ray_trainer.RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "test",
                "experiment_name": "legacy-gpu-stall",
                "logger": [],
                "val_before_train": True,
                "val_only": True,
            },
            "global_profiler": {"gpu_stall_diagnostics": {"enable": True}},
        }
    )
    trainer._dump_executor = SimpleNamespace(_shutdown=False)
    trainer._load_checkpoint = lambda: events.append("load_checkpoint")
    trainer.checkpoint_manager = _FakeCheckpointManager(events, "checkpoint")
    trainer.train_dataloader = [object()]
    trainer._validate = lambda: {"validation/reward": 1.0}
    trainer._shutdown_dump_executor = lambda: events.append("shutdown")

    def start(config):
        starts.append(config)
        return diagnostics

    monkeypatch.setattr(ray_trainer, "start_gpu_stall_diagnostics", start)
    monkeypatch.setattr(ray_trainer, "pprint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ray_trainer.SkipManager, "init", staticmethod(lambda _config: None))
    monkeypatch.setattr("verl.utils.tracking.Tracking", lambda **_kwargs: tracking)

    trainer.fit()

    assert starts == [trainer.config.global_profiler.gpu_stall_diagnostics]
    assert diagnostics_events == ["close"]
    assert events == ["load_checkpoint", "checkpoint:update_weights", "shutdown"]
    assert tracking.logs == [({"validation/reward": 1.0}, 0)]


@pytest.mark.parametrize(
    ("trainer_cls", "step_end_phase", "expected_checkpoint_event"),
    [
        (PPOTrainerSync, "weight_sync_and_wake_or_resume", "checkpoint:update_weights"),
        (PPOTrainerColocateAsync, "weight_sync_and_wake_or_resume", "checkpoint:resume_generation"),
        (PPOTrainerSeparateAsync, "standalone_rollout_weight_sync", "standalone:update_weights"),
    ],
)
def test_v1_enabled_creates_one_coordinator_merges_metrics_and_closes(
    monkeypatch, trainer_cls, step_end_phase, expected_checkpoint_event
):
    trainer, diagnostics, starts, tracking, events, diagnostic_events = _make_fit_trainer(monkeypatch, trainer_cls)

    trainer.fit(object())

    assert starts == [trainer.config.global_profiler.gpu_stall_diagnostics]
    assert diagnostic_events == [
        "begin:training_step",
        f"begin:{step_end_phase}",
        f"end:{step_end_phase}",
        "end:training_step",
        "close",
    ]
    assert expected_checkpoint_event in events
    logged_metrics, logged_step = tracking.logs[-1]
    assert logged_step == 1
    assert logged_metrics["existing"] == 1.0
    assert logged_metrics["gpu_stall/training_step/node_0/sample_count"] == 2.0
    assert logged_metrics[f"gpu_stall/{step_end_phase}/node_0/utilization_mean"] == 50.0
    assert trainer._gpu_stall_diagnostics is None
    assert diagnostics.events is diagnostic_events


def test_v1_default_off_does_not_start_diagnostics_or_change_step_order(monkeypatch):
    trainer, _diagnostics, starts, _tracking, events, diagnostic_events = _make_fit_trainer(
        monkeypatch, PPOTrainerSync, enable=False
    )
    monkeypatch.setattr(
        trainer_base,
        "gpu_stall_diagnostics_phase",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("default-off entered diagnostics phase helper")),
    )

    trainer.fit(object())

    assert starts == []
    assert diagnostic_events == []
    assert events == ["profile:start", "step", "profile:stop", "checkpoint:update_weights", "shutdown"]


def test_v1_default_off_does_not_enter_phase_helper_in_rollout_substep(monkeypatch):
    events = []
    trainer = object.__new__(PPOTrainerSync)
    trainer.config = _config(mode="sync", enable=False)
    trainer.global_steps = 1
    trainer.use_reference_policy = False
    trainer.use_critic = False
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=[])
    trainer.replay_buffer = SimpleNamespace(sample=lambda **_kwargs: (_Batch(), {}))
    trainer.checkpoint_manager = _FakeCheckpointManager(events, "checkpoint")
    trainer._gpu_stall_diagnostics = None
    trainer._balance_batch = lambda batch, **_kwargs: batch
    trainer._compute_old_log_prob = lambda batch, **_kwargs: batch
    trainer._compute_advantage = lambda batch, **_kwargs: batch
    trainer._update_actor = lambda batch, **_kwargs: events.append("actor_update") or batch
    monkeypatch.setattr(
        trainer_base,
        "gpu_stall_diagnostics_phase",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("default-off entered diagnostics phase helper")),
    )

    result = PPOTrainer._step_once(trainer, {}, {}, sample_batch_size=1)

    assert isinstance(result, _Batch)
    assert events == ["checkpoint:sleep", "actor_update"]


def test_v1_val_only_and_exception_paths_close_diagnostics(monkeypatch):
    val_trainer, _diagnostics, starts, tracking, _events, diagnostic_events = _make_fit_trainer(
        monkeypatch, PPOTrainerSync, val_before_train=True, val_only=True
    )

    val_trainer.fit(object())

    assert len(starts) == 1
    assert diagnostic_events == ["close"]
    assert tracking.logs == [({"validation/reward": 1.0}, 0)]

    validation_error = RuntimeError("validation failed")
    validation_trainer, _diagnostics, _starts, _tracking, _events, diagnostic_events = _make_fit_trainer(
        monkeypatch, PPOTrainerSync, val_before_train=True
    )
    validation_trainer._validate = lambda: (_ for _ in ()).throw(validation_error)
    with pytest.raises(RuntimeError, match="validation failed"):
        validation_trainer.fit(object())

    assert diagnostic_events == ["close"]

    error = RuntimeError("training failed")
    trainer, _diagnostics, _starts, _tracking, _events, diagnostic_events = _make_fit_trainer(
        monkeypatch, PPOTrainerSync, step_error=error
    )
    with pytest.raises(RuntimeError, match="training failed"):
        trainer.fit(object())

    assert diagnostic_events == ["begin:training_step", "end:training_step", "close"]


def test_v1_diagnostics_fail_open_for_start_phase_and_close_errors(monkeypatch):
    trainer, _diagnostics, _starts, _tracking, events, _diagnostic_events = _make_fit_trainer(
        monkeypatch, PPOTrainerSync
    )
    monkeypatch.setattr(
        trainer_base, "start_gpu_stall_diagnostics", lambda _config: (_ for _ in ()).throw(RuntimeError("start"))
    )

    trainer.fit(object())

    assert "step" in events

    trainer, diagnostics, _starts, _tracking, events, diagnostic_events = _make_fit_trainer(monkeypatch, PPOTrainerSync)
    diagnostics.begin_errors.add("training_step")
    diagnostics.end_errors.add("weight_sync_and_wake_or_resume")
    diagnostics.close_error = True

    trainer.fit(object())

    assert "step" in events
    assert diagnostic_events[-1] == "close"


@pytest.mark.parametrize(
    ("trainer_cls", "expected_release_events"),
    [
        (PPOTrainerSync, ["checkpoint:sleep"]),
        (PPOTrainerColocateAsync, ["checkpoint:abort", "checkpoint:sleep"]),
        (PPOTrainerSeparateAsync, ["switch_to_trainer"]),
    ],
)
def test_v1_real_step_once_emits_mode_accurate_phases(monkeypatch, trainer_cls, expected_release_events):
    events = []
    diagnostic_events = []
    trainer = object.__new__(trainer_cls)
    trainer.config = _config(
        mode={
            PPOTrainerSync: "sync",
            PPOTrainerColocateAsync: "colocate_async",
            PPOTrainerSeparateAsync: "separate_async",
        }[trainer_cls]
    )
    trainer.global_steps = 1
    trainer.use_reference_policy = False
    trainer.use_critic = False
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=[])
    trainer.replay_buffer = SimpleNamespace(sample=lambda **_kwargs: (_Batch(), {}))
    trainer.checkpoint_manager = _FakeCheckpointManager(events, "checkpoint")
    trainer._gpu_stall_diagnostics = _FakeDiagnostics(diagnostic_events)
    trainer._balance_batch = lambda batch, **_kwargs: batch
    trainer._compute_old_log_prob = lambda batch, **_kwargs: batch
    trainer._compute_advantage = lambda batch, **_kwargs: batch
    trainer._update_actor = lambda batch, **_kwargs: events.append("actor_update") or batch
    if trainer_cls is PPOTrainerSeparateAsync:
        trainer.current_mode = HybridEngineMode.ROLLOUT
        trainer.switch_to_trainer = lambda: events.append("switch_to_trainer")

    result = PPOTrainer._step_once(trainer, {}, {}, sample_batch_size=1)

    assert isinstance(result, _Batch)
    assert diagnostic_events == [
        "begin:rollout_wait",
        "end:rollout_wait",
        "begin:rollout_release_or_switch",
        "end:rollout_release_or_switch",
        "begin:actor_update",
        "end:actor_update",
    ]
    assert events[-len(expected_release_events) - 1 : -1] == expected_release_events
    assert events[-1] == "actor_update"


def test_v1_rollout_wait_phase_ends_when_sampling_raises():
    diagnostic_events = []
    trainer = object.__new__(PPOTrainerSync)
    trainer.config = _config(mode="sync")
    trainer.global_steps = 1
    trainer.use_reference_policy = False
    trainer.use_critic = False
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=[])
    trainer.replay_buffer = SimpleNamespace(sample=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("sample")))
    trainer._gpu_stall_diagnostics = _FakeDiagnostics(diagnostic_events)

    with pytest.raises(RuntimeError, match="sample"):
        PPOTrainer._step_once(trainer, {}, {}, sample_batch_size=1)

    assert diagnostic_events == ["begin:rollout_wait", "end:rollout_wait"]


def test_v1_metrics_aggregator_uses_nvml_sample_counts_for_multiple_async_updates():
    aggregator = MetricsAggregator()
    phase = "gpu_stall/rollout_wait/node_0"
    device = f"{phase}/gpu_0000_10_00_0"
    aggregator.add_step_metrics(
        {
            f"{phase}/sample_count": 1.0,
            f"{phase}/utilization_mean": 100.0,
            f"{phase}/zero_utilization_fraction": 0.0,
            f"{device}/sample_count": 1.0,
            f"{device}/utilization_mean": 100.0,
        },
        sample_count=64,
    )
    aggregator.add_step_metrics(
        {
            f"{phase}/sample_count": 3.0,
            f"{phase}/utilization_mean": 0.0,
            f"{phase}/zero_utilization_fraction": 1.0,
            f"{device}/sample_count": 3.0,
            f"{device}/utilization_mean": 0.0,
        },
        sample_count=64,
    )

    metrics = aggregator.get_aggregated_metrics()
    assert metrics[f"{phase}/sample_count"] == 4.0
    assert metrics[f"{phase}/utilization_mean"] == 25.0
    assert metrics[f"{phase}/zero_utilization_fraction"] == 0.75
    assert metrics[f"{device}/sample_count"] == 4.0
    assert metrics[f"{device}/utilization_mean"] == 25.0
