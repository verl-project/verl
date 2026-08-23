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

"""CPU-only unit tests for ``separate_async``'s once-per-global-step hybrid engine switching.

The hybrid engine shares GPUs with training, so every round trip costs a weight restore on the way
out and an abort plus sleep on the way back. These tests pin the round trip to one per global step:
the engine is lent to generation at the weight sync and reclaimed once the replay buffer is deep
enough, with the remaining mini-batches served by the standalone pool alone.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import verl.trainer.ppo.v1.trainer_separate_async as trainer_module
from verl.trainer.config import HybridRolloutSwitchConfig
from verl.trainer.ppo.v1.trainer_separate_async import HybridEngineMode, PPOTrainerSeparateAsync
from verl.utils.config import omega_conf_to_dataclass


class _RecordingCheckpointManager:
    def __init__(self, name: str, events: list[str]):
        self.name = name
        self.events = events
        self.update_calls: list[int] = []
        self.resume_calls = 0
        self.abort_calls = 0
        self.sleep_calls = 0

    def update_weights(self, global_steps: int):
        self.events.append(f"{self.name}_update")
        self.update_calls.append(global_steps)
        return {}

    def resume_generation_replicas(self):
        self.events.append(f"{self.name}_resume")
        self.resume_calls += 1

    def abort_replicas(self):
        self.events.append(f"{self.name}_abort")
        self.abort_calls += 1

    def sleep_replicas(self):
        self.events.append(f"{self.name}_sleep")
        self.sleep_calls += 1


class _RecordingReplayBuffer:
    poll_interval = 1.0

    def __init__(self, eviction_metrics: dict | None = None, sampleable_count: int = 0):
        self.wait_calls: list[int] = []
        self.count_calls = 0
        self.eviction_metrics = eviction_metrics or {}
        self.sampleable_count = sampleable_count

    def wait_for_sampleable(self, global_steps: int, partition_id: str, target_count: int) -> tuple[set[str], dict]:
        del global_steps, partition_id
        self.wait_calls.append(target_count)
        return set(), dict(self.eviction_metrics)

    def get_sampleable_count(self, global_steps: int, partition_id: str) -> int:
        del global_steps, partition_id
        self.count_calls += 1
        return self.sampleable_count


def _trainer(
    *,
    enable_switch: bool = True,
    switch_threshold_ratio: float = 0.25,
    adaptive_switch_threshold: bool = False,
    train_batch_size: int = 64,
    parameter_sync_step: int = 4,
    global_steps: int = 1,
    total_training_steps: int = 3,
    eviction_metrics: dict | None = None,
    sampleable_count: int = 0,
) -> PPOTrainerSeparateAsync:
    trainer = object.__new__(PPOTrainerSeparateAsync)
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": train_batch_size},
            "trainer": {"nnodes": 2, "n_gpus_per_node": 8},
            "actor_rollout_ref": {
                "rollout": {
                    "nnodes": 2,
                    "n_gpus_per_node": 8,
                    "disaggregation": {"enabled": False},
                }
            },
        }
    )
    trainer.parameter_sync_step = parameter_sync_step
    trainer.global_steps = global_steps
    trainer.total_training_steps = total_training_steps
    trainer.timing_raw = {}
    trainer.current_mode = HybridEngineMode.ROLLOUT
    trainer.replay_buffer = _RecordingReplayBuffer(eviction_metrics, sampleable_count)
    trainer.events: list[str] = []
    trainer.checkpoint_manager = _RecordingCheckpointManager("hybrid", trainer.events)
    trainer.standalone_checkpoint_manager = _RecordingCheckpointManager("standalone", trainer.events)
    trainer.balancer_calls: list[str] = []

    def add_replicas():
        trainer.events.append("add")
        trainer.balancer_calls.append("add")

    def remove_replicas():
        trainer.events.append("remove")
        trainer.balancer_calls.append("remove")

    def clear_sticky():
        trainer.events.append("clear")
        trainer.balancer_calls.append("clear")
        return {"cleared_entries": 7}

    trainer.add_replicas_to_balancer = add_replicas
    trainer.remove_replicas_from_balancer = remove_replicas
    trainer.clear_sticky_cache = clear_sticky
    trainer.hybrid_rollout_config = HybridRolloutSwitchConfig(
        enable_switch=enable_switch,
        switch_threshold_ratio=switch_threshold_ratio,
        adaptive_switch_threshold=adaptive_switch_threshold,
        switch_threshold_step_up=0.05,
        switch_threshold_step_down=0.025,
        switch_threshold_release_steps=2,
        switch_cost_window_size=3,
    )
    if enable_switch:
        trainer._init_hybrid_rollout_state()
    return trainer


def _construct_trainer(monkeypatch, *, replay_buffer, enable_switch: bool = True, disaggregation: bool = False):
    config = OmegaConf.create(
        {
            "data": {"train_batch_size": 64},
            "actor_rollout_ref": {
                "actor": {"ppo_mini_batch_size": 16},
                "rollout": {
                    "nnodes": 1,
                    "n_gpus_per_node": 8,
                    "checkpoint_engine": {"backend": "nccl"},
                    "disaggregation": {"enabled": disaggregation},
                },
            },
            "trainer": {
                "nnodes": 1,
                "n_gpus_per_node": 8,
                "v1": {
                    "separate_async": {
                        "parameter_sync_step": 4,
                        "hybrid_rollout": {
                            "_target_": "verl.trainer.config.HybridRolloutSwitchConfig",
                            "enable_switch": enable_switch,
                        },
                    }
                },
            },
            "reward": {"reward_model": {"enable": False}},
        }
    )

    def mock_base_init(trainer, trainer_config):
        trainer.config = trainer_config
        trainer.replay_buffer = replay_buffer

    monkeypatch.setattr(trainer_module.PPOTrainer, "__init__", mock_base_init)
    return PPOTrainerSeparateAsync(config)


def _run_step(trainer: PPOTrainerSeparateAsync, *, sample_wait_seconds: float) -> None:
    """Drive one global step's hooks, with the mini-batches blocked for the given total."""
    trainer.on_step_begin()
    trainer._wait_for_sampleable_and_switch()
    trainer._step_sample_wait_seconds = sample_wait_seconds
    trainer.on_step_end()


def test_hybrid_rollout_switch_config_target_instantiates_dataclass():
    config = OmegaConf.create(
        {
            "_target_": "verl.trainer.config.HybridRolloutSwitchConfig",
            "switch_threshold_ratio": 0.25,
        }
    )

    switch_config = omega_conf_to_dataclass(config)

    assert isinstance(switch_config, HybridRolloutSwitchConfig)
    assert switch_config.switch_threshold_ratio == 0.25


@pytest.mark.parametrize("window_size", [0, -1])
def test_hybrid_rollout_switch_config_rejects_nonpositive_cost_window(window_size):
    with pytest.raises(ValueError, match="switch_cost_window_size must be positive"):
        HybridRolloutSwitchConfig(switch_cost_window_size=window_size)


def test_step_lends_the_engine_out_and_reclaims_it_exactly_once():
    trainer = _trainer()
    trainer.current_mode = HybridEngineMode.TRAINER

    # Weight sync ends the previous step by lending the engine to the next one's generation.
    trainer.on_step_end()
    assert trainer.current_mode == HybridEngineMode.ROLLOUT
    assert trainer.standalone_checkpoint_manager.update_calls == [1]
    assert trainer.checkpoint_manager.update_calls == [1]
    assert trainer.checkpoint_manager.resume_calls == 1
    assert trainer.balancer_calls == ["add", "clear"]
    assert trainer.events[:5] == ["add", "clear", "standalone_update", "hybrid_update", "hybrid_resume"]
    assert trainer.timing_raw["switch_to_rollout"] >= 0.0
    assert len(trainer._to_rollout_costs) == 1
    # No wait history yet, so both halves of the benefit model are unavailable and their
    # metrics stay absent; the decision falls back to "lend whenever anything is missing".
    assert "separate_async/decision/per_sample_time_seconds" not in trainer._pending_sync_metrics
    assert "separate_async/decision/effective_switch_cost_seconds" not in trainer._pending_sync_metrics

    trainer.on_step_begin()
    trainer._wait_for_sampleable_and_switch()
    assert trainer.current_mode == HybridEngineMode.TRAINER
    assert trainer.replay_buffer.wait_calls == [16]
    assert trainer.checkpoint_manager.abort_calls == 1
    assert trainer.checkpoint_manager.sleep_calls == 1
    assert trainer.balancer_calls == ["add", "clear", "remove"]
    assert trainer.timing_raw["switch_to_trainer"] >= 0.0
    assert len(trainer._to_trainer_costs) == 1

    # Later mini-batches run through the sample hooks, which must not switch again.
    for _ in range(3):
        trainer.on_sample_begin()
        trainer.on_sample_end()
    assert trainer.checkpoint_manager.sleep_calls == 1
    assert trainer.balancer_calls == ["add", "clear", "remove"]


def test_switch_to_rollout_timing_accumulates_prepare_and_wake(monkeypatch):
    @contextmanager
    def recording_timer(name, timing_raw, **_kwargs):
        yield
        timing_raw[name] = timing_raw.get(name, 0.0) + 1.0

    monkeypatch.setattr(trainer_module, "marked_timer", recording_timer)
    trainer = _trainer()
    trainer.current_mode = HybridEngineMode.TRAINER

    trainer.on_step_end()

    assert trainer.timing_raw["switch_to_rollout"] == 2.0
    assert trainer.timing_raw["update_weights"] == 1.0


def test_step_begin_reclaims_hybrid_before_submission_when_inventory_is_ready():
    trainer = _trainer(sampleable_count=16)

    trainer.on_step_begin()

    assert trainer.current_mode == HybridEngineMode.TRAINER
    assert trainer.replay_buffer.count_calls == 1
    assert trainer.replay_buffer.wait_calls == []
    assert trainer.checkpoint_manager.abort_calls == 1
    assert trainer.checkpoint_manager.sleep_calls == 1
    assert trainer.timing_raw["switch_wait"] == 0.0
    assert trainer.timing_raw["switch_to_trainer"] >= 0.0
    assert trainer._wait_for_sampleable_and_switch() == {}


def test_wait_for_sampleable_is_a_noop_once_the_engine_is_training():
    trainer = _trainer()
    trainer.current_mode = HybridEngineMode.TRAINER

    assert trainer._wait_for_sampleable_and_switch() == {}
    assert trainer.replay_buffer.wait_calls == []
    assert trainer.balancer_calls == []


def test_eviction_metrics_from_the_wait_reach_the_step():
    metrics = {"training/off_policy/evicted_samples": 3}
    trainer = _trainer(eviction_metrics=metrics)
    trainer._add_batch_to_generate = lambda: None

    trainer.on_step_begin()
    assert trainer.prepare_step() == metrics


def test_disabled_switching_reclaims_the_engine_without_waiting():
    trainer = _trainer(enable_switch=False)

    trainer.on_step_begin()
    assert trainer._wait_for_sampleable_and_switch() == {}
    assert trainer.current_mode == HybridEngineMode.TRAINER
    assert trainer.replay_buffer.wait_calls == []
    assert trainer.checkpoint_manager.sleep_calls == 1

    trainer.on_step_end()
    assert trainer.current_mode == HybridEngineMode.TRAINER
    assert trainer.checkpoint_manager.update_calls == []


def test_last_step_does_not_lend_the_engine_out():
    trainer = _trainer(global_steps=3, total_training_steps=3)
    trainer.current_mode = HybridEngineMode.TRAINER

    trainer.on_step_end()

    assert trainer.current_mode == HybridEngineMode.TRAINER
    assert trainer.standalone_checkpoint_manager.update_calls == [3]
    assert trainer.checkpoint_manager.update_calls == []


def test_inventory_gate_skips_lending_when_the_target_is_already_buffered():
    trainer = _trainer(sampleable_count=16)
    trainer.current_mode = HybridEngineMode.TRAINER

    trainer.on_step_begin()
    trainer.on_step_end()

    assert trainer.checkpoint_manager.update_calls == []
    assert trainer.balancer_calls == []
    assert trainer.timing_raw["switch_wait"] == 0.0
    assert trainer._pending_sync_metrics["separate_async/decision/remaining"] == 0.0
    assert trainer._pending_sync_metrics["separate_async/decision/should_switch_to_rollout"] == 0.0


def test_cost_gate_skips_lending_when_the_remaining_work_is_cheaper_to_fill_on_standalone():
    trainer = _trainer(sampleable_count=15)
    trainer.current_mode = HybridEngineMode.TRAINER
    trainer._wait_seconds = 2.0
    trainer._wait_samples = 1
    trainer._to_rollout_costs.append(3.0)
    trainer._to_trainer_costs.append(4.0)

    trainer.on_step_end()

    assert trainer.checkpoint_manager.update_calls == []
    assert trainer._scaling_factor == 2.0
    assert trainer._pending_sync_metrics["separate_async/decision/remaining"] == 1.0
    assert trainer._pending_sync_metrics["separate_async/decision/per_sample_time_seconds"] == 2.0
    assert trainer._pending_sync_metrics["separate_async/decision/effective_switch_cost_seconds"] == 7.0
    # benefit = remaining * per_sample_time * (1 - 1/scaling_factor) = 1 * 2.0 * 0.5 = 1.0 < 7.0
    assert trainer._pending_sync_metrics["separate_async/decision/should_switch_to_rollout"] == 0.0


def test_cost_gate_lends_when_the_remaining_work_is_more_expensive_than_the_switch():
    trainer = _trainer(sampleable_count=15)
    trainer.current_mode = HybridEngineMode.TRAINER
    trainer._wait_seconds = 16.0
    trainer._wait_samples = 1
    trainer._to_rollout_costs.append(3.0)
    trainer._to_trainer_costs.append(4.0)

    trainer.on_step_end()

    assert trainer.checkpoint_manager.update_calls == [1]
    assert trainer._pending_sync_metrics["separate_async/decision/should_switch_to_rollout"] == 1.0


def test_switch_cost_window_forgets_cold_start():
    trainer = _trainer()
    trainer._to_rollout_costs.extend([100.0, 2.0, 4.0, 6.0])
    trainer._to_trainer_costs.extend([50.0, 1.0, 3.0, 5.0])

    assert trainer._effective_switch_cost() == pytest.approx(7.0)


@pytest.mark.parametrize(
    ("ratio", "parameter_sync_step", "expected"),
    [
        (0.5, 4, 32),
        (1.0, 4, 64),
        # Below one mini-batch the trainer would reclaim the GPUs with nothing to train on.
        (0.01, 4, 16),
        # A single local update per step leaves no mini-batch to overlap generation, so the floor
        # is the whole batch whatever the ratio says.
        (0.25, 1, 64),
    ],
)
def test_threshold_is_clamped_to_between_one_mini_batch_and_the_batch(ratio, parameter_sync_step, expected):
    trainer = _trainer(switch_threshold_ratio=ratio, parameter_sync_step=parameter_sync_step)

    assert trainer._switch_threshold() == expected


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.5])
def test_switching_rejects_a_ratio_outside_the_unit_interval(ratio):
    with pytest.raises(ValueError, match="switch_threshold_ratio"):
        _trainer(switch_threshold_ratio=ratio)


def test_switching_rejects_a_replay_buffer_that_cannot_report_depth(monkeypatch):
    with pytest.raises(TypeError, match="wait_for_sampleable"):
        _construct_trainer(monkeypatch, replay_buffer=SimpleNamespace())


def test_switching_rejects_rollout_disaggregation(monkeypatch):
    with pytest.raises(ValueError, match="does not support rollout disaggregation"):
        _construct_trainer(monkeypatch, replay_buffer=_RecordingReplayBuffer(), disaggregation=True)


def test_disabled_switching_skips_custom_replay_buffer_validation(monkeypatch):
    trainer = _construct_trainer(monkeypatch, replay_buffer=SimpleNamespace(), enable_switch=False)

    assert trainer.hybrid_rollout_config.enable_switch is False


def test_adaptive_threshold_increases_after_trainer_idle():
    trainer = _trainer(adaptive_switch_threshold=True, switch_threshold_ratio=0.5)

    ratios = []
    for _ in range(3):
        _run_step(trainer, sample_wait_seconds=2.0)
        ratios.append(trainer._switch_threshold_ratio)

    assert ratios == pytest.approx([0.5, 0.55, 0.6])


def test_adaptive_threshold_decreases_after_calm_release_interval():
    trainer = _trainer(adaptive_switch_threshold=True, switch_threshold_ratio=0.5)

    ratios = []
    for _ in range(4):
        _run_step(trainer, sample_wait_seconds=0.0)
        ratios.append(trainer._switch_threshold_ratio)

    assert ratios == pytest.approx([0.5, 0.475, 0.45, 0.425])


def test_threshold_direction_change_resets_hysteresis():
    trainer = _trainer(adaptive_switch_threshold=True, switch_threshold_ratio=0.5)

    for _ in range(3):
        _run_step(trainer, sample_wait_seconds=0.0)
    _run_step(trainer, sample_wait_seconds=2.0)
    _run_step(trainer, sample_wait_seconds=0.0)

    assert trainer._switch_threshold_ratio == pytest.approx(0.45)


def test_alternating_idle_and_calm_does_not_move_threshold():
    trainer = _trainer(adaptive_switch_threshold=True, switch_threshold_ratio=0.5)

    for sample_wait_seconds in [2.0, 0.0] * 4:
        _run_step(trainer, sample_wait_seconds=sample_wait_seconds)

    assert trainer._switch_threshold_ratio == pytest.approx(0.5)


def test_trainer_idle_is_reported():
    trainer = _trainer(switch_threshold_ratio=0.5)

    _run_step(trainer, sample_wait_seconds=2.0)

    assert trainer._pending_sync_metrics["separate_async/switch/idle"] == 1.0
    assert trainer._step_threshold == 32
    assert trainer._step_sample_wait_seconds == pytest.approx(2.0)
    assert trainer._switch_threshold_ratio == pytest.approx(0.5)
