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

"""Tests for the rollout profiler lifecycle across V1 trainer modes."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def _trainer(cls, **managers):
    trainer = cls.__new__(cls)
    trainer.config = OmegaConf.create({"global_profiler": {"profile_continuous_steps": False, "steps": [1]}})
    trainer.global_steps = 1
    trainer.trainer_mode = "sync" if cls in (PPOTrainerSync, PPOTrainerColocateAsync) else "separate_async"
    # The finish command fires only on the last profiled step; total_training_steps lets
    # _stop_profiling decide whether this is it (here step 1 is the largest profiled step).
    trainer.total_training_steps = 10
    trainer.prev_step_profile = False
    trainer.curr_step_profile = True
    trainer.next_step_profile = False
    trainer.local_trigger_step = 0
    trainer.use_reference_policy = False
    trainer.use_critic = False
    trainer.actor_rollout_wg = MagicMock()
    trainer.checkpoint_manager = MagicMock()
    for name, manager in managers.items():
        setattr(trainer, name, manager)
    return trainer


def _prepare_separate_step_end(trainer):
    trainer.hybrid_rollout_config = SimpleNamespace(enable_switch=False)
    trainer.timing_raw = {}
    trainer.standalone_checkpoint_manager = MagicMock()
    trainer.standalone_checkpoint_manager.update_weights.return_value = {}


def test_profiled_step_starts_rollout_engines():
    manager = MagicMock()
    trainer = _trainer(_StubTrainer, llm_server_manager=manager)

    trainer._start_profiling()
    manager.start_profile.assert_called_once()


def test_unprofiled_step_leaves_rollout_engines_alone():
    manager = MagicMock()
    trainer = _trainer(_StubTrainer, llm_server_manager=manager)
    trainer.curr_step_profile = False

    trainer._start_profiling()
    trainer._stop_profiling()

    manager.start_profile.assert_not_called()
    manager.stop_profile.assert_not_called()


def test_continuous_steps_leave_the_open_collection_running():
    # With profile_continuous_steps the collection opened by the previous step stays open, so a
    # continuing step must not restart it. Step boundaries are no longer marked at the trainer
    # level: the profiler is advanced once per update mini-batch inside the workers instead.
    manager = MagicMock()
    trainer = _trainer(_StubTrainer, llm_server_manager=manager)
    trainer.config = OmegaConf.create({"global_profiler": {"profile_continuous_steps": True, "steps": [1, 2]}})
    trainer.prev_step_profile = True
    trainer.curr_step_profile = True

    trainer._start_profiling()

    trainer.actor_rollout_wg.step_profile.assert_not_called()
    trainer.actor_rollout_wg.start_profile.assert_not_called()
    manager.start_profile.assert_not_called()


def test_sync_stops_rollout_after_sampling():
    manager = MagicMock()
    trainer = _trainer(PPOTrainerSync, llm_server_manager=manager)
    trainer._start_rollout_profiling()
    calls = MagicMock()
    calls.attach_mock(trainer.checkpoint_manager.sleep_replicas, "sleep")
    calls.attach_mock(manager.stop_profile, "stop")

    with patch.object(PPOTrainer, "_add_batch_to_generate") as submit:
        calls.attach_mock(submit, "submit")
        trainer._add_batch_to_generate()
        trainer.on_sample_end()

    assert calls.mock_calls == [call.submit(), call.sleep(), call.stop()]
    manager.start_profile.assert_called_once()


def test_sync_skips_rollout_stop_on_unprofiled_step():
    manager = MagicMock()
    trainer = _trainer(PPOTrainerSync, llm_server_manager=manager)
    trainer.curr_step_profile = False
    calls = MagicMock()
    calls.attach_mock(trainer.checkpoint_manager.sleep_replicas, "sleep")
    calls.attach_mock(manager.stop_profile, "stop")

    trainer.on_sample_end()

    assert calls.mock_calls == [call.sleep()]
    manager.stop_profile.assert_not_called()


def test_shared_worker_group_is_profiled_once():
    # In the hybrid engine ref_policy_wg (and sometimes critic_wg) is the *same* worker group
    # object as actor_rollout_wg. Each start/stop_profile round-trips to every rank and, on stop,
    # runs the finish hook (e.g. the user's trace-upload command). If profiling drove the shared
    # workers once per role alias, that hook -- and any upload it triggers -- would fire multiple
    # times and duplicate every trace file. It must fire exactly once per distinct process.
    manager = MagicMock()
    trainer = _trainer(_StubTrainer, llm_server_manager=manager)
    trainer.use_reference_policy = True
    trainer.use_critic = True
    trainer.ref_policy_wg = trainer.actor_rollout_wg
    trainer.critic_wg = trainer.actor_rollout_wg

    trainer._start_profiling()
    trainer._stop_profiling()

    trainer.actor_rollout_wg.start_profile.assert_called_once()
    trainer.actor_rollout_wg.stop_profile.assert_called_once()


def test_distinct_worker_groups_are_each_profiled_once():
    # When the reference/critic run in their own worker groups they must still be driven -- the
    # dedup only skips *aliases* of actor_rollout_wg, never genuinely separate processes.
    manager = MagicMock()
    trainer = _trainer(_StubTrainer, llm_server_manager=manager)
    trainer.use_reference_policy = True
    trainer.use_critic = True
    trainer.ref_policy_wg = MagicMock()
    trainer.critic_wg = MagicMock()

    trainer._start_profiling()
    trainer._stop_profiling()

    for wg in (trainer.actor_rollout_wg, trainer.ref_policy_wg, trainer.critic_wg):
        wg.start_profile.assert_called_once()
        wg.stop_profile.assert_called_once()


def test_separate_async_closes_hybrid_and_standalone_rollout_once():
    hybrid, standalone = MagicMock(), MagicMock()
    trainer = _trainer(
        PPOTrainerSeparateAsync,
        llm_server_manager=hybrid,
        standalone_server_manager=standalone,
    )
    _prepare_separate_step_end(trainer)

    trainer._start_profiling()
    trainer.on_step_end()
    trainer._stop_profiling()

    for manager in (hybrid, standalone):
        manager.start_profile.assert_called_once()
        manager.stop_profile.assert_called_once()


def test_colocate_async_stops_rollout_only_after_replicas_sleep():
    manager = MagicMock()
    trainer = _trainer(PPOTrainerColocateAsync, llm_server_manager=manager)
    trainer._start_rollout_profiling()
    calls = MagicMock()
    calls.attach_mock(trainer.checkpoint_manager.abort_replicas, "abort")
    calls.attach_mock(trainer.checkpoint_manager.sleep_replicas, "sleep")
    calls.attach_mock(manager.stop_profile, "stop")

    trainer.on_sample_end()
    trainer._stop_profiling()

    assert calls.mock_calls == [call.abort(), call.sleep(), call.stop()]
    manager.stop_profile.assert_called_once()


def test_rollout_stops_before_the_first_training_stage():
    manager = MagicMock()
    trainer = _trainer(PPOTrainerSync, llm_server_manager=manager)
    calls = MagicMock()
    calls.attach_mock(manager.start_profile, "rollout_start")
    calls.attach_mock(manager.stop_profile, "rollout_stop")

    batch = SimpleNamespace(extra_info={})
    trainer.config.actor_rollout_ref = OmegaConf.create({"rollout": {"temperature": 1.0}})
    trainer.config.trainer = OmegaConf.create({"critic_warmup": 0})
    trainer.replay_buffer = MagicMock()
    trainer.replay_buffer.sample.return_value = (batch, {})
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=object())

    for name in ("_balance_batch", "_compute_old_log_prob", "_compute_advantage", "_update_actor"):
        stage = MagicMock(return_value=batch)
        setattr(trainer, name, stage)
        calls.attach_mock(stage, name.removeprefix("_"))

    trainer._start_profiling()
    trainer._step_once({}, {}, sample_batch_size=1)

    assert [item[0] for item in calls.mock_calls] == [
        "rollout_start",
        "rollout_stop",
        "balance_batch",
        "compute_old_log_prob",
        "compute_advantage",
        "update_actor",
    ]

    trainer._stop_profiling()
    manager.stop_profile.assert_called_once()


@pytest.mark.parametrize("cls", [PPOTrainerSync, PPOTrainerColocateAsync])
@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("steps", [None, [2], [2, 3], [2, 4], [4]])
def test_fit_profiles_weight_sync_in_the_selected_step(monkeypatch, cls, continuous, steps):
    trainer, events = _fit_trainer(monkeypatch, cls, steps, continuous)

    trainer._fit(None)

    updates = [event for event in events if event[0] == "weights"]
    assert updates == [("weights", step, step in (steps or [])) for step in range(1, 5)]
    assert not trainer._test_profile_active
    for step in steps or []:
        assert events.index(("weights", step, True)) < next(
            i for i, event in enumerate(events) if i > events.index(("weights", step, True)) and event[0] == "stop"
        )


def _fit_trainer(monkeypatch, cls=PPOTrainerSync, steps=None, continuous=False):
    import verl.trainer.ppo.v1.trainer_base as module

    for name in ("SkipManager", "Tracking", "ValidationGenerationsLogger", "DapoFilteredRewardTableLogger", "tq"):
        monkeypatch.setattr(module, name, MagicMock())
    trainer = _trainer(cls, llm_server_manager=MagicMock())
    trainer.config.global_profiler.steps = steps
    trainer.config.global_profiler.profile_continuous_steps = continuous
    trainer.config.trainer = OmegaConf.create(
        dict(
            total_epochs=1,
            project_name="test",
            experiment_name="test",
            logger=["console"],
            val_before_train=False,
            save_freq=-1,
            test_freq=-1,
            rollout_data_dir=None,
        )
    )
    trainer.global_steps = 0
    trainer.steps_per_epoch = 10
    trainer.total_training_steps = 4
    trainer._test_profile_active = False
    events = []

    def start(**kwargs):
        assert not trainer._test_profile_active
        trainer._test_profile_active = True
        events.append(("start", trainer.global_steps))

    def stop(**kwargs):
        assert trainer._test_profile_active
        events.append(("stop", trainer.global_steps))
        trainer._test_profile_active = False

    trainer.actor_rollout_wg.start_profile.side_effect = start
    trainer.actor_rollout_wg.stop_profile.side_effect = stop
    trainer.checkpoint_manager.update_weights.side_effect = lambda *a, **kw: events.append(
        ("weights", trainer.global_steps, trainer._test_profile_active)
    )

    def step(metrics, timing):
        trainer.on_sample_end()
        return SimpleNamespace(keys=[], partition_id="test")

    trainer.step = MagicMock(side_effect=step)
    trainer.close_dataloaders = MagicMock()
    for name in (
        "on_train_begin",
        "on_train_end",
        "_reissue_inflight_prompts",
        "_compute_metrics",
        "_shutdown_dump_executor",
        "_shutdown_train_dataloader",
        "_save_checkpoint",
    ):
        setattr(trainer, name, MagicMock())
    return trainer, events


@pytest.mark.parametrize("cls", [PPOTrainerSync, PPOTrainerColocateAsync, PPOTrainerSeparateAsync])
@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("stage", ["step", "checkpoint", "weights", "exhausted"])
def test_fit_closes_all_profiles_on_early_exit(monkeypatch, cls, continuous, stage):
    from verl.trainer.ppo.v1.trainer_base import DataLoaderExhausted

    trainer, events = _fit_trainer(monkeypatch, cls, steps=[1, 2], continuous=continuous)
    if cls is PPOTrainerSeparateAsync:
        _prepare_separate_step_end(trainer)
        trainer.standalone_server_manager = MagicMock()
        trainer.on_step_begin = MagicMock()
        trainer.on_sample_end = MagicMock()
    if stage in ("step", "exhausted"):
        trainer.step.side_effect = DataLoaderExhausted() if stage == "exhausted" else RuntimeError("stage failed")
    elif stage == "checkpoint":
        trainer.config.trainer.save_freq = 1
        trainer._save_checkpoint.side_effect = RuntimeError("stage failed")
    elif cls is PPOTrainerSeparateAsync:
        trainer.standalone_checkpoint_manager.update_weights.side_effect = RuntimeError("stage failed")
    else:
        trainer.checkpoint_manager.update_weights.side_effect = RuntimeError("stage failed")

    if stage == "exhausted":
        trainer.fit(None)
        trainer.checkpoint_manager.update_weights.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match="stage failed"):
            trainer.fit(None)

    assert events == [("start", 1), ("stop", 1)]
    trainer.actor_rollout_wg.stop_profile.assert_called_once_with(run_command=False)
    for manager in trainer._rollout_server_managers():
        manager.start_profile.assert_called_once()
        manager.stop_profile.assert_called_once()
    assert not trainer._test_profile_active
    trainer.close_dataloaders.assert_called_once()


@pytest.mark.parametrize("current,previous", [(True, False), (False, True)])
def test_separate_async_uses_current_profile_selection(current, previous):
    manager = MagicMock()
    trainer = _trainer(PPOTrainerSeparateAsync, llm_server_manager=manager)
    _prepare_separate_step_end(trainer)
    trainer.curr_step_profile = current
    trainer.prev_step_profile = previous
    if current:
        trainer._start_rollout_profiling()

    trainer.on_step_end()

    assert manager.stop_profile.call_count == int(current)


def test_checkpoint_precedes_sync_and_final_profile_stop(monkeypatch):
    trainer, events = _fit_trainer(monkeypatch, steps=[1])
    trainer.total_training_steps = 1
    trainer.config.trainer.save_freq = 1
    trainer._save_checkpoint.side_effect = lambda: events.append(("checkpoint", trainer._test_profile_active))

    trainer._fit(None)

    assert events == [("start", 1), ("checkpoint", True), ("weights", 1, True), ("stop", 1)]
    trainer.actor_rollout_wg.stop_profile.assert_called_once_with(run_command=True)


@pytest.mark.parametrize("stage", ["checkpoint", "weights"])
def test_data_exhaustion_outside_step_is_not_swallowed(monkeypatch, stage):
    from verl.trainer.ppo.v1.trainer_base import DataLoaderExhausted

    trainer, _ = _fit_trainer(monkeypatch, steps=[1, 2], continuous=True)
    if stage == "checkpoint":
        trainer.config.trainer.save_freq = 1
        trainer._save_checkpoint.side_effect = DataLoaderExhausted()
    else:
        trainer.checkpoint_manager.update_weights.side_effect = DataLoaderExhausted()
    with pytest.raises(DataLoaderExhausted):
        trainer.fit(None)
    trainer.actor_rollout_wg.stop_profile.assert_called_once_with(run_command=False)
    trainer.close_dataloaders.assert_called_once()


def test_fit_exit_closes_continuous_window_after_metrics_error(monkeypatch):
    trainer, _ = _fit_trainer(monkeypatch, steps=[1, 2], continuous=True)
    trainer._compute_metrics.side_effect = RuntimeError("metrics failed")
    with pytest.raises(RuntimeError, match="metrics failed"):
        trainer.fit(None)
    trainer.actor_rollout_wg.stop_profile.assert_called_once_with(run_command=False)
    trainer.llm_server_manager.stop_profile.assert_called_once()


def test_final_training_step_closes_window_with_later_selected_steps(monkeypatch):
    trainer, _ = _fit_trainer(monkeypatch, steps=[1, 2], continuous=True)
    trainer.total_training_steps = 1
    trainer.fit(None)
    trainer.actor_rollout_wg.stop_profile.assert_called_once_with(run_command=True)
    trainer.llm_server_manager.stop_profile.assert_called_once()


def test_fit_closes_dataloaders_even_if_profiler_cleanup_fails(monkeypatch):
    trainer, _ = _fit_trainer(monkeypatch)
    trainer._fit = MagicMock()
    trainer._stop_profiling = MagicMock(side_effect=RuntimeError("collector failed"))
    with pytest.raises(RuntimeError, match="collector failed"):
        trainer.fit(None)
    trainer.close_dataloaders.assert_called_once()
