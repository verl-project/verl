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
    calls = MagicMock()
    calls.attach_mock(trainer.checkpoint_manager.sleep_replicas, "sleep")
    calls.attach_mock(manager.stop_profile, "stop")

    with patch.object(PPOTrainer, "_add_batch_to_generate") as submit:
        calls.attach_mock(submit, "submit")
        trainer._add_batch_to_generate()
        trainer.on_sample_end()

    assert calls.mock_calls == [call.submit(), call.sleep(), call.stop()]
    manager.start_profile.assert_not_called()


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
    trainer._stop_profiling()
    trainer.on_step_end()

    for manager in (hybrid, standalone):
        manager.start_profile.assert_called_once()
        manager.stop_profile.assert_called_once()


def test_colocate_async_stops_rollout_only_after_replicas_sleep():
    manager = MagicMock()
    trainer = _trainer(PPOTrainerColocateAsync, llm_server_manager=manager)
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
