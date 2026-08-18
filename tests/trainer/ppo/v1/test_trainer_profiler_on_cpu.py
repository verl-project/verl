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

"""CPU-only unit tests for profiling orchestration in the V1 PPO trainer."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def _make_trainer(
    trainer_cls: type[PPOTrainer],
    steps: list[int] | None,
    *,
    continuous: bool = False,
    global_step: int = 1,
    use_reference_policy: bool = False,
    use_critic: bool = False,
) -> PPOTrainer:
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.config = SimpleNamespace(
        global_profiler=SimpleNamespace(
            steps=steps,
            profile_continuous_steps=continuous,
        )
    )
    trainer.global_steps = global_step
    trainer.actor_rollout_wg = MagicMock()
    trainer.ref_policy_wg = MagicMock()
    trainer.critic_wg = MagicMock()
    trainer.llm_server_manager = MagicMock()
    trainer.standalone_server_manager = MagicMock()
    trainer.use_reference_policy = use_reference_policy
    trainer.use_critic = use_critic
    return trainer


def _profile_boundaries(trainer: PPOTrainer, max_step: int = 9) -> tuple[list[int], list[int]]:
    """Returns which steps the trainer would start and stop profiling on."""
    starts = []
    stops = []
    for step in range(max_step + 1):
        trainer.global_steps = step
        if trainer._should_start_profiling():
            starts.append(step)
        if trainer._should_stop_profiling():
            stops.append(step)
    return starts, stops


def test_continuous_profiling_uses_span_boundaries():
    steps = [1, 2, 4, 6, 7, 8]
    trainer = _make_trainer(_StubTrainer, steps, continuous=True)

    starts, stops = _profile_boundaries(trainer)

    assert starts == [1, 4, 6]
    assert stops == [2, 4, 8]


def test_discrete_profiling_uses_each_requested_step():
    steps = [1, 2, 4, 6, 7, 8]
    trainer = _make_trainer(_StubTrainer, steps, continuous=False)

    starts, stops = _profile_boundaries(trainer)

    assert starts == list(steps)
    assert stops == list(steps)


def test_profiling_disabled_without_requested_steps():
    trainer = _make_trainer(_StubTrainer, steps=None)

    starts, stops = _profile_boundaries(trainer)

    assert starts == []
    assert stops == []


def test_sync_trainer_profiles_required_components():
    trainer = _make_trainer(
        PPOTrainerSync,
        steps=[4],
        global_step=4,
        use_critic=True,
        use_reference_policy=False,
    )

    trainer._start_profiling()
    trainer._stop_profiling()

    trainer.actor_rollout_wg.start_profile.assert_called_once_with(role="e2e", profile_step=4)
    trainer.actor_rollout_wg.stop_profile.assert_called_once_with()

    trainer.llm_server_manager.start_profile.assert_called_once_with()
    trainer.llm_server_manager.stop_profile.assert_called_once_with()

    trainer.critic_wg.start_profile.assert_called_once_with(profile_step=4)
    trainer.critic_wg.stop_profile.assert_called_once_with()

    trainer.ref_policy_wg.start_profile.assert_not_called()
    trainer.ref_policy_wg.stop_profile.assert_not_called()


def test_colocate_async_trainer_profiles_required_components():
    trainer = _make_trainer(
        PPOTrainerColocateAsync,
        steps=[4],
        global_step=4,
        use_critic=False,
        use_reference_policy=True,
    )

    trainer._start_profiling()
    trainer._stop_profiling()

    trainer.actor_rollout_wg.start_profile.assert_called_once_with(role="e2e", profile_step=4)
    trainer.actor_rollout_wg.stop_profile.assert_called_once_with()

    trainer.llm_server_manager.start_profile.assert_called_once_with()
    trainer.llm_server_manager.stop_profile.assert_called_once_with()

    trainer.critic_wg.start_profile.assert_not_called()
    trainer.critic_wg.stop_profile.assert_not_called()

    trainer.ref_policy_wg.start_profile.assert_called_once_with(profile_step=4)
    trainer.ref_policy_wg.stop_profile.assert_called_once_with()


def test_separate_async_trainer_profiles_required_components():
    trainer = _make_trainer(
        PPOTrainerSeparateAsync,
        steps=[4],
        global_step=4,
        use_reference_policy=True,
        use_critic=True,
    )

    trainer._start_profiling()
    trainer._stop_profiling()

    trainer.actor_rollout_wg.start_profile.assert_called_once_with(role="e2e", profile_step=4)
    trainer.actor_rollout_wg.stop_profile.assert_called_once_with()

    trainer.ref_policy_wg.start_profile.assert_called_once_with(profile_step=4)
    trainer.ref_policy_wg.stop_profile.assert_called_once_with()

    trainer.critic_wg.start_profile.assert_called_once_with(profile_step=4)
    trainer.critic_wg.stop_profile.assert_called_once_with()

    trainer.standalone_server_manager.start_profile.assert_called_once_with()
    trainer.standalone_server_manager.stop_profile.assert_called_once_with()

    trainer.llm_server_manager.start_profile.assert_not_called()
    trainer.llm_server_manager.stop_profile.assert_not_called()


def test_rollout_managers_are_not_profiled_on_unselected_steps():
    trainers_and_managers = [
        (_make_trainer(PPOTrainerSync, steps=[4], global_step=3), "llm_server_manager"),
        (_make_trainer(PPOTrainerColocateAsync, steps=[4], global_step=3), "llm_server_manager"),
        (_make_trainer(PPOTrainerSeparateAsync, steps=[4], global_step=3), "standalone_server_manager"),
    ]

    for trainer, manager_name in trainers_and_managers:
        trainer._start_profiling()
        trainer._stop_profiling()

        trainer.actor_rollout_wg.start_profile.assert_not_called()
        trainer.actor_rollout_wg.stop_profile.assert_not_called()

        manager = getattr(trainer, manager_name)
        manager.start_profile.assert_not_called()
        manager.stop_profile.assert_not_called()
