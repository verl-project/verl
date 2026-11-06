# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import copy
from types import SimpleNamespace
from unittest.mock import Mock, call, patch, sentinel

import pytest
from omegaconf import OmegaConf

from verl.experimental.reward_loop import reward_loop
from verl.experimental.reward_loop.reward_loop import RewardLoopManager, RewardLoopWorker


@pytest.fixture
def manager():
    manager = RewardLoopManager.__new__(RewardLoopManager)
    manager.config = SimpleNamespace(reward=SimpleNamespace(profiler={}, num_workers=2))
    manager.reward_router_address = None
    # Spec the actual worker API, including the inherited profiling methods.
    manager.reward_loop_workers = [Mock(spec=RewardLoopWorker), Mock(spec=RewardLoopWorker)]
    return manager


@pytest.mark.parametrize(
    "options, expected_runtime_env",
    [({"o": "rendered_%p"}, {"nsight": {"o": "rendered_%p"}}), (None, None)],
    ids=["nsight", "no-nsight"],
)
def test_worker_creation_attaches_nsight_options(manager, options, expected_runtime_env):
    workers = manager.reward_loop_workers
    actor_class = Mock()
    actor_class.options.return_value.remote.side_effect = workers
    manager.reward_loop_workers_class = actor_class
    with (
        patch.object(
            reward_loop.ray, "nodes", return_value=[{"NodeID": "node", "Alive": True, "Resources": {"CPU": 1}}]
        ),
        patch.object(reward_loop.ray.util.scheduling_strategies, "NodeAffinitySchedulingStrategy"),
        patch.object(
            reward_loop, "resolve_ray_actor_profiler_config", return_value=sentinel.profiler_config
        ) as resolve,
        patch.object(reward_loop, "build_ray_actor_nsight_launch_options", return_value=options) as build_options,
    ):
        manager._init_reward_loop_workers()

    resolve.assert_called_once_with(manager.config.reward.profiler, manager.config)
    assert build_options.call_args_list == [
        call(sentinel.profiler_config, rank=rank, role="reward_loop") for rank in range(2)
    ]
    assert len(actor_class.options.call_args_list) == 2
    for actor_call in actor_class.options.call_args_list:
        assert actor_call.kwargs.get("runtime_env") == expected_runtime_env


@pytest.mark.parametrize(
    "enabled, stop_kwargs, run_command",
    [(False, {}, True), (True, {}, True), (True, {"run_command": False}, False)],
    ids=["disabled", "enabled", "no-finish-command"],
)
def test_manager_initializes_and_controls_worker_profilers(manager, enabled, stop_kwargs, run_command):
    for rank, worker in enumerate(manager.reward_loop_workers):
        worker.init_profiler.remote.return_value = f"init-{rank}"
        worker.start_profile.remote.return_value = f"start-{rank}"
        worker.stop_profile.remote.return_value = f"stop-{rank}"
    with (
        patch.object(
            reward_loop, "resolve_ray_actor_profiler_config", return_value=sentinel.profiler_config
        ) as resolve,
        patch.object(reward_loop, "_ray_actor_profile_enabled", return_value=enabled) as is_enabled,
        patch.object(reward_loop.ray, "get") as get,
    ):
        manager._init_reward_loop_profilers()
        manager.start_profile(profile_step=3)
        manager.stop_profile(**stop_kwargs)

    resolve.assert_called_once_with(manager.config.reward.profiler, manager.config)
    is_enabled.assert_called_once_with(sentinel.profiler_config)
    for rank, worker in enumerate(manager.reward_loop_workers):
        if enabled:
            worker.init_profiler.remote.assert_called_once_with(rank, sentinel.profiler_config, "reward_loop")
            worker.start_profile.remote.assert_called_once_with(profile_step=3)
            worker.stop_profile.remote.assert_called_once_with(run_command=run_command)
        else:
            worker.init_profiler.remote.assert_not_called()
            worker.start_profile.remote.assert_not_called()
            worker.stop_profile.remote.assert_not_called()
    if enabled:
        assert get.call_args_list == [
            call(["init-0", "init-1"]),
            call(["start-0", "start-1"]),
            call(["stop-0", "stop-1"]),
        ]


def test_constructor_resolves_a_private_reward_model_config_and_initializes_worker_profilers():
    config = OmegaConf.create(
        {
            "reward": {
                "reward_model": {"enable": True, "rollout": {"full_determinism": False, "profiler": {"tool": "nsys"}}}
            }
        }
    )
    original = copy.deepcopy(config)
    resolved = OmegaConf.create({"ray_actor_nsight_options": {"o": "resolved"}})
    events = Mock()
    with (
        patch.object(reward_loop.ray, "remote"),
        patch.object(reward_loop, "resolve_reward_manager_cls"),
        patch.object(reward_loop, "RewardModelManager") as model_manager,
        patch.object(reward_loop, "resolve_ray_actor_profiler_config", return_value=resolved) as resolve,
        patch.object(RewardLoopManager, "_init_reward_loop_workers") as create_workers,
        patch.object(RewardLoopManager, "_init_reward_loop_profilers") as init_profilers,
    ):
        events.attach_mock(create_workers, "create_workers")
        events.attach_mock(init_profilers, "init_profilers")
        RewardLoopManager(config, rm_resource_pool=sentinel.pool)

    resolve.assert_called_once_with(original.reward.reward_model.rollout.profiler, config)
    model_config = model_manager.call_args.args[0]
    assert model_config.rollout.profiler == resolved
    assert model_manager.call_args.args[1] is sentinel.pool
    resolve.call_args.args[0].tool = "modified-by-test"
    assert config == original
    assert events.mock_calls == [call.create_workers(), call.init_profilers()]
