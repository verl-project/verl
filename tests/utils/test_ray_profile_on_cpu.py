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

from unittest.mock import Mock, patch, sentinel

import pytest
from omegaconf import OmegaConf

from verl.utils.profiler import ray_profile
from verl.utils.profiler.config import NsightToolConfig, ProfilerConfig
from verl.utils.profiler.ray_profile import (
    _GpuMetricsProfilerActor,
    _GpuMetricsProfilerManager,
    _RayActorProfilerMixin,
    create_gpu_metrics_profiler,
)


@pytest.mark.parametrize(
    "config",
    [
        ProfilerConfig(
            tool="nsys", enable=True, ranks=[2], save_path="traces", tool_config={"nsys": {"discrete": False}}
        ),
        OmegaConf.create(
            {
                "tool": "nsys",
                "enable": True,
                "ranks": [2],
                "save_path": "traces",
                "tool_config": {"nsys": {"discrete": False}},
            }
        ),
    ],
    ids=["dataclass", "omegaconf"],
)
def test_init_profiler_converts_nsight_config(config):
    worker = _RayActorProfilerMixin()
    with patch("verl.utils.profiler.ray_profile.DistProfiler") as profiler:
        worker.init_profiler(2, config, role="reward_loop")

    converted = profiler.call_args.kwargs["config"]
    assert isinstance(converted, ProfilerConfig)
    assert converted.tool == "nsys"
    assert converted.enable is True
    assert converted.ranks == [2]
    assert converted.save_path == "traces"
    tool_config = profiler.call_args.kwargs["tool_config"]
    assert isinstance(tool_config, NsightToolConfig)
    assert tool_config.discrete is False
    profiler.assert_called_once_with(rank=2, config=converted, tool_config=tool_config, save_file_prefix="reward_loop")
    assert worker.profiler is profiler.return_value


def test_init_profiler_defaults_nsight_tool_config():
    worker = _RayActorProfilerMixin()
    config = ProfilerConfig(tool="nsys", tool_config=None)
    with patch("verl.utils.profiler.ray_profile.DistProfiler") as profiler:
        worker.init_profiler(0, config, role="reward_loop")

    profiler.assert_called_once_with(
        rank=0, config=config, tool_config=NsightToolConfig(), save_file_prefix="reward_loop"
    )


def test_init_profiler_ignores_nsight_config_for_other_tools():
    worker = _RayActorProfilerMixin()
    config = ProfilerConfig(tool=None, tool_config={"nsys": {"discrete": True}})
    with patch("verl.utils.profiler.ray_profile.DistProfiler") as profiler:
        worker.init_profiler(0, config, role="reward_loop")

    profiler.assert_called_once_with(rank=0, config=config, tool_config=None, save_file_prefix="reward_loop")


def test_start_profile_forwards_arguments():
    worker = _RayActorProfilerMixin()
    worker.profiler = Mock()

    worker.start_profile(role="reward_loop", profile_step="3")

    worker.profiler.start.assert_called_once_with(role="reward_loop", profile_step="3")


@pytest.mark.parametrize("kwargs, run_command", [({}, True), ({"run_command": False}, False)])
def test_stop_profile_forwards_finish_hook_control(kwargs, run_command):
    worker = _RayActorProfilerMixin()
    worker.profiler = Mock()

    worker.stop_profile(**kwargs)

    worker.profiler.stop.assert_called_once_with(run_command=run_command)


def test_actor_initializes_its_own_profiler():
    # Unwrap Ray's actor class to test construction without starting a Ray runtime.
    actor_class = _GpuMetricsProfilerActor.__ray_metadata__.modified_class
    with patch.object(actor_class, "init_profiler") as init_profiler:
        actor_class(sentinel.config)
    init_profiler.assert_called_once_with(rank=0, global_profiler_config=sentinel.config, role="gpu_metrics")


@pytest.mark.parametrize("stop_kwargs, run_command", [({}, True), ({"run_command": False}, False)])
def test_manager_waits_for_remote_profiling_actions(stop_kwargs, run_command):
    actor = Mock()
    manager = _GpuMetricsProfilerManager(actor)
    with patch.object(ray_profile.ray, "get") as get:
        manager.start_profile(profile_step=3)
        get.assert_called_once_with(actor.start_profile.remote.return_value)
        get.reset_mock()
        manager.stop_profile(**stop_kwargs)
        get.assert_called_once_with(actor.stop_profile.remote.return_value)
    actor.start_profile.remote.assert_called_once_with(profile_step=3)
    actor.stop_profile.remote.assert_called_once_with(run_command=run_command)


def test_create_gpu_metrics_profiler_skips_disabled_collection():
    config = OmegaConf.create({})
    with (
        patch.object(ray_profile, "_is_gpu_metrics_profile_enabled", return_value=False) as is_enabled,
        patch.object(ray_profile, "_build_gpu_metrics_nsight_launch_options"),
        patch.object(ray_profile, "_GpuMetricsProfilerActor") as actor_class,
    ):
        assert create_gpu_metrics_profiler(config) is None
    is_enabled.assert_called_once_with(config)
    actor_class.options.return_value.remote.assert_not_called()


@pytest.mark.parametrize(
    "hook_settings",
    [
        {},
        {
            "relocate_results": True,
            "finish_hook_cmd": "upload-traces",
            "finish_hook_all_ranks": True,
            "finish_hook_ranks": [0],
        },
    ],
    ids=["default-hooks", "configured-hooks"],
)
def test_create_gpu_metrics_profiler(hook_settings):
    config = OmegaConf.create({"global_profiler": {"save_path": "traces", **hook_settings}})
    options = {"capture-range": "cudaProfilerApi", "gpu-metrics-devices": "all"}
    with (
        patch.object(ray_profile, "_is_gpu_metrics_profile_enabled", return_value=True) as is_enabled,
        patch.object(ray_profile, "_build_gpu_metrics_nsight_launch_options", return_value=options) as build_options,
        patch.object(ray_profile, "_GpuMetricsProfilerActor") as actor_class,
        patch.object(ray_profile.ray, "get_runtime_context") as context,
        patch.object(ray_profile, "NodeAffinitySchedulingStrategy", return_value=sentinel.affinity) as affinity,
    ):
        context.return_value.get_node_id.return_value = "controller-node"
        manager = create_gpu_metrics_profiler(config)

    is_enabled.assert_called_once_with(config)
    build_options.assert_called_once_with(config)
    affinity.assert_called_once_with(node_id="controller-node", soft=False)
    actor_class.options.assert_called_once_with(
        num_cpus=0, runtime_env={"nsight": options}, scheduling_strategy=sentinel.affinity
    )
    actor_class.options.return_value.remote.assert_called_once_with(
        ProfilerConfig(
            tool="nsys",
            enable=True,
            ranks=[0],
            save_path="traces",
            tool_config={"nsys": NsightToolConfig(discrete=False)},
            ray_actor_nsight_options=options,
            **hook_settings,
        )
    )
    assert isinstance(manager, _GpuMetricsProfilerManager)
    assert manager.actor is actor_class.options.return_value.remote.return_value
