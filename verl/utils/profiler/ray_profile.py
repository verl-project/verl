# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Profiler lifecycle for independent Ray actors and GPU metrics collection."""

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.utils.config import omega_conf_to_dataclass

from .config import NsightToolConfig, ProfilerConfig
from .nsight import (
    _build_gpu_metrics_nsight_launch_options,
    _config_select,
    _is_gpu_metrics_profile_enabled,
)
from .profile import DistProfiler


class _RayActorProfilerMixin:
    """Profiler lifecycle shared by independently launched Ray actors."""

    def init_profiler(self, rank: int, global_profiler_config, role: str) -> None:
        config = omega_conf_to_dataclass(global_profiler_config, dataclass_type=ProfilerConfig)
        tool_config = (
            omega_conf_to_dataclass((config.tool_config or {}).get("nsys"), dataclass_type=NsightToolConfig)
            if config.tool == "nsys"
            else None
        )
        self.profiler = DistProfiler(rank=rank, config=config, tool_config=tool_config, save_file_prefix=role)

    def start_profile(self, **kwargs) -> None:
        self.profiler.start(**kwargs)

    def stop_profile(self, run_command: bool = True) -> None:
        self.profiler.stop(run_command=run_command)


@ray.remote
class _GpuMetricsProfilerActor(_RayActorProfilerMixin):
    """Own the single Nsight session allowed to use device-level GPU metrics."""

    def __init__(self, profiler_config) -> None:
        self.init_profiler(rank=0, global_profiler_config=profiler_config, role="gpu_metrics")


class _GpuMetricsProfilerManager:
    """Synchronously control the dedicated GPU-metrics actor from a trainer."""

    def __init__(self, actor) -> None:
        self.actor = actor

    def start_profile(self, **kwargs) -> None:
        ray.get(self.actor.start_profile.remote(**kwargs))

    def stop_profile(self, run_command: bool = True) -> None:
        ray.get(self.actor.stop_profile.remote(run_command=run_command))


def create_gpu_metrics_profiler(config) -> _GpuMetricsProfilerManager | None:
    """Create the sole GPU-metrics actor without a separate readiness RPC.

    Ray orders actor method calls after construction, and the trainer synchronously waits for
    ``start_profile``. That first start call therefore also serves as the readiness barrier.
    """
    if not _is_gpu_metrics_profile_enabled(config):
        return None

    nsight_options = _build_gpu_metrics_nsight_launch_options(config)
    global_profiler = config.global_profiler
    profiler_config = ProfilerConfig(
        tool="nsys",
        enable=True,
        ranks=[0],
        save_path=global_profiler.save_path,
        tool_config={"nsys": NsightToolConfig(discrete=False)},
        ray_actor_nsight_options=nsight_options,
        relocate_results=_config_select(global_profiler, "relocate_results", default=False),
        finish_hook_cmd=_config_select(global_profiler, "finish_hook_cmd"),
        finish_hook_all_ranks=_config_select(global_profiler, "finish_hook_all_ranks", default=False),
        finish_hook_ranks=list(_config_select(global_profiler, "finish_hook_ranks", default=[]) or []),
    )

    actor = _GpuMetricsProfilerActor.options(
        num_cpus=0,
        runtime_env={"nsight": nsight_options},
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,
        ),
    ).remote(profiler_config)
    return _GpuMetricsProfilerManager(actor)
