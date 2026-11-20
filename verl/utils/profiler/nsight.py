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

"""Nsight launch configuration for controllers, worker groups, and Ray actors."""

import copy
from dataclasses import is_dataclass, replace

from omegaconf import OmegaConf


def _count_capture_ranges(profile_steps, profile_continuous_steps: bool, *, discrete: bool = False) -> int:
    """Count captures using the worker-group defaults, merging whole-step windows."""
    steps = sorted(set(profile_steps or []))
    if discrete or not profile_continuous_steps:
        return (6 if discrete else 1) * len(steps)
    return sum(index == 0 or step != steps[index - 1] + 1 for index, step in enumerate(steps))


def _resolve_nsight_capture_options(options, profile_steps, profile_continuous_steps: bool) -> dict:
    """Copy launch options and fill an unspecified CUDA capture count."""
    options = copy.deepcopy(OmegaConf.to_container(options) if OmegaConf.is_config(options) else options)
    if options.get("capture-range") == "cudaProfilerApi" and options.get("capture-range-end") is None:
        count = _count_capture_ranges(profile_steps, profile_continuous_steps)
        if count:
            options["capture-range-end"] = f"repeat-shutdown:{count}"
    return options


def _render_nsight_output_name(output, *, rank: int, role: str) -> str:
    """Render role/rank in the output template, leaving Nsight host/PID placeholders intact."""
    if output is None:
        return f"{role}_rank_{rank}_%h_pid%p"
    output = str(output).replace("%q{RANK}", str(rank))
    if "{role}" in output or "{rank}" in output:
        return output.replace("{role}", role).replace("{rank}", str(rank))
    return f"{role}_rank_{rank}_{output}"


def _config_select(config, path: str, default=None):
    """Select from DictConfig, mappings, or structured config objects."""
    if config is None:
        return default
    if OmegaConf.is_config(config):
        return OmegaConf.select(config, path, default=default)
    value = config
    missing = object()
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part, missing)
        else:
            value = getattr(value, part, missing)
        if value is missing:
            return default
    return value


def _ray_actor_profile_enabled(profiler_config) -> bool:
    """Return whether Nsight profiling is enabled for a standalone actor."""
    return _config_select(profiler_config, "tool") == "nsys" and _config_select(
        profiler_config, "enable", default=False
    )


def _profile_rank_selected(profiler_config, rank: int, *, ranks=None) -> bool:
    """Select a rank; callers must check the overall enable gate separately."""
    if _config_select(profiler_config, "all_ranks", default=False):
        return True
    if ranks is None:
        ranks = _config_select(profiler_config, "ranks", default=[])
    if ranks:
        return rank in ranks
    # Preserve DistProfiler's default: select rank 0 only when profiling is enabled.
    return rank == 0 and _config_select(profiler_config, "enable", default=False)


def _rollout_replica_ranks(ranks, replica_world_size: int) -> list[int]:
    """Map requested global GPU ranks to their owning rollout replicas."""
    return sorted({int(rank) // replica_world_size for rank in ranks})


def resolve_ray_actor_profiler_config(profiler_config, config):
    """Resolve capture options in a copied role config, keeping output templates unrendered."""
    if _config_select(profiler_config, "tool") != "nsys":
        return profiler_config
    options_template = _config_select(profiler_config, "ray_actor_nsight_options")
    if options_template is None:
        options_template = _config_select(config, "global_profiler.global_tool_config.nsys.worker_nsight_options")
    if options_template is None:
        return profiler_config
    options_template = _resolve_nsight_capture_options(
        options_template,
        _config_select(config, "global_profiler.steps"),
        _config_select(config, "global_profiler.profile_continuous_steps", default=False),
    )
    if is_dataclass(profiler_config):
        return replace(profiler_config, ray_actor_nsight_options=options_template)
    resolved = copy.deepcopy(profiler_config)
    if isinstance(resolved, dict):
        resolved["ray_actor_nsight_options"] = options_template
    else:
        resolved.ray_actor_nsight_options = options_template
    return resolved


def _render_ray_actor_nsight_options(profiler_config, *, rank: int, role: str) -> dict:
    """Copy shared options and render the output template for one actor."""
    options_template = _config_select(profiler_config, "ray_actor_nsight_options")
    if options_template is None:
        raise ValueError("Nsight worker launch options are required for standalone Ray actor profiling")
    options = copy.deepcopy(
        OmegaConf.to_container(options_template) if OmegaConf.is_config(options_template) else options_template
    )
    options["o"] = _render_nsight_output_name(options.get("o"), rank=rank, role=role)
    return options


def build_ray_actor_nsight_launch_options(
    profiler_config,
    *,
    rank: int,
    role: str,
) -> dict | None:
    """Build Ray ``runtime_env.nsight`` options for a selected standalone actor.

    Unlike a :class:`RayWorkerGroup`, reward-loop workers and rollout
    servers are created as individual Ray actors. They therefore need the Nsight launch
    options attached at their own ``.options(...)`` call. ``{role}`` and ``{rank}`` in the
    configured output name are expanded for these actors. Managers resolve the global
    capture window with ``resolve_ray_actor_profiler_config`` before calling this helper.
    """
    if not _ray_actor_profile_enabled(profiler_config) or not _profile_rank_selected(profiler_config, rank):
        return None
    if _config_select(profiler_config, "tool_config.nsys.discrete", default=False):
        raise ValueError("Standalone Ray actor nsys profiling requires discrete=False")
    return _render_ray_actor_nsight_options(profiler_config, rank=rank, role=role)


def _build_rollout_ray_actor_nsight_launch_options(
    profiler_config,
    *,
    replica_rank: int,
    replica_world_size: int,
    role: str,
) -> dict | None:
    """Build Nsight options for the server processes of one rollout replica."""
    if not _ray_actor_profile_enabled(profiler_config):
        return None
    ranks = _rollout_replica_ranks(_config_select(profiler_config, "ranks", default=[]) or [], replica_world_size)
    if not _profile_rank_selected(profiler_config, replica_rank, ranks=ranks):
        return None
    if _config_select(profiler_config, "tool_config.nsys.discrete", default=False):
        raise ValueError("Standalone Ray actor nsys profiling requires discrete=False")
    return _render_ray_actor_nsight_options(profiler_config, rank=replica_rank, role=role)


def is_controller_nsys_api_control_enabled(config) -> bool:
    """Return whether the controller uses CUDA-profiler range control."""
    controller_options = (
        OmegaConf.select(
            config,
            "global_profiler.global_tool_config.nsys.controller_nsight_options",
            default={},
        )
        or {}
    )
    return (
        OmegaConf.select(config, "global_profiler.tool") == "nsys"
        and controller_options.get("capture-range") == "cudaProfilerApi"
    )


def build_controller_nsight_launch_options(config) -> dict:
    """Build controller launch options by resolving the configured capture window."""
    return _resolve_nsight_capture_options(
        config.global_profiler.global_tool_config.nsys.controller_nsight_options,
        config.global_profiler.steps,
        config.global_profiler.profile_continuous_steps,
    )


def _is_gpu_metrics_profile_enabled(config) -> bool:
    """Return whether a dedicated Ray actor should collect system-wide GPU metrics."""
    options = OmegaConf.select(
        config,
        "global_profiler.global_tool_config.nsys.gpu_metrics_nsight_options",
        default={},
    )
    steps = OmegaConf.select(config, "global_profiler.steps")
    return OmegaConf.select(config, "global_profiler.tool") == "nsys" and bool(steps) and bool(options)


def _build_gpu_metrics_nsight_launch_options(config) -> dict:
    """Build launch options for the sole GPU-metrics collection actor."""
    options = OmegaConf.select(
        config,
        "global_profiler.global_tool_config.nsys.gpu_metrics_nsight_options",
        default={},
    )
    if not options:
        raise ValueError("gpu_metrics_nsight_options must not be empty when GPU metrics profiling is enabled")
    if options.get("capture-range") != "cudaProfilerApi":
        raise ValueError("GPU metrics profiling requires capture-range=cudaProfilerApi")
    return _resolve_nsight_capture_options(
        options, config.global_profiler.steps, config.global_profiler.profile_continuous_steps
    )
