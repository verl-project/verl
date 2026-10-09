# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Private Ray backend configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

from verl.runtime.config import parse_env_vars


@dataclass(frozen=True, slots=True)
class RayRuntimeConfig:
    """Store validated Ray backend configuration.

    Attributes:
        ray_init: Keyword arguments for ``ray.init`` when Ray is not yet initialized.
            Must not include ``runtime_env.env_vars``; those come from root ``env_vars``.
        env_vars: Environment variables from root RuntimeConfig (injected).
        timeline_json_file: Optional Ray timeline output written when Runtime closes.
        profile_steps: Training steps to profile, or None to disable worker profiling.
        worker_nsight_options: ``runtime_env["nsight"]`` options for worker actors.
            Only applied when ``profile_steps`` is non-empty.
    """

    ray_init: Mapping[str, Any]
    env_vars: Mapping[str, str]
    timeline_json_file: str | None
    placement_ready_timeout_s: float
    profile_steps: tuple[int, ...] | None = None
    worker_nsight_options: Mapping[str, Any] | None = None


def parse_ray_section(section: Mapping[str, object] | None) -> RayRuntimeConfig:
    """Convert the selected ``ray`` section into typed private config.

    Args:
        section: Selected backend section mapping after root ``env_vars`` injection,
            or None when absent.

    Returns:
        Validated private Ray configuration.
    """
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise TypeError(f'RuntimeConfig["ray"] must be a mapping, got {type(section)!r}')

    ray_init_raw = section.get("ray_init", {})
    if ray_init_raw is None:
        ray_init_raw = {}
    if not isinstance(ray_init_raw, Mapping):
        raise TypeError(f'RuntimeConfig["ray"]["ray_init"] must be a mapping, got {type(ray_init_raw)!r}')
    ray_init = {str(k): v for k, v in ray_init_raw.items()}
    # Root env_vars is authoritative; drop any nested copy under ray_init.
    runtime_env = ray_init.get("runtime_env")
    if isinstance(runtime_env, Mapping) and "env_vars" in runtime_env:
        runtime_env = {k: v for k, v in runtime_env.items() if k != "env_vars"}
        if runtime_env:
            ray_init["runtime_env"] = runtime_env
        else:
            ray_init.pop("runtime_env", None)

    timeline_json_file = section.get("timeline_json_file")
    if timeline_json_file is not None and not isinstance(timeline_json_file, str):
        raise TypeError(
            f'RuntimeConfig["ray"]["timeline_json_file"] must be a str or None, got {type(timeline_json_file)!r}'
        )

    placement_ready_timeout_s = section.get("placement_ready_timeout_s", 300.0)
    if isinstance(placement_ready_timeout_s, bool) or not isinstance(placement_ready_timeout_s, int | float):
        raise TypeError('RuntimeConfig["ray"]["placement_ready_timeout_s"] must be a positive finite number')
    placement_ready_timeout_s = float(placement_ready_timeout_s)
    if placement_ready_timeout_s <= 0 or not isfinite(placement_ready_timeout_s):
        raise ValueError('RuntimeConfig["ray"]["placement_ready_timeout_s"] must be a positive finite number')

    profile_steps = section.get("profile_steps")
    if profile_steps is not None:
        if isinstance(profile_steps, str) or not isinstance(profile_steps, Sequence):
            raise TypeError(
                f'RuntimeConfig["ray"]["profile_steps"] must be a sequence of int or None, got {type(profile_steps)!r}'
            )
        if any(isinstance(step, bool) or not isinstance(step, int) for step in profile_steps):
            raise TypeError('RuntimeConfig["ray"]["profile_steps"] must contain only int values')
        profile_steps = tuple(profile_steps)

    worker_nsight_options = section.get("worker_nsight_options")
    if worker_nsight_options is not None:
        if not isinstance(worker_nsight_options, Mapping):
            raise TypeError(
                f'RuntimeConfig["ray"]["worker_nsight_options"] must be a mapping, got {type(worker_nsight_options)!r}'
            )
        worker_nsight_options = {str(k): v for k, v in worker_nsight_options.items()}

    return RayRuntimeConfig(
        ray_init=ray_init,
        env_vars=parse_env_vars(section.get("env_vars")),
        timeline_json_file=timeline_json_file,
        placement_ready_timeout_s=placement_ready_timeout_s,
        profile_steps=profile_steps,
        worker_nsight_options=worker_nsight_options,
    )
