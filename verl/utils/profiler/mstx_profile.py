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

# Inspired from https://gitee.com/ascend/MindSpeed-RL/blob/master/mindspeed_rl/utils/utils.py
import functools
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Optional

import torch_npu
from packaging import version
from torch_npu.npu import mstx

from .config import NPUToolConfig
from .profile import DistProfiler, ProfilerConfig


def mark_start_range(message: Optional[str] = None) -> None:
    """Start a mark range in the profiler.

    Args:
        message (str, optional):
            The message to be displayed in the profiler. Defaults to None.
    """
    return mstx.range_start(message=message)


def mark_end_range(range_id: str) -> None:
    """End a mark range in the profiler.

    Args:
        range_id (str):
            The id of the mark range to end.
    """
    return mstx.range_end(range_id)


def mark_annotate(message: Optional[str] = None) -> Callable:
    """Decorate a function to annotate a mark range along with the function life cycle.

    Args:
        message (str, optional):
            The message to be displayed in the profiler. Defaults to None.
    """

    def decorator(func):
        profile_message = message or func.__name__
        return mstx.mstx_range(profile_message)(func)

    return decorator


@contextmanager
def marked_timer(name: str, timing_raw: dict[str, float], *args: Any, **kwargs: Any) -> None:
    """Context manager for timing with MSTX markers.

    This utility function measures the execution time of code within its context,
    accumulates the timing information, and adds MSTX markers for profiling.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    if args:
        logging.warning(f"Args are not supported in mstx_profile, but received: {args}")
    if kwargs:
        logging.warning(f"Kwargs are not supported in mstx_profile, but received: {kwargs}")
    mark_range = mark_start_range(message=name)
    from .performance import _timer

    yield from _timer(name, timing_raw)
    mark_end_range(mark_range)


def get_npu_profiler(
    contents: list[str],
    profile_level: str,
    profile_save_path: str,
    analysis: bool,
    role: Optional[str] = None,
    profile_step: Optional[str] = None,
):
    """Generate and return an NPU profiler object.

    Args:
        contents (list[str]):
            A list of options to control the collection content,
            such as npu, cpu, memory, shapes, module, stack.
        profile_level (str):
            The collection level, which can be set to level_none,
            level0, level1 and level2.
        profile_save_path (str):
            The path to save the collected data.
        analysis (bool):
            Whether to enables automatic data parsing.
        role (str, optional):
            The role of the current data collection. Defaults to None.
        profile_step(str, optional):
            The current training step. Defaults to None.
    """
    if profile_level == "level_none":
        level = torch_npu.profiler.ProfilerLevel.Level_none
    elif profile_level == "level0":
        level = torch_npu.profiler.ProfilerLevel.Level0
    elif profile_level == "level1":
        level = torch_npu.profiler.ProfilerLevel.Level1
    elif profile_level == "level2":
        level = torch_npu.profiler.ProfilerLevel.Level2
    else:
        raise ValueError(f"level only supports level0, 1, 2, and level_none, but gets {profile_level}")

    if profile_step:
        profile_save_path = os.path.join(profile_save_path, profile_step)
    if role:
        profile_save_path = os.path.join(profile_save_path, role)

    # The ability to filter communication via mstx_domain_exclude requires torch_npu==2.1 or higher.
    if version.parse(torch_npu.__version__) < version.parse("2.1"):
        raise RuntimeError("torch_npu==2.1 or higher is required to use mstx_domain_exclude")

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=level,
        export_type=torch_npu.profiler.ExportType.Db,
        data_simplification=True,
        msprof_tx=True,
        mstx_domain_exclude=["communication"],
    )

    activites = []
    if contents is None or "npu" in contents:
        activites.append(torch_npu.profiler.ProfilerActivity.NPU)
    if contents is None or "cpu" in contents:
        activites.append(torch_npu.profiler.ProfilerActivity.CPU)

    prof = torch_npu.profiler.profile(
        with_modules=contents is None or "module" in contents,
        with_stack=contents is None or "stack" in contents,
        record_shapes=contents is None or "shapes" in contents,
        profile_memory=contents is None or "memory" in contents,
        activities=activites,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_save_path, analyse_flag=analysis),
        experimental_config=experimental_config,
    )
    return prof


class NPUProfiler(DistProfiler):
    """
    NPU profiler. Initialized in a worker to control the NPU profiler.
    """

    _define_count = 0

    def __init__(self, rank: int, config: ProfilerConfig, tool_config: NPUToolConfig, **kwargs):
        """Initialize the NsightSystemsProfiler.

        Args:
            rank (int): The rank of the current process.
            config (Optional[ProfilerConfig]): Configuration for the profiler. If None, a default configuration is used.
            tool_config (NPUToolConfig): The config to control npu profiler behavior.
        """
        if not config:
            config = ProfilerConfig(ranks=[], enable=False)
        if not tool_config:
            assert not config.enable, "tool_config must be set when profiler is enabled"
        self.discrete: bool = tool_config.discrete
        self.profile_npu = None
        self.profile_contents = tool_config.contents
        self.profile_level = tool_config.level
        self.profile_save_path = config.save_path
        self.analysis = tool_config.analysis

    def start(self, **kwargs):
        role = kwargs.get("role", None)
        if not self.discrete and NPUProfiler._define_count == 0:
            self.profile_npu = get_npu_profiler(
                contents=self.profile_contents,
                profile_level=self.profile_level,
                profile_save_path=self.profile_save_path,
                analysis=self.analysis,
                role=role,
            )
            self.profile_npu.start()
            NPUProfiler._define_count += 1

    def stop(self):
        if not self.discrete and NPUProfiler._define_count == 1:
            self.profile_npu.step()
            self.profile_npu.stop()
            NPUProfiler._define_count -= 1

    def annotate(self, message: Optional[str] = None, role: Optional[str] = None, **kwargs_outer) -> Callable:
        """Decorate a Worker member function to profile the current rank in the current training step.

        Requires the target function to be a member function of a Worker,
        which has a member field `profiler` with NPUProfiler type.

        Args:
            message (str, optional):
                The message to be displayed in the profiler. Defaults to None.
            role (str, optional):
                The role of the current data collection. Defaults to None.
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs_inner):
                profile_name = message or func.__name__
                discrete_mode = self.discrete

                if not discrete_mode:
                    mark_range = mark_start_range(message=profile_name)
                else:
                    profile_npu = get_npu_profiler(
                        contents=self.profile_contents,
                        profile_level=self.profile_level,
                        profile_save_path=self.profile_save_path,
                        analysis=self.analysis,
                        role=role,
                    )
                    profile_npu.start()
                    mark_range = mark_start_range(message=profile_name)

                result = func(*args, **kwargs_inner)

                if not discrete_mode:
                    mark_end_range(mark_range)
                else:
                    mark_end_range(mark_range)
                    profile_npu.step()
                    profile_npu.stop()

                return result

            return wrapper

        return decorator
