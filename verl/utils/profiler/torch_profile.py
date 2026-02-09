# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import functools
import os
from typing import Callable, Optional

import torch

from .config import ProfilerConfig, TorchProfilerToolConfig
from .profile import DistProfiler


def get_torch_profiler(
    contents: list[str],
    save_path: str,
    role: Optional[str] = None,
    save_file_prefix: Optional[str] = None,
    rank: int = 0,
):
    if role:
        save_path = os.path.join(save_path, role)

    os.makedirs(save_path, exist_ok=True)

    save_file_name = f"prof_rank-{rank}.json.gz"
    if save_file_prefix:
        save_file_name = f"{save_file_prefix}_{save_file_name}"
    save_path = os.path.join(save_path, save_file_name)

    def _trace_handler(prof):
        print(f"[Profiler] Saving trace to {save_path}")
        prof.export_chrome_trace(save_path)

    contents = set(contents) if contents else set()
    activities = []
    if not contents or "cpu" in contents:
        activities.append(torch.profiler.ProfilerActivity.CPU)
    if not contents or "cuda" in contents:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    return torch.profiler.profile(
        activities=activities,
        with_stack="stack" in contents,
        record_shapes="shapes" in contents,
        profile_memory="memory" in contents,
        on_trace_ready=_trace_handler,
    )


class Profiler(DistProfiler):
    """A PyTorch profiler wrapper class for collecting performance metrics.

    This profiler provides a convenient interface for profiling PyTorch operations,
    with support for:

    - CPU and CUDA activity profiling
    - Configurable profiling schedule (wait/warmup/active steps)
    - Multi-rank profiling support
    - Chrome trace export

    Args:
        config: Configuration object containing profiling parameters
    """

    _define_count = 0

    def __init__(
        self,
        rank,
        config: ProfilerConfig,
        tool_config: Optional[TorchProfilerToolConfig] = None,
        save_file_prefix=None,
    ):
        # note : if we do not set use_profile, it will be set as None, so that all function will be skip
        config = config or ProfilerConfig(ranks=[], enable=False)
        self.save_file_prefix = save_file_prefix

        if not tool_config:
            assert not config.enable, "tool_config must be provided when profiler is enabled"

        self.prof = None
        self.rank = rank
        self.config = config
        self.tool_config = tool_config
        self.contents = self.tool_config.contents
        self.save_path = self.config.save_path
        # Align with other profilers: read discrete mode, default to False for torch profiler
        self.discrete = getattr(self.tool_config, "discrete", False)

    def check(self):
        return self.prof is not None

    def start(self, **kwargs):
        role = kwargs.get("role", None)
        if not self.discrete and Profiler._define_count == 0:
            self.prof = get_torch_profiler(
                contents=self.contents,
                save_path=self.save_path,
                role=role,
                save_file_prefix=self.save_file_prefix,
                rank=self.rank,
            )
            print(f"[Profiler] started for rank {self.rank}")
            self.prof.start()
            Profiler._define_count += 1

    def step(self):
        if self.check():
            self.prof.step()

    def stop(self):
        if not self.discrete and Profiler._define_count == 1:
            self.step()
            print(f"[Profiler] stopped for rank {self.rank}")
            self.prof.stop()
            Profiler._define_count -= 1

    def annotate(self, message: Optional[str] = None, role: Optional[str] = None, **kwargs_outer) -> Callable:
        """Decorate a Worker member function to profile the current rank in the current training step.

        Requires the target function to be a member function of a Worker,
        which has a member field `profiler` with Profiler type.

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

                if not self.discrete:
                    # In continuous mode, we just record function, profiler started globally
                    with torch.profiler.record_function(profile_name):
                        return func(*args, **kwargs_inner)

                # In discrete mode, we start/stop profiler around the function
                prof = get_torch_profiler(
                    contents=self.contents,
                    save_path=self.save_path,
                    role=role,
                    save_file_prefix=self.save_file_prefix,
                    rank=self.rank,
                )
                prof.start()
                with torch.profiler.record_function(profile_name):
                    result = func(*args, **kwargs_inner)
                prof.stop()
                return result

            return wrapper

        return decorator
