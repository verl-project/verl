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

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig


@dataclass
class NsightToolConfig(BaseConfig):
    """Nsight tool config."""

    "True for each task has its own database, False for all tasks in one training step share one database."
    discrete: bool = False
    name: str = "nsight"

    def __post_init__(self) -> None:
        pass


@dataclass
class TorchProfilerToolConfig(BaseConfig):
    """Torch profiler tool config."""

    # options: cuda, cpu, memory, shapes, stack
    contents: list[str] = field(default_factory=list)
    discrete: bool = False
    name: str = "torch"

    def __post_init__(self) -> None:
        """config validation logics go here"""
        __support_contents = ["cuda", "cpu", "memory", "shapes", "stack"]
        for content in self.contents:
            assert content in __support_contents, (
                f"Profiler contents only supports {__support_contents}, but gets {content}"
            )
        assert isinstance(self.contents, list), f"Profiler contents must be of type list, got {type(self.contents)}"


@dataclass
class TorchMemoryToolConfig(BaseConfig):
    """Torch memory profiler tool config.

    Args:
        trace_alloc_max_entries (int): Maximum number of memory allocation entries to track.
        stack_depth (int): Stack trace depth for memory allocations.
    """

    trace_alloc_max_entries: int = 100_000
    stack_depth: int = 32
    name: str = "torch_memory"

    def __post_init__(self) -> None:
        """config validation logics go here"""
        assert isinstance(self.trace_alloc_max_entries, int), (
            f"trace_alloc_max_entries must be int, got {type(self.trace_alloc_max_entries)}"
        )
        assert isinstance(self.stack_depth, int), f"stack_depth must be int, got {type(self.stack_depth)}"
        assert self.trace_alloc_max_entries > 0, (
            f"trace_alloc_max_entries must be positive, got {self.trace_alloc_max_entries}"
        )
        assert self.stack_depth > 0, f"stack_depth must be positive, got {self.stack_depth}"


@dataclass
class NPUToolConfig(NsightToolConfig):
    """NPU profiler too; config."""

    # options: npu, cpu, memory, shapes, module, stack
    contents: list[str] = field(default_factory=list)

    # Collection level, optional values: level_none, level0, level1, level2.
    level: str = "level0"

    # Whether to automatically parse the data.
    analysis: bool = False

    name: str = "npu"

    def __post_init__(self) -> None:
        """config validation logics go here"""
        assert isinstance(self.contents, list), f"Profiler contents must be of type list, got {type(self.contents)}"
        assert isinstance(self.level, str), f"Profiler level must be of type str, got {type(self.level)}"
        assert isinstance(self.analysis, bool), f"Profiler analysis must be of type bool, got {type(self.analysis)}"
        for content in self.contents:
            assert content in ["npu", "cpu", "memory", "shapes", "module", "stack"], (
                f"Profiler contents only supports npu, cpu, memory, shapes, module, stack, but gets {content}"
            )
        assert self.level in ["level_none", "level0", "level1", "level2"], (
            f"Profiler level only supports level0, 1, 2, and level_none, but gets {self.level}"
        )


@dataclass
class ProfilerConfig(BaseConfig):
    """Worker profiler config.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        discrete (bool): True for each task has its own database, False for all tasks in one training step
          share one database.
        all_ranks (bool): Whether to profile all ranks.
        ranks (list[int]): The ranks that will be profiled. Defaults to [].
        global_tool_config (Any): Global tool configuration for all profiling tools.
    """

    tool: Optional[str] = MISSING
    enable: bool = False
    all_ranks: bool = False
    ranks: list[int] = field(default_factory=list)
    save_path: Optional[str] = MISSING
    tool_config: Any = MISSING  # Just a placeholder, will use configs above directly
    global_tool_config: Optional[Any] = None  # Global tool configuration for all profiling tools

    def union(self, other: "ProfilerConfig") -> "ProfilerConfig":
        assert self.tool == other.tool, f"Cannot union ProfilerConfig with different tools: {self.tool} vs {other.tool}"
        return ProfilerConfig(
            tool=self.tool,
            enable=self.enable or other.enable,
            all_ranks=self.all_ranks or other.all_ranks,
            ranks=list(set(self.ranks or []) | set(other.ranks or [])),
            save_path=self.save_path,
            tool_config=self.tool_config,
            global_tool_config=self.global_tool_config or other.global_tool_config,
        )

    def intersect(self, other: "ProfilerConfig") -> "ProfilerConfig":
        assert self.tool == other.tool, (
            f"Cannot intersect ProfilerConfig with different tools: {self.tool} vs {other.tool}"
        )
        return ProfilerConfig(
            tool=self.tool,
            enable=self.enable and other.enable,
            all_ranks=self.all_ranks and other.all_ranks,
            ranks=list(set(self.ranks or []) & set(other.ranks or [])),
            save_path=self.save_path,
            tool_config=self.tool_config,
            global_tool_config=self.global_tool_config if self.global_tool_config else other.global_tool_config,
        )

    def __post_init__(self) -> None:
        """config validation logics go here"""
        assert isinstance(self.ranks, set | list | tuple), (
            f"Profiler ranks must be of type list, got {type(self.ranks)}"
        )


def build_vllm_profiler_args(profiler_config: ProfilerConfig, tool_config: BaseConfig, rank: int) -> dict:
    """
    Build arguments and environment variables for vLLM profiler.

    Acts as an adapter to bridge verl's unified profiler config and vLLM's specific requirements.
    It sets environment variables for compatibility and constructs arguments for vLLM >= 0.13.0.

    Args:
        profiler_config (ProfilerConfig): The unified profiler configuration.
        tool_config (BaseConfig): The tool configuration.
        rank (int): The rank of the replica.

    Returns:
        dict: A dictionary of arguments to be passed to vLLM's start_profile method.
    """
    if not profiler_config or not tool_config or not hasattr(tool_config, "contents"):
        return {}

    contents = tool_config.contents
    with_stack = True if "stack" in contents or "module" in contents else False
    record_shapes = True if "shapes" in contents else False
    with_memory = True if "memory" in contents else False
    save_path = os.path.join(profiler_config.save_path, f"agent_loop_rollout_replica_{rank}")

    # vLLM < 0.13.0 supports controlling profiler via environment variables
    os.environ["VLLM_TORCH_PROFILER_DIR"] = save_path
    os.environ["VLLM_TORCH_PROFILER_WITH_STACK"] = "1" if with_stack else "0"
    os.environ["VLLM_TORCH_PROFILER_RECORD_SHAPES"] = "1" if record_shapes else "0"
    os.environ["VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY"] = "1" if with_memory else "0"

    # vLLM >= 0.13.0 supports controlling profiler via arguments.
    # While it maintains backward compatibility with environment variables,
    # we provide arguments explicitly to align with the new API style.
    return {
        "profiler_config": json.dumps(
            {
                "profiler": "torch",
                "torch_profiler_dir": save_path,
                "torch_profiler_with_memory": with_memory,
                "torch_profiler_with_stack": with_stack,
                "torch_profiler_record_shapes": record_shapes,
            }
        )
    }


def build_sglang_profiler_args(profiler_config: ProfilerConfig, tool_config: BaseConfig, rank: int) -> dict:
    """
    Build arguments for SGLang profiler.

    Args:
        profiler_config (ProfilerConfig): The unified profiler configuration.
        tool_config (BaseConfig): The tool configuration.
        rank (int): The rank of the replica.

    Returns:
        dict: A dictionary of arguments suitable for starting the SGLang profiler.
    """
    if not profiler_config or not tool_config or not hasattr(tool_config, "contents"):
        return {}

    contents = tool_config.contents
    if "memory" in contents:
        warnings.warn("SGLang profiler does not support memory profiling. Ignoring memory content.", stacklevel=2)

    return {
        "output_dir": os.path.join(profiler_config.save_path, f"agent_loop_rollout_replica_{rank}"),
        "with_stack": "stack" in contents or "module" in contents,
        "record_shapes": "shapes" in contents,
    }
