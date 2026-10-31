# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import json
import os
from unittest.mock import patch

import pytest

from verl.utils.profiler.config import (
    NPUToolConfig,
    NsightToolConfig,
    ProfilerConfig,
    TorchProfilerToolConfig,
    build_sglang_profiler_args,
    build_vllm_profiler_args,
    rollout_trace_dir,
)


class TestServerProfilerArgs:
    @pytest.mark.parametrize(
        "legacy_env, expected_env",
        [
            (
                True,
                {
                    "VLLM_TORCH_PROFILER_DIR": "/tmp/test/agent_loop_rollout_replica_0",
                    "VLLM_TORCH_PROFILER_WITH_STACK": "1",
                    "VLLM_TORCH_PROFILER_RECORD_SHAPES": "1",
                    "VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY": "1",
                },
            ),
            (False, {}),
        ],
        ids=["legacy-env", "no-legacy-env"],
    )
    def test_build_vllm_profiler_args(self, legacy_env, expected_env):
        tool_config = TorchProfilerToolConfig(contents=["stack", "shapes", "memory"])
        config = ProfilerConfig(tool="torch", save_path="/tmp/test", tool_config=tool_config)

        with patch.dict(os.environ, {}, clear=True):
            args = build_vllm_profiler_args(config, tool_config, rank=0, legacy_env=legacy_env)
            assert dict(os.environ) == expected_env
        assert json.loads(args["profiler_config"]) == {
            "profiler": "torch",
            "torch_profiler_dir": "/tmp/test/agent_loop_rollout_replica_0",
            "torch_profiler_with_stack": True,
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_memory": True,
            "delay_iterations": 0,
            "max_iterations": 0,
            "ignore_frontend": "true",
        }

    @pytest.mark.parametrize(
        "tool_config",
        [
            TorchProfilerToolConfig(contents=["stack"], profile_token_start=12, profile_token_end=46),
            NPUToolConfig(contents=["npu"], profile_token_start=12, profile_token_end=46),
        ],
        ids=["torch", "npu"],
    )
    def test_build_vllm_profiler_args_with_profile_window(self, tool_config):
        config = ProfilerConfig(save_path="/tmp/test", tool_config=tool_config)
        args = build_vllm_profiler_args(config, tool_config, rank=0, legacy_env=False)
        profiler_config = json.loads(args["profiler_config"])
        assert profiler_config["delay_iterations"] == 12
        assert profiler_config["max_iterations"] == 34

    def test_build_vllm_nsight_profiler_args(self):
        config = ProfilerConfig(tool="nsys")
        with patch.dict(os.environ, {}, clear=True):
            args = build_vllm_profiler_args(config, NsightToolConfig(), rank=0)
            assert dict(os.environ) == {}
        assert set(args) == {"profiler_config"}
        assert json.loads(args["profiler_config"]) == {"profiler": "cuda"}

    def test_build_vllm_profiler_args_without_tool_config(self):
        assert build_vllm_profiler_args(ProfilerConfig(tool="nsys"), None, rank=0) == {}

    def test_rollout_trace_dir_matches_what_the_engines_write_to(self):
        """The finish hook derives its directory from this helper, so it must not drift."""
        tool_config = TorchProfilerToolConfig(contents=["cpu"])
        config = ProfilerConfig(save_path="/tmp/test", tool_config=tool_config)
        expected = rollout_trace_dir(config, rank=2)

        assert build_sglang_profiler_args(config, tool_config, rank=2)["output_dir"] == expected
        with patch.dict(os.environ, {}, clear=True):
            args = build_vllm_profiler_args(config, tool_config, rank=2, legacy_env=False)
        assert json.loads(args["profiler_config"])["torch_profiler_dir"] == expected

    def test_build_sglang_profiler_args(self):
        # Case 1: Basic features
        tool_config = TorchProfilerToolConfig(contents=["stack", "shapes", "memory"])
        config = ProfilerConfig(save_path="/tmp/test", tool_config=tool_config)
        with pytest.warns(UserWarning):
            args = build_sglang_profiler_args(config, tool_config, rank=0)
        assert args["output_dir"] == "/tmp/test/agent_loop_rollout_replica_0"
        assert args["with_stack"] is True
        assert args["record_shapes"] is True
        assert args["start_step"] is None
        assert args["num_steps"] is None

    def test_build_sglang_profiler_args_with_profile_window(self):
        tool_config = TorchProfilerToolConfig(contents=["stack"], profile_token_start=7, profile_token_end=16)
        config = ProfilerConfig(save_path="/tmp/test", tool_config=tool_config)
        args = build_sglang_profiler_args(config, tool_config, rank=0)
        assert args["start_step"] == 7
        assert args["num_steps"] == 9
