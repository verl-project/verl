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

import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from verl.utils.profiler.config import (
    ProfilerConfig,
    TorchProfilerToolConfig,
    build_sglang_profiler_args,
    build_vllm_profiler_args,
)


class TestServerProfilerArgs(unittest.TestCase):
    def test_build_vllm_profiler_args(self):
        # Case 1: All features enabled
        tool_config = TorchProfilerToolConfig(contents=["stack", "shapes", "memory"])
        config = ProfilerConfig(save_path="/tmp/test", tool_config=tool_config)

        # Patch environ to avoid side effects and verify calls
        with patch.dict(os.environ, {}, clear=True):
            args = build_vllm_profiler_args(config, tool_config, rank=0)

            # Check Env vars (backward compatibility)
            self.assertEqual(os.environ.get("VLLM_TORCH_PROFILER_DIR"), "/tmp/test/agent_loop_rollout_replica_0")
            self.assertEqual(os.environ.get("VLLM_TORCH_PROFILER_WITH_STACK"), "1")
            self.assertEqual(os.environ.get("VLLM_TORCH_PROFILER_RECORD_SHAPES"), "1")
            self.assertEqual(os.environ.get("VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY"), "1")

            # Check Args (new API)
            self.assertIn("profiler_config", args)
            profiler_config_dict = json.loads(args["profiler_config"])
            self.assertEqual(profiler_config_dict["torch_profiler_dir"], "/tmp/test/agent_loop_rollout_replica_0")
            self.assertTrue(profiler_config_dict["torch_profiler_with_stack"])
            self.assertTrue(profiler_config_dict["torch_profiler_record_shapes"])
            self.assertTrue(profiler_config_dict["torch_profiler_with_memory"])

    def test_build_sglang_profiler_args(self):
        # Case 1: Basic features
        tool_config = TorchProfilerToolConfig(contents=["stack", "shapes", "memory"])
        config = ProfilerConfig(save_path="/tmp/test", tool_config=tool_config)
        with self.assertWarns(UserWarning):
            args = build_sglang_profiler_args(config, tool_config, rank=0)
        self.assertEqual(args["output_dir"], "/tmp/test/agent_loop_rollout_replica_0")
        self.assertTrue(args["with_stack"])
        self.assertTrue(args["record_shapes"])


class TestServerProfilerFunctionality(unittest.IsolatedAsyncioTestCase):
    async def test_vllm_start_stop_profile(self):
        try:
            # Import strictly inside test to avoid import errors if dependencies missing
            from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
        except ImportError:
            self.skipTest("vllm or dependencies not installed")
            return

        # Mock dependencies
        mock_profiler = MagicMock()
        mock_profiler.check_enable.return_value = True
        mock_profiler.check_this_rank.return_value = True
        mock_profiler.is_discrete_mode.return_value = True

        mock_engine = AsyncMock()

        # Mock self object
        mock_self = MagicMock()
        mock_self.profiler_controller = mock_profiler
        mock_self.engine = mock_engine

        # Test start_profile using the unbound method
        await vLLMHttpServer.start_profile(mock_self)
        mock_engine.start_profile.assert_called_once()

        # Test stop_profile
        await vLLMHttpServer.stop_profile(mock_self)
        mock_engine.stop_profile.assert_called_once()

    async def test_sglang_start_stop_profile(self):
        try:
            # Import strictly inside test to avoid import errors if dependencies missing
            from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer
        except ImportError:
            self.skipTest("sglang or dependencies not installed")
            return

        # Mock dependencies
        mock_profiler = MagicMock()
        mock_profiler.check_enable.return_value = True
        mock_profiler.check_this_rank.return_value = True
        mock_profiler.is_discrete_mode.return_value = True
        mock_profiler.config = MagicMock()
        mock_profiler.tool_config = MagicMock()

        mock_tokenizer_manager = AsyncMock()

        mock_self = MagicMock()
        mock_self.profiler_controller = mock_profiler
        mock_self.tokenizer_manager = mock_tokenizer_manager
        mock_self.replica_rank = 0

        # Mock build_sglang_profiler_args to return known dict
        with patch("verl.workers.rollout.sglang_rollout.async_sglang_server.build_sglang_profiler_args") as mock_build:
            mock_args = {"arg1": "val1"}
            mock_build.return_value = mock_args

            # Test start_profile
            await SGLangHttpServer.start_profile(mock_self)

            mock_build.assert_called_once_with(mock_profiler.config, mock_profiler.tool_config, mock_self.replica_rank)
            mock_tokenizer_manager.start_profile.assert_called_once_with(**mock_args)

            # Test stop_profile
            await SGLangHttpServer.stop_profile(mock_self)
            mock_tokenizer_manager.stop_profile.assert_called_once()


if __name__ == "__main__":
    unittest.main()
