# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

import unittest
from unittest.mock import MagicMock, patch

from verl.utils import omega_conf_to_dataclass
from verl.utils.profiler.config import NsightToolConfig, ProfilerConfig
from verl.utils.profiler.profile import DistProfiler


class TestProfilerConfig(unittest.TestCase):
    def test_config_init(self):
        import os

        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
            cfg = compose(config_name="ppo_trainer")
        for config in [
            cfg.actor_rollout_ref.actor.profiler,
            cfg.actor_rollout_ref.rollout.profiler,
            cfg.actor_rollout_ref.ref.profiler,
            cfg.critic.profiler,
            cfg.reward_model.profiler,
        ]:
            profiler_config = omega_conf_to_dataclass(config)
            self.assertEqual(profiler_config.tool, config.tool)
            self.assertEqual(profiler_config.enable, config.enable)
            self.assertEqual(profiler_config.all_ranks, config.all_ranks)
            self.assertEqual(profiler_config.ranks, config.ranks)
            self.assertEqual(profiler_config.save_path, config.save_path)
            self.assertEqual(profiler_config.ranks, config.ranks)
            assert isinstance(profiler_config, ProfilerConfig)
            with self.assertRaises(AttributeError):
                _ = profiler_config.non_existing_key
            assert config.get("non_existing_key") == profiler_config.get("non_existing_key")
            assert config.get("non_existing_key", 1) == profiler_config.get("non_existing_key", 1)

    def test_frozen_config(self):
        """Test that modifying frozen keys in ProfilerConfig raises exceptions."""
        from dataclasses import FrozenInstanceError

        from verl.utils.profiler.config import ProfilerConfig

        # Create a new ProfilerConfig instance
        config = ProfilerConfig(all_ranks=False, ranks=[0])

        with self.assertRaises(FrozenInstanceError):
            config.all_ranks = True

        with self.assertRaises(FrozenInstanceError):
            config.ranks = [1, 2, 3]

        with self.assertRaises(TypeError):
            config["all_ranks"] = True

        with self.assertRaises(TypeError):
            config["ranks"] = [1, 2, 3]


class TestNsightSystemsProfiler(unittest.TestCase):
    """Test suite for NsightSystemsProfiler functionality.

    Test Plan:
    1. Initialization: Verify profiler state after creation
    2. Basic Profiling: Test start/stop functionality
    3. Discrete Mode: TODO: Test discrete profiling behavior
    4. Annotation: Test the annotate decorator in both normal and discrete modes
    5. Config Validation: Verify proper config initialization from OmegaConf
    """

    def setUp(self):
        self.config = ProfilerConfig(tool="nsys", enable=True, all_ranks=True)
        self.rank = 0
        self.profiler = DistProfiler(self.rank, self.config, tool_config=NsightToolConfig(discrete=False))

    def test_initialization(self):
        self.assertEqual(self.profiler.check_this_rank(), True)
        self.assertEqual(self.profiler.check_this_step(), False)

    def test_start_stop_profiling(self):
        with patch("torch.cuda.profiler.start") as mock_start, patch("torch.cuda.profiler.stop") as mock_stop:
            # Test start
            self.profiler.start()
            self.assertTrue(self.profiler.check_this_step())
            mock_start.assert_called_once()

            # Test stop
            self.profiler.stop()
            self.assertFalse(self.profiler.check_this_step())
            mock_stop.assert_called_once()

    # def test_discrete_profiling(self):
    #     discrete_config = ProfilerConfig(discrete=True, all_ranks=True)
    #     profiler = NsightSystemsProfiler(self.rank, discrete_config)

    #     with patch("torch.cuda.profiler.start") as mock_start, patch("torch.cuda.profiler.stop") as mock_stop:
    #         profiler.start()
    #         self.assertTrue(profiler.this_step)
    #         mock_start.assert_not_called()  # Shouldn't start immediately in discrete mode

    #         profiler.stop()
    #         self.assertFalse(profiler.this_step)
    #         mock_stop.assert_not_called()  # Shouldn't stop immediately in discrete mode

    def test_annotate_decorator(self):
        mock_self = MagicMock()
        mock_self.profiler = self.profiler
        mock_self.profiler.start()
        decorator = mock_self.profiler.annotate(message="test")

        @decorator
        def test_func(self, *args, **kwargs):
            return "result"

        with (
            patch("torch.cuda.profiler.start") as mock_start,
            patch("torch.cuda.profiler.stop") as mock_stop,
            patch("verl.utils.profiler.nvtx_profile.mark_start_range") as mock_start_range,
            patch("verl.utils.profiler.nvtx_profile.mark_end_range") as mock_end_range,
        ):
            result = test_func(mock_self)
            self.assertEqual(result, "result")
            mock_start_range.assert_called_once()
            mock_end_range.assert_called_once()
            mock_start.assert_not_called()  # Not discrete mode
            mock_stop.assert_not_called()  # Not discrete mode

    # def test_annotate_discrete_mode(self):
    #     discrete_config = ProfilerConfig(discrete=True, all_ranks=True)
    #     profiler = NsightSystemsProfiler(self.rank, discrete_config)
    #     mock_self = MagicMock()
    #     mock_self.profiler = profiler
    #     mock_self.profiler.this_step = True

    #     @NsightSystemsProfiler.annotate(message="test")
    #     def test_func(self, *args, **kwargs):
    #         return "result"

    #     with (
    #         patch("torch.cuda.profiler.start") as mock_start,
    #         patch("torch.cuda.profiler.stop") as mock_stop,
    #         patch("verl.utils.profiler.nvtx_profile.mark_start_range") as mock_start_range,
    #         patch("verl.utils.profiler.nvtx_profile.mark_end_range") as mock_end_range,
    #     ):
    #         result = test_func(mock_self)
    #         self.assertEqual(result, "result")
    #         mock_start_range.assert_called_once()
    #         mock_end_range.assert_called_once()
    #         mock_start.assert_called_once()  # Should start in discrete mode
    #         mock_stop.assert_called_once()  # Should stop in discrete mode


if __name__ == "__main__":
    unittest.main()
