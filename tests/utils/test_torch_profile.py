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

import os
import unittest
from unittest.mock import MagicMock, patch

import torch

from verl.utils.profiler.config import ProfilerConfig, TorchProfilerScheduleConfig, TorchProfilerToolConfig
from verl.utils.profiler.profile import DistProfiler
from verl.utils.profiler.torch_profile import (
    Profiler,
    build_trace_basename,
    get_torch_profiler,
)


class TestTorchProfile(unittest.TestCase):
    def setUp(self):
        # Reset process-global Profiler class state so tests don't leak into each other.
        Profiler._define_count = 0
        Profiler._active_prof = None

    def tearDown(self):
        Profiler._define_count = 0
        Profiler._active_prof = None

    @patch("torch.profiler.profile")
    def test_get_torch_profiler(self, mock_profile):
        # Test wrapper function
        get_torch_profiler(contents=["cpu", "cuda", "stack"], save_path="/tmp/test", rank=0)
        mock_profile.assert_called_once()
        _, kwargs = mock_profile.call_args

        # Verify activities
        activities = kwargs["activities"]
        self.assertIn(torch.profiler.ProfilerActivity.CPU, activities)
        self.assertIn(torch.profiler.ProfilerActivity.CUDA, activities)

        # Verify options
        self.assertTrue(kwargs["with_stack"])
        self.assertFalse(kwargs["record_shapes"])
        self.assertFalse(kwargs["profile_memory"])

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_profiler_lifecycle(self, mock_get_profiler):
        # Mock the underlying torch profiler object
        mock_prof_instance = MagicMock()
        mock_get_profiler.return_value = mock_prof_instance

        # Initialize
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)

        # Test Start
        profiler.start()
        mock_get_profiler.assert_called_once()
        mock_prof_instance.start.assert_called_once()

        # Test Step
        profiler.step()
        mock_prof_instance.step.assert_called_once()

        # Test Stop
        profiler.stop()
        mock_prof_instance.stop.assert_called_once()

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_discrete_mode(self, mock_get_profiler):
        # Mock for discrete mode
        mock_prof_instance = MagicMock()
        mock_get_profiler.return_value = mock_prof_instance

        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=True)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)

        # In discrete mode, start/stop shouldn't trigger global profiler immediately
        profiler.start()
        mock_get_profiler.assert_not_called()

        profiler.stop()
        mock_prof_instance.stop.assert_not_called()

    @patch("torch.profiler.schedule")
    @patch("torch.profiler.profile")
    def test_get_torch_profiler_with_schedule(self, mock_profile, mock_schedule):
        # When a schedule dict is provided, torch.profiler.schedule must be built and forwarded.
        sentinel_schedule = object()
        mock_schedule.return_value = sentinel_schedule
        schedule = {"skip_first": 1, "wait": 2, "warmup": 1, "active": 3, "repeat": 2}

        get_torch_profiler(contents=["cpu"], save_path="/tmp/test", rank=0, schedule=schedule)

        mock_schedule.assert_called_once_with(**schedule)
        _, kwargs = mock_profile.call_args
        self.assertIs(kwargs["schedule"], sentinel_schedule)

    @patch("torch.profiler.schedule")
    @patch("torch.profiler.profile")
    def test_get_torch_profiler_without_schedule(self, mock_profile, mock_schedule):
        # Without a schedule, the profiler runs in continuous mode (no schedule kwarg).
        get_torch_profiler(contents=["cpu"], save_path="/tmp/test", rank=0)

        mock_schedule.assert_not_called()
        _, kwargs = mock_profile.call_args
        self.assertNotIn("schedule", kwargs)

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_scheduled_profiler_lifecycle(self, mock_get_profiler):
        mock_prof_instance = MagicMock()
        mock_get_profiler.return_value = mock_prof_instance

        schedule_cfg = TorchProfilerScheduleConfig(wait=1, warmup=1, active=2, repeat=1)
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False, schedule=schedule_cfg)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)

        # Start forwards the resolved schedule kwargs and records the active profiler.
        profiler.start()
        _, kwargs = mock_get_profiler.call_args
        self.assertEqual(
            kwargs["schedule"],
            {"skip_first": 0, "wait": 1, "warmup": 1, "active": 2, "repeat": 1},
        )
        self.assertIs(Profiler._active_prof, mock_prof_instance)

        # Each step advances the active torch profiler.
        profiler.step()
        profiler.step()
        self.assertEqual(mock_prof_instance.step.call_count, 2)

        # With a schedule, stop must NOT emit an extra implicit step (stepping is per mini-batch).
        profiler.stop()
        self.assertEqual(mock_prof_instance.step.call_count, 2)
        mock_prof_instance.stop.assert_called_once()
        self.assertIsNone(Profiler._active_prof)

    def test_build_trace_basename_encodes_role_rank_and_parallelism(self):
        # Filename stem must embed the worker role, scope role, rank/world size and the
        # tp/pp/dp/cp parallel ranks so per-process traces are self-describing.
        name = build_trace_basename(
            rank=5,
            role="e2e",
            save_file_prefix="actor",
            topology={"rank": 5, "world_size": 16, "tp": 1, "pp": 0, "dp": 2, "cp": 0},
        )
        self.assertTrue(name.startswith("actor_e2e_"))
        self.assertIn("rank5-of-16", name)
        self.assertIn("tp1-pp0-dp2-cp0", name)
        self.assertIn(f"pid{os.getpid()}", name)

    def test_build_trace_basename_distinguishes_roles_same_rank(self):
        # The original scheme collided ref/critic at the same rank; the role prefix fixes it.
        topo = {"rank": 5, "world_size": 16}
        ref_name = build_trace_basename(rank=5, save_file_prefix="ref", topology=topo)
        critic_name = build_trace_basename(rank=5, save_file_prefix="value_model", topology=topo)
        self.assertTrue(ref_name.startswith("ref_rank5-of-16_"))
        # Underscores in labels are normalized to hyphens (underscore is the field separator).
        self.assertTrue(critic_name.startswith("value-model_rank5-of-16_"))
        self.assertNotEqual(ref_name, critic_name)

    def test_build_trace_basename_minimal_topology(self):
        # With no distributed topology, fall back to the passed rank and omit parallel dims.
        name = build_trace_basename(rank=3, topology={})
        self.assertTrue(name.startswith("rank3_"))
        self.assertNotIn("-of-", name)
        for dim in ("tp", "pp", "dp", "cp"):
            self.assertNotIn(f"{dim}0", name)

    def test_build_trace_basename_sanitizes_labels(self):
        # Slashes/spaces in labels must not leak into the filename.
        name = build_trace_basename(rank=0, role="update actor", save_file_prefix="actor/rollout", topology={})
        self.assertNotIn("/", name)
        self.assertNotIn(" ", name)
        self.assertIn("actor-rollout", name)
        self.assertIn("update-actor", name)

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_dist_profiler_forwards_save_file_prefix(self, mock_get_profiler):
        # DistProfiler must forward save_file_prefix down to the torch backend so it
        # ends up in the trace filename.
        mock_get_profiler.return_value = MagicMock()
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(
            tool="torch", enable=True, all_ranks=True, save_path="/tmp/test", tool_config=tool_config
        )
        dist_profiler = DistProfiler(rank=0, config=config, tool_config=tool_config, save_file_prefix="actor")
        self.assertEqual(dist_profiler._impl.save_file_prefix, "actor")

        dist_profiler.start()
        _, kwargs = mock_get_profiler.call_args
        self.assertEqual(kwargs["save_file_prefix"], "actor")
        dist_profiler.stop()

    def test_dist_profiler_step_noop_backend(self):
        # A backend without scheduling support (no-op impl) must make step() a safe no-op.
        config = ProfilerConfig(tool=None, enable=True, all_ranks=True, save_path="/tmp/test", tool_config=None)
        dist_profiler = DistProfiler(rank=0, config=config)
        self.assertIsNone(dist_profiler.step())

    def test_dist_profiler_step_disabled(self):
        # When disabled, step() must not touch the backend at all.
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(
            tool="torch", enable=False, all_ranks=True, save_path="/tmp/test", tool_config=tool_config
        )
        dist_profiler = DistProfiler(rank=0, config=config, tool_config=tool_config)
        dist_profiler._impl = MagicMock()
        self.assertIsNone(dist_profiler.step())
        dist_profiler._impl.step.assert_not_called()

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_dist_profiler_step_torch_delegates(self, mock_get_profiler):
        mock_prof_instance = MagicMock()
        mock_get_profiler.return_value = mock_prof_instance

        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(
            tool="torch", enable=True, all_ranks=True, save_path="/tmp/test", tool_config=tool_config
        )
        dist_profiler = DistProfiler(rank=0, config=config, tool_config=tool_config)

        dist_profiler.start()
        dist_profiler.step()
        mock_prof_instance.step.assert_called_once()

        dist_profiler.stop()


if __name__ == "__main__":
    unittest.main()
