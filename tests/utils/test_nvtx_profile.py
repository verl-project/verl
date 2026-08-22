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

import io
import os
import socket
import tempfile
import unittest
from contextlib import redirect_stdout
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

    def test_role_configs_inherit_global_finish_hook(self):
        """Setting the finish hook on global_profiler must reach every role's profiler.

        Roles read only their own `profiler` block at worker construction time, so without this
        inheritance a globally configured hook would be silently ignored.
        """
        import os

        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
            cfg = compose(
                config_name="ppo_trainer",
                overrides=[
                    "global_profiler.finish_hook_cmd=upload.sh",
                    "global_profiler.finish_hook_all_ranks=True",
                    "global_profiler.finish_hook_ranks=[0,8]",
                    "global_profiler.relocate_results=True",
                ],
            )
        for config in [
            cfg.actor_rollout_ref.actor.profiler,
            cfg.actor_rollout_ref.rollout.profiler,
            cfg.actor_rollout_ref.ref.profiler,
            cfg.critic.profiler,
        ]:
            self.assertEqual(config.finish_hook_cmd, "upload.sh")
            self.assertTrue(config.finish_hook_all_ranks)
            self.assertEqual(list(config.finish_hook_ranks), [0, 8])
            self.assertTrue(config.relocate_results)

    def test_finish_hook_field_defaults_and_validation(self):
        """Finish-hook fields default to disabled and validate their types."""
        config = ProfilerConfig()
        self.assertFalse(config.relocate_results)
        self.assertIsNone(config.finish_hook_cmd)
        self.assertFalse(config.finish_hook_all_ranks)
        self.assertEqual(list(config.finish_hook_ranks), [])

        with self.assertRaises(AssertionError):
            ProfilerConfig(finish_hook_cmd=123)
        with self.assertRaises(AssertionError):
            ProfilerConfig(finish_hook_ranks="0,1")

    def test_union_intersect_carry_finish_hook_fields(self):
        """union/intersect must propagate the finish-hook fields."""
        a = ProfilerConfig(
            tool="nsys",
            relocate_results=True,
            finish_hook_cmd="cmd_a",
            finish_hook_all_ranks=False,
            finish_hook_ranks=[0, 1],
        )
        b = ProfilerConfig(
            tool="nsys",
            relocate_results=False,
            finish_hook_cmd=None,
            finish_hook_all_ranks=True,
            finish_hook_ranks=[1, 2],
        )

        union = a.union(b)
        self.assertTrue(union.relocate_results)
        self.assertEqual(union.finish_hook_cmd, "cmd_a")
        self.assertTrue(union.finish_hook_all_ranks)
        self.assertEqual(set(union.finish_hook_ranks), {0, 1, 2})

        intersect = a.intersect(b)
        self.assertFalse(intersect.relocate_results)
        self.assertEqual(intersect.finish_hook_cmd, "cmd_a")
        self.assertFalse(intersect.finish_hook_all_ranks)
        self.assertEqual(set(intersect.finish_hook_ranks), {1})

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
        with patch("verl.utils.profiler.nvtx_profile.get_platform") as mock_get_platform:
            mock_platform = MagicMock()
            mock_get_platform.return_value = mock_platform
            # Test start
            self.profiler.start()
            self.assertTrue(self.profiler.check_this_step())
            mock_platform.profiler_start.assert_called_once()

            # Test stop
            self.profiler.stop()
            self.assertFalse(self.profiler.check_this_step())
            mock_platform.profiler_stop.assert_called_once()

    def test_step_is_noop_and_does_not_raise(self):
        # Regression: the dispatcher DistProfiler.step() delegates to self._impl.step().
        # NsightSystemsProfiler subclasses DistProfiler without running its __init__, so a
        # missing step() override used to resolve to the inherited DistProfiler.step and
        # crash with "AttributeError: 'NsightSystemsProfiler' object has no attribute
        # '_enable'". It must now be a clean no-op.
        with patch("verl.utils.profiler.nvtx_profile.get_platform") as mock_get_platform:
            mock_platform = MagicMock()
            mock_get_platform.return_value = mock_platform
            self.profiler.start()
            self.profiler.step()
            self.profiler.stop()
            # step() must not drive the underlying platform profiler.
            mock_platform.profiler_start.assert_called_once()
            mock_platform.profiler_stop.assert_called_once()

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
        with patch("verl.utils.profiler.nvtx_profile.get_platform") as mock_get_platform:
            mock_platform = MagicMock()
            mock_get_platform.return_value = mock_platform
            mock_self.profiler.start()
        decorator = mock_self.profiler.annotate(message="test")

        @decorator
        def test_func(self, *args, **kwargs):
            return "result"

        with (
            patch("verl.utils.profiler.nvtx_profile.get_platform") as mock_get_platform,
            patch("verl.utils.profiler.nvtx_profile.mark_start_range") as mock_start_range,
            patch("verl.utils.profiler.nvtx_profile.mark_end_range") as mock_end_range,
        ):
            mock_platform = MagicMock()
            mock_get_platform.return_value = mock_platform
            result = test_func(mock_self)
            self.assertEqual(result, "result")
            mock_start_range.assert_called_once()
            mock_end_range.assert_called_once()
            mock_platform.profiler_start.assert_not_called()  # Not discrete mode
            mock_platform.profiler_stop.assert_not_called()  # Not discrete mode

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


class TestProfilerFinishHook(unittest.TestCase):
    """Tests for the finish-hook command dispatched from DistProfiler.stop().

    The hook runs real shell commands here, so the assertions cover both the dispatch
    decision and the report printed for the command (its output and exit code).
    """

    def _stop_and_capture(self, profiler):
        """Call stop() with the platform backend stubbed, returning printed output and that stub."""
        buffer = io.StringIO()
        with (
            patch("verl.utils.profiler.nvtx_profile.get_platform") as mock_get_platform,
            redirect_stdout(buffer),
        ):
            profiler.stop()
        return buffer.getvalue(), mock_get_platform

    def test_finish_hook_cmd_runs_on_default_ranks(self):
        from verl.utils.profiler.nvtx_profile import RAY_NSIGHT_LOG_DIR

        config = ProfilerConfig(
            tool="nsys",
            enable=True,
            all_ranks=True,
            finish_hook_cmd='echo "ctx $VERL_PROFILE_RANK $VERL_PROFILE_TOOL '
            '$VERL_PROFILE_SAVE_PATH $VERL_PROFILE_RAY_NSIGHT_DIR"',
            save_path="/tmp/prof",
        )
        profiler = DistProfiler(rank=0, config=config, tool_config=NsightToolConfig(discrete=False))
        out, _ = self._stop_and_capture(profiler)

        self.assertIn(f"ctx 0 nsys /tmp/prof {RAY_NSIGHT_LOG_DIR}", out)
        self.assertIn("command exited with 0", out)

    def test_finish_hook_cmd_respects_finish_hook_ranks(self):
        # Selected finish-hook rank is 1, so rank 0 must not run the command.
        config = ProfilerConfig(
            tool="nsys", enable=True, all_ranks=True, finish_hook_cmd="echo marker", finish_hook_ranks=[1]
        )
        profiler = DistProfiler(rank=0, config=config, tool_config=NsightToolConfig(discrete=False))
        out, _ = self._stop_and_capture(profiler)

        self.assertNotIn("running command", out)
        self.assertIn("rank not selected", out)

    def test_finish_hook_cmd_runs_on_non_profiled_rank(self):
        # Profile only rank 0, but the finish hook targets rank 1: the command runs on rank 1
        # even though its backend profiler never started/stopped.
        config = ProfilerConfig(
            tool="nsys", enable=True, ranks=[0], finish_hook_cmd="echo marker", finish_hook_ranks=[1]
        )
        profiler = DistProfiler(rank=1, config=config, tool_config=NsightToolConfig(discrete=False))
        out, mock_get_platform = self._stop_and_capture(profiler)

        mock_get_platform.return_value.profiler_stop.assert_not_called()
        # The bare "rank 1: marker" line is the command's own output, not the echoed command.
        self.assertIn("rank 1: marker", out)

    def test_finish_hook_noop_when_unconfigured(self):
        config = ProfilerConfig(tool="nsys", enable=True, all_ranks=True)
        profiler = DistProfiler(rank=0, config=config, tool_config=NsightToolConfig(discrete=False))
        out, _ = self._stop_and_capture(profiler)

        self.assertNotIn("running command", out)

    def test_finish_hook_reports_failure_without_raising(self):
        # A failing command must report its output and exit code, and never break training.
        config = ProfilerConfig(
            tool="nsys", enable=True, all_ranks=True, finish_hook_cmd="echo boom >&2; exit 3", save_path="/tmp/prof"
        )
        profiler = DistProfiler(rank=0, config=config, tool_config=NsightToolConfig(discrete=False))
        out, _ = self._stop_and_capture(profiler)

        self.assertIn("boom", out)
        self.assertIn("command exited with 3", out)

    def test_finish_hook_launch_failure_is_swallowed(self):
        config = ProfilerConfig(tool="nsys", enable=True, all_ranks=True, finish_hook_cmd="whatever")
        profiler = DistProfiler(rank=0, config=config, tool_config=NsightToolConfig(discrete=False))
        buffer = io.StringIO()
        with (
            patch("verl.utils.profiler.nvtx_profile.get_platform"),
            patch("verl.utils.profiler.profile.subprocess.Popen", side_effect=OSError("nope")),
            redirect_stdout(buffer),
        ):
            profiler.stop()

        self.assertIn("failed to launch", buffer.getvalue())


class TestNsightRelocateResults(unittest.TestCase):
    """Tests for moving Ray's fixed-location Nsight reports into save_path."""

    def _make_profiler(self):
        from verl.utils.profiler.nvtx_profile import NsightSystemsProfiler

        return NsightSystemsProfiler(
            rank=0,
            config=ProfilerConfig(tool="nsys", enable=True, all_ranks=True),
            tool_config=NsightToolConfig(discrete=False),
        )

    def test_relocate_moves_only_current_pid_files(self):
        prof = self._make_profiler()
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            pid = os.getpid()
            mine = os.path.join(src, f"worker_process_{pid}.nsys-rep")
            mine_ranged = os.path.join(src, f"worker_process_{pid}.1.nsys-rep")
            other = os.path.join(src, f"worker_process_{pid + 1}.nsys-rep")
            for path in (mine, mine_ranged, other):
                with open(path, "w") as fh:
                    fh.write("x")

            moved = prof.relocate_results(dst, rank=0, save_file_prefix="actor", source_dir=src)

            self.assertEqual(len(moved), 2)
            host = socket.gethostname()
            self.assertTrue(os.path.exists(os.path.join(dst, f"actor_{host}_worker_process_{pid}.nsys-rep")))
            self.assertTrue(os.path.exists(os.path.join(dst, f"actor_{host}_worker_process_{pid}.1.nsys-rep")))
            # Files owned by another PID stay put; ours are moved (not copied).
            self.assertTrue(os.path.exists(other))
            self.assertFalse(os.path.exists(mine))

    def test_relocate_no_matching_files_returns_empty(self):
        prof = self._make_profiler()
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            self.assertEqual(prof.relocate_results(dst, source_dir=src), [])

    def test_relocate_without_save_path_is_noop(self):
        prof = self._make_profiler()
        with tempfile.TemporaryDirectory() as src:
            self.assertEqual(prof.relocate_results(None, source_dir=src), [])

    def test_relocate_via_stop_when_enabled(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            pid = os.getpid()
            report = os.path.join(src, f"worker_process_{pid}.nsys-rep")
            with open(report, "w") as fh:
                fh.write("x")

            config = ProfilerConfig(tool="nsys", enable=True, all_ranks=True, relocate_results=True, save_path=dst)
            profiler = DistProfiler(rank=0, config=config, tool_config=NsightToolConfig(discrete=False))
            with (
                patch("verl.utils.profiler.nvtx_profile.get_platform"),
                patch("verl.utils.profiler.nvtx_profile.RAY_NSIGHT_LOG_DIR", src),
            ):
                profiler.stop()

            host = socket.gethostname()
            self.assertTrue(os.path.exists(os.path.join(dst, f"{host}_worker_process_{pid}.nsys-rep")))
            self.assertFalse(os.path.exists(report))


if __name__ == "__main__":
    unittest.main()
