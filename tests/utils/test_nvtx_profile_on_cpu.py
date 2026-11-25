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

import pytest

from verl.utils.profiler.config import NsightToolConfig, ProfilerConfig
from verl.utils.profiler.profile import DistProfiler


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


class TestNsightRelocateResults:
    """Tests for moving Ray's fixed-location Nsight reports into save_path."""

    def _make_profiler(self, options=None):
        from verl.utils.profiler.nvtx_profile import NsightSystemsProfiler

        return NsightSystemsProfiler(
            rank=0,
            config=ProfilerConfig(tool="nsys", enable=True, all_ranks=True, ray_actor_nsight_options=options),
            tool_config=NsightToolConfig(discrete=False),
        )

    @pytest.mark.parametrize(
        "options, rendered_output, moved_names, retained_names",
        [
            (
                None,
                "unused_%p",
                ["worker_process_42.nsys-rep", "worker_process_42.1.nsys-rep"],
                ["worker_process_43.nsys-rep", "worker_process_421.nsys-rep"],
            ),
            (
                {},
                "default_%h_pid%p",
                ["default_node_pid42.nsys-rep", "default_node_pid42.1.nsys-rep"],
                ["default_node_pid43.nsys-rep"],
            ),
            (
                {"o": "template"},
                "custom[trace]_%h_pid%p",
                ["custom[trace]_node_pid42.nsys-rep", "custom[trace]_node_pid42.1.nsys-rep"],
                # The second name would match [trace] if the glob were not escaped.
                ["custom[trace]_node_pid43.nsys-rep", "customt_node_pid42.nsys-rep"],
            ),
            (
                {"o": "group_rank_%q{RANK}_%h_pid%p"},
                "actor_prefix_%p",
                ["group_rank_2_node_pid42.nsys-rep"],
                ["group_rank_3_node_pid42.nsys-rep", "group_rank_2_node_pid43.nsys-rep"],
            ),
        ],
        ids=["legacy", "actor-default", "actor-custom", "worker-group"],
    )
    def test_relocate_moves_only_matching_reports(
        self, tmp_path, options, rendered_output, moved_names, retained_names
    ):
        prof = self._make_profiler(options)
        src = tmp_path / "source"
        dst = tmp_path / "destination"
        src.mkdir()
        for name in moved_names + retained_names:
            (src / name).write_text("x")

        with (
            patch("verl.utils.profiler.nvtx_profile.os.getpid", return_value=42),
            patch("verl.utils.profiler.nvtx_profile.socket.gethostname", return_value="node"),
            patch(
                "verl.utils.profiler.nsight._render_nsight_output_name", return_value=rendered_output
            ) as render_output,
        ):
            moved = prof.relocate_results(str(dst), rank=2, save_file_prefix="actor", source_dir=str(src))

        if options is not None:
            render_output.assert_any_call(options.get("o"), rank=2, role="actor")
        expected = {f"actor_node_{name}" for name in moved_names}
        assert set(moved) == {str(dst / name) for name in expected}
        assert {path.name for path in dst.iterdir()} == expected
        assert {path.name for path in src.iterdir()} == set(retained_names)

    def test_relocate_does_not_claim_custom_output_without_pid(self):
        prof = self._make_profiler({"o": "shared_output"})
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            report = os.path.join(src, "shared_output.nsys-rep")
            with open(report, "w") as fh:
                fh.write("x")
            with patch("verl.utils.profiler.nsight._render_nsight_output_name", return_value="shared_output"):
                assert prof.relocate_results(dst, rank=0, save_file_prefix="actor", source_dir=src) == []
            assert os.path.exists(report)

    def test_relocate_no_matching_files_returns_empty(self):
        prof = self._make_profiler()
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            assert prof.relocate_results(dst, source_dir=src) == []

    def test_relocate_without_save_path_is_noop(self):
        prof = self._make_profiler()
        with tempfile.TemporaryDirectory() as src:
            assert prof.relocate_results(None, source_dir=src) == []

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
            assert os.path.exists(os.path.join(dst, f"{host}_worker_process_{pid}.nsys-rep"))
            assert not os.path.exists(report)


if __name__ == "__main__":
    unittest.main()
