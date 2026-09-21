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

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.profiler.config import MemrayToolConfig, ProfilerConfig, build_role_profiler_tool_config
from verl.utils.profiler.profile import DistProfiler


class _FakeTracker:
    """Minimal Memray tracker that emits a real file when the window closes."""

    def __init__(self, file_name: str, native_traces: bool, payload: bytes):
        self.output_path = Path(file_name)
        self.native_traces = native_traces
        self.payload = payload
        self.active = False
        self.exit_args = None
        self.fail_on_exit = False

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, *args):
        self.active = False
        self.exit_args = args
        if self.fail_on_exit:
            raise OSError("trace flush failed")
        self.output_path.write_bytes(self.payload)


class TestMemrayProfiler(unittest.TestCase):
    def setUp(self):
        self.trackers = []
        self.memray = SimpleNamespace(Tracker=MagicMock(side_effect=self._new_tracker))
        self.memray_patcher = patch.dict(sys.modules, {"memray": self.memray})
        self.memray_patcher.start()
        self.addCleanup(self.memray_patcher.stop)

    def _new_tracker(self, *, file_name: str, native_traces: bool):
        tracker = _FakeTracker(
            file_name=file_name,
            native_traces=native_traces,
            payload=f"trace-{len(self.trackers)}".encode(),
        )
        self.trackers.append(tracker)
        return tracker

    def _config(self, save_path: str = "/tmp/profiles") -> ProfilerConfig:
        return ProfilerConfig(enable=True, tool="memray", ranks=[0], save_path=save_path)

    def test_dist_profiler_closes_and_moves_a_multi_step_trace(self):
        tool_config = MemrayToolConfig(memory_snapshot_num_steps=2)
        with tempfile.TemporaryDirectory() as out_dir:
            profiler = DistProfiler(rank=0, config=self._config(out_dir), tool_config=tool_config)

            profiler.start(profile_step=4)
            self.assertEqual(len(self.trackers), 1)
            tracker = self.trackers[0]
            self.assertTrue(tracker.active)
            self.assertTrue(tracker.native_traces)
            self.assertEqual(tracker.output_path.parent, Path(out_dir) / ".memray")
            self.assertRegex(tracker.output_path.name, r"memray_rank0_pid\d+\.bin")

            profiler.stop(run_command=False)
            self.assertTrue(tracker.active)
            self.assertIsNone(tracker.exit_args)
            self.assertFalse((Path(out_dir) / "steps4-5").exists())

            profiler.start(profile_step=5)
            self.assertEqual(len(self.trackers), 1, "one tracker must span the complete window")
            profiler.stop(run_command=False)

            final_path = Path(out_dir) / "steps4-5" / tracker.output_path.name
            self.assertFalse(tracker.active)
            self.assertEqual(tracker.exit_args, (None, None, None))
            self.assertEqual(final_path.read_bytes(), b"trace-0")
            self.assertFalse(tracker.output_path.exists())

    def test_consecutive_windows_produce_independent_trace_files(self):
        with tempfile.TemporaryDirectory() as out_dir:
            profiler = DistProfiler(rank=0, config=self._config(out_dir), tool_config=MemrayToolConfig())

            for step in (4, 5):
                profiler.start(profile_step=step)
                profiler.stop(run_command=False)

            self.assertEqual(len(self.trackers), 2)
            first_path = Path(out_dir) / "step4" / self.trackers[0].output_path.name
            second_path = Path(out_dir) / "step5" / self.trackers[1].output_path.name
            self.assertEqual(first_path.read_bytes(), b"trace-0")
            self.assertEqual(second_path.read_bytes(), b"trace-1")
            self.assertEqual(list((Path(out_dir) / ".memray").iterdir()), [])

    def test_unselected_rank_does_not_create_a_trace(self):
        with tempfile.TemporaryDirectory() as out_dir:
            profiler = DistProfiler(rank=1, config=self._config(out_dir), tool_config=MemrayToolConfig())
            profiler.start(profile_step=4)
            profiler.stop(run_command=False)

            self.memray.Tracker.assert_not_called()
            self.assertEqual(list(Path(out_dir).iterdir()), [])

    def test_missing_memray_is_nonfatal(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with (
                patch.dict(sys.modules, {"memray": None}),
                self.assertLogs("verl.utils.profiler.memray_profile", level="WARNING") as logs,
            ):
                profiler = DistProfiler(rank=0, config=self._config(out_dir), tool_config=MemrayToolConfig())
                profiler.start(profile_step=4)
                profiler.stop(run_command=False)

            self.assertTrue(any("memray is not installed" in entry for entry in logs.output))
            self.assertEqual(list(Path(out_dir).iterdir()), [])

    def test_tracker_start_failure_is_nonfatal(self):
        self.memray.Tracker.side_effect = RuntimeError("tracker unavailable")
        with (
            tempfile.TemporaryDirectory() as out_dir,
            self.assertLogs("verl.utils.profiler.memray_profile", level="WARNING") as logs,
        ):
            profiler = DistProfiler(rank=0, config=self._config(out_dir), tool_config=MemrayToolConfig())
            profiler.start(profile_step=4)
            profiler.stop(run_command=False)

        self.assertTrue(any("tracker unavailable" in entry for entry in logs.output))
        self.assertIsNone(profiler._impl._tracker)
        self.assertTrue(profiler._impl._disabled)

    def test_tracker_stop_failure_is_nonfatal_and_next_window_recovers(self):
        with tempfile.TemporaryDirectory() as out_dir:
            profiler = DistProfiler(rank=0, config=self._config(out_dir), tool_config=MemrayToolConfig())
            profiler.start(profile_step=4)
            self.trackers[0].fail_on_exit = True
            with self.assertLogs("verl.utils.profiler.memray_profile", level="WARNING") as logs:
                profiler.stop(run_command=False)

            self.assertTrue(any("trace flush failed" in entry for entry in logs.output))
            self.assertIsNone(profiler._impl._tracker)
            self.assertIsNone(profiler._impl._output_path)

            profiler.start(profile_step=5)
            profiler.stop(run_command=False)

            self.assertEqual(len(self.trackers), 2)
            recovered_path = Path(out_dir) / "step5" / self.trackers[1].output_path.name
            self.assertEqual(recovered_path.read_bytes(), b"trace-1")


class TestMemrayWorkerConfig(unittest.TestCase):
    def test_role_worker_config_preserves_multi_step_window(self):
        omega_config = OmegaConf.create(
            {
                "_target_": "verl.utils.profiler.ProfilerConfig",
                "tool": "memray",
                "enable": True,
                "all_ranks": False,
                "ranks": [0],
                "save_path": "/tmp/test_memray_profile",
                "tool_config": {
                    "memray": {
                        "_target_": "verl.utils.profiler.config.MemrayToolConfig",
                        "memory_snapshot_num_steps": 3,
                    },
                },
            }
        )

        profiler_config = omega_conf_to_dataclass(omega_config, dataclass_type=ProfilerConfig)
        tool_config = build_role_profiler_tool_config(omega_config)
        profiler = DistProfiler(rank=0, config=profiler_config, tool_config=tool_config)

        self.assertIsInstance(tool_config, MemrayToolConfig)
        self.assertEqual(tool_config.memory_snapshot_num_steps, 3)
        self.assertEqual(profiler._impl.memory_snapshot_num_steps, 3)


@unittest.skipUnless(importlib.util.find_spec("memray"), "requires the optional memray dependency")
class TestMemrayProfilerIntegration(unittest.TestCase):
    def test_real_trace_contains_profiled_allocations(self):
        import memray

        with tempfile.TemporaryDirectory() as out_dir:
            config = ProfilerConfig(enable=True, tool="memray", ranks=[0], save_path=out_dir)
            profiler = DistProfiler(rank=0, config=config, tool_config=MemrayToolConfig())

            profiler.start(profile_step=4)
            allocations = [bytearray(1024 * 1024) for _ in range(4)]
            profiler.stop(run_command=False)

            traces = list((Path(out_dir) / "step4").glob("memray_rank0_pid*.bin"))
            self.assertEqual(len(traces), 1)
            records = memray.FileReader(str(traces[0])).get_allocation_records()
            self.assertTrue(any(record.size >= 1024 * 1024 for record in records))
            self.assertEqual(len(allocations), 4)  # Keep allocations alive until after reading the trace.
