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

import unittest
from unittest.mock import MagicMock, patch

from verl.utils.profiler.config import ProfilerConfig, TorchMemoryToolConfig
from verl.utils.profiler.profile import DistProfiler
from verl.utils.profiler.torch_memory_profile import TorchMemoryProfiler


class TestTorchMemoryProfiler(unittest.TestCase):
    def setUp(self):
        self.attach_observer = MagicMock()
        for patcher in (
            patch("verl.utils.profiler.torch_memory_profile.enable_memory_visualize"),
            patch("verl.utils.profiler.torch_memory_profile.is_cuda_available", True),
            patch(
                "verl.utils.profiler.torch_memory_profile.torch._C._cuda_attach_out_of_memory_observer",
                self.attach_observer,
                create=True,
            ),
            patch.object(TorchMemoryProfiler, "_memory_history_enabled", False),
            patch.object(TorchMemoryProfiler, "_oom_observer_attached", False),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _config(self) -> ProfilerConfig:
        return ProfilerConfig(enable=True, ranks=[0], save_path="/tmp/profiles")

    def test_oom_observer_is_automatic_and_dumps_without_sync(self):
        profiler = TorchMemoryProfiler(rank=0, config=self._config(), tool_config=TorchMemoryToolConfig())

        self.attach_observer.assert_called_once()
        observer = self.attach_observer.call_args.args[0]
        with (
            patch("verl.utils.profiler.torch_memory_profile.traceback.format_stack", return_value=["stack"]),
            patch(
                "verl.utils.profiler.torch_memory_profile.get_memory_info", return_value={"allocated": 1234}
            ) as memory_info,
            patch.object(profiler.sampler, "dump_memory_snapshot") as dump_snapshot,
        ):
            with self.assertLogs("verl.utils.profiler.torch_memory_profile", level="ERROR") as logs:
                observer(0, 4096, 1234, 5678)

        dump_snapshot.assert_called_once()
        memory_info.assert_called_once()
        self.assertTrue(any("requested=4096 total=1234 free=5678" in entry for entry in logs.output))
        self.assertTrue(any("Python stack at OOM:\nstack" in entry for entry in logs.output))
        self.assertTrue(any("allocator memory at OOM" in entry for entry in logs.output))
        kwargs = dump_snapshot.call_args.kwargs
        self.assertEqual(kwargs["out_dir"], "/tmp/profiles")
        self.assertEqual(kwargs["tag"], "torch_memory_oom")
        self.assertTrue(kwargs["sub_dir"].startswith("oom_"))
        self.assertFalse(kwargs["synchronize"])

    def test_oom_observer_is_automatic_without_tool_config(self):
        TorchMemoryProfiler(rank=0, config=None)
        self.attach_observer.assert_called_once()

    def test_oom_observer_skips_unselected_ranks(self):
        TorchMemoryProfiler(rank=1, config=self._config())
        self.attach_observer.assert_not_called()

    def test_oom_observer_is_registered_once_per_process(self):
        TorchMemoryProfiler(rank=0, config=self._config())
        TorchMemoryProfiler(rank=0, config=self._config())
        self.attach_observer.assert_called_once()

    def test_non_cuda_device_skips_oom_observer(self):
        with (
            patch("verl.utils.profiler.torch_memory_profile.is_cuda_available", False),
            self.assertLogs("verl.utils.profiler.torch_memory_profile", level="WARNING"),
        ):
            TorchMemoryProfiler(rank=0, config=self._config())
        self.attach_observer.assert_not_called()
        self.assertFalse(TorchMemoryProfiler._oom_observer_attached)

    def test_missing_oom_observer_api_is_nonfatal(self):
        with (
            patch(
                "verl.utils.profiler.torch_memory_profile.torch._C._cuda_attach_out_of_memory_observer",
                None,
                create=True,
            ),
            self.assertLogs("verl.utils.profiler.torch_memory_profile", level="WARNING"),
        ):
            TorchMemoryProfiler(rank=0, config=self._config())
        self.attach_observer.assert_not_called()
        self.assertFalse(TorchMemoryProfiler._oom_observer_attached)

    def test_oom_observer_registration_failure_is_nonfatal(self):
        self.attach_observer.side_effect = RuntimeError("allocator does not support OOM observers")
        with self.assertLogs("verl.utils.profiler.torch_memory_profile", level="WARNING"):
            TorchMemoryProfiler(rank=0, config=self._config())
        self.attach_observer.assert_called_once()
        self.assertFalse(TorchMemoryProfiler._oom_observer_attached)

    def test_unselected_memory_tool_does_not_register_oom_observer(self):
        DistProfiler(rank=0, config=ProfilerConfig(tool=None))
        self.attach_observer.assert_not_called()
        self.assertFalse(TorchMemoryProfiler._memory_history_enabled)

    def test_snapshot_window_keeps_history_until_the_configured_step_count(self):
        tool_config = TorchMemoryToolConfig(memory_snapshot_num_steps=2)
        with patch("verl.utils.profiler.torch_memory_profile.clear_memory_history") as clear_memory_history:
            profiler = TorchMemoryProfiler(rank=0, config=self._config(), tool_config=tool_config)
            with patch.object(profiler.sampler, "dump_memory_snapshot") as dump_snapshot:
                profiler.start(profile_step=4)
                profiler.stop()
                dump_snapshot.assert_not_called()
                clear_memory_history.assert_not_called()

                profiler.start(profile_step=5)
                profiler.stop()

            dump_snapshot.assert_called_once_with(out_dir="/tmp/profiles", tag="torch_memory", sub_dir="steps4-5")
            clear_memory_history.assert_called_once_with(trace_alloc_max_entries=100_000, stack_depth=32)
