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

import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from verl.utils.profiler.config import ProfilerConfig, TorchMemoryToolConfig
from verl.utils.profiler.profile import DistProfiler
from verl.utils.profiler.torch_memory_profile import TorchMemoryProfiler


class TestTorchMemoryProfiler(unittest.TestCase):
    device_name = "cuda"

    def setUp(self):
        self.attach_observer = MagicMock()
        self.device = MagicMock()
        self.device.is_available.return_value = True
        self.npu = SimpleNamespace(_C=SimpleNamespace())
        self.backend = self.npu._C if self.device_name == "npu" else torch._C
        self.observer_api = f"_{self.device_name}_attach_out_of_memory_observer"
        for patcher in (
            patch("verl.utils.profiler.torch_memory_profile.enable_memory_visualize"),
            patch("verl.utils.profiler.torch_memory_profile.get_device_name", return_value=self.device_name),
            patch("verl.utils.profiler.torch_memory_profile.get_torch_device", return_value=self.device),
            patch.dict(sys.modules, {"torch_npu": self.npu}),
            patch.object(self.backend, self.observer_api, self.attach_observer, create=True),
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
        total_label = "total_or_limit" if self.device_name == "npu" else "total"
        self.assertTrue(any(f"{self.device_name.upper()} OOM on device 0" in entry for entry in logs.output))
        self.assertTrue(any(f"requested=4096 {total_label}=1234 free=5678" in entry for entry in logs.output))
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

    def test_unavailable_device_skips_oom_observer(self):
        self.device.is_available.return_value = False
        with self.assertLogs("verl.utils.profiler.torch_memory_profile", level="WARNING"):
            TorchMemoryProfiler(rank=0, config=self._config())
        self.attach_observer.assert_not_called()
        self.assertFalse(TorchMemoryProfiler._oom_observer_attached)

    def test_missing_oom_observer_api_is_nonfatal(self):
        with (
            patch.object(self.backend, self.observer_api, None),
            self.assertLogs("verl.utils.profiler.torch_memory_profile", level="WARNING"),
        ):
            TorchMemoryProfiler(rank=0, config=self._config())
        self.attach_observer.assert_not_called()
        self.assertFalse(TorchMemoryProfiler._oom_observer_attached)

    def test_unsupported_device_skips_oom_observer(self):
        with (
            patch("verl.utils.profiler.torch_memory_profile.get_device_name", return_value="cpu"),
            patch("verl.utils.profiler.torch_memory_profile.get_torch_device", return_value=self.device) as get_device,
            patch.object(torch._C, "_cpu_attach_out_of_memory_observer", create=True) as unsupported_attach,
            self.assertLogs("verl.utils.profiler.torch_memory_profile", level="WARNING") as logs,
        ):
            TorchMemoryProfiler(rank=0, config=self._config())
        get_device.assert_not_called()
        unsupported_attach.assert_not_called()
        self.assertTrue(any("only available on CUDA/NPU devices" in entry for entry in logs.output))
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

    def test_oom_diagnostics_and_dump_failures_are_nonfatal(self):
        profiler = TorchMemoryProfiler(rank=0, config=self._config())
        observer = self.attach_observer.call_args.args[0]
        with (
            patch("verl.utils.profiler.torch_memory_profile.get_memory_info", side_effect=RuntimeError("stats failed")),
            patch.object(profiler.sampler, "dump_memory_snapshot", side_effect=OSError("disk full")) as dump_snapshot,
            self.assertLogs("verl.utils.profiler.torch_memory_profile", level="WARNING") as logs,
        ):
            observer(0, 4096, 1234, 5678)
        dump_snapshot.assert_called_once()
        self.assertTrue(any("stats failed" in entry for entry in logs.output))
        self.assertTrue(any("disk full" in entry for entry in logs.output))


class TestNpuTorchMemoryProfiler(TestTorchMemoryProfiler):
    device_name = "npu"

    def test_missing_torch_npu_is_nonfatal(self):
        with (
            patch.dict(sys.modules, {"torch_npu": None}),
            self.assertLogs("verl.utils.profiler.torch_memory_profile", level="WARNING"),
        ):
            TorchMemoryProfiler(rank=0, config=self._config())
        self.attach_observer.assert_not_called()
        self.assertFalse(TorchMemoryProfiler._oom_observer_attached)

    def test_npu_callback_writes_pickle_without_sync_or_native_dump(self):
        snapshot = {"segments": [], "device_traces": [[{"action": "oom", "size": 4096}]]}
        self.device.memory._snapshot.return_value = snapshot
        with (
            tempfile.TemporaryDirectory() as out_dir,
            patch("verl.utils.memory_utils.get_device_name", return_value="npu"),
            patch("verl.utils.memory_utils.get_torch_device", return_value=self.device),
            patch("verl.utils.profiler.torch_memory_profile.get_memory_info", return_value={}),
            patch(
                "verl.utils.profiler.torch_memory_profile.torch._C._cuda_attach_out_of_memory_observer", create=True
            ) as cuda_attach,
        ):
            TorchMemoryProfiler(rank=0, config=ProfilerConfig(save_path=out_dir))
            observer = self.attach_observer.call_args.args[0]
            with self.assertLogs("verl.utils.profiler.torch_memory_profile", level="ERROR"):
                observer(0, 4096, 1234, 5678)

            paths = list(Path(out_dir).glob("oom_*/torch_memory_oom_rank*_pid*.pickle"))
            self.assertEqual(len(paths), 1)
            with paths[0].open("rb") as f:
                self.assertEqual(pickle.load(f), snapshot)
            cuda_attach.assert_not_called()
        self.device.memory._snapshot.assert_called_once_with()
        self.device.memory._dump_snapshot.assert_not_called()
        self.device.synchronize.assert_not_called()
