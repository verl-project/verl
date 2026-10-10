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
"""Verify a failing stage flushes its real NPU profiler before returning."""

import json
import multiprocessing
from pathlib import Path
from unittest.mock import patch

import pytest


def _check_failure_trace(save_path):
    import torch
    import torch_npu  # noqa: F401 - registers the NPU backend

    from verl.utils.profiler import mstx_profile as module
    from verl.utils.profiler.config import NPUToolConfig, ProfilerConfig
    from verl.utils.profiler.profile import DistProfiler

    torch.npu.set_device(0)
    inputs = torch.randn(64, 64, device="npu:0")
    backend_factory = module.get_npu_profiler
    backends = []

    def capture_backend(*args, **kwargs):
        backend = backend_factory(*args, **kwargs)
        backends.append(backend)
        return backend

    failure = RuntimeError("training stage failed")

    class Worker:
        def __init__(self):
            self.calls = 0
            self.profiler = DistProfiler(
                0,
                ProfilerConfig(tool="npu", enable=True, all_ranks=True, save_path=save_path),
                NPUToolConfig(discrete=True, contents=["npu"], analysis=False, level="level0"),
            )
            self.profiler.start()

        @DistProfiler.annotate(role="update_actor")
        def stage(self, fail):
            self.calls += 1
            result = inputs @ inputs
            torch.npu.synchronize()
            if fail:
                raise failure
            return result

    worker = Worker()
    try:
        with (
            patch.object(module, "get_npu_profiler", side_effect=capture_backend),
            patch.object(module, "mark_end_range", wraps=module.mark_end_range) as end_range,
        ):
            with pytest.raises(RuntimeError) as raised:
                worker.stage(fail=True)
            assert raised.value is failure
            assert worker.calls == 1
            assert len(backends) == 1
            assert backends[0].stopped
            assert end_range.call_count == 1
            framework_traces = list(Path(save_path).rglob("torch.op_range"))
            assert len(framework_traces) == 1
            assert framework_traces[0].stat().st_size > 0

            # Parse the failed stage's real trace; its marker must have an end.
            trace_path = str(Path(save_path) / "failed_stage.json")
            backends[0].export_chrome_trace(trace_path)
            trace = json.loads(Path(trace_path).read_text())
            events = trace["traceEvents"] if isinstance(trace, dict) else trace
            markers = [event for event in events if event.get("name") == "update_actor"]
            assert markers, "The failed stage's MSTX marker was not flushed."
            assert all(event.get("ph") == "X" and event.get("dur", 0) > 0 for event in markers)
            assert any(event.get("name") in {"aten::mm", "aten::matmul"} for event in events)

            result = worker.stage(fail=False)
            torch.testing.assert_close(result, inputs @ inputs)
            assert worker.calls == 2
            assert len(backends) == 2 and backends[1].stopped
            assert end_range.call_count == 2
            assert len(list(Path(save_path).rglob("torch.op_range"))) == 2
            print(f"failed_stage_stopped=True, closed_markers={len(markers)}, subsequent_stage_stopped=True")
    finally:
        # Keep failed baseline runs and assertions from leaking the global profiler.
        for backend in backends:
            if not backend.stopped:
                backend.stop()


def test_failure_flushes_trace_and_allows_next_stage(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Requires an Ascend NPU.")
    process = multiprocessing.get_context("spawn").Process(target=_check_failure_trace, args=(str(tmp_path),))
    process.start()
    try:
        process.join(timeout=60)
        assert process.exitcode == 0, f"NPU profiler regression failed or timed out (exitcode={process.exitcode})."
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        process.close()
