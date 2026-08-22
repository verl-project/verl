# Copyright 2026 Amazon.com Inc and/or its affiliates
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
"""Isolated checks for the nccl_parallel engine's riskiest stream mechanisms.

Runs on ONE GPU (local ray init), far cheaper than the full correctness test:
1. `_force_nonblocking_comm_streams()` patches ray's stream factory, and the
   streams ray then creates really do come from the patched factory.
2. `_comm_stream(group)` returns a handle for a live group -- without it the
   sender falls back to a device-wide sync and the senders re-serialize.
3. A cupy event recorded on that stream completes, so the engine's
   per-transfer event polling can make progress.

A failure here means a real run would silently lose sender overlap (2/3) or
hang (1), so this is the first thing to run when triaging the engine on a new
stack (ray/cupy/torch version bumps).
"""

import time

import pytest
import ray
import torch


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="needs one CUDA device")
def test_nonblocking_stream_mechanisms():
    from verl.checkpoint_engine import nccl_parallel_checkpoint_engine as eng

    assert eng._force_nonblocking_comm_streams(), "stream-factory patch did not apply in the driver process"

    ray.init(num_gpus=1, ignore_reinit_error=True)
    try:

        @ray.remote(num_gpus=1)
        class Solo:
            def run(self):
                import cupy as cp
                import ray.util.collective as collective

                from verl.checkpoint_engine import nccl_parallel_checkpoint_engine as e

                out = {}
                out["patched"] = e._force_nonblocking_comm_streams()
                collective.init_collective_group(1, 0, "nccl", "mech_check")
                collective.barrier("mech_check")
                out["streams_created"] = e._NONBLOCKING_STREAMS_CREATED
                stream = e._comm_stream("mech_check")
                out["stream_found"] = stream is not None
                if stream is not None:
                    buf = torch.ones(1 << 20, dtype=torch.uint8, device="cuda")
                    collective.broadcast(buf, src_rank=0, group_name="mech_check")
                    ev = cp.cuda.Event()
                    ev.record(stream)
                    deadline = time.monotonic() + 30
                    out["event_completed"] = False
                    while time.monotonic() < deadline:
                        if ev.done:
                            out["event_completed"] = True
                            break
                        time.sleep(0.001)
                return out

        res = ray.get(Solo.remote().run.remote())
        assert res["patched"], "stream-factory patch did not apply in the worker process"
        assert res["streams_created"] >= 1, "ray created comm streams outside the patched factory"
        assert res["stream_found"], "_comm_stream() returned None -> sender falls back to device-wide sync"
        assert res["event_completed"], "cupy event on the comm stream never completed"
    finally:
        ray.shutdown()
