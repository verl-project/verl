# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Check CuPy staging release across real two-GPU delta broadcasts.

Run from the repository root with two visible GPUs::

    PYTHONPATH=. python tests/special_distributed/test_delta_staging_lifecycle.py

This isolates the production send/release methods, with a metadata sink and a
real Ray NCCL receiver. It does not exercise model loading or the ZMQ handshake.
Payloads alternate seed, empty, sparse and near-full sends. Receiver values and
positions must match exactly. Every sender release must return its dedicated
CuPy pool to zero used/total bytes. A fresh pair of actors repeats the sequence.
"""

import json
import time

import ray


@ray.remote(num_gpus=1)
class Endpoint:
    def run(self, rank, group):
        import cupy as cp
        import ray.util.collective as collective
        import torch

        from verl.checkpoint_engine.delta_checkpoint_engine import DeltaShardedCheckpointEngine
        from verl.checkpoint_engine.delta_sync.encode import DeltaFlush

        torch.cuda.set_device(0)
        collective.init_collective_group(2, rank, "nccl", group)

        class ManifestSink:
            def send_string(self, *a, **k):
                pass

            def send_pyobj(self, *a, **k):
                pass

        engine = object.__new__(DeltaShardedCheckpointEngine)
        engine.group_name = group
        engine.topic = "probe"
        engine.socket = ManifestSink()
        engine.encoding = "indices"
        pool = cp.get_default_memory_pool()
        records = []
        try:
            for step in range(40):
                phase = ["seed", "zero", "sparse", "near_full"][step % 4]
                n = {"seed": 8 * 1024 * 1024, "zero": 0, "sparse": 128 * 1024, "near_full": 4 * 1024 * 1024}[phase]
                start = time.perf_counter()
                if rank == 0:
                    values = (torch.arange(n, device="cuda", dtype=torch.int32) % 128 + step).to(torch.bfloat16)
                    if phase == "seed":
                        engine._publish_values_flush([], values, True)
                    elif phase == "zero":
                        engine._publish_terminal(False)
                    else:
                        pos = torch.arange(n, device="cuda", dtype=torch.int32).view(torch.uint8)
                        engine._publish_flush(DeltaFlush("indices", [], pos, values, 0), False, True)
                        del pos
                    del values
                    before = pool.total_bytes()
                    engine._release_staging_pool(phase)
                    assert pool.used_bytes() == 0, (step, phase, pool.used_bytes())
                    assert pool.total_bytes() == 0, (step, phase, pool.total_bytes())
                    records.append(
                        {
                            "step": step,
                            "phase": phase,
                            "pool_before": before,
                            "pool_used": pool.used_bytes(),
                            "pool_total": pool.total_bytes(),
                            "seconds": time.perf_counter() - start,
                        }
                    )
                elif n:
                    if phase != "seed":
                        positions = cp.empty(n * 4, dtype=cp.uint8)
                        collective.broadcast(positions, 0, group)
                        actual_pos = torch.as_tensor(positions, device="cuda").view(torch.int32)
                        assert torch.equal(actual_pos, torch.arange(n, device="cuda", dtype=torch.int32))
                        del actual_pos, positions
                    data = cp.empty(n * 2, dtype=cp.uint8)
                    collective.broadcast(data, 0, group)
                    actual = torch.as_tensor(data, device="cuda").view(torch.bfloat16)
                    expected = (torch.arange(n, device="cuda", dtype=torch.int32) % 128 + step).to(torch.bfloat16)
                    assert torch.equal(actual, expected), (step, phase)
                    del actual, expected, data
                    pool.free_all_blocks()
            # The value comparison synchronizes the receiving stream only; sender
            # uses the production release path without added device synchronization.
            return {"rank": rank, "status": "PASS", "records": records}
        finally:
            collective.destroy_collective_group(group)


def main():
    """Run two isolated sender/receiver lifetimes and fail on any mismatch."""
    import torch

    if torch.cuda.device_count() < 2:
        raise RuntimeError("This probe requires two visible CUDA GPUs")
    ray.init(num_cpus=4, num_gpus=2, include_dashboard=False, object_store_memory=256 * 1024 * 1024)
    results = []
    try:
        for restart in range(2):
            actors = [Endpoint.remote() for _ in range(2)]
            try:
                result = ray.get([a.run.remote(i, f"staging-{restart}") for i, a in enumerate(actors)], timeout=180)
                results.append(result)
            finally:
                for a in actors:
                    ray.kill(a)
        print(json.dumps({"rounds": results}), flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
