# Copyright 2026 Individual Contributor
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

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.distributed as dist

from verl.checkpoint_engine import nccl_checkpoint_engine as nccl


@pytest.mark.asyncio
@pytest.mark.parametrize("num_senders", [1, 2], ids=["single-sender", "relay-root"])
async def test_broadcast_host_return_precedes_next_export_gather(monkeypatch, num_senders):
    loop = asyncio.get_running_loop()
    shard_sizes = [2, 2, 2, 2, 1]  # two ranks make four full 4-byte buckets and a 2-byte tail
    store = dist.HashStore()
    entered = [threading.Event() for _ in shard_sizes]
    trace, payloads, device_fences = [], [], []

    def export(rank):
        group = dist.ProcessGroupGloo(store, rank, 2, timedelta(seconds=5))
        for i, size in enumerate(shard_sizes):
            shard = torch.full((size,), 2 * i + rank, dtype=torch.uint8)
            parts = [torch.empty_like(shard) for _ in range(2)]
            if rank == 0:
                # Weight i-1 triggered the overflow that scheduled bucket i-2.
                if i >= 2:
                    assert entered[i - 2].wait(5)
                trace.append(f"gather {i}")
            group.allgather([parts], [shard]).wait()
            yield f"weight_{i}", torch.cat(parts)
        if rank == 0:
            assert entered[-2].wait(5)
            trace.append("export exhausted")

    def broadcast(bucket, *, src_rank, group_name):
        seq = len(payloads)
        trace.append(f"enter {seq}")
        entered[seq].set()
        # The event loop can release the real executor thread only when send_weights yields.
        release = threading.Event()
        loop.call_soon_threadsafe(release.set)
        assert release.wait(5)
        payloads.append(bucket.tobytes())
        trace.append(f"return {seq}")

    monkeypatch.setattr(nccl, "collective", SimpleNamespace(broadcast=broadcast))
    monkeypatch.setattr(nccl, "cp", np)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: device_fences.append(trace[-1]))
    monkeypatch.setattr(nccl.NCCLCheckpointEngine, "_start_zmq_server", lambda self: None)
    root = nccl.NCCLCheckpointEngine(bucket_size=4, is_master=True)
    root.rank, root.num_senders, root.socket = 0, num_senders, Mock()
    root.send_buf, root.recv_buf = np.empty(4, dtype=np.uint8), np.empty(4, dtype=np.uint8)

    with ThreadPoolExecutor(1) as peer:
        other_rank = peer.submit(list, export(1))
        await asyncio.wait_for(root.send_weights(export(0)), timeout=15)
        other_rank.result(timeout=5)

    logging.getLogger(__name__).info("Root host order: %s", " -> ".join(trace))
    for seq in range(len(shard_sizes) - 1):
        next_export = f"gather {seq + 2}" if seq + 2 < len(shard_sizes) else "export exhausted"
        assert trace.index(f"return {seq}") < trace.index(next_export), trace
    assert payloads == [bytes([2 * i]) * n + bytes([2 * i + 1]) * n for i, n in enumerate(shard_sizes)]
    metadata = [call.args[0] for call in root.socket.send_pyobj.call_args_list]
    assert [(m["length"], m["is_last"]) for m in metadata] == [(4, False)] * 4 + [(2, True)]
    assert len(device_fences) == len(shard_sizes) + 1
    assert device_fences[-1] == "return 4"
