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
"""Spawn a CPU worker with the production combined-request functions."""

import sys
from functools import partial
from pathlib import Path
from typing import NamedTuple

import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verl/utils/megatron"))
from async_checkpoint_batch import coalesce_async_requests  # noqa: E402


class Request(NamedTuple):
    async_fn: object
    async_fn_args: tuple
    finalize_fns: list
    async_fn_kwargs: dict
    preload_fn: object
    is_frozen: bool = False


def preload(buckets):
    return [
        (name, key, (byte_data, [(item, tensor.clone()) for item, tensor in tensor_data]))
        for name, key, (byte_data, tensor_data) in buckets
    ]


def write(rank, staged, output):
    output.put((rank, staged[0][2][1][0][1].tolist()))


def child(request_queue, ready, release):
    request = request_queue.get(timeout=10)
    staged = request.preload_fn()
    ready.set()
    if not release.wait(timeout=10):
        raise TimeoutError("parent did not release writer")
    request.async_fn(request.async_fn_args[0], staged, None)


def main():
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    request_queue = ctx.Queue()
    output = manager.Queue()
    ready, release = ctx.Event(), ctx.Event()
    values = [torch.tensor([3.0]), torch.tensor([7.0])]
    requests = []
    for rank, tensor in enumerate(values):
        buckets = [("file", "key", ([], [("tensor", tensor)]))]
        requests.append(Request(write, (rank, buckets, output), [], {}, partial(preload, buckets)))
    batch, _ = coalesce_async_requests(requests, 1024, lambda eligible: eligible)
    assert len(batch) == 1
    process = ctx.Process(target=child, args=(request_queue, ready, release))
    process.start()
    try:
        request_queue.put(batch[0])
        if not ready.wait(timeout=15):
            raise TimeoutError("spawned writer did not preload")
        for value in values:
            value.fill_(99)
        release.set()
        result = [output.get(timeout=10), output.get(timeout=10)]
        process.join(timeout=10)
        assert process.exitcode == 0 and result == [(0, [3.0]), (1, [7.0])], result
        print("SPAWN_BATCH_OK: combined request preserved both snapshots and write order")
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        request_queue.close()
        manager.shutdown()


if __name__ == "__main__":
    main()
