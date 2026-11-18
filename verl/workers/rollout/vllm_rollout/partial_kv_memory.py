# Copyright 2026 Individual Contributor: HaitaoWuTJU
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
"""Physical allocation accounting for partial-rollout CPU preservation."""

import fcntl
import os
import time
from bisect import bisect_right
from contextlib import contextmanager
from pathlib import Path

RUNTIME_TAG = "verl_partial_runtime_state"
STATE_ROOTS = ("model_state", "sampler", "req_states", "input_buffers", "block_tables", "input_batch")
SKIP = {"model", "model_runner", "vllm_config", "compilation_config", "config", "_parameters", "_modules"}


def runtime_tensors(runner):
    """Find live V1/V2 request tensors without descending into model weights."""
    import torch

    seen, result = set(), []

    def visit(value, path, depth):
        if value is None or isinstance(value, str | bytes | int | float | bool | type):
            return
        if torch.is_tensor(value):
            if value.is_cuda and value.numel():
                result.append((path, value))
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        if depth > 12 or len(seen) > 20000:
            raise RuntimeError("Runtime tensor ownership traversal exceeded its bound")
        if isinstance(value, dict):
            items = value.items()
        elif isinstance(value, list | tuple):
            items = enumerate(value)
        elif hasattr(value, "__dict__"):
            items = vars(value).items()
        else:
            return
        for name, item in items:
            if name not in SKIP:
                visit(item, f"{path}.{name}", depth + 1)

    for name in STATE_ROOTS:
        if hasattr(runner, name):
            visit(getattr(runner, name), name, 0)
    # V1 also keeps scheduling tensors directly on GPUModelRunner.
    for name, value in vars(runner).items():
        if torch.is_tensor(value):
            visit(value, name, 0)
    if hasattr(runner, "model_state"):
        for name in ("model_state", "sampler"):
            if not any(path.startswith(name + ".") for path, _ in result):
                cls = type(runner.model_state)
                # Native DefaultModelState owns no CUDA tensors for ordinary
                # 1D-position text transformers without prompt embeddings.
                # Recurrent/unknown state implementations still must expose
                # their live CUDA state; do not broadly waive the ownership gate.
                if name == "model_state" and (cls.__module__, cls.__name__) == (
                    "vllm.v1.worker.gpu.model_states.default",
                    "DefaultModelState",
                ):
                    continue
                raise RuntimeError(f"Native {name} has no inspectable CUDA state")
    elif not hasattr(runner, "input_batch"):
        raise RuntimeError("Unsupported runner: no inspectable native request state")
    return result


def allocations_for_views(views, allocations):
    """Classify tensor addresses; mixed parameter/state allocations stay whole."""
    ordered = sorted(allocations)
    starts = [row[0] for row in ordered]
    found = {}
    for path, ptr in views:
        index = bisect_right(starts, ptr) - 1
        if index < 0:
            continue
        start, size, tag = ordered[index]
        if start <= ptr < start + size and tag in ("weights", RUNTIME_TAG):
            row = found.setdefault(start, dict(pointer=start, bytes=size, original_tag=tag, paths=[]))
            row["paths"].append(path)
    return list(found.values())


def protect_runtime_state(runner, allocator):
    """Retag entire allocations holding live CUDA request/sampler state."""
    tensors = runtime_tensors(runner)
    entries = allocations_for_views(
        [(path, tensor.data_ptr()) for path, tensor in tensors],
        [(ptr, int(data.handle[1]), data.tag) for ptr, data in allocator.pointer_to_data.items()],
    )
    if any(allocator.pointer_to_data[row["pointer"]].is_asleep for row in entries):
        raise RuntimeError("Protect live request state before any allocation sleeps")
    for row in entries:
        allocator.pointer_to_data[row["pointer"]].tag = RUNTIME_TAG
    return dict(
        tag=RUNTIME_TAG,
        protected_allocation_bytes=sum(row["bytes"] for row in entries),
        protected_allocations=entries,
        accounting="whole physical allocations, possibly including parameters and padding",
    )


def host_memory_available():
    """Return the smaller of host MemAvailable and the cgroup's memory headroom."""
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(values["MemAvailable"].split()[0]) * 1024
    for limit, used in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        if Path(limit).is_file() and Path(used).is_file():
            cap = Path(limit).read_text().strip()
            if cap != "max":
                available = min(available, int(cap) - int(Path(used).read_text()))
    return available


@contextmanager
def admit_cpu_copy(required, headroom, *, timeout=300, directory="/tmp"):
    """Serialize each node's copies and recheck capacity after acquiring the lock.

    The lock encloses allocator copies only, never Worker.sleep's NCCL waits.
    The allocator owns one physical CPU backup; admission adds no KV payload.
    """
    if type(required) is not int or required < 0 or type(headroom) is not int or headroom < 0:
        raise ValueError("Invalid CPU KV byte budget")
    path = Path(directory) / f"verl-partial-kv-{os.getuid()}.lock"
    started = time.monotonic()
    with path.open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - started >= timeout:
                    raise TimeoutError("CPU KV copy admission timed out") from None
                time.sleep(0.02)
        try:
            if host_memory_available() < required + headroom:
                raise RuntimeError("Insufficient host/cgroup memory for full-pool KV backup")
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
