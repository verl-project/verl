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
"""Benchmark phase scheduling for collectives on overlapping process groups.

The benchmark deliberately has no dependency on the rest of verl. A typical
four-GPU run is::

    torchrun --standalone --nproc-per-node=4 \
      scripts/benchmark_collective_phase_sweep.py \
      --comm-a all-to-all --comm-b reduce-scatter \
      --group-layout auto --offset-us -4000 -2000 -1000 -500 0 500 1000 2000 4000 \
      --warmup 20 --iters 200 --output-json phase_sweep.json

``offset`` uses an absolute host clock gate. It is a measurement/search tool,
not a recommendation to add a fixed sleep to a training runtime. On CUDA the
reported realized offset and operation durations are CUDA-event brackets around
collective eligibility and observed completion. They are not exact NCCL kernel
boundaries; use an Nsight Systems/CUPTI trace for kernel-observed evidence.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import math
import os
import platform
import random
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

SCHEMA_VERSION = 2
COMM_A_CHOICES = ("all_to_all", "ulysses_all_to_all")
COMM_B_CHOICES = ("all_reduce", "reduce_scatter", "all_gather")
POLICY_CHOICES = ("isolated", "concurrent", "serialized", "offset")
SUMMARY_METRIC_NAMES = tuple(
    f"{name}_p{percent}"
    for name in (
        "comm_a_ms",
        "comm_b_ms",
        "pair_completion_ms",
        "actual_overlap_ms",
        "realized_gpu_offset_us",
        "api_launch_offset_us",
        "rank_start_skew_us",
        "rank_finish_skew_us",
        "launch_anchor_lateness_us",
    )
    for percent in (50, 95, 99)
) + ("pairs_per_second",)
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_HOST_GATE = threading.Event()


@dataclasses.dataclass(frozen=True)
class GroupSpec:
    group_id: str
    ranks: tuple[int, ...]


@dataclasses.dataclass
class GroupContext:
    group_a: dist.ProcessGroup
    group_b: dist.ProcessGroup
    group_id_a: str
    group_id_b: str
    resolved_layout: str
    created_groups: list[dist.ProcessGroup]


@dataclasses.dataclass
class LocalObservation:
    a_start_us: float | None = None
    a_end_us: float | None = None
    b_start_us: float | None = None
    b_end_us: float | None = None
    a_api_launch_ns: int | None = None
    b_api_launch_ns: int | None = None
    launch_anchor_lateness_us: float | None = None
    collectives: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class CudaLaunch:
    start: torch.cuda.Event
    end: torch.cuda.Event
    api_launch_ns: int
    api_return_ns: int
    sequence_id: int
    stream_id: int


def policy_cell_id(mode: str, offset_us: float) -> str:
    """Separate offset candidates even when they share one process invocation."""
    if not math.isfinite(offset_us):
        raise ValueError("offset must be finite")
    return f"offset/{offset_us:.17g}us" if mode == "offset" else mode


class NativeTraceWriter:
    """Stream lossless rank-local records outside the measured trial region.

    Transfer leases describe permission to reuse persistent benchmark buffers,
    not physical CUDA allocation/free events. CPU timings remain host timings.
    """

    def __init__(self, path: Path, *, rank: int, world_size: int, run_id: str, backend: str):
        self.path = path
        self.rank, self.world_size, self.run_id, self.backend = rank, world_size, run_id, backend
        path.parent.mkdir(parents=True, exist_ok=True)
        self.output = path.open("x", encoding="utf-8")
        self.records = 0

    def emit(self, observation: LocalObservation, *, mode: str, offset_us: float, step: int, phase: str) -> None:
        """Preserve measured boundaries, group membership and exact sequence IDs."""
        for side, measured in observation.collectives.items():
            pair_id = f"{self.run_id}/trial/{step}"
            logical_id = f"{pair_id}/{side}"
            record = {
                "schema_version": 3,
                "record_type": "collective",
                "framework": "verl",
                "run_id": self.run_id,
                "process_launch_id": self.run_id,
                "hostname": socket.gethostname(),
                "rank": self.rank,
                "world_size": self.world_size,
                "step": step,
                "direction": "benchmark",
                "policy_id": policy_cell_id(mode, offset_us),
                "sample_phase": phase,
                "pair_id": pair_id,
                "pair_role": side,
                "logical_operation_id": logical_id,
                "requested_offset_us": offset_us,
                "transport": self.backend,
                "topology_class": "single-node-measured",
                "gpu_timestamp_semantics": "event-bracket" if self.backend == "nccl" else "not-applicable",
                "timestamp_domain": (
                    "single-node-perf-counter-projected-cuda-event"
                    if self.backend == "nccl"
                    else "single-node-perf-counter"
                ),
                "clock_sync_error_bound_us": None,
                "kernel_observed": False,
                **measured,
            }
            if self.backend != "nccl":
                # Host collective durations are never renamed to GPU timestamps.
                record["gpu_start_timestamp_ns"] = record["gpu_end_timestamp_ns"] = None
            self.output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            self.records += 1
        self.output.flush()

    def close(self) -> None:
        """Close this writer without touching any other rank's artifact."""
        self.output.close()


def create_native_trace_writer(
    template: str, *, rank: int, world_size: int, run_id: str, backend: str, summary_path: Path
) -> NativeTraceWriter:
    """Reject collisions or local filesystem errors on every rank before trials."""
    writer = None
    error = None
    path = None
    resolved = None
    try:
        path = Path(template.format(rank=rank))
        if "{rank}" not in template:
            path = path.with_name(f"{path.stem}.rank-{rank}{path.suffix}")
        resolved = str(path.resolve())
        if path.resolve() == summary_path.resolve():
            raise ValueError("raw trace and summary output paths must differ")
    except (OSError, ValueError, KeyError, IndexError) as exc:
        error = str(exc)
    choices = [None] * world_size
    dist.all_gather_object(choices, (resolved, error))
    if any(item[1] for item in choices) or len({item[0] for item in choices}) != world_size:
        raise ValueError(f"invalid raw trace paths: {choices}")
    try:
        writer = NativeTraceWriter(path, rank=rank, world_size=world_size, run_id=run_id, backend=backend)
    except OSError as exc:
        error = str(exc)
    dist.all_gather_object(choices, error)
    if any(choices):
        if writer is not None:
            writer.close()
        raise ValueError(f"raw trace files must be fresh and writable on every rank: {choices}")
    return writer


class SequenceTracker:
    """Track the logical launch sequence for one communicator."""

    def __init__(self) -> None:
        self.count = 0
        self._digest = hashlib.sha256()

    def record(self, operation: str) -> int:
        sequence_id = self.count
        self._digest.update(f"{sequence_id}:{operation};".encode())
        self.count += 1
        return sequence_id

    @property
    def digest(self) -> str:
        return self._digest.hexdigest()


def parse_size(value: str) -> int:
    """Parse a byte count, accepting binary suffixes such as 64MiB."""

    normalized = value.strip().lower().replace(" ", "")
    suffixes = {
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "b": 1,
    }
    multiplier = 1
    for suffix, candidate in suffixes.items():
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            multiplier = candidate
            break
    try:
        parsed = float(normalized) * multiplier
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid byte count: {value!r}") from exc
    if not math.isfinite(parsed) or parsed <= 0 or not parsed.is_integer():
        raise argparse.ArgumentTypeError(f"byte count must be a positive integer: {value!r}")
    return int(parsed)


def normalize_collective(value: str) -> str:
    normalized = value.replace("-", "_").lower()
    if normalized not in COMM_A_CHOICES + COMM_B_CHOICES:
        choices = ", ".join(name.replace("_", "-") for name in COMM_A_CHOICES + COMM_B_CHOICES)
        raise argparse.ArgumentTypeError(f"unknown collective {value!r}; choose one of: {choices}")
    return normalized


def _automatic_mesh_shape(world_size: int) -> tuple[int, int] | None:
    candidates = [
        (rows, world_size // rows)
        for rows in range(2, math.isqrt(world_size) + 1)
        if world_size % rows == 0 and world_size // rows >= 2
    ]
    return min(candidates, key=lambda shape: abs(shape[0] - shape[1])) if candidates else None


def resolve_group_layout(
    layout: str, world_size: int, mesh_shape: tuple[int, int] | None = None
) -> tuple[str, tuple[int, int] | None]:
    """Resolve a general group layout; ``ep3-dp2`` is only a mesh shorthand."""

    if world_size < 2:
        raise ValueError("the benchmark requires at least two ranks")
    if layout == "world":
        if mesh_shape is not None:
            raise ValueError("--mesh-shape cannot be combined with --group-layout world")
        return "world", None
    shorthand = re.fullmatch(r"ep(\d+)-dp(\d+)", layout)
    if shorthand:
        if mesh_shape is not None:
            raise ValueError("--mesh-shape cannot be combined with an epN-dpM shorthand")
        columns, rows = (int(value) for value in shorthand.groups())
        shape = (rows, columns)
    elif layout in ("auto", "mesh"):
        shape = mesh_shape or _automatic_mesh_shape(world_size)
        if shape is None:
            if layout == "auto":
                return "world", None
            raise ValueError(f"world size {world_size} has no non-degenerate 2-D mesh; use --group-layout world")
    else:
        raise ValueError("--group-layout must be auto, mesh, world, or an epN-dpM shorthand")
    rows, columns = shape
    if rows < 2 or columns < 2:
        raise ValueError("both dimensions of --mesh-shape must be at least two")
    if rows * columns != world_size:
        raise ValueError(f"mesh shape {rows}x{columns} does not match world size {world_size}")
    return f"mesh-{rows}x{columns}", shape


def build_group_specs(
    layout: str, world_size: int, mesh_shape: tuple[int, int] | None = None
) -> tuple[list[GroupSpec], list[GroupSpec]]:
    """Return process-group specs in the globally consistent creation order."""

    resolved_layout, shape = resolve_group_layout(layout, world_size, mesh_shape)
    if resolved_layout == "world":
        ranks = tuple(range(world_size))
        return [GroupSpec("world-a", ranks)], [GroupSpec("world-b", ranks)]
    rows, columns = shape
    groups_a = [
        GroupSpec(f"mesh-row-{row}", tuple(row * columns + column for column in range(columns))) for row in range(rows)
    ]
    groups_b = [
        GroupSpec(f"mesh-column-{column}", tuple(row * columns + column for row in range(rows)))
        for column in range(columns)
    ]
    return groups_a, groups_b


def _create_groups(layout: str, world_size: int, rank: int, mesh_shape: tuple[int, int] | None = None) -> GroupContext:
    resolved_layout, _ = resolve_group_layout(layout, world_size, mesh_shape)
    specs_a, specs_b = build_group_specs(layout, world_size, mesh_shape)
    created: list[dist.ProcessGroup] = []
    local_a: tuple[dist.ProcessGroup, str] | None = None
    local_b: tuple[dist.ProcessGroup, str] | None = None
    for spec in (*specs_a, *specs_b):
        group = dist.new_group(ranks=list(spec.ranks))
        if group != dist.GroupMember.NON_GROUP_MEMBER:
            created.append(group)
            if rank in spec.ranks:
                if spec in specs_a:
                    local_a = (group, spec.group_id)
                else:
                    local_b = (group, spec.group_id)
    if local_a is None or local_b is None:
        raise RuntimeError(f"rank {rank} did not receive both process groups")
    return GroupContext(local_a[0], local_b[0], local_a[1], local_b[1], resolved_layout, created)


class CollectiveBuffer:
    """Own input/output tensors and correctness checks for one collective."""

    def __init__(
        self,
        operation: str,
        requested_bytes: int,
        dtype: torch.dtype,
        device: torch.device,
        group: dist.ProcessGroup,
    ) -> None:
        self.operation = operation
        self.dtype = dtype
        self.device = device
        self.group = group
        self.group_size = dist.get_world_size(group)
        self.group_rank = dist.get_rank(group)
        element_size = torch.empty((), dtype=dtype).element_size()
        requested_elements = max(1, math.ceil(requested_bytes / element_size))
        alignment = self.group_size if operation in (*COMM_A_CHOICES, "reduce_scatter") else 1
        self.input_elements = math.ceil(requested_elements / alignment) * alignment
        self.message_bytes = self.input_elements * element_size
        self.input = torch.empty(self.input_elements, dtype=dtype, device=device)
        if operation in COMM_A_CHOICES:
            self.output = torch.empty_like(self.input)
        elif operation == "all_reduce":
            self.output = self.input
        elif operation == "reduce_scatter":
            self.output = torch.empty(self.input_elements // self.group_size, dtype=dtype, device=device)
        elif operation == "all_gather":
            self.output = torch.empty(self.input_elements * self.group_size, dtype=dtype, device=device)
        else:
            raise ValueError(f"unsupported collective: {operation}")
        self.reset()

    def reset(self) -> None:
        if self.operation in COMM_A_CHOICES:
            chunks = self.input.view(self.group_size, -1)
            for destination in range(self.group_size):
                chunks[destination].fill_(self.group_rank * self.group_size + destination + 1)
            self.output.zero_()
        else:
            self.input.fill_(self.group_rank + 1)
            if self.output is not self.input:
                self.output.zero_()

    def launch(self) -> dist.Work:
        if self.operation == "all_to_all":
            return dist.all_to_all_single(self.output, self.input, group=self.group, async_op=True)
        if self.operation == "ulysses_all_to_all":
            input_chunks = list(self.input.chunk(self.group_size))
            output_chunks = list(self.output.chunk(self.group_size))
            return dist.all_to_all(output_chunks, input_chunks, group=self.group, async_op=True)
        if self.operation == "all_reduce":
            return dist.all_reduce(self.input, group=self.group, async_op=True)
        if self.operation == "reduce_scatter":
            return dist.reduce_scatter_tensor(self.output, self.input, group=self.group, async_op=True)
        if self.operation == "all_gather":
            return dist.all_gather_into_tensor(self.output, self.input, group=self.group, async_op=True)
        raise AssertionError(f"unreachable collective: {self.operation}")

    def is_correct(self) -> bool:
        if self.operation in COMM_A_CHOICES:
            expected = torch.empty_like(self.output).view(self.group_size, -1)
            for source in range(self.group_size):
                expected[source].fill_(source * self.group_size + self.group_rank + 1)
            return bool(torch.equal(self.output, expected.view_as(self.output)))
        reduced_value = self.group_size * (self.group_size + 1) // 2
        if self.operation in ("all_reduce", "reduce_scatter"):
            expected = torch.full_like(self.output, reduced_value)
            return bool(torch.equal(self.output, expected))
        expected = torch.empty_like(self.output).view(self.group_size, -1)
        for source in range(self.group_size):
            expected[source].fill_(source + 1)
        return bool(torch.equal(self.output, expected.view_as(self.output)))


class BenchmarkRunner:
    def __init__(
        self,
        buffer_a: CollectiveBuffer,
        buffer_b: CollectiveBuffer,
        device: torch.device,
        validate: bool,
        launch_anchor_lead_us: int,
        trace_writer: NativeTraceWriter | None = None,
    ) -> None:
        self.buffer_a = buffer_a
        self.buffer_b = buffer_b
        self.device = device
        self.validate = validate
        self.launch_anchor_lead_us = launch_anchor_lead_us
        self.trace_writer = trace_writer
        self.trial_index = 0
        self.sequence_a = SequenceTracker()
        self.sequence_b = SequenceTracker()
        if device.type == "cuda":
            self.stream_a = torch.cuda.Stream(device=device)
            self.stream_b = torch.cuda.Stream(device=device)

    def run_trial(self, mode: str, requested_offset_us: float = 0.0, *, phase: str = "measurement") -> LocalObservation:
        if mode not in ("isolated_a", "isolated_b", "concurrent", "serialized", "offset"):
            raise ValueError(f"unsupported trial mode: {mode}")
        self.buffer_a.reset()
        self.buffer_b.reset()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        dist.barrier()
        anchor_ns = self._shared_launch_anchor_ns()
        if self.device.type == "cuda":
            observation = self._run_cuda_trial(mode, requested_offset_us, anchor_ns)
        else:
            observation = self._run_cpu_trial(mode, requested_offset_us, anchor_ns)
        self._validate(mode, observation)
        if self.trace_writer is not None:
            # Every transfer has physically completed. After optional payload
            # consumption, this is the boundary permitting the next buffer reset.
            for record in observation.collectives.values():
                record["buffer_reuse_release_timestamp_ns"] = time.perf_counter_ns()
            self.trace_writer.emit(
                observation, mode=mode, offset_us=requested_offset_us, step=self.trial_index, phase=phase
            )
        self.trial_index += 1
        return observation

    def _shared_launch_anchor_ns(self) -> int:
        anchor = [time.perf_counter_ns() + self.launch_anchor_lead_us * 1000 if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(anchor, src=0)
        return int(anchor[0])

    def _launch_cuda(
        self,
        buffer: CollectiveBuffer,
        tracker: SequenceTracker,
        stream: torch.cuda.Stream,
        ready: torch.cuda.Event,
    ) -> CudaLaunch:
        sequence_id = tracker.record(buffer.operation)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        api_launch_ns = time.perf_counter_ns()
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            start.record(stream)
            work = buffer.launch()
            api_return_ns = time.perf_counter_ns()
            # NCCL work.wait() inserts a dependency from the NCCL stream into the
            # current CUDA stream without requiring a device-wide synchronize.
            work.wait()
            end.record(stream)
        return CudaLaunch(start, end, api_launch_ns, api_return_ns, sequence_id, int(stream.cuda_stream))

    def _raw_fields(
        self,
        buffer: CollectiveBuffer,
        sequence_id: int,
        stream_id: str,
        launch_ns: int,
        return_ns: int,
        completion_ns: int,
    ) -> dict[str, Any]:
        ranks = dist.get_process_group_ranks(buffer.group)
        # Group names distinguish two communicators with identical membership.
        group_name = getattr(buffer.group, "group_name", None)
        if group_name is None:
            group_name = dist._get_process_group_name(buffer.group)
        return {
            "operation": buffer.operation,
            "process_group_id": f"{dist.get_backend(buffer.group)}:{group_name}:ranks-{','.join(map(str, ranks))}",
            "process_group_ranks": ranks,
            "communicator_sequence_id": sequence_id,
            "stream_id": stream_id,
            "message_bytes": buffer.message_bytes,
            "api_launch_timestamp_ns": launch_ns,
            "api_return_timestamp_ns": return_ns,
            "completion_timestamp_ns": completion_ns,
            "consumer_timestamp_ns": None,
            "buffer_reuse_acquire_timestamp_ns": launch_ns,
            "resource_scope": "persistent-buffer-transfer-lease",
            "completion_semantics": "host-observed-physical-completion",
        }

    def _run_cuda_trial(self, mode: str, requested_offset_us: float, anchor_host_ns: int) -> LocalObservation:
        _wait_until_ns(anchor_host_ns)
        gate_release_ns = time.perf_counter_ns()
        ready = torch.cuda.Event(enable_timing=True)
        ready.record(torch.cuda.current_stream(self.device))
        launch_a: CudaLaunch | None = None
        launch_b: CudaLaunch | None = None
        if mode == "isolated_a":
            launch_a = self._launch_cuda(self.buffer_a, self.sequence_a, self.stream_a, ready)
        elif mode == "isolated_b":
            launch_b = self._launch_cuda(self.buffer_b, self.sequence_b, self.stream_b, ready)
        elif mode == "serialized":
            launch_a = self._launch_cuda(self.buffer_a, self.sequence_a, self.stream_a, ready)
            self.stream_b.wait_event(launch_a.end)
            launch_b = self._launch_cuda(self.buffer_b, self.sequence_b, self.stream_b, ready)
        elif mode == "concurrent":
            launch_a = self._launch_cuda(self.buffer_a, self.sequence_a, self.stream_a, ready)
            launch_b = self._launch_cuda(self.buffer_b, self.sequence_b, self.stream_b, ready)
        elif requested_offset_us >= 0:
            launch_a = self._launch_cuda(self.buffer_a, self.sequence_a, self.stream_a, ready)
            _wait_until_ns(anchor_host_ns + int(requested_offset_us * 1000))
            launch_b = self._launch_cuda(self.buffer_b, self.sequence_b, self.stream_b, ready)
        else:
            launch_b = self._launch_cuda(self.buffer_b, self.sequence_b, self.stream_b, ready)
            _wait_until_ns(anchor_host_ns + int(-requested_offset_us * 1000))
            launch_a = self._launch_cuda(self.buffer_a, self.sequence_a, self.stream_a, ready)

        raw = {}
        for side, launch, buffer in (("a", launch_a, self.buffer_a), ("b", launch_b, self.buffer_b)):
            if launch is not None:
                launch.end.synchronize()
                if self.trace_writer is not None:
                    raw[side] = self._raw_fields(
                        buffer,
                        launch.sequence_id,
                        str(launch.stream_id),
                        launch.api_launch_ns,
                        launch.api_return_ns,
                        time.perf_counter_ns(),
                    )
                    raw[side].update(
                        gpu_start_timestamp_ns=round(_event_timestamp_us(ready, launch.start, anchor_host_ns) * 1000),
                        gpu_end_timestamp_ns=round(_event_timestamp_us(ready, launch.end, anchor_host_ns) * 1000),
                    )
        return LocalObservation(
            a_start_us=_event_timestamp_us(ready, launch_a.start, anchor_host_ns) if launch_a else None,
            a_end_us=_event_timestamp_us(ready, launch_a.end, anchor_host_ns) if launch_a else None,
            b_start_us=_event_timestamp_us(ready, launch_b.start, anchor_host_ns) if launch_b else None,
            b_end_us=_event_timestamp_us(ready, launch_b.end, anchor_host_ns) if launch_b else None,
            a_api_launch_ns=launch_a.api_launch_ns if launch_a else None,
            b_api_launch_ns=launch_b.api_launch_ns if launch_b else None,
            launch_anchor_lateness_us=(gate_release_ns - anchor_host_ns) / 1000,
            collectives=raw,
        )

    def _run_cpu_trial(self, mode: str, requested_offset_us: float, anchor_ns: int) -> LocalObservation:
        observation = LocalObservation()
        lock = threading.Lock()
        go = threading.Event()
        start_barrier: threading.Barrier | None = None
        worker_failures: list[BaseException] = []

        def launch(which: str, delay_us: float) -> None:
            if start_barrier is not None:
                start_barrier.wait()
            go.wait()
            _wait_until_ns(anchor_ns + int(delay_us * 1000))
            buffer = self.buffer_a if which == "a" else self.buffer_b
            tracker = self.sequence_a if which == "a" else self.sequence_b
            sequence_id = tracker.record(buffer.operation)
            api_ns = time.perf_counter_ns()
            start_us = api_ns / 1000
            work = buffer.launch()
            return_ns = time.perf_counter_ns()
            work.wait()
            completion_ns = time.perf_counter_ns()
            end_us = completion_ns / 1000
            with lock:
                setattr(observation, f"{which}_api_launch_ns", api_ns)
                setattr(observation, f"{which}_start_us", start_us)
                setattr(observation, f"{which}_end_us", end_us)
                if self.trace_writer is not None:
                    observation.collectives[which] = self._raw_fields(
                        buffer, sequence_id, f"host-thread-{which}", api_ns, return_ns, completion_ns
                    )

        def guarded_launch(which: str, delay_us: float) -> None:
            try:
                launch(which, delay_us)
            except BaseException as error:
                # Exceptions in Python threads otherwise do not reach the
                # caller, which could publish a partially observed trial.
                with lock:
                    worker_failures.append(error)

        if mode in ("isolated_a", "isolated_b", "serialized"):
            go.set()
        if mode == "isolated_a":
            launch("a", 0)
        elif mode == "isolated_b":
            launch("b", 0)
        elif mode == "serialized":
            launch("a", 0)
            launch("b", (time.perf_counter_ns() - anchor_ns) / 1000)
        else:
            if mode == "concurrent":
                delay_a_us = delay_b_us = 0.0
            else:
                delay_a_us = max(0.0, -requested_offset_us)
                delay_b_us = max(0.0, requested_offset_us)
            start_barrier = threading.Barrier(3)
            threads = [
                threading.Thread(target=guarded_launch, args=("a", delay_a_us)),
                threading.Thread(target=guarded_launch, args=("b", delay_b_us)),
            ]
            for thread in threads:
                thread.start()
            start_barrier.wait()
            go.set()
            for thread in threads:
                thread.join()
            if worker_failures:
                raise RuntimeError("CPU collective worker failed") from worker_failures[0]
        starts = [value for value in (observation.a_api_launch_ns, observation.b_api_launch_ns) if value is not None]
        observation.launch_anchor_lateness_us = (min(starts) - anchor_ns) / 1000
        return observation

    def _validate(self, mode: str, observation: LocalObservation) -> None:
        if not self.validate:
            return
        correct = True
        if mode != "isolated_b":
            if "a" in observation.collectives:
                observation.collectives["a"]["consumer_timestamp_ns"] = time.perf_counter_ns()
            correct = correct and self.buffer_a.is_correct()
        if mode != "isolated_a":
            if "b" in observation.collectives:
                observation.collectives["b"]["consumer_timestamp_ns"] = time.perf_counter_ns()
            correct = correct and self.buffer_b.is_correct()
        flag = torch.tensor(int(correct), dtype=torch.int32, device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if not bool(flag.item()):
            raise RuntimeError(f"collective correctness check failed in {mode}")


def _wait_until_ns(target_ns: int) -> None:
    """Wait for an absolute monotonic deadline, then finish with a short spin."""

    spin_ns = 100_000
    while True:
        remaining_ns = target_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > spin_ns:
            _HOST_GATE.wait((remaining_ns - spin_ns) / 1e9)


def _event_timestamp_us(anchor: torch.cuda.Event, event: torch.cuda.Event, anchor_host_ns: int) -> float:
    return anchor_host_ns / 1000 + anchor.elapsed_time(event) * 1000


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _rounded(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def summarize_observations(iterations: list[list[dict[str, Any]]]) -> dict[str, float | None]:
    """Summarize gathered rank timelines into end-to-end distributions."""

    comm_a_ms: list[float] = []
    comm_b_ms: list[float] = []
    pair_completion_ms: list[float] = []
    actual_overlap_ms: list[float] = []
    realized_offsets_us: list[float] = []
    api_offsets_us: list[float] = []
    rank_start_skews_us: list[float] = []
    rank_finish_skews_us: list[float] = []
    launch_anchor_lateness_us: list[float] = []
    for rank_observations in iterations:
        a_durations = [
            row["a_end_us"] - row["a_start_us"] for row in rank_observations if row["a_start_us"] is not None
        ]
        b_durations = [
            row["b_end_us"] - row["b_start_us"] for row in rank_observations if row["b_start_us"] is not None
        ]
        if a_durations:
            comm_a_ms.append(max(a_durations) / 1000)
        if b_durations:
            comm_b_ms.append(max(b_durations) / 1000)

        pair_rows = [
            row for row in rank_observations if row["a_start_us"] is not None and row["b_start_us"] is not None
        ]
        if pair_rows:
            starts = [min(row["a_start_us"], row["b_start_us"]) for row in pair_rows]
            finishes = [max(row["a_end_us"], row["b_end_us"]) for row in pair_rows]
            pair_completion_ms.append((max(finishes) - min(starts)) / 1000)
            overlaps = [
                max(0.0, min(row["a_end_us"], row["b_end_us"]) - max(row["a_start_us"], row["b_start_us"]))
                for row in pair_rows
            ]
            actual_overlap_ms.append(sum(overlaps) / len(overlaps) / 1000)
            realized_offsets_us.extend(row["b_start_us"] - row["a_start_us"] for row in pair_rows)
            api_offsets_us.extend((row["b_api_launch_ns"] - row["a_api_launch_ns"]) / 1000 for row in pair_rows)

        start_skews = []
        finish_skews = []
        for prefix in ("a", "b"):
            starts = [row[f"{prefix}_start_us"] for row in rank_observations if row[f"{prefix}_start_us"] is not None]
            finishes = [row[f"{prefix}_end_us"] for row in rank_observations if row[f"{prefix}_end_us"] is not None]
            if starts:
                start_skews.append(max(starts) - min(starts))
                finish_skews.append(max(finishes) - min(finishes))
        if start_skews:
            rank_start_skews_us.append(max(start_skews))
            rank_finish_skews_us.append(max(finish_skews))
        launch_anchor_lateness_us.extend(row["launch_anchor_lateness_us"] for row in rank_observations)

    result: dict[str, float | None] = {}
    metrics = {
        "comm_a_ms": comm_a_ms,
        "comm_b_ms": comm_b_ms,
        "pair_completion_ms": pair_completion_ms,
        "actual_overlap_ms": actual_overlap_ms,
        "realized_gpu_offset_us": realized_offsets_us,
        "api_launch_offset_us": api_offsets_us,
        "rank_start_skew_us": rank_start_skews_us,
        "rank_finish_skew_us": rank_finish_skews_us,
        "launch_anchor_lateness_us": launch_anchor_lateness_us,
    }
    for name, values in metrics.items():
        for percent in (50, 95, 99):
            result[f"{name}_p{percent}"] = _rounded(percentile(values, percent))
    pair_p50 = result["pair_completion_ms_p50"]
    result["pairs_per_second"] = _rounded(1000 / pair_p50) if pair_p50 else None
    return result


def _gather_observation(observation: LocalObservation, rank: int, world_size: int) -> list[dict[str, Any]] | None:
    destination = [None] * world_size if rank == 0 else None
    dist.gather_object(dataclasses.asdict(observation), destination, dst=0)
    return destination


def _run_series(
    runner: BenchmarkRunner,
    mode: str,
    offset_us: float,
    warmup: int,
    iterations: int,
    rank: int,
    world_size: int,
) -> dict[str, float | None] | None:
    for _ in range(warmup):
        runner.run_trial(mode, offset_us, phase="warmup")
    gathered_iterations = []
    for _ in range(iterations):
        observation = runner.run_trial(mode, offset_us)
        gathered = _gather_observation(observation, rank, world_size)
        if rank == 0:
            gathered_iterations.append(gathered)
    return summarize_observations(gathered_iterations) if rank == 0 else None


def _sequence_status(tracker: SequenceTracker, group: dist.ProcessGroup) -> dict[str, Any]:
    local = {"count": tracker.count, "digest": tracker.digest}
    gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, local, group=group)
    consistent = all(item == local for item in gathered)
    return {"consistent": consistent, **local}


def _topology_has_nvlink(topo_matrix: str | None) -> bool:
    """Detect an actual NVLink matrix entry, not the ``NV#`` legend text."""

    return bool(topo_matrix and re.search(r"\bNV\d+\b", topo_matrix))


def _describe_topology(device: torch.device, rank: int, world_size: int) -> dict[str, Any] | None:
    local = {
        "rank": rank,
        "hostname": socket.gethostname(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "cpu",
    }
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if rank != 0:
        return None
    topo_matrix = None
    if device.type == "cuda":
        try:
            topo_matrix = subprocess.run(
                ["nvidia-smi", "topo", "-m"], check=True, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
    hosts = {item["hostname"] for item in gathered}
    if len(hosts) > 1:
        topology_class = "multi-node"
    elif _topology_has_nvlink(topo_matrix):
        topology_class = "single-node-nvlink"
    elif device.type == "cuda":
        topology_class = "single-node-pcie"
    else:
        topology_class = "single-node-cpu"
    return {"topology_class": topology_class, "ranks": gathered, "nvidia_smi_topology": topo_matrix}


def _require_single_node(topology: dict[str, Any] | None, rank: int) -> None:
    """Reject a clock domain that cannot support the benchmark's absolute gate."""

    supported = [topology["topology_class"] != "multi-node" if rank == 0 else None]
    dist.broadcast_object_list(supported, src=0)
    if not supported[0]:
        raise RuntimeError(
            "collective phase sweep currently requires one node because rank-0 "
            "perf_counter_ns anchors are not portable across host clock domains"
        )


def _record_for_policy(
    policy: str,
    offset_us: float | None,
    summary: dict[str, float | None],
    isolated_a_ms: float | None,
    isolated_b_ms: float | None,
    concurrent: dict[str, float | None] | None,
) -> dict[str, Any]:
    comm_a_ms = summary.get("comm_a_ms_p50")
    comm_b_ms = summary.get("comm_b_ms_p50")
    record: dict[str, Any] = {
        "policy": policy,
        "requested_offset_us": offset_us,
        **{name: summary.get(name) for name in SUMMARY_METRIC_NAMES},
    }
    record.update(
        {
            "comm_a_isolated_ms": isolated_a_ms,
            "comm_b_isolated_ms": isolated_b_ms,
            "comm_a_contended_ms": concurrent.get("comm_a_ms_p50") if concurrent else None,
            "comm_b_contended_ms": concurrent.get("comm_b_ms_p50") if concurrent else None,
            "comm_a_offset_ms": comm_a_ms if policy == "offset" else None,
            "comm_b_offset_ms": comm_b_ms if policy == "offset" else None,
            "pair_completion_ms": summary.get("pair_completion_ms_p50"),
            "actual_overlap_ms": summary.get("actual_overlap_ms_p50"),
            "stretch_a": _rounded(comm_a_ms / isolated_a_ms) if comm_a_ms and isolated_a_ms else None,
            "stretch_b": _rounded(comm_b_ms / isolated_b_ms) if comm_b_ms and isolated_b_ms else None,
        }
    )
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comm-a", type=normalize_collective, choices=COMM_A_CHOICES, default="all_to_all")
    parser.add_argument("--comm-b", type=normalize_collective, choices=COMM_B_CHOICES, default="reduce_scatter")
    parser.add_argument(
        "--group-layout",
        default="auto",
        help="auto, mesh, world, or an epN-dpM shorthand such as ep2-dp2",
    )
    parser.add_argument(
        "--mesh-shape",
        type=int,
        nargs=2,
        metavar=("ROWS", "COLUMNS"),
        help="2-D mesh shape; A uses rows and B uses columns (product must equal world size)",
    )
    parser.add_argument(
        "--offset-us", type=float, nargs="+", default=[-4000, -2000, -1000, -500, 0, 500, 1000, 2000, 4000]
    )
    parser.add_argument("--policies", nargs="+", choices=POLICY_CHOICES, default=list(POLICY_CHOICES))
    parser.add_argument("--message-bytes-a", type=parse_size, default=parse_size("128MiB"))
    parser.add_argument("--message-bytes-b", type=parse_size, default=parse_size("64MiB"))
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--backend", choices=("auto", "nccl", "gloo"), default="auto")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument(
        "--launch-anchor-lead-us",
        type=int,
        default=20_000,
        help="lead time for the rank-0-broadcast absolute launch anchor",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--trace-jsonl", help="Optional fresh rank-local raw trace path, with a {rank} placeholder")
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-offsets", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters must be positive")
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")
    if args.launch_anchor_lead_us <= 0:
        raise ValueError("--launch-anchor-lead-us must be positive")
    if any(not math.isfinite(offset) for offset in args.offset_us):
        raise ValueError("--offset-us values must be finite")
    if args.comm_a not in COMM_A_CHOICES:
        raise ValueError("--comm-a must be an all-to-all collective")
    if args.comm_b not in COMM_B_CHOICES:
        raise ValueError("--comm-b must be a DP/FSDP-style collective")
    if args.mesh_shape is not None and any(dimension < 2 for dimension in args.mesh_shape):
        raise ValueError("both dimensions of --mesh-shape must be at least two")


def _resolve_runtime(args: argparse.Namespace) -> tuple[torch.device, str, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("launch this benchmark with torchrun")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    backend = ("nccl" if use_cuda else "gloo") if args.backend == "auto" else args.backend
    if backend == "nccl" and not use_cuda:
        raise ValueError("the NCCL backend requires --device cuda")
    if backend == "gloo" and use_cuda:
        raise ValueError("use --device cpu with the Gloo backend")
    return device, backend, rank, world_size


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    device, backend, rank, world_size = _resolve_runtime(args)
    mesh_shape = tuple(args.mesh_shape) if args.mesh_shape else None
    specs_a, specs_b = build_group_specs(args.group_layout, world_size, mesh_shape)
    init_kwargs: dict[str, Any] = {
        "backend": backend,
        "timeout": datetime.timedelta(seconds=args.timeout_s),
    }
    if backend == "nccl":
        init_kwargs["device_id"] = device
    dist.init_process_group(**init_kwargs)
    groups: GroupContext | None = None
    trace_writer = None
    try:
        run_ids = [str(uuid.uuid4()) if rank == 0 else None]
        dist.broadcast_object_list(run_ids, src=0)
        run_id = run_ids[0]
        if args.trace_jsonl:
            trace_writer = create_native_trace_writer(
                args.trace_jsonl,
                rank=rank,
                world_size=world_size,
                run_id=run_id,
                backend=backend,
                summary_path=args.output_json,
            )
        groups = _create_groups(args.group_layout, world_size, rank, mesh_shape)
        dtype = DTYPES[args.dtype]
        buffer_a = CollectiveBuffer(args.comm_a, args.message_bytes_a, dtype, device, groups.group_a)
        buffer_b = CollectiveBuffer(args.comm_b, args.message_bytes_b, dtype, device, groups.group_b)
        runner = BenchmarkRunner(buffer_a, buffer_b, device, args.validate, args.launch_anchor_lead_us, trace_writer)
        topology = _describe_topology(device, rank, world_size)
        _require_single_node(topology, rank)

        if rank == 0:
            print("Measuring isolated A baseline...", file=sys.stderr, flush=True)
        isolated_a = _run_series(runner, "isolated_a", 0, args.warmup, args.iters, rank, world_size)
        if rank == 0:
            print("Measuring isolated B baseline...", file=sys.stderr, flush=True)
        isolated_b = _run_series(runner, "isolated_b", 0, args.warmup, args.iters, rank, world_size)

        concurrent = None
        if any(policy in args.policies for policy in ("concurrent", "offset")):
            if rank == 0:
                print("Measuring concurrent baseline...", file=sys.stderr, flush=True)
            concurrent = _run_series(runner, "concurrent", 0, args.warmup, args.iters, rank, world_size)

        results: list[dict[str, Any]] = []
        if rank == 0:
            isolated_a_ms = isolated_a["comm_a_ms_p50"]
            isolated_b_ms = isolated_b["comm_b_ms_p50"]
            if "isolated" in args.policies:
                isolated_summary = {
                    **{key: value for key, value in isolated_a.items() if key.startswith("comm_a_")},
                    **{key: value for key, value in isolated_b.items() if key.startswith("comm_b_")},
                }
                results.append(
                    _record_for_policy("isolated", None, isolated_summary, isolated_a_ms, isolated_b_ms, concurrent)
                )
            if "concurrent" in args.policies:
                results.append(
                    _record_for_policy("concurrent", 0, concurrent, isolated_a_ms, isolated_b_ms, concurrent)
                )
        else:
            isolated_a_ms = isolated_b_ms = None

        if "serialized" in args.policies:
            if rank == 0:
                print("Measuring serialized policy...", file=sys.stderr, flush=True)
            serialized = _run_series(runner, "serialized", 0, args.warmup, args.iters, rank, world_size)
            if rank == 0:
                results.append(
                    _record_for_policy("serialized", 0, serialized, isolated_a_ms, isolated_b_ms, concurrent)
                )

        offsets = list(dict.fromkeys(args.offset_us))
        if args.shuffle_offsets:
            random.Random(args.seed).shuffle(offsets)
        if "offset" in args.policies:
            offset_results = []
            for offset_us in offsets:
                if rank == 0:
                    print(f"Measuring offset policy at {offset_us:g} us...", file=sys.stderr, flush=True)
                summary = _run_series(runner, "offset", offset_us, args.warmup, args.iters, rank, world_size)
                if rank == 0:
                    offset_results.append(
                        _record_for_policy("offset", offset_us, summary, isolated_a_ms, isolated_b_ms, concurrent)
                    )
            if rank == 0:
                results.extend(sorted(offset_results, key=lambda item: item["requested_offset_us"]))

        sequence_a = _sequence_status(runner.sequence_a, groups.group_a)
        sequence_b = _sequence_status(runner.sequence_b, groups.group_b)
        local_sequence = {
            "group_id_a": groups.group_id_a,
            "group_id_b": groups.group_id_b,
            "a": sequence_a,
            "b": sequence_b,
        }
        all_sequences = [None] * world_size if rank == 0 else None
        dist.gather_object(local_sequence, all_sequences, dst=0)
        if not sequence_a["consistent"] or not sequence_b["consistent"]:
            raise RuntimeError("communicator logical sequence diverged across ranks")

        if rank == 0:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "raw_trace_schema_version": 3 if trace_writer else None,
                "raw_trace_enabled": trace_writer is not None,
                "framework": "verl",
                "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "world_size": world_size,
                "backend": backend,
                "device_type": device.type,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "nccl_version": torch.cuda.nccl.version() if device.type == "cuda" else None,
                "comm_a": args.comm_a,
                "comm_b": args.comm_b,
                "group_layout": groups.resolved_layout,
                "requested_group_layout": args.group_layout,
                "process_group_id_a_by_rank": [item["group_id_a"] for item in all_sequences],
                "process_group_id_b_by_rank": [item["group_id_b"] for item in all_sequences],
                "process_groups": {
                    "a": [dataclasses.asdict(spec) for spec in specs_a],
                    "b": [dataclasses.asdict(spec) for spec in specs_b],
                },
                "requested_message_bytes_a": args.message_bytes_a,
                "requested_message_bytes_b": args.message_bytes_b,
                "message_bytes_a": buffer_a.message_bytes,
                "message_bytes_b": buffer_b.message_bytes,
                "dtype": args.dtype,
                "warmup": args.warmup,
                "iters": args.iters,
                "seed": args.seed,
                "launch_anchor_lead_us": args.launch_anchor_lead_us,
                "offset_execution_order_us": offsets if "offset" in args.policies else [],
                "timestamp_domain": (
                    "single-node-perf-counter-projected-cuda-event"
                    if device.type == "cuda"
                    else "single-node-perf-counter"
                ),
                "gpu_timestamp_semantics": "event-bracket",
                "kernel_observed": False,
                "timing_sources": {
                    "api_launch_offset": "host_perf_counter",
                    "realized_gpu_offset": (
                        "cuda_event_bracket_start" if device.type == "cuda" else "host_call_bracket_start"
                    ),
                    "rank_skew": (
                        "single_node_host_anchored_cuda_event_bracket"
                        if device.type == "cuda"
                        else "single_node_host_perf_counter"
                    ),
                },
                "topology": topology,
                "sequence_validation": {
                    "all_groups_consistent": all(
                        item[side]["consistent"] for item in all_sequences for side in ("a", "b")
                    ),
                    "by_rank": all_sequences,
                },
                "baselines": {"isolated_a": isolated_a, "isolated_b": isolated_b, "concurrent": concurrent},
                "results": results,
            }
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(json.dumps(payload, sort_keys=True))
        dist.barrier()
    finally:
        if trace_writer is not None:
            trace_writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
