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
"""Opt-in execution of an approved collective plan on configured NIC lanes.

The caller supplies validated NetworkPolicyEligibility and current telemetry
objects from the offline policy tooling. Runtime code does not import scripts.
Rail isolation is provided by operator-installed, explicitly named NCCL network
plugins. Stock IB has no per-communicator HCA filter; changing NCCL_IB_HCA around
new_group is deliberately unsupported. Network configuration is never edited.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class NetworkLane:
    """Operator mapping of a measured logical rail/class to a NCCL plugin."""

    rank: int
    rail_id: str
    traffic_class: str
    net_name: str
    traffic_class_value: int


@dataclass(frozen=True)
class NetworkOperation:
    """One collective slot in the repeated, immutable per-step launch sequence."""

    operation: str
    kind: str
    message_bytes: int
    dtype: str
    source_rank: int = 0


def network_plan_digest(
    eligibility: Any, lanes: tuple[NetworkLane, ...], operations: tuple[NetworkOperation, ...], evidence_sha256: str
) -> str:
    """Address every runtime decision and the reviewed source artifact by content."""
    payload = {
        "executor_abi": 1,
        "eligibility": eligibility.to_dict(),
        "lanes": [
            asdict(lane) for lane in sorted(lanes, key=lambda lane: (lane.rank, lane.rail_id, lane.traffic_class))
        ],
        "operations": [asdict(operation) for operation in operations],
        "evidence_sha256": evidence_sha256,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _check_environment() -> None:
    overrides = {"NCCL_NET", "NCCL_IB_TC", "NCCL_IB_SL", "NCCL_IB_HCA"}
    if overrides.intersection(os.environ):
        raise ValueError("NCCL environment overrides conflict with communicator lane settings")
    paths = {Path("/etc/nccl.conf"), Path.home() / ".nccl.conf"}
    if "NCCL_CONF_FILE" in os.environ:
        paths.add(Path(os.environ["NCCL_CONF_FILE"]))
    for path in paths:
        if path.is_file():
            for line in path.read_text().splitlines():
                if line.strip().split("=", 1)[0].strip() in overrides:
                    raise ValueError(f"NCCL configuration file overrides communicator lanes: {path}")


class NetworkCollectiveWork:
    """Own transfer buffers and establish completion on each consuming CUDA stream."""

    def __init__(self, work: Any, result: torch.Tensor, owned: tuple[torch.Tensor, ...]):
        self.work, self.result, self.owned = work, result, owned
        self._event = None
        self._complete = False
        self._error: BaseException | None = None
        self._lock = threading.Lock()

    def wait(self) -> torch.Tensor:
        """Wait once, then fence any subsequent CUDA consumer using an event."""
        with self._lock:
            return self._wait()

    def _wait(self) -> torch.Tensor:
        if self._error is not None:
            raise self._error
        if not self._complete:
            try:
                self.work.wait()
                if self.result.is_cuda:
                    self._event = torch.cuda.Event()
                    self._event.record(torch.cuda.current_stream(self.result.device))
                self._complete = True
            except BaseException as exc:
                self._error = exc
                raise
        elif self._event is not None:
            torch.cuda.current_stream(self.result.device).wait_event(self._event)
        if self.result.is_cuda:
            for tensor in self.owned:
                tensor.record_stream(torch.cuda.current_stream(tensor.device))
        return self.result

    def synchronize(self) -> None:
        """Observe physical completion before releasing buffers or publishing state."""
        with self._lock:
            self._wait()
            if self._event is not None:
                self._event.synchronize()


class NetworkCollectiveExecutor:
    """Create per-operation NCCL lanes and run audited broadcast/AR/A2A slots.

    All global ranks call construction, begin_step, launch and finish_step in
    the same order. The separate Gloo control group must contain the world.
    Every validation error is exchanged before a data collective is launched.
    Use at a quiescent, post-autograd transfer boundary, from one host thread.
    A telemetry observer must inspect each initialized communicator's actual
    lane (including plugin and numeric class); returning requested settings is
    not network evidence. This executor does not promise cross-lane overlap.
    """

    def __init__(
        self,
        *,
        eligibility: Any,
        target_telemetry: Any,
        lanes: tuple[NetworkLane, ...],
        operations: tuple[NetworkOperation, ...],
        evidence_sha256: str,
        approved_digest: str,
        control_group: Any,
        observe_binding: Callable[[Any, str], NetworkLane],
        timeout: timedelta = timedelta(seconds=120),
    ) -> None:
        self.control_group = control_group
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.groups: dict[str, Any] = {}
        self.operations = tuple(operations)
        self._step = None
        self._active = False
        self._index = 0
        self._works: list[NetworkCollectiveWork] = []
        self._failed = False
        self._closed = False
        error = None
        digest = ""
        self._assignments = {}
        self._lanes = {}
        try:
            eligibility.validate()
            target_telemetry.validate()
            if not eligibility.telemetry.compare(target_telemetry).compatible:
                raise ValueError("network telemetry does not match the reviewed policy cell")
            if eligibility.telemetry.capability.topology.world_size != self.world_size:
                raise ValueError("network policy world size differs from runtime")
            if dist.get_backend(control_group) != "gloo" or dist.get_process_group_ranks(control_group) != list(
                range(self.world_size)
            ):
                raise ValueError("network executor requires a full-world Gloo control group")
            if len(evidence_sha256) != 64 or any(c not in "0123456789abcdef" for c in evidence_sha256):
                raise ValueError("source evidence must have a SHA-256 content digest")
            if not operations or len({op.operation for op in operations}) != len(operations):
                raise ValueError("operation slots must be nonempty and uniquely named")
            for op in operations:
                if op.kind not in {"broadcast", "all_reduce", "all_to_all_single"}:
                    raise ValueError("unsupported network collective kind")
                if type(op.message_bytes) is not int or op.message_bytes <= 0:
                    raise ValueError("network collective requires positive message bytes")
                if type(op.source_rank) is not int or not 0 <= op.source_rank < self.world_size:
                    raise ValueError("invalid collective source rank")
                if not op.operation or not op.dtype:
                    raise ValueError("operation and dtype must be explicit")
            for lane in lanes:
                key = (lane.rank, lane.rail_id, lane.traffic_class)
                if key in self._lanes:
                    raise ValueError("duplicate network lane")
                if not eligibility.telemetry.capability.supports(*key):
                    raise ValueError("lane is absent from the measured capability inventory")
                max_class = 15 if eligibility.telemetry.capability.network_fabric == "infiniband" else 255
                if type(lane.traffic_class_value) is not int or not 0 <= lane.traffic_class_value <= max_class:
                    raise ValueError(f"traffic class must be an explicit value in [0,{max_class}]")
                if not lane.net_name or lane.net_name.lower() in {"ib", "socket", "auto"}:
                    raise ValueError("rail pinning requires an explicitly named rail-specific network plugin")
                self._lanes[key] = lane
            plugin_rails = {}
            for lane in lanes:
                key = (lane.rank, lane.net_name.lower())
                if plugin_rails.setdefault(key, lane.rail_id) != lane.rail_id:
                    raise ValueError("distinct rails cannot alias the same network plugin")
            for assignment in eligibility.required_assignments:
                key = (assignment.rank, assignment.rail_id, assignment.traffic_class)
                if key not in self._lanes:
                    raise ValueError("a measured assignment has no runtime lane binding")
                self._assignments[(assignment.operation, assignment.rank)] = key
            if {op.operation for op in operations} != {key[0] for key in self._assignments}:
                raise ValueError("runtime operations differ from reviewed assignment coverage")
            digest = network_plan_digest(eligibility, lanes, operations, evidence_sha256)
            if digest != approved_digest:
                raise ValueError("runtime plan does not match the operator-approved digest")
            _check_environment()
            options = dist.ProcessGroupNCCL.Options()
            if not all(hasattr(options.config, field) for field in ("net_name", "traffic_class")):
                raise ValueError("this PyTorch/NCCL build cannot configure communicator traffic classes")
        except (ValueError, TypeError, AttributeError, OSError) as exc:
            error = str(exc)
        startup = self._agreement((digest, socket.gethostname()), error, require_same=False)
        if len({item[0] for item in startup}) != 1:
            raise ValueError("ranks supplied different network execution plans")
        observed_nodes: dict[str, list[int]] = {}
        for rank, (_, hostname) in enumerate(startup):
            observed_nodes.setdefault(hostname, []).append(rank)
        topology = target_telemetry.capability.topology
        if tuple(sorted(map(tuple, observed_nodes.values()))) != topology.rank_groups:
            raise ValueError("runtime rank placement differs from measured multi-node topology")
        self.digest = digest
        # All ranks build the same operation communicators in the same order.
        try:
            for op in operations:
                lane = self._lanes[self._assignments[(op.operation, self.rank)]]
                options = dist.ProcessGroupNCCL.Options()
                options.config.net_name = lane.net_name
                options.config.traffic_class = lane.traffic_class_value
                group = dist.new_group(
                    ranks=list(range(self.world_size)), backend="nccl", pg_options=options, timeout=timeout
                )
                self.groups[op.operation] = group
                probe = torch.zeros(1, dtype=torch.uint8, device="cuda")
                dist.all_reduce(probe, group=group)
                torch.cuda.synchronize()
                error = None
                try:
                    if observe_binding(group, op.operation) != lane:
                        raise ValueError("observed communicator rail/class differs from approved assignment")
                except Exception as exc:
                    error = str(exc)
                self._agreement((self.digest, op.operation), error)
        except BaseException:
            self._failed = True
            self.close()
            raise

    def _agreement(self, value: Any, error: str | None, *, require_same: bool = True) -> list[Any]:
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, (value, error), group=self.control_group)
        errors = [(rank, message) for rank, (_, message) in enumerate(gathered) if message is not None]
        if errors or (require_same and any(item[0] != gathered[0][0] for item in gathered)):
            self._failed = True
            raise ValueError(f"network execution preflight failed: {errors or 'rank sequence/digest mismatch'}")
        return [item[0] for item in gathered]

    def begin_step(self, step: int) -> None:
        """Open the next strictly increasing step after the preceding step drains."""
        error = None
        if self._failed or self._closed or self._active:
            error = "executor is failed, closed, or has an incomplete preceding step"
        if type(step) is not int or step < 0 or (self._step is not None and step <= self._step):
            error = "step must increase monotonically"
        self._agreement((self.digest, "begin", step), error)
        self._step, self._index = step, 0
        self._active = True

    def launch(self, operation: str, tensor: torch.Tensor) -> NetworkCollectiveWork:
        """Launch the next collective on its configured lane after all-rank preflight."""
        error = None
        spec = self.operations[min(self._index, len(self.operations) - 1)]
        if self._failed or self._closed or not self._active or self._index >= len(self.operations):
            error = "executor is not ready for another collective"
        elif operation != spec.operation:
            error = "collective does not match the canonical operation sequence"
        elif not isinstance(tensor, torch.Tensor):
            error = "runtime transfer requires a tensor"
        elif not tensor.is_cuda or not tensor.is_contiguous() or tensor.requires_grad:
            error = "runtime transfer requires a contiguous detached CUDA tensor"
        elif tensor.numel() * tensor.element_size() != spec.message_bytes or str(tensor.dtype) != spec.dtype:
            error = "tensor differs from reviewed message-size/dtype cell"
        elif spec.kind == "all_to_all_single" and (tensor.ndim == 0 or tensor.shape[0] % self.world_size):
            error = "equal-split all-to-all requires a divisible first dimension"
        shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else None
        self._agreement((self.digest, self._step, self._index, operation, shape), error)
        # Follow ProcessGroupNCCL's explicit cross-communicator wait contract.
        # The dependency is placed on this launch stream, with no host timer.
        group = self.groups[operation]
        result = tensor
        try:
            if self._works:
                self._works[-1].wait()
            if spec.kind == "broadcast":
                work = dist.broadcast(tensor, src=spec.source_rank, group=group, async_op=True)
            elif spec.kind == "all_reduce":
                work = dist.all_reduce(tensor, group=group, async_op=True)
            else:
                result = torch.empty_like(tensor)
                work = dist.all_to_all_single(result, tensor, group=group, async_op=True)
        except BaseException:
            self._failed = True
            raise
        handle = NetworkCollectiveWork(work, result, (tensor, result))
        self._works.append(handle)
        self._index += 1
        return handle

    def finish_step(self) -> None:
        """Fence every transfer before the caller publishes the new state."""
        error = None
        if self._failed or self._closed or not self._active or self._index != len(self.operations):
            error = "executor is not active or step omitted planned collectives"
        self._agreement((self.digest, "finish", self._step, self._index), error)
        try:
            for work in self._works:
                work.synchronize()
        except BaseException:
            self._failed = True
            raise
        self._works.clear()
        self._active = False

    def close(self) -> None:
        """Drain owned work and destroy only this executor's process groups."""
        if self._closed:
            return
        for work in self._works:
            work.synchronize()
        for group in reversed(list(self.groups.values())):
            dist.destroy_process_group(group)
        self._works.clear()
        self.groups.clear()
        self._closed = True
