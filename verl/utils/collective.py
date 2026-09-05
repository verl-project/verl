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
"""Common primitives for asynchronous distributed collectives."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, cast

import torch
import torch.distributed as dist

T = TypeVar("T")
_UNSET = object()
_SEQUENCE_LOCK = threading.Lock()
_SEQUENCE_IDS: dict[tuple[int, str], int] = {}


class CollectiveWork(Protocol):
    """Minimum interface implemented by ``torch.distributed.Work``."""

    def wait(self) -> object: ...


def _accelerator_stream_key(device: torch.device) -> tuple[int, int]:
    """Return the current accelerator stream identity for a concrete device."""

    from verl.utils.device import get_device_id, get_device_name, get_torch_device, is_device_available

    if not is_device_available():
        raise RuntimeError("an accelerator consumer device requires an available device runtime")
    if device.type != get_device_name():
        raise RuntimeError(f"unsupported consumer device type: {device.type}")
    device_index = device.index
    if device_index is None:
        device_index = get_device_id()
    current_stream = getattr(get_torch_device(), "current_stream", None)
    if not callable(current_stream):
        raise RuntimeError("the accelerator runtime does not expose current_stream()")
    stream = current_stream(device_index)
    stream_id = getattr(stream, "stream_id", None)
    if stream_id is None:
        stream_id = getattr(stream, f"{get_device_name()}_stream", None)
    if stream_id is None:
        raise RuntimeError("the accelerator stream does not expose a stable identity")
    return device_index, int(stream_id)


def resolve_process_group_id(group: dist.ProcessGroup | None = None) -> str:
    """Return a useful process-group identifier without requiring private APIs."""

    if group is None:
        if not dist.is_initialized():
            return "world"
        group = dist.group.WORLD
    name = getattr(group, "group_name", None)
    if name is not None:
        return str(name)
    return "world" if group is dist.group.WORLD else f"group-size-{dist.get_world_size(group)}"


def next_collective_sequence_id(group: dist.ProcessGroup | None = None, process_group_id: str | None = None) -> int:
    """Allocate a rank-local sequence number for one communicator.

    Ranks that launch collectives in the same order obtain matching IDs. A
    trace or benchmark can use the IDs to detect divergent launch order.
    """

    if group is None and dist.is_initialized():
        group = dist.group.WORLD
    resolved_group_id = process_group_id if process_group_id is not None else resolve_process_group_id(group)
    key = (id(group), resolved_group_id)
    with _SEQUENCE_LOCK:
        sequence_id = _SEQUENCE_IDS.get(key, 0)
        _SEQUENCE_IDS[key] = sequence_id + 1
    return sequence_id


@dataclass(slots=True)
class AsyncCollectiveHandle(Generic[T]):
    """Own an async collective and its post-communication transformation.

    ``wait_collective`` calls ``Work.wait`` and records ``complete_event``
    before any concat/reshape finalizer runs. For CUDA work, this establishes
    ordering on one consumer stream; it does not claim CPU-visible physical
    kernel completion. The first wait binds the handle to the current stream
    of ``consumer_device``. Later waits from a different CUDA stream fail
    loudly instead of silently omitting that stream's dependency.

    Callers that need separate measurements can invoke ``wait_collective`` and
    ``finalize_result`` independently on that same stream; ``wait`` provides
    the usual combined behavior. The collective wait and finalizer each run at
    most once. A finalizer exception is cached and re-raised without repeating
    its side effects. ``owned_resources`` keeps asynchronous input, output,
    and staging objects alive for at least the handle lifetime.
    """

    work: CollectiveWork
    finalize: Callable[[], T]
    comm_kind: str
    process_group_id: str
    sequence_id: int
    launch_event: Any | None = None
    complete_event: Any | None = None
    consumer_device: torch.device | str | None = None
    owned_resources: tuple[Any, ...] = ()
    _collective_complete: bool = field(default=False, init=False, repr=False)
    _result: object = field(default=_UNSET, init=False, repr=False)
    _finalization_attempted: bool = field(default=False, init=False, repr=False)
    _finalization_error: BaseException | None = field(default=None, init=False, repr=False)
    _consumer_stream_key: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.consumer_device is not None:
            self.consumer_device = torch.device(self.consumer_device)
        if type(self.owned_resources) is not tuple:
            raise TypeError("owned_resources must be an immutable tuple")

    @property
    def collective_complete(self) -> bool:
        return self._collective_complete

    @property
    def finalized(self) -> bool:
        return self._result is not _UNSET

    @property
    def finalization_attempted(self) -> bool:
        return self._finalization_attempted

    @property
    def finalization_error(self) -> BaseException | None:
        return self._finalization_error

    def _bind_consumer_stream(self) -> None:
        device = self.consumer_device
        if device is None or device.type == "cpu":
            return
        stream_key = _accelerator_stream_key(device)
        if self._consumer_stream_key is None:
            self._consumer_stream_key = stream_key
        elif self._consumer_stream_key != stream_key:
            raise RuntimeError(
                "AsyncCollectiveHandle supports one CUDA consumer stream; "
                "wait and finalize on the stream that performed the first wait"
            )

    def wait_collective(self) -> None:
        """Establish collective completion ordering on the bound stream once."""

        with self._lock:
            self._bind_consumer_stream()
            if self._collective_complete:
                return
            record = None
            if self.complete_event is not None:
                record = getattr(self.complete_event, "record", None)
                if not callable(record):
                    raise TypeError("complete_event must provide a callable record() method")
            self.work.wait()
            if record is not None:
                record()
            self._collective_complete = True

    def finalize_result(self) -> T:
        """Wait for communication if needed, then run the finalizer once."""

        self.wait_collective()
        with self._lock:
            if self._finalization_attempted:
                if self._finalization_error is not None:
                    raise self._finalization_error
            else:
                self._finalization_attempted = True
                try:
                    self._result = self.finalize()
                except BaseException as exc:
                    self._finalization_error = exc
                    raise
            if self._result is _UNSET:
                raise RuntimeError("collective finalizer did not produce a result")
            return cast(T, self._result)

    def wait(self) -> T:
        """Wait for communication and return the finalized result."""

        return self.finalize_result()
