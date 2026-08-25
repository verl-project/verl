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

"""Bounded-memory pipelined tensor storage for node-local disks."""

from __future__ import annotations

import atexit
import errno
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch

from verl.utils.device import get_device_name, get_torch_device, is_device_available

logger = logging.getLogger(__name__)

_COMPONENTS = frozenset({"param", "grad", "optimizer"})
_DIRECTIONS = ("offload", "onload")
_ALIGNMENT = 4096
_METRIC_DECIMAL_PLACES = 4


def _safe_segment(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "unknown"


def _runtime_job_id() -> str:
    for name in ("RAY_JOB_ID", "SLURM_JOB_ID", "JOB_ID"):
        value = os.environ.get(name)
        if value:
            return _safe_segment(value)
    return f"pid-{os.getpid()}"


def _align(value: int, alignment: int = _ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class TensorDiskMetadata:
    key: str
    offset: int
    nbytes: int
    numel: int
    dtype: str
    shape: tuple[int, ...]
    device_type: str

    @classmethod
    def from_dict(cls, value: dict) -> TensorDiskMetadata:
        value = dict(value)
        value["shape"] = tuple(value["shape"])
        # Version-1 scratch manifests written before device-aware optimizer
        # offload contained accelerator tensors only.
        value.setdefault("device_type", get_device_name())
        return cls(**value)


@dataclass
class DiskOffloadIOStats:
    """Accumulated wall time and payload bytes for one disk I/O operation type."""

    seconds: float = 0.0
    nbytes: int = 0


@dataclass
class _StagingSlot:
    tensor: torch.Tensor
    buffer: memoryview
    io_future: Future[None] | None = None
    copy_event: object | None = None


def aggregate_disk_offload_metrics(
    stats: dict[tuple[str, str], DiskOffloadIOStats], device: str | torch.device
) -> dict[str, float]:
    """Aggregate rank-local disk stats into phase-critical distributed metrics."""

    shape = (len(_DIRECTIONS), len(_COMPONENTS))
    seconds = torch.zeros(shape, dtype=torch.float32, device=device)
    nbytes = torch.zeros(shape, dtype=torch.int64, device=device)
    components = sorted(_COMPONENTS)
    for direction_index, direction in enumerate(_DIRECTIONS):
        for component_index, component in enumerate(components):
            current = stats.get((direction, component))
            if current is not None:
                seconds[direction_index, component_index] = current.seconds
                nbytes[direction_index, component_index] = current.nbytes

    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        torch.distributed.all_reduce(seconds, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(nbytes, op=torch.distributed.ReduceOp.SUM)

    metrics = {}
    for direction_index, direction in enumerate(_DIRECTIONS):
        for component_index, component in enumerate(components):
            component_nbytes = nbytes[direction_index, component_index].item()
            if component_nbytes <= 0:
                continue
            component_seconds = seconds[direction_index, component_index].item()
            component_gib = component_nbytes / (1 << 30)
            metrics[f"disk_{direction}_s/{component}"] = round(component_seconds, _METRIC_DECIMAL_PLACES)
            metrics[f"disk_{direction}_gib/{component}"] = round(component_gib, _METRIC_DECIMAL_PLACES)
            if component_seconds > 0:
                metrics[f"disk_{direction}_gib_s/{component}"] = round(
                    component_gib / component_seconds, _METRIC_DECIMAL_PLACES
                )
    return metrics


class DiskOffloadStore:
    """Store component tensors in one reusable flat file per rank.

    Calls remain synchronous at the API boundary, while two fixed-size CPU
    staging tensors pipeline accelerator copies with file I/O.  Disk offload
    therefore does not create a full host-memory replica.  A generation marker
    is removed before overwriting data and recreated only after the manifest is
    committed.  Callers must release accelerator storage only after
    :meth:`write_tensors` returns successfully.
    """

    def __init__(
        self,
        path: str,
        *,
        rank: int,
        chunk_size_mb: int = 64,
        cleanup_on_exit: bool = True,
        job_id: str | None = None,
    ) -> None:
        if not path:
            raise ValueError("disk offload path must not be empty")
        if chunk_size_mb <= 0:
            raise ValueError("disk offload chunk_size_mb must be positive")

        self.chunk_size = chunk_size_mb << 20
        self.cleanup_on_exit = cleanup_on_exit
        self._lock = threading.RLock()
        self._closed = False
        self._staging_slots: list[_StagingSlot] | None = None
        self._copy_stream = None
        self._io_executor: ThreadPoolExecutor | None = None
        self._io_stats: dict[tuple[str, str], DiskOffloadIOStats] = {}
        self._owner_token = uuid.uuid4().hex

        job_segment = _safe_segment(job_id or _runtime_job_id())
        self.root = (
            Path(path).expanduser().resolve() / f"job_{job_segment}" / f"rank_{rank:06d}" / f"store_{self._owner_token}"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self._owner_path = self.root / ".owner"
        self._owner_path.write_text(self._owner_token, encoding="utf-8")
        if cleanup_on_exit:
            atexit.register(self.close)

    def _paths(self, component: str) -> tuple[Path, Path, Path]:
        if component not in _COMPONENTS:
            raise ValueError(f"Unknown offload component: {component!r}")
        component_dir = self.root / component
        component_dir.mkdir(parents=True, exist_ok=True)
        return component_dir / "state.bin", component_dir / "manifest.json", component_dir / "generation"

    def _get_staging_slots(self) -> list[_StagingSlot]:
        if self._staging_slots is not None:
            return self._staging_slots
        pin_memory = is_device_available()
        try:
            tensors = [
                torch.empty(self.chunk_size, dtype=torch.uint8, device="cpu", pin_memory=pin_memory) for _ in range(2)
            ]
        except RuntimeError:
            logger.warning("Pinned staging allocation failed; falling back to pageable host memory")
            tensors = [torch.empty(self.chunk_size, dtype=torch.uint8, device="cpu") for _ in range(2)]
        self._staging_slots = [_StagingSlot(tensor=tensor, buffer=memoryview(tensor.numpy())) for tensor in tensors]
        return self._staging_slots

    def _get_io_executor(self) -> ThreadPoolExecutor:
        if self._io_executor is None:
            self._io_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="verl-disk-offload")
        return self._io_executor

    def _get_copy_stream(self):
        if self._copy_stream is None:
            self._copy_stream = get_torch_device().Stream()
        return self._copy_stream

    @staticmethod
    def _tensor_nbytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    @staticmethod
    def _byte_view(tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.is_contiguous():
            raise ValueError("disk offload requires contiguous tensors")
        return tensor.detach().view(torch.uint8).reshape(-1)

    @staticmethod
    def _load_manifest(path: Path) -> tuple[int, dict[str, TensorDiskMetadata]]:
        if not path.exists():
            return 0, {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = {key: TensorDiskMetadata.from_dict(value) for key, value in payload["entries"].items()}
        return int(payload["generation"]), entries

    @staticmethod
    def _reserve(fd: int, length: int) -> None:
        if length <= 0:
            return
        try:
            os.posix_fallocate(fd, 0, length)
        except AttributeError:
            os.ftruncate(fd, length)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
                raise
            os.ftruncate(fd, length)

    @staticmethod
    def _pwrite_all(fd: int, buffer: memoryview, file_offset: int, size: int) -> None:
        written = 0
        while written < size:
            view = buffer[written:size]
            count = (
                os.pwritev(fd, [view], file_offset + written)
                if hasattr(os, "pwritev")
                else os.pwrite(fd, view, file_offset + written)
            )
            if count <= 0:
                raise OSError("short write while offloading tensor")
            written += count

    @staticmethod
    def _pread_all(fd: int, buffer: memoryview, file_offset: int, size: int, key: str) -> None:
        read = 0
        while read < size:
            view = buffer[read:size]
            if hasattr(os, "preadv"):
                count = os.preadv(fd, [view], file_offset + read)
            else:
                data = os.pread(fd, size - read, file_offset + read)
                count = len(data)
                view[:count] = data
            if count <= 0:
                raise OSError(f"short read for {key!r}: expected {size} bytes, received {read}")
            read += count

    def _iter_write_chunks(
        self, tensors: Iterable[tuple[torch.Tensor, int]]
    ) -> Iterator[tuple[torch.Tensor, int, int]]:
        for tensor, file_offset in tensors:
            source = self._byte_view(tensor)
            tensor_offset = 0
            while tensor_offset < source.numel():
                size = min(self.chunk_size, source.numel() - tensor_offset)
                yield source[tensor_offset : tensor_offset + size], file_offset + tensor_offset, size
                tensor_offset += size

    def _iter_read_chunks(
        self, tensors: Iterable[tuple[torch.Tensor, TensorDiskMetadata]]
    ) -> Iterator[tuple[torch.Tensor, int, int, str]]:
        for tensor, metadata in tensors:
            target = self._byte_view(tensor)
            tensor_offset = 0
            while tensor_offset < metadata.nbytes:
                size = min(self.chunk_size, metadata.nbytes - tensor_offset)
                yield (
                    target[tensor_offset : tensor_offset + size],
                    metadata.offset + tensor_offset,
                    size,
                    metadata.key,
                )
                tensor_offset += size

    @classmethod
    def _write_after_copy(
        cls,
        copy_event,
        fd: int,
        buffer: memoryview,
        file_offset: int,
        size: int,
    ) -> None:
        if copy_event is not None:
            copy_event.synchronize()
        cls._pwrite_all(fd, buffer, file_offset, size)

    @classmethod
    def _read_after_copy(
        cls,
        copy_event,
        fd: int,
        buffer: memoryview,
        file_offset: int,
        size: int,
        key: str,
    ) -> None:
        if copy_event is not None:
            copy_event.synchronize()
        cls._pread_all(fd, buffer, file_offset, size, key)

    @staticmethod
    def _drain_slots(slots: Iterable[_StagingSlot], *, suppress_errors: bool) -> None:
        for slot in slots:
            if slot.io_future is None:
                continue
            try:
                slot.io_future.result()
            except Exception:
                if not suppress_errors:
                    raise
            finally:
                slot.io_future = None

    def _write_tensors_pipelined(self, fd: int, tensors: Iterable[tuple[torch.Tensor, int]]) -> None:
        slots = self._get_staging_slots()
        executor = self._get_io_executor()
        copy_stream = None
        try:
            for chunk_index, (source, file_offset, size) in enumerate(self._iter_write_chunks(tensors)):
                slot = slots[chunk_index % len(slots)]
                if slot.io_future is not None:
                    slot.io_future.result()
                    slot.io_future = None

                copy_event = None
                if source.device.type == "cpu":
                    slot.tensor[:size].copy_(source, non_blocking=False)
                else:
                    if copy_stream is None:
                        copy_stream = self._get_copy_stream()
                        copy_stream.wait_stream(get_torch_device().current_stream())
                    with get_torch_device().stream(copy_stream):
                        slot.tensor[:size].copy_(source, non_blocking=True)
                        if slot.copy_event is None:
                            slot.copy_event = get_torch_device().Event()
                        slot.copy_event.record(copy_stream)
                    copy_event = slot.copy_event

                slot.io_future = executor.submit(
                    self._write_after_copy,
                    copy_event,
                    fd,
                    slot.buffer,
                    file_offset,
                    size,
                )
            self._drain_slots(slots, suppress_errors=False)
        except BaseException:
            self._drain_slots(slots, suppress_errors=True)
            if copy_stream is not None:
                copy_stream.synchronize()
            raise

    def _read_tensors_pipelined(self, fd: int, tensors: Iterable[tuple[torch.Tensor, TensorDiskMetadata]]) -> None:
        slots = self._get_staging_slots()
        executor = self._get_io_executor()
        chunks = iter(self._iter_read_chunks(tensors))
        pending = deque()
        copy_stream = None

        def submit_read(slot: _StagingSlot, chunk: tuple[torch.Tensor, int, int, str]) -> None:
            target, file_offset, size, key = chunk
            slot.io_future = executor.submit(
                self._read_after_copy,
                slot.copy_event,
                fd,
                slot.buffer,
                file_offset,
                size,
                key,
            )
            pending.append((slot, target, size))

        try:
            for slot in slots:
                try:
                    submit_read(slot, next(chunks))
                except StopIteration:
                    break

            while pending:
                slot, target, size = pending.popleft()
                assert slot.io_future is not None
                slot.io_future.result()
                slot.io_future = None

                if target.device.type == "cpu":
                    target.copy_(slot.tensor[:size], non_blocking=False)
                    slot.copy_event = None
                else:
                    if copy_stream is None:
                        copy_stream = self._get_copy_stream()
                        copy_stream.wait_stream(get_torch_device().current_stream())
                    with get_torch_device().stream(copy_stream):
                        target.copy_(slot.tensor[:size], non_blocking=True)
                        if slot.copy_event is None:
                            slot.copy_event = get_torch_device().Event()
                        slot.copy_event.record(copy_stream)

                try:
                    submit_read(slot, next(chunks))
                except StopIteration:
                    pass

            if copy_stream is not None:
                copy_stream.synchronize()
        except BaseException:
            self._drain_slots(slots, suppress_errors=True)
            if copy_stream is not None:
                copy_stream.synchronize()
            raise

    def write_tensors(self, component: str, tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Synchronously write a complete component generation."""

        started = time.perf_counter()
        tensor_list = [(key, tensor) for key, tensor in tensors if tensor.numel() > 0]
        total_nbytes = sum(self._tensor_nbytes(tensor) for _, tensor in tensor_list)
        keys = [key for key, _ in tensor_list]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate tensor keys in {component} offload generation")

        with self._lock:
            data_path, manifest_path, generation_path = self._paths(component)
            previous_generation, layout = self._load_manifest(manifest_path)
            next_offset = max((entry.offset + entry.nbytes for entry in layout.values()), default=0)

            for key, tensor in tensor_list:
                nbytes = self._tensor_nbytes(tensor)
                shape = tuple(tensor.shape)
                dtype = str(tensor.dtype)
                device_type = tensor.device.type
                existing = layout.get(key)
                if existing is not None:
                    if (existing.nbytes, existing.shape, existing.dtype) != (nbytes, shape, dtype):
                        raise ValueError(
                            f"Tensor layout changed for {key!r}: "
                            f"disk={(existing.shape, existing.dtype, existing.nbytes)}, "
                            f"current={(shape, dtype, nbytes)}"
                        )
                    if existing.device_type != device_type:
                        layout[key] = TensorDiskMetadata(
                            key=existing.key,
                            offset=existing.offset,
                            nbytes=existing.nbytes,
                            numel=existing.numel,
                            dtype=existing.dtype,
                            shape=existing.shape,
                            device_type=device_type,
                        )
                    continue
                next_offset = _align(next_offset)
                layout[key] = TensorDiskMetadata(
                    key=key,
                    offset=next_offset,
                    nbytes=nbytes,
                    numel=tensor.numel(),
                    dtype=dtype,
                    shape=shape,
                    device_type=device_type,
                )
                next_offset += nbytes

            generation = previous_generation + 1
            generation_path.unlink(missing_ok=True)
            fd = os.open(data_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                self._reserve(fd, max((entry.offset + entry.nbytes for entry in layout.values()), default=0))
                if tensor_list:
                    self._write_tensors_pipelined(fd, ((tensor, layout[key].offset) for key, tensor in tensor_list))
            finally:
                os.close(fd)

            payload = {
                "version": 1,
                "generation": generation,
                "entries": {key: asdict(value) for key, value in layout.items()},
            }
            manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
            manifest_tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            os.replace(manifest_tmp, manifest_path)

            generation_tmp = generation_path.with_name(f".{generation_path.name}.{os.getpid()}.tmp")
            generation_tmp.write_text(str(generation), encoding="utf-8")
            os.replace(generation_tmp, generation_path)

        if total_nbytes > 0:
            self._record_io("offload", component, time.perf_counter() - started, total_nbytes)

    def read_tensors(self, component: str, tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Synchronously restore tensors from the latest committed generation."""

        started = time.perf_counter()
        tensor_list = [(key, tensor) for key, tensor in tensors if tensor.numel() > 0]
        total_nbytes = sum(self._tensor_nbytes(tensor) for _, tensor in tensor_list)
        with self._lock:
            data_path, manifest_path, generation_path = self._paths(component)
            generation, layout = self._load_manifest(manifest_path)
            if not generation_path.exists() or generation_path.read_text(encoding="utf-8") != str(generation):
                raise RuntimeError(f"No committed {component} disk-offload generation under {self.root}")

            restore_entries = []
            for key, tensor in tensor_list:
                metadata = layout.get(key)
                if metadata is None:
                    raise KeyError(f"Tensor {key!r} is missing from the {component} disk manifest")
                current = (self._tensor_nbytes(tensor), tuple(tensor.shape), str(tensor.dtype))
                expected = (metadata.nbytes, metadata.shape, metadata.dtype)
                if current != expected:
                    raise ValueError(f"Restore target mismatch for {key!r}: expected {expected}, got {current}")
                restore_entries.append((tensor, metadata))

            fd = os.open(data_path, os.O_RDONLY)
            try:
                if restore_entries:
                    self._read_tensors_pipelined(fd, restore_entries)
            finally:
                os.close(fd)

        if total_nbytes > 0:
            self._record_io("onload", component, time.perf_counter() - started, total_nbytes)

    def _record_io(self, direction: str, component: str, seconds: float, nbytes: int) -> None:
        with self._lock:
            stats = self._io_stats.setdefault((direction, component), DiskOffloadIOStats())
            stats.seconds += seconds
            stats.nbytes += nbytes

    def pop_io_stats(self) -> dict[tuple[str, str], DiskOffloadIOStats]:
        """Return and clear successful, non-empty disk I/O statistics."""

        with self._lock:
            stats = self._io_stats
            self._io_stats = {}
            return stats

    def metadata(self, component: str, key: str) -> TensorDiskMetadata:
        _, manifest_path, generation_path = self._paths(component)
        generation, layout = self._load_manifest(manifest_path)
        if not generation_path.exists() or generation_path.read_text(encoding="utf-8") != str(generation):
            raise RuntimeError(f"No committed {component} disk-offload generation under {self.root}")
        try:
            return layout[key]
        except KeyError as exc:
            raise KeyError(f"Tensor {key!r} is missing from the {component} disk manifest") from exc

    def invalidate(self, component: str) -> None:
        """Invalidate a component whose live value was deliberately discarded."""

        with self._lock:
            _, _, generation_path = self._paths(component)
            generation_path.unlink(missing_ok=True)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._io_executor is not None:
                self._io_executor.shutdown(wait=True)
                self._io_executor = None
            self._staging_slots = None
            self._copy_stream = None
            if not self.cleanup_on_exit or not self._owner_path.exists():
                return
            if self._owner_path.read_text(encoding="utf-8") != self._owner_token:
                logger.warning("Refusing to clean disk offload directory with a mismatched owner marker: %s", self.root)
                return
            shutil.rmtree(self.root)
