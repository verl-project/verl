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

"""Bounded-memory synchronous tensor storage for node-local disks."""

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

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
        value.setdefault("device_type", "cuda")
        return cls(**value)


@dataclass
class DiskOffloadIOStats:
    """Accumulated wall time and payload bytes for one disk I/O operation type."""

    seconds: float = 0.0
    nbytes: int = 0


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

    Writes are synchronous and use a fixed-size CPU staging tensor, so disk
    offload does not create a full host-memory replica.  A generation marker is
    removed before overwriting data and recreated only after the manifest is
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
        self._staging: torch.Tensor | None = None
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

    def _get_staging(self) -> torch.Tensor:
        if self._staging is not None:
            return self._staging
        pin_memory = torch.cuda.is_available()
        try:
            self._staging = torch.empty(self.chunk_size, dtype=torch.uint8, device="cpu", pin_memory=pin_memory)
        except RuntimeError:
            logger.warning("Pinned staging allocation failed; falling back to pageable host memory")
            self._staging = torch.empty(self.chunk_size, dtype=torch.uint8, device="cpu")
        return self._staging

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

    def _write_tensor(self, fd: int, tensor: torch.Tensor, file_offset: int) -> None:
        source = self._byte_view(tensor)
        staging = self._get_staging()
        staging_memory = memoryview(staging.numpy())
        tensor_offset = 0
        while tensor_offset < source.numel():
            size = min(self.chunk_size, source.numel() - tensor_offset)
            staging[:size].copy_(source[tensor_offset : tensor_offset + size], non_blocking=False)
            written = 0
            while written < size:
                count = os.pwrite(
                    fd,
                    staging_memory[written:size],
                    file_offset + tensor_offset + written,
                )
                if count <= 0:
                    raise OSError("short write while offloading tensor")
                written += count
            tensor_offset += size

    def _read_tensor(self, fd: int, tensor: torch.Tensor, metadata: TensorDiskMetadata) -> None:
        target = self._byte_view(tensor)
        staging = self._get_staging()
        tensor_offset = 0
        while tensor_offset < metadata.nbytes:
            size = min(self.chunk_size, metadata.nbytes - tensor_offset)
            data = os.pread(fd, size, metadata.offset + tensor_offset)
            if len(data) != size:
                raise OSError(f"short read for {metadata.key!r}: expected {size} bytes, received {len(data)}")
            staging[:size].copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
            target[tensor_offset : tensor_offset + size].copy_(staging[:size], non_blocking=False)
            tensor_offset += size

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
                for key, tensor in tensor_list:
                    self._write_tensor(fd, tensor, layout[key].offset)
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

            fd = os.open(data_path, os.O_RDONLY)
            try:
                for key, tensor in tensor_list:
                    metadata = layout.get(key)
                    if metadata is None:
                        raise KeyError(f"Tensor {key!r} is missing from the {component} disk manifest")
                    current = (self._tensor_nbytes(tensor), tuple(tensor.shape), str(tensor.dtype))
                    expected = (metadata.nbytes, metadata.shape, metadata.dtype)
                    if current != expected:
                        raise ValueError(f"Restore target mismatch for {key!r}: expected {expected}, got {current}")
                    self._read_tensor(fd, tensor, metadata)
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
            self._staging = None
            if not self.cleanup_on_exit or not self._owner_path.exists():
                return
            if self._owner_path.read_text(encoding="utf-8") != self._owner_token:
                logger.warning("Refusing to clean disk offload directory with a mismatched owner marker: %s", self.root)
                return
            shutil.rmtree(self.root)
