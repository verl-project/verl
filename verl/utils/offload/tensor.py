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

"""Tensor-storage adapters shared by sharded disk-offload backends.

FSDP2 and VeOmni expose parameters as ``DTensor`` objects, while FSDP1
exposes views into flat parameters.  Replacing ``tensor.data`` would destroy
the former's placements and the latter's aliasing.  These helpers therefore
persist each unique local storage once and resize that same storage to zero
after the disk generation commits.  Restoring the storage in place keeps all
Tensor/Parameter/DTensor objects and views valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .disk import DiskOffloadStore, TensorDiskMetadata

try:
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover - supported torch versions provide DTensor
    DTensor = None


def local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the rank-local tensor without changing a distributed wrapper."""

    if DTensor is not None and isinstance(tensor, DTensor):
        local = tensor.to_local()
        if local is None:
            raise ValueError("Cannot disk-offload a DTensor whose local tensor is not materialized")
        return local
    return tensor


@dataclass
class StorageOffloadRef:
    """A stable reference to one unique local storage and its full byte span."""

    key: str
    tensor: torch.Tensor
    storage: torch.UntypedStorage
    nbytes: int

    def release(self) -> None:
        current_nbytes = self.storage.nbytes()
        if current_nbytes not in (0, self.nbytes):
            raise RuntimeError(
                f"Storage size changed before releasing {self.key!r}: "
                f"expected {self.nbytes} bytes, found {current_nbytes}"
            )
        if current_nbytes:
            self.storage.resize_(0)

    def validate_restore(self, metadata: TensorDiskMetadata) -> None:
        expected = (self.nbytes, tuple(self.tensor.shape), str(self.tensor.dtype))
        actual = (metadata.nbytes, metadata.shape, metadata.dtype)
        if actual != expected:
            raise ValueError(f"Storage layout changed for {self.key!r}: expected {expected}, found {actual}")
        current_nbytes = self.storage.nbytes()
        if current_nbytes not in (0, self.nbytes):
            raise RuntimeError(
                f"Storage size changed before restoring {self.key!r}: "
                f"expected 0 or {self.nbytes} bytes, found {current_nbytes}"
            )

    def prepare_restore(self, metadata: TensorDiskMetadata) -> torch.Tensor:
        self.validate_restore(metadata)
        current_nbytes = self.storage.nbytes()
        if current_nbytes == 0:
            self.storage.resize_(self.nbytes)
        return self.tensor


def storage_offload_refs(tensors: Iterable[tuple[str, torch.Tensor]]) -> list[StorageOffloadRef]:
    """Coalesce tensors by backing storage while preserving deterministic keys.

    A flat FSDP buffer can back many parameter views.  Writing the full storage
    once preserves padding and aliasing and avoids duplicate disk traffic.
    """

    refs: list[StorageOffloadRef] = []
    seen: dict[int, StorageOffloadRef] = {}
    used_keys: set[str] = set()
    for key, wrapped in tensors:
        tensor = local_tensor(wrapped)
        if tensor.numel() == 0:
            continue

        storage = tensor.untyped_storage()
        storage_id = storage._cdata
        existing = seen.get(storage_id)
        if existing is not None:
            if existing.tensor.dtype != tensor.dtype or existing.tensor.device != tensor.device:
                raise ValueError(f"Storage {key!r} is aliased through incompatible tensor views")
            continue

        nbytes = storage.nbytes()
        if nbytes == 0:
            raise RuntimeError(f"Cannot create a disk-offload reference from released storage: {key!r}")
        if nbytes % tensor.element_size() != 0:
            raise ValueError(f"Storage size for {key!r} is not aligned to {tensor.dtype}")

        storage_key = f"{key}.__storage__"
        if storage_key in used_keys:
            raise ValueError(f"Duplicate disk-offload storage key: {storage_key!r}")
        used_keys.add(storage_key)

        full_storage_tensor = torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
            storage,
            0,
            (nbytes // tensor.element_size(),),
            (1,),
        )
        ref = StorageOffloadRef(
            key=storage_key,
            tensor=full_storage_tensor,
            storage=storage,
            nbytes=nbytes,
        )
        seen[storage_id] = ref
        refs.append(ref)
    return refs


def write_storage_refs(
    store: DiskOffloadStore,
    component: str,
    tensors: Iterable[tuple[str, torch.Tensor]],
) -> list[StorageOffloadRef]:
    """Commit a component generation, then release all referenced storages."""

    refs = storage_offload_refs(tensors)
    store.write_tensors(component, ((ref.key, ref.tensor) for ref in refs))
    release_storage_refs(store, component, refs)
    return refs


def release_storage_refs(
    store: DiskOffloadStore,
    component: str,
    refs: Iterable[StorageOffloadRef],
) -> None:
    """Release resident storages while retaining their committed disk generation."""

    refs = list(refs)
    try:
        for ref in refs:
            ref.release()
    except Exception:
        # A backend-specific storage may reject resize. Restore every ref so a
        # failed offload never leaves the live model partially released.
        read_storage_refs(store, component, refs)
        raise


def read_storage_refs(store: DiskOffloadStore, component: str, refs: Iterable[StorageOffloadRef]) -> None:
    """Resize previously released storages and restore the committed bytes."""

    resolved = [(ref, store.metadata(component, ref.key)) for ref in refs]
    # Validate the complete generation before resizing any live storage. A
    # stale/missing later entry must not leave earlier tensors half-restored.
    for ref, metadata in resolved:
        ref.validate_restore(metadata)
    targets = [(ref.key, ref.prepare_restore(metadata)) for ref, metadata in resolved]
    store.read_tensors(component, targets)
