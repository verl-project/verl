# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Bounded process-local snapshots of exact immutable TorchStore row selections."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from verl.single_controller.monarch.object_store.store import TensorRowRange, TorchStoreReference

CacheKey = tuple[str, str, tuple[int, ...], tuple[tuple[int, int], ...]]


@dataclass(frozen=True)
class _Entry:
    pieces: tuple[torch.Tensor, ...]
    nbytes: int


def _segments(selection: TensorRowRange) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, stop in selection.segments or ((selection.start, selection.stop),):
        if merged and merged[-1][1] == start:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    return tuple(merged)


def _key(reference: TorchStoreReference, selection: TensorRowRange) -> CacheKey:
    return reference.store_name, reference.key, selection.shape, _segments(selection)


class LocalCache:
    """Metadata is lock-protected; clones and native reads never hold the lock."""

    def __init__(self, capacity_bytes: int) -> None:
        if isinstance(capacity_bytes, bool) or not isinstance(capacity_bytes, int) or capacity_bytes < 0:
            raise ValueError("local_cache_bytes must be a nonnegative integer")
        self.capacity_bytes = capacity_bytes
        self._entries: OrderedDict[CacheKey, _Entry] = OrderedDict()
        self._size_bytes = 0
        self._generation = 0
        self._lock = threading.Lock()

    def generation(self) -> int:
        with self._lock:
            return self._generation

    def get(self, reference: TorchStoreReference, selection: TensorRowRange) -> torch.Tensor | None:
        key = _key(reference, selection)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
        # A live entry reference survives eviction; callers never see its tensors.
        if key[3] == ((selection.start, selection.stop),):
            return entry.pieces[0].clone()
        result = entry.pieces[0].new_zeros((selection.stop - selection.start, *selection.shape[1:]))
        for (start, stop), piece in zip(key[3], entry.pieces, strict=True):
            result[start - selection.start : stop - selection.start].copy_(piece)
        return result

    def put(self, reference: TorchStoreReference, selection: TensorRowRange, value: object, generation: int) -> None:
        if (
            type(value) is not torch.Tensor
            or value.device.type != "cpu"
            or value.layout != torch.strided
            or not value.is_contiguous()
            or tuple(value.shape) != (selection.stop - selection.start, *selection.shape[1:])
        ):
            return
        key = _key(reference, selection)
        row_elements = 1
        for dimension in selection.shape[1:]:
            row_elements *= dimension
        nbytes = sum(stop - start for start, stop in key[3]) * row_elements * value.element_size()
        if not nbytes or nbytes > self.capacity_bytes:
            return
        pieces = tuple(
            value[start - selection.start : stop - selection.start].detach().clone() for start, stop in key[3]
        )
        entry = _Entry(pieces, nbytes)
        with self._lock:
            if generation != self._generation:
                return
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._size_bytes -= previous.nbytes
            self._entries[key] = entry
            self._size_bytes += entry.nbytes
            while self._size_bytes > self.capacity_bytes:
                _, removed = self._entries.popitem(last=False)
                self._size_bytes -= removed.nbytes

    def invalidate(self, addresses: set[tuple[str, str]]) -> None:
        with self._lock:
            self._generation += 1
            for key in list(self._entries):
                if key[:2] in addresses:
                    self._size_bytes -= self._entries.pop(key).nbytes

    def clear(self) -> None:
        with self._lock:
            self._generation += 1
            self._entries.clear()
            self._size_bytes = 0
