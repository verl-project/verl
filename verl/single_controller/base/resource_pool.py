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

"""Ordered reusable rank placement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal


def normalize_pool_ranks(ranks: int | slice, world_size: int) -> tuple[int, int]:
    """Normalize an int or contiguous slice selection against world_size.

    Args:
        ranks: Zero-based rank or contiguous Python slice.
        world_size: Size of the pool or view being indexed.

    Returns:
        Inclusive-exclusive ``(start, stop)`` range in rank order.

    Raises:
        TypeError: Selection type is invalid.
        ValueError: Slice step is invalid or the normalized range is empty.
        IndexError: Integer rank is out of range.
    """
    if isinstance(ranks, bool) or not isinstance(ranks, int | slice):
        raise TypeError(f"ranks must be an int or slice, got {type(ranks)!r}")
    if isinstance(ranks, int):
        if ranks < 0 or ranks >= world_size:
            raise IndexError(f"rank {ranks} out of range for world_size={world_size}")
        return ranks, ranks + 1
    if ranks.step not in (None, 1):
        raise ValueError(f"ResourcePool.slice only supports step=1, got {ranks.step!r}")
    start, stop, _step = ranks.indices(world_size)
    if stop <= start:
        raise ValueError(f"ResourcePool.slice produced an empty range for {ranks!r} with world_size={world_size}")
    return start, stop


def rectangular_pool_view(
    ranks: int | slice,
    *,
    nnodes: int,
    processes_per_node: int,
) -> tuple[int, int, int, int]:
    """Resolve a contiguous rank selection into a rectangular node-local view.

    Returns ``(node_start, node_count, process_start, process_count)``. A view
    may select any contiguous processes on one node, or complete contiguous
    nodes. A range that cuts across node boundaries is rejected because it
    would give the selected nodes different local process counts.
    """
    start, stop = normalize_pool_ranks(ranks, nnodes * processes_per_node)
    first_node, process_start = divmod(start, processes_per_node)
    last_node, last_process = divmod(stop - 1, processes_per_node)
    if first_node == last_node:
        return first_node, 1, process_start, stop - start
    if process_start != 0 or last_process != processes_per_node - 1:
        raise ValueError(
            "ResourcePool.slice must preserve a homogeneous process count on every selected node; "
            f"range [{start}, {stop}) cuts across node boundaries"
        )
    return first_node, last_node - first_node + 1, 0, processes_per_node


class ResourcePool(ABC):
    """Represent an ordered reusable placement of process ranks."""

    def _bind_runtime_scope(self, runtime_id: str, root_name: str) -> None:
        owner = getattr(self, "_runtime_owner_id", None)
        if owner is not None and owner != runtime_id:
            raise ValueError("ResourcePool already belongs to another Runtime")
        self._runtime_owner_id = runtime_id
        self._runtime_root_name = root_name

    def _runtime_owner(self) -> str | None:
        return getattr(self, "_runtime_owner_id", None)

    def _runtime_root(self) -> str:
        return getattr(self, "_runtime_root_name", "parent")

    def _inherit_runtime_scope(self, view: ResourcePool) -> ResourcePool:
        owner = self._runtime_owner()
        if owner is not None:
            view._bind_runtime_scope(owner, self._runtime_root())
        return view

    @property
    @abstractmethod
    def nnodes(self) -> int:
        """Return the number of physical nodes in this pool or view."""

    @property
    @abstractmethod
    def world_size(self) -> int:
        """Return the number of ranks selected by this pool or view."""

    @property
    @abstractmethod
    def processes_per_node(self) -> int:
        """Return the identical process count selected on every node."""

    @property
    @abstractmethod
    def device_type(self) -> Literal["gpu", "cpu"]:
        """Return the logical device class shared by every selected rank."""

    @abstractmethod
    def slice(self, ranks: int | slice) -> ResourcePool:
        """Return a non-owning view over one rank or a contiguous rank range.

        Args:
            ranks: A zero-based rank or a contiguous Python slice to select.

        Returns:
            A rank-ordered ResourcePool view backed by the same allocation.
        """


def split_resource_pool(resource_pool: ResourcePool, split_size: int | list[int]) -> list[ResourcePool]:
    """Split an ordered pool into contiguous backend-neutral views."""
    if isinstance(split_size, int):
        if split_size <= 0 or resource_pool.world_size % split_size != 0:
            raise ValueError("split_size must be positive and divide world_size")
        sizes = [split_size] * (resource_pool.world_size // split_size)
    else:
        sizes = list(split_size)
        if any(size <= 0 for size in sizes):
            raise ValueError("split sizes must be positive")
    if sum(sizes) != resource_pool.world_size:
        raise ValueError("split sizes must sum to world_size")

    views: list[ResourcePool] = []
    start = 0
    for size in sizes:
        views.append(resource_pool.slice(slice(start, start + size)))
        start += size
    return views
