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

"""Monarch ResourcePool backed by a Job-owned HostMesh."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from verl.single_controller.base.resource_pool import ResourcePool, normalize_pool_ranks, rectangular_pool_view

if TYPE_CHECKING:
    from monarch.actor import HostMesh


class MonarchResourcePool(ResourcePool):
    """Ordered non-owning HostMesh placement with a process demand."""

    def __init__(
        self,
        *,
        monarch_host_mesh: HostMesh,
        nnodes: int,
        processes_per_node: int,
        device_type: Literal["gpu", "cpu"],
        gpu_offset: int = 0,
        gpu_count: int | None = None,
        bound: bool = True,
    ) -> None:
        self._monarch_host_mesh = monarch_host_mesh
        self._nnodes = nnodes
        self._processes_per_node = processes_per_node
        self._device_type = device_type
        self._gpu_offset = gpu_offset
        self._gpu_count = processes_per_node if gpu_count is None and device_type == "gpu" else gpu_count or 0
        self._bound = bound

    @property
    def world_size(self) -> int:
        if not self._bound:
            raise RuntimeError("root ResourcePool has no process ranks")
        return self._nnodes * self._processes_per_node

    @property
    def nnodes(self) -> int:
        return self._nnodes

    @property
    def processes_per_node(self) -> int:
        """Return one process per host for roots, or the bound rank count."""
        return self._processes_per_node if self._bound else 1

    @property
    def device_type(self) -> Literal["gpu", "cpu"]:
        if not self._bound:
            raise RuntimeError("root ResourcePool has no device_type")
        return self._device_type

    def slice(self, ranks: int | slice) -> MonarchResourcePool:
        if not self._bound:
            start, stop = normalize_pool_ranks(ranks, self._nnodes)
            mesh = self._monarch_host_mesh.flatten("host").slice(host=slice(start, stop))
            return self._inherit_runtime_scope(
                MonarchResourcePool(
                    monarch_host_mesh=mesh,
                    nnodes=stop - start,
                    processes_per_node=1,
                    device_type="cpu",
                    bound=False,
                )
            )
        node_start, node_count, process_start, process_count = rectangular_pool_view(
            ranks,
            nnodes=self._nnodes,
            processes_per_node=self._processes_per_node,
        )
        mesh = self._monarch_host_mesh.flatten("host").slice(host=slice(node_start, node_start + node_count))
        return self._inherit_runtime_scope(
            MonarchResourcePool(
                monarch_host_mesh=mesh,
                nnodes=node_count,
                processes_per_node=process_count,
                device_type=self._device_type,
                gpu_offset=self._gpu_offset + process_start,
                gpu_count=process_count if self._gpu_count else 0,
                bound=True,
            )
        )

    def _with_processes(
        self,
        *,
        processes_per_node: int,
        device_type: Literal["gpu", "cpu"],
    ) -> MonarchResourcePool:
        return MonarchResourcePool(
            monarch_host_mesh=self._monarch_host_mesh,
            nnodes=self._nnodes,
            processes_per_node=processes_per_node,
            device_type=device_type,
            gpu_offset=self._gpu_offset,
            gpu_count=(processes_per_node if device_type == "gpu" else self._gpu_count),
            bound=True,
        )

    def _select_device_range(self, start: int, end: int) -> MonarchResourcePool:
        if self.device_type != "gpu":
            raise ValueError("device ranges require a GPU ResourcePool")
        if start < 0 or end <= start or end > self._gpu_count:
            raise ValueError(f"device range must satisfy 0 <= start < end <= {self._gpu_count}, got {(start, end)!r}")
        return MonarchResourcePool(
            monarch_host_mesh=self._monarch_host_mesh,
            nnodes=self._nnodes,
            processes_per_node=end - start,
            device_type="gpu",
            gpu_offset=self._gpu_offset + start,
            gpu_count=end - start,
            bound=True,
        )

    @property
    def gpu_ids(self) -> tuple[int, ...]:
        return tuple(range(self._gpu_offset, self._gpu_offset + self._gpu_count))


__all__ = ["MonarchResourcePool"]
