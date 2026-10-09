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

"""Partition declared DevicePools from cluster HostMeshes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from verl.single_controller.base.topology import Topology

if TYPE_CHECKING:
    from monarch.actor import HostMesh


def required_hosts(topology: Topology) -> int:
    """Return the minimum homogeneous host inventory required by Ray-equivalent placement."""
    pools_by_cluster = {cluster.name: 0 for cluster in topology.clusters}
    for device_pool in topology.device_pools:
        pools_by_cluster[device_pool.cluster] += device_pool.nnodes

    required = 1
    claimed = 0
    for cluster in topology.clusters:
        required = max(required, claimed + cluster.nnodes)
        claimed += pools_by_cluster[cluster.name]
    return required


def resolve_topology(topology: Topology, host_mesh: HostMesh) -> dict[str, HostMesh]:
    """Apply Ray-equivalent ordered claiming to one homogeneous HostMesh."""
    result: dict[str, HostMesh] = {}
    pools_by_cluster = {cluster.name: [] for cluster in topology.clusters}
    for device_pool in topology.device_pools:
        pools_by_cluster[device_pool.cluster].append(device_pool)

    size = int(host_mesh.size())
    required = required_hosts(topology)
    if size < required:
        raise ValueError(f"Monarch HostMesh has {size} nodes, but topology placement requires at least {required}")
    root_mesh = host_mesh.flatten("host")
    claimed = 0
    for cluster in topology.clusters:
        cluster_mesh = root_mesh.slice(host=slice(claimed, claimed + cluster.nnodes))
        pool_cursor = 0
        for device_pool in pools_by_cluster[cluster.name]:
            result[device_pool.name] = cluster_mesh.slice(host=slice(pool_cursor, pool_cursor + device_pool.nnodes))
            pool_cursor += device_pool.nnodes
        claimed += pool_cursor
    return result


__all__ = ["required_hosts", "resolve_topology"]
