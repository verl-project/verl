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

"""Partition Ray nodes and compile DevicePools into standard resource bundles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from verl.plugin.platform import get_platform
from verl.single_controller.base.errors import PlacementUnavailableError
from verl.single_controller.base.topology import Cluster, DevicePool, Topology

_CPUS_PER_BUNDLE = 10.0
_NODE_RESOURCE_REQUEST = 1e-3


@dataclass(frozen=True, slots=True)
class PlacementStrategy:
    """Placement-group inputs for one DevicePool."""

    name: str
    nodes: tuple[tuple[str, tuple[Mapping[str, float], ...]], ...]
    strategy: str = "STRICT_PACK"


@dataclass(frozen=True, slots=True)
class RayNode:
    node_id: str
    address: str
    node_resource: str
    resources: Mapping[str, float]


def _inventory(nodes: Sequence[Mapping[str, object]]) -> tuple[RayNode, ...]:
    result = []
    for node in nodes:
        if not node.get("Alive", True):
            continue
        resources = {str(name): float(value) for name, value in dict(node.get("Resources", {})).items()}
        address = str(node.get("NodeManagerAddress", ""))
        node_resource = f"node:{address}"
        if node_resource not in resources:
            candidates = sorted(name for name in resources if name.startswith("node:") and "internal" not in name)
            if len(candidates) != 1:
                raise PlacementUnavailableError(
                    f"Ray node {node.get('NodeID')!r} has no unique node resource; available: {candidates!r}"
                )
            node_resource = candidates[0]
        result.append(
            RayNode(
                node_id=str(node["NodeID"]),
                address=address,
                node_resource=node_resource,
                resources=MappingProxyType(resources),
            )
        )
    return tuple(sorted(result, key=lambda item: (item.address, item.node_id)))


def _eligible_nodes(cluster: Cluster, nodes: tuple[RayNode, ...]) -> tuple[RayNode, ...]:
    gpu = get_platform().ray_resource_name()
    eligible = []
    for node in nodes:
        if node.resources.get(gpu, 0.0) < cluster.n_gpus_per_node:
            continue
        if cluster.name != "default" and node.resources.get(cluster.name, 0.0) < cluster.n_gpus_per_node:
            continue
        eligible.append(node)
    return tuple(eligible)


def _node_placement(
    device_pool: DevicePool,
    cluster: Cluster,
    node: RayNode,
) -> tuple[str, tuple[Mapping[str, float], ...]]:
    gpu = get_platform().ray_resource_name()
    bundle: dict[str, float] = {
        "CPU": _CPUS_PER_BUNDLE,
        gpu: 1.0,
        node.node_resource: _NODE_RESOURCE_REQUEST,
    }
    if cluster.name != "default":
        bundle[cluster.name] = 1.0
    frozen = MappingProxyType(bundle)
    return node.node_id, tuple(frozen for _ in range(device_pool.n_gpus_per_node))


def resolve_topology(
    topology: Topology,
    nodes: Sequence[Mapping[str, object]],
) -> dict[str, PlacementStrategy]:
    """Partition one Ray node snapshot into non-overlapping DevicePools."""
    inventory = _inventory(nodes)
    pools_by_cluster: dict[str, list[DevicePool]] = {cluster.name: [] for cluster in topology.clusters}
    for device_pool in topology.device_pools:
        pools_by_cluster[device_pool.cluster].append(device_pool)

    result: dict[str, PlacementStrategy] = {}
    claimed_node_ids: set[str] = set()
    for cluster in topology.clusters:
        cluster_nodes = [node for node in _eligible_nodes(cluster, inventory) if node.node_id not in claimed_node_ids]
        if len(cluster_nodes) < cluster.nnodes:
            raise PlacementUnavailableError(
                f"Ray cluster {cluster.name!r} has {len(cluster_nodes)} unclaimed eligible nodes; "
                f"topology declares {cluster.nnodes}"
            )
        cluster_nodes = cluster_nodes[: cluster.nnodes]
        cursor = 0
        for device_pool in pools_by_cluster[cluster.name]:
            selected = cluster_nodes[cursor : cursor + device_pool.nnodes]
            if len(selected) != device_pool.nnodes:
                raise PlacementUnavailableError(
                    f"Ray cluster {cluster.name!r} cannot assign {device_pool.nnodes} unclaimed nodes "
                    f"to device pool {device_pool.name!r}"
                )
            cursor += device_pool.nnodes
            claimed_node_ids.update(node.node_id for node in selected)
            result[device_pool.name] = PlacementStrategy(
                name=device_pool.name,
                nodes=tuple(_node_placement(device_pool, cluster, node) for node in selected),
            )
    return result


__all__ = ["PlacementStrategy", "resolve_topology"]
