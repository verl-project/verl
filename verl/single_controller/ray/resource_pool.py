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

"""Ray ResourcePool backed by placement-group resource bundles."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any, Literal

import ray
from ray.util.placement_group import PlacementGroup, placement_group, placement_group_table
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from verl.plugin.platform import get_platform
from verl.single_controller.base.errors import ExceptionGroup, PlacementUnavailableError
from verl.single_controller.base.resource_pool import ResourcePool, normalize_pool_ranks, rectangular_pool_view
from verl.single_controller.ray.topology import PlacementStrategy

_UNSET = object()


BundleRef = tuple[PlacementGroup, int]


def get_random_string(length: int) -> str:
    """Return ``length`` random ASCII letters and digits, used for unique name prefixes."""
    import random
    import string

    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def sort_placement_group_by_node_ip(pgs: list[PlacementGroup]) -> list[PlacementGroup]:
    """
    Sort the placement groups by node ip, all bundles in a single placement group should be on the same node.

    FSDPCheckpointManager saves sharded model states and optimizer states in local storage, which requires RANK
    to be consistent across nodes when resume from checkpoint.

    With this function, if there's only one resource pool and there's no node change, RANK should be consistent
    across nodes in multiple ray jobs, even if the whole ray cluster is restarted.
    """
    node_ip = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes()}
    ranked = []
    for original_index, group in enumerate(pgs):
        # all bundles should be on the same node
        first_node = _placement_group_node_ids(group)[0]
        ranked.append((str(node_ip[first_node]), first_node, original_index, group))
    return [group for _address, _node_id, _index, group in sorted(ranked, key=lambda item: item[:3])]


def _placement_group_node_ids(group: PlacementGroup) -> tuple[str, ...]:
    """Read one ready placement group's bundle assignment via Ray's public API."""
    assignment = placement_group_table(group).get("bundles_to_node_id")
    if not isinstance(assignment, dict):
        raise PlacementUnavailableError("Ray placement-group table did not expose bundle node assignments")
    bundle_count = int(getattr(group, "bundle_count", len(assignment)))
    node_ids = []
    for bundle_index in range(bundle_count):
        node_id = assignment.get(bundle_index, assignment.get(str(bundle_index)))
        if not node_id:
            raise PlacementUnavailableError(
                f"Ray placement-group bundle {bundle_index} has no assigned node after readiness"
            )
        node_ids.append(str(node_id))
    return tuple(node_ids)


@contextmanager
def _acquire_placement_groups(error_message: str) -> Iterator[list[PlacementGroup]]:
    """Transfer acquired groups on success, otherwise roll back in creation order."""
    groups: list[PlacementGroup] = []
    try:
        try:
            yield groups
        except Exception as primary:
            raise PlacementUnavailableError(error_message) from primary
    except BaseException:
        # Wrap allocation errors before cleanup, retaining cleanup interruption chains.
        for group in groups:
            with suppress(Exception):
                ray.util.remove_placement_group(group)
        raise


class RayResourcePool(ResourcePool):
    """Ordered Ray placement where every rank selects one bundle reference."""

    def __init__(
        self,
        process_on_nodes: list[int] | None = None,
        use_gpu: bool = True,
        name_prefix: str | None = None,
        max_colocate_count: int | object = _UNSET,
        detached: bool | object = _UNSET,
        *,
        device_type: Literal["gpu", "cpu"] | None = None,
        node_ids: list[str] | None = None,
        gpu_ids: tuple[int, ...] | None = None,
        placement_groups: tuple[PlacementGroup, ...] = (),
        bundle_refs: tuple[BundleRef, ...] = (),
        bound: bool | None = None,
        accelerator_type: str | None = None,
    ) -> None:
        if max_colocate_count is _UNSET:
            max_colocate_count = 10
        elif isinstance(max_colocate_count, bool) or not isinstance(max_colocate_count, int):
            raise TypeError(f"max_colocate_count must be an int, got {type(max_colocate_count)!r}")
        if max_colocate_count <= 0:
            raise ValueError(f"max_colocate_count must be positive, got {max_colocate_count}")
        if detached is _UNSET:
            detached = False
        elif not isinstance(detached, bool):
            raise TypeError(f"detached must be a bool, got {type(detached)!r}")
        self._store = list(process_on_nodes or [])
        if self._store and len(set(self._store)) != 1:
            raise ValueError(f"RayResourcePool requires homogeneous per-node bundles, got {self._store!r}")
        self._bound = bool(self._store) if bound is None else bound
        self.use_gpu = use_gpu
        self.name_prefix = get_random_string(6) if name_prefix is None else name_prefix
        self.max_colocate_count = max_colocate_count
        self.detached = detached
        self._device_type: Literal["gpu", "cpu"] = (
            device_type if device_type is not None else ("gpu" if use_gpu else "cpu")
        )
        self._node_ids = list(node_ids or [])
        self._gpu_ids = gpu_ids or ()
        self._placement_groups = placement_groups
        self._bundle_refs = bundle_refs
        self._release_reservation = False
        self._closed = False
        self.accelerator_type = accelerator_type

    @classmethod
    def _from_placement_strategy(cls, strategy: PlacementStrategy) -> RayResourcePool:
        """Create placement groups without waiting for scheduling."""
        with _acquire_placement_groups(f"failed to allocate Ray DevicePool {strategy.name!r}") as groups:
            for node_index, (_node_id, bundles) in enumerate(strategy.nodes):
                group = placement_group(
                    bundles=[dict(bundle) for bundle in bundles],
                    strategy=strategy.strategy,
                    name=f"{strategy.name}:{node_index}",
                )
                groups.append(group)

        refs = tuple(
            (group, index)
            for group, (_node_id, bundles) in zip(groups, strategy.nodes, strict=True)
            for index in range(len(bundles))
        )
        per_node = [len(bundles) for _node_id, bundles in strategy.nodes]
        pool = cls(
            process_on_nodes=per_node,
            use_gpu=True,
            name_prefix=strategy.name,
            device_type="gpu",
            placement_groups=tuple(groups),
            bundle_refs=refs,
            gpu_ids=tuple(range(per_node[0])),
            bound=True,
        )
        pool._release_reservation = True
        return pool

    @classmethod
    def _from_cluster(cls) -> RayResourcePool:
        """Discover the live Ray nodes as a non-executable root domain."""
        nodes = sorted(
            (str(node.get("NodeManagerAddress", "")), str(node["NodeID"]))
            for node in ray.nodes()
            if node.get("Alive", True)
        )
        if not nodes:
            raise PlacementUnavailableError("Ray cluster has no live nodes")
        return cls(
            process_on_nodes=[1] * len(nodes),
            use_gpu=False,
            device_type="cpu",
            node_ids=[node_id for _address, node_id in nodes],
            bound=False,
        )

    @classmethod
    def _for_controller(cls, cluster: RayResourcePool) -> RayResourcePool:
        node_id = ray.get_runtime_context().get_node_id()
        if node_id not in cluster.node_ids:
            raise PlacementUnavailableError(f"Ray controller node {node_id!r} is not in the live cluster")
        return cls(
            process_on_nodes=[1],
            use_gpu=False,
            device_type="cpu",
            node_ids=[node_id],
            bound=False,
        )

    def _derive(
        self,
        *,
        nnodes: int,
        processes_per_node: int,
        device_type: Literal["gpu", "cpu"],
    ) -> RayResourcePool:
        """Compatibility allocation for callers not yet using declared DevicePools."""
        if nnodes <= 0 or processes_per_node <= 0:
            raise ValueError("nnodes and processes_per_node must be positive")
        resource_name = "CPU" if device_type == "cpu" else get_platform().ray_resource_name()
        required = 1 if device_type == "cpu" else processes_per_node
        allowed = set(self.node_ids)
        eligible = sorted(
            (str(node.get("NodeManagerAddress", "")), str(node["NodeID"]))
            for node in ray.nodes()
            if node.get("Alive", True)
            and str(node["NodeID"]) in allowed
            and float(node.get("Resources", {}).get(resource_name, 0)) >= required
        )
        if len(eligible) < nnodes:
            raise PlacementUnavailableError(f"ResourcePool has {len(eligible)} eligible nodes; requested {nnodes}")
        selected = [node_id for _address, node_id in eligible[:nnodes]]
        if device_type == "gpu":
            # Legacy V0 pools reserve one Ray bundle per rank. Keeping this
            # pool unpinned lets Ray place the STRICT_PACK groups while the
            # bundles prevent fractional-GPU actors from sharing one device.
            return RayResourcePool(
                process_on_nodes=[processes_per_node] * nnodes,
                use_gpu=True,
                device_type="gpu",
                gpu_ids=tuple(range(processes_per_node)),
                bound=True,
            )
        return RayResourcePool(
            process_on_nodes=[processes_per_node] * nnodes,
            use_gpu=False,
            device_type=device_type,
            node_ids=[node_id for node_id in selected for _ in range(processes_per_node)],
            bound=True,
        )

    def _with_processes(
        self,
        *,
        processes_per_node: int,
        device_type: Literal["gpu", "cpu"],
    ) -> RayResourcePool:
        if device_type != "cpu":
            raise ValueError("per-WorkerGroup placement overrides currently support CPU processes only")
        source_ranks = tuple(
            node * self.processes_per_node + (local_rank % self.processes_per_node if self._bundle_refs else 0)
            for node in range(self.nnodes)
            for local_rank in range(processes_per_node)
        )
        return RayResourcePool(
            process_on_nodes=[processes_per_node] * self.nnodes,
            use_gpu=False,
            device_type="cpu",
            placement_groups=self._placement_groups if self._bundle_refs else (),
            bundle_refs=tuple(self._bundle_refs[rank] for rank in source_ranks) if self._bundle_refs else (),
            node_ids=[self._node_ids[rank] for rank in source_ranks],
            gpu_ids=self.gpu_ids,
            bound=True,
        )

    def _select_device_range(self, start: int, end: int) -> RayResourcePool:
        if self.device_type != "gpu":
            raise ValueError("device ranges require a GPU ResourcePool")
        if start < 0 or end <= start or end > self.processes_per_node:
            raise ValueError(
                f"device range must satisfy 0 <= start < end <= {self.processes_per_node}, got {(start, end)!r}"
            )
        width = end - start
        ranks = [
            node * self.processes_per_node + local_rank
            for node in range(self.nnodes)
            for local_rank in range(start, end)
        ]
        return RayResourcePool(
            process_on_nodes=[width] * self.nnodes,
            use_gpu=True,
            name_prefix=f"{self.name_prefix}_devices_{start}_{end}",
            max_colocate_count=self.max_colocate_count,
            detached=self.detached,
            device_type="gpu",
            node_ids=[self._node_ids[rank] for rank in ranks],
            gpu_ids=self.gpu_ids[start:end],
            placement_groups=self._placement_groups,
            bundle_refs=tuple(self._bundle_refs[rank] for rank in ranks) if self._bundle_refs else (),
            bound=True,
        )

    @property
    def store(self) -> list[int]:
        return self._store

    @property
    def world_size(self) -> int:
        if not self._bound:
            raise RuntimeError("root ResourcePool has no process ranks")
        return sum(self._store)

    @property
    def nnodes(self) -> int:
        return len(self._store)

    @property
    def processes_per_node(self) -> int:
        if not self._bound or not self._store:
            raise RuntimeError("root ResourcePool has no processes_per_node")
        return self._store[0]

    @property
    def device_type(self) -> Literal["gpu", "cpu"]:
        if not self._bound:
            raise RuntimeError("root ResourcePool has no device_type")
        return self._device_type

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(self._node_ids)

    @property
    def gpu_ids(self) -> tuple[int, ...]:
        return self._gpu_ids

    @property
    def _reserved_groups(self) -> tuple[PlacementGroup, ...]:
        return self._placement_groups

    def _scheduling_strategy(self, rank: int):
        if self._bundle_refs:
            placement_group, bundle_index = self._bundle_refs[rank]
            return PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=bundle_index,
                placement_group_capture_child_tasks=True,
            )
        if self._node_ids:
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            return NodeAffinitySchedulingStrategy(node_id=self._node_ids[rank], soft=False)
        raise RuntimeError("Ray ResourcePool has no placement for its ranks")

    def get_placement_groups(self, strategy="STRICT_PACK", name=None, device_name=None) -> list[PlacementGroup]:
        if self._placement_groups:
            return list(self._placement_groups)
        if self._node_ids:
            return []
        # DEPRECATED: device_name is retained for compatibility; resource naming follows the active platform.
        _ = device_name
        current_platform = get_platform()
        resource_name = current_platform.ray_resource_name()
        bundle: dict[str, float] = {"CPU": float(self.max_colocate_count)}
        if self.use_gpu:
            bundle[resource_name] = 1.0
            if self.accelerator_type is not None:
                bundle[self.accelerator_type] = 1e-4
        lifetime = "detached" if self.detached else None
        pg_name_prefix = (
            name if name else f"{self.name_prefix}verl_group_{'_'.join(str(count) for count in self._store)}:"
        )
        with _acquire_placement_groups(f"failed to allocate Ray ResourcePool {self.name_prefix!r}") as groups:
            for node_index, size in enumerate(self._store):
                groups.append(
                    placement_group(
                        bundles=[dict(bundle) for _ in range(size)],
                        strategy=strategy,
                        name=f"{pg_name_prefix}{node_index}",
                        lifetime=lifetime,
                    )
                )
            ray.get([group.ready() for group in groups])
        self._placement_groups = tuple(groups)
        self._finalize_placement(sort_groups=True)
        self._release_reservation = True
        return list(self._placement_groups)

    def _finalize_placement(self, *, sort_groups: bool) -> None:
        """Publish bundle refs and actual node IDs after all groups are ready."""
        groups = list(self._placement_groups)
        if sort_groups:
            groups = sort_placement_group_by_node_ip(groups)
        assignments = [_placement_group_node_ids(group) for group in groups]
        refs = []
        node_ids = []
        for group, size, group_node_ids in zip(groups, self._store, assignments, strict=True):
            for index in range(size):
                refs.append((group, index))
                node_ids.append(group_node_ids[index])
        self._placement_groups = tuple(groups)
        self._bundle_refs = tuple(refs)
        self._node_ids = node_ids

    @property
    def pgs(self) -> list[PlacementGroup] | None:
        return list(self._placement_groups) if self._placement_groups else None

    @pgs.setter
    def pgs(self, value: list[PlacementGroup] | None) -> None:
        self._placement_groups = tuple(value or ())

    def slice(self, ranks: int | slice) -> RayResourcePool:
        if not self._bound:
            start, stop = normalize_pool_ranks(ranks, self.nnodes)
            return self._inherit_runtime_scope(
                RayResourcePool(
                    process_on_nodes=[1] * (stop - start),
                    use_gpu=False,
                    device_type="cpu",
                    node_ids=self._node_ids[start:stop],
                    bound=False,
                )
            )
        node_start, node_count, process_start, process_count = rectangular_pool_view(
            ranks,
            nnodes=self.nnodes,
            processes_per_node=self.processes_per_node,
        )
        start = node_start * self.processes_per_node + process_start
        stop = start + node_count * process_count
        return self._inherit_runtime_scope(
            SubRayResourcePool(
                source=self,
                start=start,
                stop=stop,
                nnodes=node_count,
                processes_per_node=process_count,
                gpu_ids=self.gpu_ids[process_start : process_start + process_count],
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        errors: list[Exception] = []
        remaining: list[PlacementGroup] = []
        if self._release_reservation:
            for group in self._placement_groups:
                try:
                    ray.util.remove_placement_group(group)
                except Exception as exc:
                    errors.append(exc)
                    remaining.append(group)
            self._placement_groups = tuple(remaining)
            self._bundle_refs = tuple(ref for ref in self._bundle_refs if ref[0] in remaining)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("runtime failures", errors)
        self._closed = True


class SubRayResourcePool(RayResourcePool):
    """Non-owning contiguous bundle view."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "source" in kwargs:
            source = kwargs["source"]
            start = kwargs["start"]
            stop = kwargs["stop"]
            nnodes = kwargs["nnodes"]
            processes_per_node = kwargs["processes_per_node"]
            gpu_ids = kwargs.get("gpu_ids", ())
            self._init_from_source(
                source,
                start=start,
                stop=stop,
                nnodes=nnodes,
                processes_per_node=processes_per_node,
                gpu_ids=gpu_ids,
            )
            return

        placement_groups = args[0] if args else kwargs.pop("placement_groups")
        start_bundle_index = args[1] if len(args) > 1 else kwargs.pop("start_bundle_index")
        subgroup_world_size = args[2] if len(args) > 2 else kwargs.pop("subgroup_world_size")
        if not placement_groups:
            raise ValueError("legacy SubRayResourcePool requires placement_groups")
        bundle_counts = [int(getattr(group, "bundle_count", 0)) for group in placement_groups]
        if any(count <= 0 for count in bundle_counts):
            raise ValueError("legacy SubRayResourcePool placement groups must expose positive bundle_count")
        if len(set(bundle_counts)) != 1:
            raise ValueError(f"legacy SubRayResourcePool requires homogeneous per-node bundles, got {bundle_counts!r}")
        processes_per_node = bundle_counts[0]
        nnodes = len(placement_groups)
        if start_bundle_index < 0 or subgroup_world_size <= 0:
            raise ValueError("legacy SubRayResourcePool bundle range must be non-negative and non-empty")
        if start_bundle_index + subgroup_world_size > sum(bundle_counts):
            raise ValueError("legacy SubRayResourcePool bundle range exceeds its placement groups")
        process_on_nodes = kwargs.get("process_on_nodes") or bundle_counts
        if len(set(process_on_nodes)) != 1:
            raise ValueError(
                f"legacy SubRayResourcePool requires homogeneous per-node bundles, got {process_on_nodes!r}"
            )
        super().__init__(
            process_on_nodes=process_on_nodes,
            use_gpu=kwargs.get("use_gpu", True),
            name_prefix=kwargs.get("name_prefix"),
            max_colocate_count=kwargs.get("max_colocate_count", _UNSET),
            detached=kwargs.get("detached", _UNSET),
            device_type=kwargs.get("device_type"),
            placement_groups=tuple(placement_groups),
            bound=True,
            accelerator_type=kwargs.get("accelerator_type"),
        )
        self._source = None
        self._rank_start = start_bundle_index
        self._rank_stop = start_bundle_index + subgroup_world_size
        self.start_bundle_index = start_bundle_index
        self.subgroup_world_size = subgroup_world_size
        self._legacy_subgroup_world_size = subgroup_world_size

    def _init_from_source(
        self,
        source: RayResourcePool,
        *,
        start: int,
        stop: int,
        nnodes: int,
        processes_per_node: int,
        gpu_ids: tuple[int, ...],
    ) -> None:
        super().__init__(
            process_on_nodes=[processes_per_node] * nnodes,
            use_gpu=source.use_gpu,
            name_prefix=f"{source.name_prefix}_view_{start}_{stop}",
            max_colocate_count=source.max_colocate_count,
            detached=source.detached,
            device_type=source._device_type,
            node_ids=list(source.node_ids[start:stop]),
            gpu_ids=gpu_ids,
            placement_groups=source._reserved_groups,
            bundle_refs=source._bundle_refs[start:stop],
            bound=True,
            accelerator_type=source.accelerator_type,
        )
        self._source = source
        self._rank_start = start
        self._rank_stop = stop
        self.start_bundle_index = start
        self.subgroup_world_size = stop - start
        self._legacy_subgroup_world_size = None

    @property
    def world_size(self) -> int:
        if getattr(self, "_legacy_subgroup_world_size", None) is not None:
            return self._legacy_subgroup_world_size
        return super().world_size

    def slice(self, ranks: int | slice) -> RayResourcePool:
        start, stop = normalize_pool_ranks(ranks, self.world_size)
        if self._source is not None:
            return self._source.slice(slice(self._rank_start + start, self._rank_start + stop))
        return super().slice(ranks)

    def close(self) -> None:
        return


# split a RayResourcePool or SubRayResourcePool into multiple SubRayResourcePool
def split_resource_pool(
    resource_pool: RayResourcePool | SubRayResourcePool, split_size: int | list[int]
) -> list[SubRayResourcePool]:
    """
    Split a RayResourcePool into multiple SubRayResourcePool.
    resouce_pool can also be a SubRayResourcePool (have been splited) for multiple-time spliting.

    Args:
        resource_pool (RayResourcePool | SubRayResourcePool): The resource pool to split.
        split_size (int | list[int]): The size of each split. If int, all splits will have the same size.
            If list[int], each element in the list represents the size of a split.

    Returns:
        list[SubRayResourcePool]: A list of non-owning SubRayResourcePool views after splitting.
    """
    if isinstance(split_size, int):
        if resource_pool.world_size % split_size != 0:
            raise ValueError("split_size must divide world_size")
        sizes = [split_size] * (resource_pool.world_size // split_size)
    else:
        sizes = list(split_size)
    if sum(sizes) != resource_pool.world_size:
        raise ValueError("split_size must sum to world_size")
    views = []
    start = 0
    for size in sizes:
        views.append(resource_pool.slice(slice(start, start + size)))
        start += size
    return views


def merge_resource_pool(rp1: RayResourcePool, rp2: RayResourcePool) -> RayResourcePool:
    """Concatenate two compatible Ray resource pools into one bound pool.

    Args:
        rp1: Pool whose processes come first in the merged rank order.
        rp2: Pool whose processes follow ``rp1``.

    Returns:
        A bound :class:`RayResourcePool` reusing both pools' placement groups.

    Raises:
        ValueError: The pools differ in GPU use, colocation count, detachment,
            device type, processes per node, or accelerator type.
    """
    if rp1.use_gpu != rp2.use_gpu:
        raise ValueError("cannot merge ResourcePools with different use_gpu")
    if rp1.max_colocate_count != rp2.max_colocate_count:
        raise ValueError("cannot merge ResourcePools with different max_colocate_count")
    if rp1.detached != rp2.detached:
        raise ValueError("Detached ResourcePool cannot be merged with non-detached ResourcePool")
    if rp1.device_type != rp2.device_type:
        raise ValueError("cannot merge ResourcePools with different device types")
    if rp1.processes_per_node != rp2.processes_per_node:
        raise ValueError("cannot merge ResourcePools with different processes_per_node")
    if rp1.accelerator_type != rp2.accelerator_type:
        raise ValueError("cannot merge ResourcePools with different accelerator_type")
    return RayResourcePool(
        process_on_nodes=rp1.store + rp2.store,
        use_gpu=rp1.use_gpu,
        name_prefix=f"{rp1.name_prefix}_{rp2.name_prefix}",
        max_colocate_count=rp1.max_colocate_count,
        detached=rp1.detached,
        device_type=rp1.device_type,
        node_ids=list(rp1.node_ids + rp2.node_ids),
        gpu_ids=rp1.gpu_ids if rp1.gpu_ids == rp2.gpu_ids else (),
        placement_groups=rp1._reserved_groups + rp2._reserved_groups,
        bundle_refs=rp1._bundle_refs + rp2._bundle_refs,
        bound=True,
        accelerator_type=rp1.accelerator_type,
    )


__all__ = [
    "RayResourcePool",
    "SubRayResourcePool",
    "merge_resource_pool",
    "split_resource_pool",
    "sort_placement_group_by_node_ip",
]
