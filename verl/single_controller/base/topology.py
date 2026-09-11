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

"""Declarative model placement from https://github.com/verl-project/verl/issues/7269."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verl.single_controller.base.resource_pool import ResourcePool


def _validate_name(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty str, got {value!r}")


def _validate_capacity(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-bool int, got {type(value)!r}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True, slots=True)
class Cluster:
    """One gpu type and its available capacity."""

    name: str
    nnodes: int
    n_gpus_per_node: int

    def __post_init__(self) -> None:
        _validate_name("cluster.name", self.name)
        _validate_capacity("cluster.nnodes", self.nnodes)
        _validate_capacity("cluster.n_gpus_per_node", self.n_gpus_per_node)


@dataclass(frozen=True, slots=True)
class DevicePool:
    """A named group of gpus selected from one cluster."""

    name: str
    cluster: str
    nnodes: int
    n_gpus_per_node: int
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_name("device_pool.name", self.name)
        _validate_name("device_pool.cluster", self.cluster)
        _validate_capacity("device_pool.nnodes", self.nnodes)
        _validate_capacity("device_pool.n_gpus_per_node", self.n_gpus_per_node)
        if not isinstance(self.attributes, Mapping):
            raise TypeError(f"device_pool.attributes must be a mapping, got {type(self.attributes)!r}")
        attributes = dict(self.attributes)
        object.__setattr__(self, "attributes", MappingProxyType(attributes))
        if attributes:
            warnings.warn(
                f"device pool {self.name!r} attributes are preserved but not enforced: {sorted(attributes)!r}",
                stacklevel=2,
            )


@dataclass(frozen=True, slots=True)
class Model:
    """One model instance assigned to a resource pool."""

    name: str
    worker: str
    config_key: str
    resource_pool: str
    device_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        _validate_name("model.name", self.name)
        _validate_name("model.worker", self.worker)
        _validate_name("model.config_key", self.config_key)
        _validate_name("model.resource_pool", self.resource_pool)
        if self.device_range is not None:
            if len(self.device_range) != 2:
                raise ValueError("device_range must contain exactly two integers")
            start, end = self.device_range
            if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)):
                raise TypeError("device_range values must be non-bool integers")
            if start < 0 or end <= start:
                raise ValueError("device_range must satisfy 0 <= start < end")


@dataclass(frozen=True, slots=True)
class Topology:
    """The cluster -> device_pool -> model declaration."""

    clusters: tuple[Cluster, ...] = ()
    device_pools: tuple[DevicePool, ...] = ()
    models: tuple[Model, ...] = ()

    def __post_init__(self) -> None:
        for kind, entries in (
            ("cluster", self.clusters),
            ("device pool", self.device_pools),
            ("model", self.models),
        ):
            if len({entry.name for entry in entries}) != len(entries):
                raise ValueError(f"{kind} names must be unique")
        self._validate_pool_capacity()
        self._validate_models()
        self._validate_model_overlap()

    def _validate_pool_capacity(self) -> None:
        clusters = {cluster.name: cluster for cluster in self.clusters}
        selected_nodes = {name: 0 for name in clusters}
        for device_pool in self.device_pools:
            if device_pool.cluster not in clusters:
                raise ValueError(f"device pool {device_pool.name!r} references unknown cluster {device_pool.cluster!r}")
            cluster = clusters[device_pool.cluster]
            if device_pool.n_gpus_per_node > cluster.n_gpus_per_node:
                raise ValueError(f"device pool {device_pool.name!r} exceeds cluster {cluster.name!r} gpu capacity")
            selected_nodes[device_pool.cluster] += device_pool.nnodes
        for cluster_name, nnodes in selected_nodes.items():
            if nnodes > clusters[cluster_name].nnodes:
                raise ValueError(f"device pools exceed cluster {cluster_name!r} node capacity")

    def _validate_models(self) -> None:
        device_pools = {pool.name: pool for pool in self.device_pools}
        for model in self.models:
            if model.resource_pool not in device_pools:
                raise ValueError(f"model {model.name!r} references unknown resource_pool {model.resource_pool!r}")
            if (
                model.device_range is not None
                and model.device_range[1] > device_pools[model.resource_pool].n_gpus_per_node
            ):
                raise ValueError(f"model {model.name!r}.device_range exceeds its device pool")

    def _model_placements(self) -> dict[str, tuple[str, tuple[int, int]]]:
        """Return declared pool/range identities in model declaration order."""
        device_pools = {pool.name: pool for pool in self.device_pools}
        return {
            model.name: (
                model.resource_pool,
                model.device_range or (0, device_pools[model.resource_pool].n_gpus_per_node),
            )
            for model in self.models
        }

    def _validate_model_overlap(self) -> None:
        models_by_pool: dict[str, list[tuple[str, tuple[int, int]]]] = {}
        for name, (pool_name, selected) in self._model_placements().items():
            models_by_pool.setdefault(pool_name, []).append((name, selected))
        for pool_name, placements in models_by_pool.items():
            for (left_model, left), (right_model, right) in combinations(placements, 2):
                if left == right or left[1] <= right[0] or right[1] <= left[0]:
                    continue
                raise ValueError(
                    f"models {left_model!r} and {right_model!r} have overlapping "
                    f"device ranges {left!r} and {right!r} on resource pool {pool_name!r}"
                )

    @classmethod
    def from_mapping(cls, value: object | None) -> Topology:
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError(f"topology must be a mapping, got {type(value)!r}")
        unknown = set(value) - {"clusters", "device_pools", "models"}
        if unknown:
            raise ValueError(f"unknown topology fields: {sorted(unknown)!r}")

        raw_clusters = value.get("clusters", ())
        raw_device_pools = value.get("device_pools", ())
        raw_models = value.get("models", ())
        for name, entries in (
            ("clusters", raw_clusters),
            ("device_pools", raw_device_pools),
            ("models", raw_models),
        ):
            if isinstance(entries, str) or not isinstance(entries, Sequence):
                raise TypeError(f"topology.{name} must be a sequence")
            for index, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    raise TypeError(f"topology.{name}[{index}] must be a mapping")

        clusters = tuple(Cluster(**dict(entry)) for entry in raw_clusters)
        device_pools = tuple(DevicePool(**dict(entry)) for entry in raw_device_pools)
        models = []
        for entry in raw_models:
            payload = dict(entry)
            device_range = payload.get("device_range")
            if device_range is not None:
                if isinstance(device_range, str | bytes) or not isinstance(device_range, Sequence):
                    raise TypeError("model.device_range must be a two-item sequence")
                payload["device_range"] = tuple(device_range)
            models.append(Model(**payload))
        return cls(clusters=clusters, device_pools=device_pools, models=tuple(models))


def _format_topology(
    topology: Topology,
    model_resource_pools: Mapping[str, ResourcePool] | None = None,
) -> str:
    """Format the declared or compiled model placement report."""
    pool_specs = {pool.name: pool for pool in topology.device_pools}
    declared_placements = topology._model_placements()
    declared_counts: dict[tuple[str, tuple[int, int]], int] = {}
    actual_pools: dict[str, ResourcePool] = {}
    actual_counts: dict[int, int] = {}
    if model_resource_pools is None:
        for placement in declared_placements.values():
            declared_counts[placement] = declared_counts.get(placement, 0) + 1
    else:
        actual_pools = {model.name: model_resource_pools[model.name] for model in topology.models}
        for pool in actual_pools.values():
            actual_counts[id(pool)] = actual_counts.get(id(pool), 0) + 1
    actor_model = next((model for model in topology.models if model.worker == "actor"), None)

    rows = [("CLUSTER", "POOL", "GPUS", "MODEL", "WORKER", "PROCESS", "MODE")]
    for model in topology.models:
        pool_spec = pool_specs[model.resource_pool]
        if model_resource_pools is None:
            placement = declared_placements[model.name]
            process = "colocated" if declared_counts[placement] > 1 else "dedicated"
            with_actor = actor_model is not None and placement == declared_placements[actor_model.name]
            pool = None
        else:
            pool = actual_pools[model.name]
            process = "colocated" if actual_counts[id(pool)] > 1 else "dedicated"
            with_actor = actor_model is not None and pool is actual_pools[actor_model.name]
        mode = "-"
        if model.worker == "rollout":
            mode = "HYBRID" if with_actor else "STANDALONE"
        elif model.worker in {"rm", "teacher"}:
            mode = "COLOCATED" if with_actor else "DEDICATED"
        rows.append(
            (
                pool_spec.cluster,
                model.resource_pool,
                _format_model_gpus(model, pool_spec, pool),
                model.name,
                model.worker,
                process,
                mode,
            )
        )
    return _format_table(rows)


def _format_table(rows: Sequence[tuple[str, ...]]) -> str:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return "\n".join("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)).rstrip() for row in rows)


def _format_model_gpus(
    model: Model,
    pool_spec: DevicePool,
    pool: ResourcePool | None,
) -> str:
    start, end = model.device_range or (0, pool_spec.n_gpus_per_node)
    if pool is not None and pool.processes_per_node != end - start:
        raise RuntimeError(
            f"compiled pool for model {model.name!r} has {pool.processes_per_node} processes per node; "
            f"expected {end - start} from device_range"
        )
    nnodes = pool_spec.nnodes if pool is None else pool.nnodes
    ranges = [
        (node * pool_spec.n_gpus_per_node + start, node * pool_spec.n_gpus_per_node + end) for node in range(nnodes)
    ]
    merged: list[tuple[int, int]] = []
    for current_start, current_end in ranges:
        if merged and merged[-1][1] == current_start:
            merged[-1] = (merged[-1][0], current_end)
        else:
            merged.append((current_start, current_end))
    return ",".join(
        str(range_start) if range_end - range_start == 1 else f"{range_start}-{range_end - 1}"
        for range_start, range_end in merged
    )


__all__ = ["Cluster", "DevicePool", "Model", "Topology"]
