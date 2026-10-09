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

"""PPO role-to-ResourcePool composition over the common Runtime contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from verl.runtime import ResourcePool, current_runtime
from verl.trainer.ppo.utils import Role


class ResourcePoolManagerProtocol(Protocol):
    """Resource-pool operations consumed by the legacy PPO trainer."""

    resource_pool_dict: dict[str, ResourcePool]

    def create_resource_pool(self) -> None: ...

    def get_resource_pool(self, role: Role) -> ResourcePool: ...

    def get_n_gpus(self) -> int: ...


@dataclass
class RuntimeResourcePoolManager:
    """Create named GPU pools through the owning Runtime.

    ``resource_pool_spec`` is used only by the legacy path. A non-empty Runtime
    topology already owns concrete DevicePools and model views.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, ResourcePool] = field(default_factory=dict)

    def create_resource_pool(self) -> None:
        if self.resource_pool_dict:
            return
        runtime = current_runtime()
        if runtime.topology.models:
            for model in runtime.topology.models:
                self.resource_pool_dict[model.name] = runtime.model_resource_pool(model.name)
            return
        for name, processes_per_node in self.resource_pool_spec.items():
            if not processes_per_node or len(set(processes_per_node)) != 1:
                raise ValueError(
                    f"resource pool {name!r} must use one homogeneous processes_per_node value; "
                    f"got {processes_per_node!r}"
                )
            self.resource_pool_dict[name] = runtime.create_resource_pool(
                nnodes=len(processes_per_node),
                processes_per_node=processes_per_node[0],
                device_type="gpu",
            )

    def get_resource_pool(self, role: Role) -> ResourcePool:
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        runtime = current_runtime()
        if runtime.topology.models:
            return sum(pool.nnodes * pool.n_gpus_per_node for pool in runtime.topology.device_pools)
        return sum(count for per_host in self.resource_pool_spec.values() for count in per_host)


__all__ = ["ResourcePoolManagerProtocol", "RuntimeResourcePoolManager"]
