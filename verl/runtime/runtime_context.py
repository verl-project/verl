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

"""Backend-injected identity for one process-local Runtime tree node."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from verl.runtime.config import RuntimeConfig, select_backend

if TYPE_CHECKING:
    from verl.single_controller.base.resource_pool import ResourcePool
    from verl.single_controller.base.topology import Topology

_RUNTIME_CONTEXT: RuntimeContext | None = None


@dataclass(frozen=True, slots=True)
class AttachSpec:
    """Everything a backend-spawned process needs to build its Runtime node.

    The parent assembles this once per WorkerGroup and the backend delivers it
    as a single spawn payload: the child's identity (``runtime_id``,
    ``node_path``), the static ``config`` used to rebuild the backend, and the
    runtime state the root already decided (``topology``, ``resource_pools``).
    ``object_store_config`` carries the root backend's client configuration so
    every attached process starts the same process-global ObjectStore.
    """

    runtime_id: str
    node_path: tuple[str, ...]
    config: RuntimeConfig
    topology: Topology
    resource_pools: Mapping[str, ResourcePool]
    object_store_config: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Serializable identity for one Runtime tree node."""

    runtime_id: str
    node_path: tuple[str, ...]
    config: RuntimeConfig
    object_store_config: dict[str, object] = field(default_factory=dict, compare=False, hash=False)

    def child_attach_spec(
        self,
        *,
        node_path: tuple[str, ...],
        topology: Topology,
        resource_pools: Mapping[str, ResourcePool],
    ) -> AttachSpec:
        """Build the backend-serialized state for one child Runtime node."""
        return AttachSpec(
            runtime_id=self.runtime_id,
            node_path=node_path,
            config=self.config,
            topology=topology,
            resource_pools=resource_pools,
            object_store_config=dict(self.object_store_config),
        )

    @property
    def backend(self) -> str:
        return select_backend(self.config)


def runtime_context() -> RuntimeContext:
    """Return the initialized process RuntimeContext."""
    if _RUNTIME_CONTEXT is None:
        raise RuntimeError("RuntimeContext is not initialized in this process")
    return _RUNTIME_CONTEXT


def install_runtime_context(context: RuntimeContext) -> RuntimeContext:
    global _RUNTIME_CONTEXT
    current = _RUNTIME_CONTEXT
    if current is not None:
        if current != context:
            raise RuntimeError(
                f"process already belongs to Runtime node {'/'.join(current.node_path)!r}; "
                f"cannot attach {'/'.join(context.node_path)!r}"
            )
        return current
    _RUNTIME_CONTEXT = context
    return context


def clear_runtime_context(expected: RuntimeContext) -> None:
    global _RUNTIME_CONTEXT
    if _RUNTIME_CONTEXT is not None and _RUNTIME_CONTEXT == expected:
        _RUNTIME_CONTEXT = None


__all__ = ["AttachSpec", "RuntimeContext", "runtime_context"]
