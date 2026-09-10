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

"""Public backend-neutral Runtime and ObjectStore API."""

from __future__ import annotations

from verl.runtime.config import RuntimeConfig, parse_env_vars, select_backend
from verl.runtime.core import Runtime, current_runtime
from verl.runtime.object_store import delete, delete_many, get, get_many, put, put_many
from verl.single_controller.base.actor import ClassWithInitArgs
from verl.single_controller.base.decorator import Dispatch, Execute, make_nd_compute_dataproto_dispatch_fn, register
from verl.single_controller.base.errors import (
    ExceptionGroup,
    PlacementUnavailableError,
    RPCError,
    RPCRemoteError,
    RPCTimeoutError,
    RPCTransportError,
    RPCUnavailableError,
)
from verl.single_controller.base.remote_call import RemoteCall
from verl.single_controller.base.remote_worker_group import RemoteWorkerGroup
from verl.single_controller.base.resource_pool import ResourcePool, split_resource_pool
from verl.single_controller.base.topology import Cluster, DevicePool, Model, Topology
from verl.single_controller.base.worker import Worker
from verl.single_controller.base.worker_group import WorkerGroup

__all__ = [
    "ClassWithInitArgs",
    "Cluster",
    "DevicePool",
    "Dispatch",
    "Execute",
    "ExceptionGroup",
    "Model",
    "PlacementUnavailableError",
    "RPCError",
    "RPCRemoteError",
    "RPCTimeoutError",
    "RPCTransportError",
    "RPCUnavailableError",
    "RemoteCall",
    "RemoteWorkerGroup",
    "ResourcePool",
    "Runtime",
    "RuntimeConfig",
    "Topology",
    "Worker",
    "WorkerGroup",
    "current_runtime",
    "delete",
    "delete_many",
    "get",
    "get_many",
    "make_nd_compute_dataproto_dispatch_fn",
    "put",
    "put_many",
    "parse_env_vars",
    "register",
    "select_backend",
    "split_resource_pool",
]
