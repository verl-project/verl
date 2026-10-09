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

"""Backend-neutral foundational Runtime types and helpers."""

from __future__ import annotations

from verl.single_controller.base.actor import ClassWithInitArgs, WorkerContainer
from verl.single_controller.base.decorator import Dispatch, Execute, make_nd_compute_dataproto_dispatch_fn, register
from verl.single_controller.base.errors import (
    PlacementUnavailableError,
    RPCError,
    RPCRemoteError,
    RPCTimeoutError,
    RPCTransportError,
    RPCUnavailableError,
)
from verl.single_controller.base.remote_call import RemoteCall
from verl.single_controller.base.remote_worker_group import RemoteWorkerGroup
from verl.single_controller.base.resource_pool import ResourcePool
from verl.single_controller.base.worker import DistGlobalInfo, DistRankInfo, Worker, WorkerHelper
from verl.single_controller.base.worker_group import WorkerGroup

__all__ = [
    "ClassWithInitArgs",
    "Dispatch",
    "DistGlobalInfo",
    "DistRankInfo",
    "Execute",
    "PlacementUnavailableError",
    "RPCError",
    "RPCRemoteError",
    "RPCTimeoutError",
    "RPCTransportError",
    "RPCUnavailableError",
    "RemoteCall",
    "RemoteWorkerGroup",
    "ResourcePool",
    "Worker",
    "WorkerContainer",
    "WorkerGroup",
    "WorkerHelper",
    "make_nd_compute_dataproto_dispatch_fn",
    "register",
]
