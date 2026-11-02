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

"""Collective group adapters used by the NCCL checkpoint engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from threading import Lock

import ray.util.collective as ray_collective
import torch

from verl.utils.device import get_device_id


class NCCLCollectiveGroup(ABC):
    """Small collective surface required by the checkpoint engines."""

    requires_torch_tensor = False

    def __init__(self, group_name: str):
        if not group_name:
            raise ValueError("collective group_name must be non-empty")
        self.group_name = group_name

    @property
    @abstractmethod
    def initialized(self) -> bool:
        """Whether this process has initialized the group."""

    @abstractmethod
    def initialize(self, world_size: int, rank: int, master_addr: str, master_port: int | None):
        """Initialize this rank in the collective group."""

    @abstractmethod
    def broadcast(self, tensor, src_rank: int = 0):
        """Broadcast one contiguous tensor."""

    @abstractmethod
    def barrier(self):
        """Wait until every rank has initialized the group."""

    @abstractmethod
    def destroy(self):
        """Release group resources owned by this process."""


class RayNCCLCollectiveGroup(NCCLCollectiveGroup):
    """Existing Ray/CuPy NCCL implementation."""

    @property
    def initialized(self) -> bool:
        return ray_collective.is_group_initialized(self.group_name)

    def initialize(self, world_size: int, rank: int, master_addr: str, master_port: int | None):
        del master_addr, master_port
        ray_collective.init_collective_group(world_size, rank, "nccl", self.group_name)

    def broadcast(self, tensor, src_rank: int = 0):
        ray_collective.broadcast(tensor, src_rank=src_rank, group_name=self.group_name)

    def barrier(self):
        ray_collective.barrier(self.group_name)

    def destroy(self):
        if self.initialized:
            ray_collective.destroy_collective_group(self.group_name)


@dataclass
class _TorchNCCLGroupState:
    store: object
    process_group: object


class TorchNCCLCollectiveGroup(NCCLCollectiveGroup):
    """Named ProcessGroupNCCL implementation for environments without a compatible CuPy NCCL binding."""

    requires_torch_tensor = True
    _groups: dict[str, _TorchNCCLGroupState] = {}
    _groups_lock = Lock()

    def __init__(self, group_name: str, timeout_s: float = 300):
        super().__init__(group_name)
        if timeout_s <= 0:
            raise ValueError(f"torch_nccl timeout must be positive, got {timeout_s}")
        self.timeout = timedelta(seconds=timeout_s)

    @property
    def initialized(self) -> bool:
        with self._groups_lock:
            return self.group_name in self._groups

    def _state(self) -> _TorchNCCLGroupState:
        with self._groups_lock:
            state = self._groups.get(self.group_name)
        if state is None:
            raise RuntimeError(f"collective group {self.group_name!r} is not initialized")
        return state

    def initialize(self, world_size: int, rank: int, master_addr: str, master_port: int | None):
        if master_port is None:
            raise ValueError("torch_nccl requires a TCPStore port")
        if self.initialized:
            raise RuntimeError(f"collective group {self.group_name!r} is already initialized")

        tcp_store = torch.distributed.TCPStore(
            master_addr,
            master_port,
            world_size,
            rank == 0,
            self.timeout,
        )
        store = torch.distributed.PrefixStore(self.group_name, tcp_store)
        options = torch.distributed.ProcessGroupNCCL.Options()
        options._timeout = self.timeout
        process_group = torch.distributed.ProcessGroupNCCL(store, rank, world_size, options)
        eager_connect = getattr(process_group, "eager_connect_single_device", None)
        if eager_connect is not None:
            eager_connect(torch.device("cuda", get_device_id()))

        with self._groups_lock:
            if self.group_name in self._groups:
                process_group.abort()
                raise RuntimeError(f"collective group {self.group_name!r} was initialized concurrently")
            self._groups[self.group_name] = _TorchNCCLGroupState(store=store, process_group=process_group)

    def _check_tensor(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"torch_nccl requires torch.Tensor, got {type(tensor)!r}")
        if tensor.device.type == "cpu":
            raise ValueError(f"torch_nccl requires an accelerator tensor, got device {tensor.device}")

    def broadcast(self, tensor, src_rank: int = 0):
        self._check_tensor(tensor)
        options = torch.distributed.BroadcastOptions()
        options.rootRank = src_rank
        options.rootTensor = 0
        self._state().process_group.broadcast([tensor], options).wait()

    def barrier(self):
        options = torch.distributed.BarrierOptions()
        options.device_ids = [get_device_id()]
        self._state().process_group.barrier(options).wait()

    def destroy(self):
        with self._groups_lock:
            state = self._groups.pop(self.group_name, None)
        if state is not None:
            state.process_group.abort()


def create_nccl_collective_group(
    backend: str, group_name: str, backend_kwargs: dict | None = None
) -> NCCLCollectiveGroup:
    """Create a checkpoint collective without changing the Ray default path."""

    backend_kwargs = dict(backend_kwargs or {})
    if backend == "ray_nccl":
        if backend_kwargs:
            raise ValueError(f"ray_nccl does not accept backend kwargs: {sorted(backend_kwargs)}")
        return RayNCCLCollectiveGroup(group_name)
    if backend == "torch_nccl":
        unknown = backend_kwargs.keys() - {"timeout_s"}
        if unknown:
            raise ValueError(f"torch_nccl received unsupported backend kwargs: {sorted(unknown)}")
        return TorchNCCLCollectiveGroup(group_name, **backend_kwargs)
    raise ValueError(f"unsupported NCCL collective backend {backend!r}; expected 'ray_nccl' or 'torch_nccl'")
