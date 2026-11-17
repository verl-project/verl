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

import json
import os
import socket
import time
from contextlib import contextmanager

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["NCCL_DEBUG"] = "WARN"

import ray
import torch
import torch.distributed

from verl.single_controller.base.worker import Worker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.device import get_device_name, get_torch_device


def _log_collective_state(actor, stage, error=None):
    """Print local metadata without adding a collective or synchronizing the device."""
    if os.getenv("VERL_TEST_DISTRIBUTED_DEBUG") != "1":
        return
    try:
        record = {
            "time": time.time(),
            "stage": stage,
            "actor": type(actor).__name__,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "rank": actor.rank,
            "world_size": actor.world_size,
            "torch_version": torch.__version__,
            "ray_version": ray.__version__,
            "env": {
                key: os.getenv(key)
                for key in (
                    "RANK",
                    "LOCAL_RANK",
                    "WORLD_SIZE",
                    "RAY_LOCAL_WORLD_SIZE",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                    "CUDA_VISIBLE_DEVICES",
                    "ASCEND_VISIBLE_DEVICES",
                    "ASCEND_RT_VISIBLE_DEVICES",
                    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
                    "HCCL_IF_IP",
                    "HCCL_SOCKET_IFNAME",
                    "HCCL_PORT_RANGE",
                    "HCCL_CONNECT_TIMEOUT",
                    "HCCL_EXEC_TIMEOUT",
                )
            },
        }
        # A failed diagnostic query must not hide the original HCCL error.
        for name, probe in {
            "ray_accelerator_ids": lambda: ray.get_runtime_context().get_accelerator_ids(),
            "device_type": get_device_name,
            "current_device": lambda: (
                get_torch_device().current_device() if get_torch_device().is_initialized() else None
            ),
            "device_count": lambda: get_torch_device().device_count(),
            "backend": lambda: torch.distributed.get_backend() if torch.distributed.is_initialized() else None,
        }.items():
            try:
                record[name] = probe()
            except Exception as exc:
                record[name] = f"unavailable: {type(exc).__name__}: {exc}"
        tensor = getattr(actor, "tensor", None)
        if tensor is not None:
            record["tensor"] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device)}
        if error is not None:
            record["error"] = f"{type(error).__name__}: {error}"
        print("[VERL_ALL_GATHER_DIAGNOSTICS] " + json.dumps(record, default=str), flush=True)
    except Exception:
        # Diagnostics are best-effort, including when the worker is already failing.
        pass


@contextmanager
def _trace_collective_stage(actor, stage):
    _log_collective_state(actor, f"{stage}:begin")
    try:
        yield
    except Exception as exc:
        _log_collective_state(actor, f"{stage}:error", error=exc)
        raise
    else:
        _log_collective_state(actor, f"{stage}:success")


@ray.remote
class TestAllGatherActor(Worker):
    def __init__(self, size) -> None:
        super().__init__()
        self.size = size

    def init(self):
        with _trace_collective_stage(self, "init_process_group"):
            torch.distributed.init_process_group()
        with _trace_collective_stage(self, "create_input"):
            self.tensor = torch.zeros(size=(self.size,), dtype=torch.int64, device=get_device_name())
            self.tensor += self.rank

    def all_gather(self):
        world_size = self._world_size
        output = torch.zeros(
            size=(self.tensor.shape[0] * world_size,), dtype=self.tensor.dtype, device=self.tensor.device
        )
        with _trace_collective_stage(self, "all_gather_into_tensor"):
            torch.distributed.all_gather_into_tensor(output, self.tensor, async_op=False)
        return output


@ray.remote
class TestAllGatherActorV2(Worker):
    def __init__(self, size) -> None:
        super().__init__()
        self.size = size

        with _trace_collective_stage(self, "init_process_group"):
            torch.distributed.init_process_group()
        with _trace_collective_stage(self, "create_input"):
            self.tensor = torch.zeros(size=(self.size,), dtype=torch.int64, device=get_device_name())
            self.tensor += self.rank

    def all_gather(self):
        world_size = self._world_size
        output = torch.zeros(
            size=(self.tensor.shape[0] * world_size,), dtype=self.tensor.dtype, device=self.tensor.device
        )
        with _trace_collective_stage(self, "all_gather_into_tensor"):
            torch.distributed.all_gather_into_tensor(output, self.tensor, async_op=False)
        return output


def test_all_gather_torch():
    """
    Test all_gather with four workers by default, or the count requested by CI.
    """
    ray.init()

    world_size = int(os.getenv("VERL_TEST_N_GPUS_PER_NODE", "4"))
    resource_pool = RayResourcePool([world_size], use_gpu=True)
    class_with_args = RayClassWithInitArgs(cls=TestAllGatherActor, size=2)

    worker_group = RayWorkerGroup(
        resource_pool, class_with_args, name_prefix="worker_group_torch", device_name=get_device_name()
    )

    worker_group.execute_all_sync("init")
    # Each result comes back on its worker's device; those indices only coincide when the runtime masks
    # every actor down to device 0, so compare on the host.
    output = [o.cpu() for o in worker_group.execute_all_sync("all_gather")]
    for i in range(1, len(output)):
        assert torch.all(output[i] == output[0])

    output = output[0]
    print(output)
    assert torch.all(output == torch.arange(world_size, dtype=torch.int64).repeat_interleave(2))

    ray.shutdown()


def test_all_gather_torch_v2():
    """
    Test all_gather with four workers by default, or the count requested by CI.
    """
    ray.init()

    world_size = int(os.getenv("VERL_TEST_N_GPUS_PER_NODE", "4"))
    resource_pool = RayResourcePool([world_size], use_gpu=True)
    class_with_args = RayClassWithInitArgs(cls=TestAllGatherActorV2, size=2)

    worker_group = RayWorkerGroup(
        resource_pool, class_with_args, name_prefix="worker_group_torch", device_name=get_device_name()
    )

    output = [o.cpu() for o in worker_group.execute_all_sync("all_gather")]
    for i in range(1, len(output)):
        assert torch.all(output[i] == output[0])

    output = output[0]
    print(output)
    assert torch.all(output == torch.arange(world_size, dtype=torch.int64).repeat_interleave(2))

    ray.shutdown()
