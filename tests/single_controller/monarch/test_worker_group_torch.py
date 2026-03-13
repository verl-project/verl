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

import os

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["NCCL_DEBUG"] = "WARN"

import torch
import torch.distributed
from monarch.actor import endpoint, proc_mesh
from verl.single_controller.monarch.base import (
    MonarchClassWithInitArgs,
    MonarchResourcePool,
    MonarchWorker,
    MonarchWorkerGroup,
)


class TestAllGatherActor(MonarchWorker):
    def __init__(self, size) -> None:
        super().__init__()
        self.size = size

    @endpoint
    def init(self):
        torch.distributed.init_process_group()
        self.tensor = torch.zeros(size=(self.size,), dtype=torch.int64, device="cuda")
        self.tensor += self.rank

    @endpoint
    def all_gather(self):
        world_size = self._world_size
        output = torch.zeros(
            size=(self.tensor.shape[0] * world_size,),
            dtype=self.tensor.dtype,
            device=self.tensor.device,
        )
        torch.distributed.all_gather_into_tensor(output, self.tensor, async_op=False)
        return output


class TestAllGatherActorV2(MonarchWorker):
    def __init__(self, size) -> None:
        super().__init__()
        self.size = size

        torch.distributed.init_process_group()
        self.tensor = torch.zeros(size=(self.size,), dtype=torch.int64, device="cuda")
        self.tensor += self.rank

    @endpoint
    def all_gather(self):
        world_size = self._world_size
        output = torch.zeros(
            size=(self.tensor.shape[0] * world_size,),
            dtype=self.tensor.dtype,
            device=self.tensor.device,
        )
        torch.distributed.all_gather_into_tensor(output, self.tensor, async_op=False)
        return output


def test_all_gather_torch():
    """
    In this test, we instantiate 4 GPUs in a group and test the all_gather
    """

    # create 4 MonarchWorkers, each hold a GPU
    pm = proc_mesh(gpus=4).get()
    resource_pool = MonarchResourcePool(pm)
    class_with_args = MonarchClassWithInitArgs(cls=TestAllGatherActor, size=2)

    MonarchWorker_group = MonarchWorkerGroup(
        resource_pool, class_with_args, name_prefix="MonarchWorker_group_torch"
    )

    MonarchWorker_group.execute_all_sync("init")
    output = MonarchWorker_group.execute_all_sync("all_gather")
    for i in range(1, len(output)):
        assert torch.all(output[i] == output[0])

    output = output[0].cpu()
    print(output)
    assert torch.all(
        output == torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64)
    )
    # asyncio.run(pm.stop())


def test_all_gather_torch_v2():
    """
    In this test, we instantiate 4 GPUs in a group and test the all_gather
    """

    # create 4 MonarchWorkers, each hold a GPU
    pm = proc_mesh(gpus=4).get()
    resource_pool = MonarchResourcePool(pm)
    class_with_args = MonarchClassWithInitArgs(cls=TestAllGatherActorV2, size=2)

    MonarchWorker_group = MonarchWorkerGroup(
        resource_pool, class_with_args, name_prefix="MonarchWorker_group_torch"
    )

    output = MonarchWorker_group.execute_all_sync("all_gather")
    for i in range(1, len(output)):
        assert torch.all(output[i] == output[0])

    output = output[0].cpu()
    print(output)
    assert torch.all(
        output == torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64)
    )
    # TODO(papercut): this leads to a watchdog timeout
    # TODO(papercut): this needs a non-async variant
    # asyncio.run(pm.stop())


if __name__ == "__main__":
    test_all_gather_torch()
    # TODO(papercut) dist init doesn't really work
    # test_all_gather_torch_v2()
