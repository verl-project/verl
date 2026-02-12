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
"""
In this test, we instantiate a data parallel worker with 8 GPUs
"""

import tensordict
import torch
from codetiming import Timer
from monarch.actor import endpoint, proc_mesh
from torch import distributed as dist
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.monarch.base import (
    get,
    MonarchClassWithInitArgs,
    MonarchResourcePool,
    MonarchWorker,
    MonarchWorkerGroup,
)


class DummyWorker(MonarchWorker):
    def __init__(self):
        super().__init__()
        dist.init_process_group()

    @endpoint
    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def do_nothing(self, data):
        for key in data.batch.keys():
            data.batch[key] += 1
        if tensordict.__version__ >= "0.5.0":
            data.batch = data.batch.consolidate()
        return data


def test_data_transfer():
    # construct resource pool
    pm = proc_mesh(gpus=8).get()
    resource_pool = MonarchResourcePool(pm)
    cls_with_init = MonarchClassWithInitArgs(cls=DummyWorker)
    # construct worker group
    wg = MonarchWorkerGroup(resource_pool, cls_with_init)

    # this is real dataset size
    batch_size = 4096
    seqlen = 32768

    data_dict = {}

    for i in range(2):
        data_dict[str(i)] = torch.randint(0, 10000, (batch_size, seqlen))

    data = DataProto.from_dict(tensors=data_dict)

    print(data)

    # we manually split data here and send to each worker
    data_list = data.chunk(wg.world_size)

    for i in range(wg.world_size):
        # consolidate is necessary
        if tensordict.__version__ >= "0.5.0":
            data_list[i].batch = data_list[i].batch.consolidate()

    with Timer(name="monarch.serialize", initial_text=True):
        # Note: Monarch may use different serialization mechanism
        import pickle

        for i in range(wg.world_size):
            pickle.dumps(data_list[i])

    with Timer(name="raw.pickle", initial_text=True):
        import pickle

        for i in range(wg.world_size):
            pickle.dumps(data_list[i])

    # Note: Monarch handles data transfer differently than Ray's put/get
    with Timer(name="launch", initial_text=True):
        output_ref = wg.do_nothing(data_list)

    with Timer(name="get", initial_text=True):
        output_lst = get(output_ref)

    for input_data, output_data in zip(data_list, output_lst):
        for key in input_data.batch.keys():
            assert torch.all(
                torch.eq(input_data.batch[key] + 1, output_data.batch[key])
            ), (
                input_data.batch[key],
                output_data.batch[key],
                key,
            )
