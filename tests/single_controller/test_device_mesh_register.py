# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import numpy as np
import ray
import torch
from tensordict import TensorDict

import verl.utils.tensordict_utils as tu
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_device_name, get_nccl_backend


@ray.remote
class TestActor(Worker):
    def __init__(self):
        super().__init__()

        import torch.distributed

        torch.distributed.init_process_group(backend=get_nccl_backend())
        self.infer_device_mesh = torch.distributed.device_mesh.init_device_mesh(
            device_type=get_device_name(), mesh_shape=[2, 4], mesh_dim_names=["dp", "tp"]
        )
        self.train_device_mesh = torch.distributed.device_mesh.init_device_mesh(
            device_type=get_device_name(), mesh_shape=[2, 2, 2], mesh_dim_names=["pp", "dp", "tp"]
        )

        self._register_dispatch_collect_info(
            "infer",
            dp_rank=self.infer_device_mesh["dp"].get_local_rank(),
            is_collect=self.infer_device_mesh["tp"].get_local_rank() == 0,
        )
        self._register_dispatch_collect_info(
            "train",
            dp_rank=self.train_device_mesh["dp"].get_local_rank(),
            is_collect=self.train_device_mesh["tp"].get_local_rank() == 0
            and self.train_device_mesh["pp"].get_local_rank() == 1,
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="infer"))
    def generate_data_proto(self, data: DataProto):
        tp_rank = self.infer_device_mesh["tp"].get_local_rank()
        dp_rank = self.infer_device_mesh["dp"].get_local_rank()
        data.batch["a"] += (tp_rank + 1) * dp_rank
        return data

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="infer"))
    def generate_tensordict(self, data: TensorDict):
        tp_rank = self.infer_device_mesh["tp"].get_local_rank()
        dp_rank = self.infer_device_mesh["dp"].get_local_rank()
        data["a"] += (tp_rank + 1) * dp_rank
        return data

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"))
    def train_data_proto(self, data: DataProto):
        tp_rank = self.train_device_mesh["tp"].get_local_rank()
        dp_rank = self.train_device_mesh["dp"].get_local_rank()
        pp_rank = self.train_device_mesh["pp"].get_local_rank()
        data.batch["a"] += (tp_rank + 1) * (dp_rank + 2) * (pp_rank + 3)
        # tp rank 0, pp rank 1, dp rank 0, output data added: 8 + 3 = 11
        # tp rank 0, pp rank 1, dp rank 1, output data added: 12 + 4 = 16
        return data

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"))
    def train_tensordict(self, data: TensorDict):
        tp_rank = self.train_device_mesh["tp"].get_local_rank()
        dp_rank = self.train_device_mesh["dp"].get_local_rank()
        pp_rank = self.train_device_mesh["pp"].get_local_rank()
        data["a"] += (tp_rank + 1) * (dp_rank + 2) * (pp_rank + 3)
        # tp rank 0, pp rank 1, dp rank 0, output data added: 8 + 3 = 11
        # tp rank 0, pp rank 1, dp rank 1, output data added: 12 + 4 = 16
        return data

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="infer"))
    def generate_nested_tensor(self, data: TensorDict):
        tp_rank = self.infer_device_mesh["tp"].get_local_rank()
        dp_rank = self.infer_device_mesh["dp"].get_local_rank()
        assert data.shape[0] == 8
        data["input_ids"] += tp_rank + dp_rank

        print(data)
        return data


def test_dist_global_info_wg():
    # create a worker group with size 8
    # register a infer dist info with tp=4, dp=2
    # register a train dist info with tp=2, dp=2, pp=2
    # test the correctness of data dispatch and computation
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

    ray.init()

    ray_cls = RayClassWithInitArgs(TestActor)
    resource_pool = RayResourcePool(process_on_nodes=[8])
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls, device_name=get_device_name())

    infer_input_data_proto = DataProto.from_single_dict(data={"a": torch.tensor([1, 2])})
    infer_output_data_proto = wg.generate_data_proto(infer_input_data_proto)

    assert wg._dispatch_info["infer"] == [0, 0, 0, 0, 1, 1, 1, 1]

    assert torch.all(torch.eq(infer_output_data_proto.batch["a"], torch.tensor([1, 3])))

    infer_input_tensordict = infer_input_data_proto.to_tensordict()
    infer_output_tensordict = wg.generate_tensordict(infer_input_tensordict)
    assert torch.all(torch.eq(infer_output_tensordict["a"], torch.tensor([1, 3])))

    train_input_data_proto = DataProto.from_single_dict(data={"a": torch.tensor([3, 4])})
    train_output_data_proto = wg.train_data_proto(train_input_data_proto)

    assert wg._dispatch_info["train"] == [0, 0, 1, 1, 0, 0, 1, 1]

    assert torch.all(torch.eq(train_output_data_proto.batch["a"], torch.tensor([11, 16])))

    train_input_tensordict = train_input_data_proto.to_tensordict()
    train_output_tensordict = wg.train_tensordict(train_input_tensordict)
    assert torch.all(torch.eq(train_output_tensordict["a"], torch.tensor([11, 16])))

    # create a batch size of input_ids
    input_ids = [
        torch.randint(low=0, high=128, size=(np.random.randint(low=1, high=10, dtype=np.int64),)) for _ in range(16)
    ]
    input_ids = torch.nested.as_nested_tensor(input_ids, layout=torch.jagged)
    data = tu.get_tensordict(tensor_dict={"input_ids": input_ids})
    output = wg.generate_nested_tensor(data)

    input_ids_chunked = list(input_ids.chunk(2))

    print(input_ids_chunked)

    input_ids_chunked[0] += 0
    input_ids_chunked[1] += 1

    expected = tu.concat_nested_tensors(input_ids_chunked)

    assert torch.all(torch.eq(output["input_ids"].values(), expected.values()))

    ray.shutdown()


if __name__ == "__main__":
    test_dist_global_info_wg()
