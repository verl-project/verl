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
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device


def _broadcast_preprocess(mesh_name: str):
    def preprocess(worker, *args, **kwargs):
        is_mp_src = worker._query_dispatch_info(mesh_name).is_mp_src
        values = (*args, *kwargs.values())
        if is_mp_src:
            assert values and all(value is not None for value in values), (
                "MP source rank must receive the dispatched shard"
            )
        else:
            assert values and all(value is None for value in values), (
                "Non-source ranks must receive None before the model-parallel broadcast"
            )
        return worker.broadcast_data(mesh_name, *args, **kwargs)

    return preprocess


def _double_postprocess(mesh_name: str):
    """Double the method output so the collected value shows the hook ran after compute."""

    def postprocess(worker, output):
        worker._query_dispatch_info(mesh_name)
        if isinstance(output, DataProto):
            output.batch["a"] *= 2
            return output
        key = "a" if "a" in output else "input_ids"
        output[key] *= 2
        return output

    return postprocess


@ray.remote
class TestActor(Worker):
    def __init__(self):
        super().__init__()

        import torch.distributed

        torch.distributed.init_process_group(backend=get_nccl_backend())
        ngpus = torch.distributed.get_world_size()
        self.infer_device_mesh = torch.distributed.device_mesh.init_device_mesh(
            device_type=get_device_name(), mesh_shape=[2, ngpus // 2], mesh_dim_names=["dp", "tp"]
        )
        self.train_device_mesh = torch.distributed.device_mesh.init_device_mesh(
            device_type=get_device_name(), mesh_shape=[2, 2, ngpus // 4], mesh_dim_names=["pp", "dp", "tp"]
        )

        self._register_dispatch_collect_info(
            "infer",
            dp_rank=self.infer_device_mesh["dp"].get_local_rank(),
            is_collect=self.infer_device_mesh["tp"].get_local_rank() == 0,
            is_mp_src=self.infer_device_mesh["tp"].get_local_rank() == 0,
        )
        self._register_dispatch_collect_info(
            "train",
            dp_rank=self.train_device_mesh["dp"].get_local_rank(),
            is_collect=self.train_device_mesh["tp"].get_local_rank() == 0
            and self.train_device_mesh["pp"].get_local_rank() == 1,
            is_mp_src=self.train_device_mesh["pp"].get_local_rank() == 0
            and self.train_device_mesh["tp"].get_local_rank() == 0,
        )

    def broadcast_data(self, mesh_name: str, *args, **kwargs):
        """Broadcast arguments from the model-parallel source across existing mesh groups."""
        import torch.distributed as dist

        mesh = self.infer_device_mesh if mesh_name == "infer" else self.train_device_mesh

        def _broadcast(value, group):
            device = torch.device(get_device_name(), get_torch_device().current_device())
            objects = [value]
            dist.broadcast_object_list(objects, group=group, group_src=0, device=device)
            return objects[0]

        for dim_name in mesh.mesh_dim_names:
            if dim_name == "dp":
                continue
            group = mesh.get_group(dim_name)
            args = tuple(_broadcast(arg, group) for arg in args)
            kwargs = {key: _broadcast(value, group) for key, value in kwargs.items()}
        return args, kwargs

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

    @register(
        dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="infer", dispatch_mp_src_only=True),
        worker_preprocess_fn=_broadcast_preprocess("infer"),
    )
    def generate_data_proto_mp_src(self, data: DataProto):
        return self.generate_data_proto(data)

    @register(
        dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="infer", dispatch_mp_src_only=True),
        worker_preprocess_fn=_broadcast_preprocess("infer"),
    )
    def generate_tensordict_mp_src(self, data: TensorDict):
        return self.generate_tensordict(data)

    @register(
        dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train", dispatch_mp_src_only=True),
        worker_preprocess_fn=_broadcast_preprocess("train"),
    )
    def train_data_proto_mp_src(self, data: DataProto):
        return self.train_data_proto(data)

    @register(
        dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train", dispatch_mp_src_only=True),
        worker_preprocess_fn=_broadcast_preprocess("train"),
    )
    def train_tensordict_mp_src(self, data: TensorDict):
        return self.train_tensordict(data)

    @register(
        dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="infer", dispatch_mp_src_only=True),
        worker_preprocess_fn=_broadcast_preprocess("infer"),
    )
    def generate_nested_tensor_mp_src(self, data: TensorDict):
        return self.generate_nested_tensor(data)

    @register(
        dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="infer", dispatch_mp_src_only=True),
        worker_preprocess_fn=_broadcast_preprocess("infer"),
        worker_postprocess_fn=_double_postprocess("infer"),
    )
    def generate_data_proto_mp_src_postprocess(self, data: DataProto):
        return self.generate_data_proto(data)

    @register(
        dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="infer", dispatch_mp_src_only=True),
        worker_preprocess_fn=_broadcast_preprocess("infer"),
        worker_postprocess_fn=_double_postprocess("infer"),
    )
    def generate_tensordict_mp_src_postprocess(self, data: TensorDict):
        return self.generate_tensordict(data)

    @register(
        dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train", dispatch_mp_src_only=True),
        worker_preprocess_fn=_broadcast_preprocess("train"),
        worker_postprocess_fn=_double_postprocess("train"),
    )
    def train_data_proto_mp_src_postprocess(self, data: DataProto):
        return self.train_data_proto(data)

    @register(
        dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train", dispatch_mp_src_only=True),
        worker_preprocess_fn=_broadcast_preprocess("train"),
        worker_postprocess_fn=_double_postprocess("train"),
    )
    def train_tensordict_mp_src_postprocess(self, data: TensorDict):
        return self.train_tensordict(data)

    @register(
        dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="infer", dispatch_mp_src_only=True),
        worker_preprocess_fn=_broadcast_preprocess("infer"),
        worker_postprocess_fn=_double_postprocess("infer"),
    )
    def generate_nested_tensor_mp_src_postprocess(self, data: TensorDict):
        return self.generate_nested_tensor(data)


def test_dist_global_info_wg():
    # create a worker group with size 8
    # register a infer dist info with tp=4, dp=2
    # register a train dist info with tp=2, dp=2, pp=2
    # test the correctness of data dispatch and computation
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

    ray.init()
    ngpus = get_torch_device().device_count()
    infer_tp = ngpus // 2
    train_tp = ngpus // 4

    ray_cls = RayClassWithInitArgs(TestActor)
    resource_pool = RayResourcePool(process_on_nodes=[ngpus])
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls, device_name=get_device_name())

    infer_input_data_proto = DataProto.from_single_dict(data={"a": torch.tensor([1, 2])})
    infer_output_data_proto = wg.generate_data_proto(infer_input_data_proto)

    expected_infer_dispatch = [d for d in range(2) for _ in range(infer_tp)]
    assert [info.dp_rank for info in wg._dispatch_info["infer"]] == expected_infer_dispatch

    assert torch.all(torch.eq(infer_output_data_proto.batch["a"], torch.tensor([1, 3])))
    infer_output_data_proto_mp_src = wg.generate_data_proto_mp_src(infer_input_data_proto)
    assert torch.all(torch.eq(infer_output_data_proto_mp_src.batch["a"], torch.tensor([1, 3])))
    # Doubling after compute yields [2, 6]. Doubling first would yield [2, 5].
    infer_output_data_proto_post = wg.generate_data_proto_mp_src_postprocess(infer_input_data_proto)
    assert torch.all(torch.eq(infer_output_data_proto_post.batch["a"], torch.tensor([2, 6])))

    infer_input_tensordict = infer_input_data_proto.to_tensordict()
    infer_output_tensordict = wg.generate_tensordict(infer_input_tensordict)
    assert torch.all(torch.eq(infer_output_tensordict["a"], torch.tensor([1, 3])))
    infer_output_tensordict_mp_src = wg.generate_tensordict_mp_src(infer_input_tensordict)
    assert torch.all(torch.eq(infer_output_tensordict_mp_src["a"], torch.tensor([1, 3])))
    infer_output_tensordict_post = wg.generate_tensordict_mp_src_postprocess(infer_input_tensordict)
    assert torch.all(torch.eq(infer_output_tensordict_post["a"], torch.tensor([2, 6])))

    train_input_data_proto = DataProto.from_single_dict(data={"a": torch.tensor([3, 4])})
    train_output_data_proto = wg.train_data_proto(train_input_data_proto)

    expected_train_dispatch = [d for _ in range(2) for d in range(2) for _ in range(train_tp)]
    assert [info.dp_rank for info in wg._dispatch_info["train"]] == expected_train_dispatch

    assert torch.all(torch.eq(train_output_data_proto.batch["a"], torch.tensor([11, 16])))
    train_output_data_proto_mp_src = wg.train_data_proto_mp_src(train_input_data_proto)
    assert torch.all(torch.eq(train_output_data_proto_mp_src.batch["a"], torch.tensor([11, 16])))
    # Train collection uses pp rank 1, while the source is pp rank 0.
    train_output_data_proto_post = wg.train_data_proto_mp_src_postprocess(train_input_data_proto)
    assert torch.all(torch.eq(train_output_data_proto_post.batch["a"], torch.tensor([22, 32])))

    train_input_tensordict = train_input_data_proto.to_tensordict()
    train_output_tensordict = wg.train_tensordict(train_input_tensordict)
    assert torch.all(torch.eq(train_output_tensordict["a"], torch.tensor([11, 16])))
    train_output_tensordict_mp_src = wg.train_tensordict_mp_src(train_input_tensordict)
    assert torch.all(torch.eq(train_output_tensordict_mp_src["a"], torch.tensor([11, 16])))
    train_output_tensordict_post = wg.train_tensordict_mp_src_postprocess(train_input_tensordict)
    assert torch.all(torch.eq(train_output_tensordict_post["a"], torch.tensor([22, 32])))

    # create a batch size of input_ids
    input_ids = [
        torch.randint(low=0, high=128, size=(np.random.randint(low=1, high=10, dtype=np.int64),)) for _ in range(16)
    ]
    input_ids = torch.nested.as_nested_tensor(input_ids, layout=torch.jagged)
    data = tu.get_tensordict(tensor_dict={"input_ids": input_ids})
    output = wg.generate_nested_tensor(data)
    output_mp_src = wg.generate_nested_tensor_mp_src(data)

    input_ids_chunked = list(input_ids.clone().chunk(2))

    print(input_ids_chunked)

    input_ids_chunked[0] += 0
    input_ids_chunked[1] += 1

    expected = tu.concat_nested_tensors(input_ids_chunked)

    assert torch.all(torch.eq(output["input_ids"].values(), expected.values()))
    assert torch.all(torch.eq(output_mp_src["input_ids"].values(), expected.values()))
    output_post = wg.generate_nested_tensor_mp_src_postprocess(data)
    assert torch.all(torch.eq(output_post["input_ids"].values(), expected.values() * 2))

    ray.shutdown()


if __name__ == "__main__":
    test_dist_global_info_wg()
