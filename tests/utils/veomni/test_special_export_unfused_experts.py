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

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer

from verl.workers.engine.veomni.utils import MOE_PARAM_HANDERS


def get_by_path(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def set_by_path(obj, path: str, value):
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def get_per_tensor_param(model, device_mesh):
    ep_rank, ep_size, ep_group = (
        device_mesh["ep"].get_local_rank(),
        device_mesh["ep"].size(),
        device_mesh["ep"].get_group(),
    )
    process_func = MOE_PARAM_HANDERS["qwen3_5_moe"]

    for name, param in model.named_parameters():
        if not isinstance(param, DTensor):
            continue
        unsharded_tensor = param.full_tensor()
        buffer = torch.empty_like(unsharded_tensor)  # [num_experts/ep_size, H, I]
        for src_ep_rank in range(ep_size):
            tensor = unsharded_tensor if src_ep_rank == ep_rank else buffer
            torch.distributed.broadcast(tensor, group_src=src_ep_rank, group=ep_group)
            yield from process_func(name, tensor, ep_rank=src_ep_rank)


def test_veomni_export_unfused_experts():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # init process group
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", init_method=f"tcp://{master_addr}:{master_port}", world_size=world_size, rank=rank
    )

    ep_device_mesh = dist.device_mesh.init_device_mesh("cuda", mesh_shape=(2, 4), mesh_dim_names=["efsdp", "ep"])

    config = Qwen3_5MoeTextConfig()
    layer = Qwen3_5MoeDecoderLayer(config, layer_idx=0)
    expert_param_names = [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]

    # 0. random init experts
    for param_name in expert_param_names:
        param = get_by_path(layer, param_name)
        param.data.normal_(mean=0.0, std=0.01)
    state_dict = {k: v.clone() for k, v in layer.state_dict().items()}

    # 1. split experts by ep_size in dim=0
    for param_name in expert_param_names:
        param = get_by_path(layer, param_name)
        dtensor = distribute_tensor(
            param, device_mesh=ep_device_mesh["ep"], placements=(Shard(dim=0),), src_data_rank=None
        )
        local_tensor = dtensor.to_local()
        local_param = torch.nn.Parameter(local_tensor, requires_grad=param.requires_grad)
        set_by_path(layer, param_name, local_param)

    # 2. fully shard local experts by ep_fsdp_mesh in dim=1
    ep_mesh_dim_names = ep_device_mesh.mesh_dim_names
    ep_fsdp_mesh = ep_device_mesh[ep_mesh_dim_names[:-1]]
    shard_placement_fn = lambda x: Shard(dim=1)
    fully_shard(layer.mlp.experts, mesh=ep_fsdp_mesh, shard_placement_fn=shard_placement_fn, reshard_after_forward=True)

    exported_params = {}
    for name, param in get_per_tensor_param(layer, ep_device_mesh):
        if dist.get_rank() == 0:
            print(f"rank: {dist.get_rank()}, name: {name}, param: {param.shape}")
        exported_params[name] = param.clone()

    gate, up, down = [], [], []
    for name, param in exported_params.items():
        # mlp.experts.{idx}.down_proj.weight
        idx = int(name.split(".")[-3])
        if "gate_proj" in name:
            gate.append((idx, param))
        elif "up_proj" in name:
            up.append((idx, param))
        elif "down_proj" in name:
            down.append((idx, param))
    gate = sorted(gate, key=lambda x: x[0])
    up = sorted(up, key=lambda x: x[0])
    down = sorted(down, key=lambda x: x[0])
    gate = torch.stack([param for _, param in gate], dim=0)
    up = torch.stack([param for _, param in up], dim=0)
    down_proj = torch.stack([param for _, param in down], dim=0)
    gate_up_proj = torch.cat((gate, up), dim=1)

    assert torch.equal(gate_up_proj, state_dict["mlp.experts.gate_up_proj"].to("cuda")), "gate_up_proj is not equal"
    assert torch.equal(down_proj, state_dict["mlp.experts.down_proj"].to("cuda")), "down_proj is not equal"


if __name__ == "__main__":
    test_veomni_export_unfused_experts()
