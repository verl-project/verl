# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --standalone --nproc-per-node=4 on the VeOmni GPU image.

Exercises the actual VeOmni export methods with DTensor shards and EP2/DP2.
This is a small distributed contract test, not a full trainer benchmark.
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.tensor import Shard, distribute_tensor


def add_parameter(module, name, value):
    parts = name.split(".")
    for part in parts[:-1]:
        if part not in module._modules:
            module.add_module(part, torch.nn.Module())
        module = module._modules[part]
    module.register_parameter(parts[-1], torch.nn.Parameter(value, requires_grad=False))


def main():
    from verl.workers.engine.veomni import transformer_impl as implementation

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        assert dist.get_world_size() == 4
        device = torch.device("cuda", torch.cuda.current_device())
        mesh = dist.init_device_mesh("cuda", (2, 2), mesh_dim_names=("ep", "dp"))
        ep_rank = mesh.get_local_rank("ep")
        module = torch.nn.Module()
        module.config = SimpleNamespace(model_type="qwen3_moe", num_experts=4)
        prefix = "model.layers.0."
        sources = {}
        for index, name in enumerate(
            [prefix + "self_attn." + part + "_proj.weight" for part in "qkv"]
            + [prefix + "mlp.gate.weight", prefix + "mlp.experts.gate_up_proj", prefix + "mlp.experts.down_proj"]
        ):
            expert = ".experts." in name
            shape = (2, 8, 8) if expert else (8, 8)
            value = torch.arange(torch.tensor(shape).prod().item(), device=device).reshape(shape) / 113.0
            value += index * 0.003 + (ep_rank * 1.003 if expert else 0)
            tensor = distribute_tensor(value, mesh["dp"], [Shard(0)])
            sources[name] = tensor.to("cpu")
            add_parameter(module, name, sources[name])
        module.register_buffer("export_buffer", torch.tensor([1.003, -0.0]))
        buffers = [{"name": "export_buffer", "dtype": "torch.float32", "shape": [2]}]
        receiver = [
            {"name": prefix + "self_attn.qkv_proj.weight", "dtype": "torch.bfloat16", "shape": [8, 8]},
            {"name": prefix + "mlp.gate.weight", "dtype": "torch.float32", "shape": [4, 8]},
            {"name": prefix + "mlp.experts.routed_experts.w13_weight", "dtype": "torch.bfloat16", "shape": [4, 8, 8]},
            {"name": prefix + "mlp.experts.routed_experts.w2_weight", "dtype": "torch.bfloat16", "shape": [4, 8, 8]},
        ]
        metadata = [
            dict(
                rank=rank,
                model_type="qwen3_moe",
                model_class="Qwen3MoeForCausalLM",
                model_dtype="torch.bfloat16",
                quantization=None,
                parameters=receiver,
                buffers=buffers,
            )
            for rank in range(4)
        ]
        engine = SimpleNamespace(module=module, _is_offload_param=True)
        state = SimpleNamespace(ep_enabled=True, ep_rank=ep_rank, ep_size=2, ep_group=mesh["ep"].get_group())
        source_bytes = {name: tensor.to_local().view(torch.uint8).clone() for name, tensor in sources.items()}
        with patch.object(implementation.parallel_state, "get_parallel_state", return_value=state):
            with patch.object(dist, "all_gather_into_tensor", side_effect=AssertionError("preflight collective")):
                dtypes, actual_receiver = implementation.VeOmniEngine.prepare_receiver_export_dtypes(
                    engine, metadata, 4
                )
            assert actual_receiver == {row["name"]: row["dtype"] for row in receiver}
            reference = dict(implementation.VeOmniEngine.get_per_tensor_param(engine)[0])
            candidate = dict(implementation.VeOmniEngine.get_per_tensor_param(engine, receiver_export_dtypes=dtypes)[0])
        assert reference.keys() == candidate.keys()
        for name, value in candidate.items():
            dtype = torch.float32 if name in (prefix + "mlp.gate.weight", "export_buffer") else torch.bfloat16
            expected = reference[name].to(dtype).contiguous().view(torch.uint8)
            assert value.dtype == dtype and torch.equal(value.contiguous().view(torch.uint8), expected), name
        for name, tensor in sources.items():
            assert tensor.device.type == "cpu" and tensor.dtype == torch.float32
            assert torch.equal(tensor.to_local().view(torch.uint8), source_bytes[name]), name
        assert torch.equal(module.export_buffer, torch.tensor([1.003, -0.0]))
        dist.barrier()
        print(
            json.dumps(
                dict(
                    rank=dist.get_rank(),
                    status="PASS",
                    EP=2,
                    DP=2,
                    exported_tensors=len(candidate),
                    source_unchanged=True,
                    receiver_bytes_equal=True,
                    fp32_router_and_buffer_preserved=True,
                )
            ),
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
