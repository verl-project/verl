# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import time

import torch
import torch.distributed as dist

from verl.utils.device import get_device_id, get_torch_device

from .saver import _megatron_calc_global_rank


def _megatron_calc_layer_map(config):
    """Calculate the mapping of global layer_idx to local layer_idx
    Returns:
        layer_map (Dict: int -> tuple(int, int, int)):
            mapping from the global layer index to
            a tuple of (pp_rank, virtual_pp_rank, layer_idx inside model)
    """
    from megatron.core import mpu

    pp_size = mpu.get_pipeline_model_parallel_world_size()
    virtual_pp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1

    layer_map = dict()
    num_layers_per_model = config.num_hidden_layers // pp_size // virtual_pp_size
    assert num_layers_per_model * pp_size * virtual_pp_size == config.num_hidden_layers

    for pp_rank_idx in range(pp_size):
        for virtual_pp_rank_idx in range(virtual_pp_size):
            layer_offset = (
                virtual_pp_rank_idx * (config.num_hidden_layers // virtual_pp_size) + pp_rank_idx * num_layers_per_model
            )
            for layer_idx in range(num_layers_per_model):
                layer_map[layer_offset + layer_idx] = (
                    pp_rank_idx,
                    virtual_pp_rank_idx,
                    layer_idx,
                )
    return layer_map


def load_state_dict_to_megatron_gptmodel(state_dict, wrapped_models, config, params_dtype, is_value_model=False):
    """Load merged state_dict to sharded Megatron module in training."""
    from megatron.core import DistributedDataParallel as LocalDDP
    from megatron.core import mpu
    from megatron.core.transformer.module import Float16Module
    from torch.nn.parallel import DistributedDataParallel as torchDDP

    from verl.utils.logger import print_rank_0
    from verl.utils.megatron_utils import unwrap_model

    start_time = time.time()

    def _get_gpt_model(model):
        return model

    def broadcast_params(module):
        for param in module.parameters():
            torch.distributed.broadcast(
                param.data, src=mpu.get_data_parallel_src_rank(), group=mpu.get_data_parallel_group()
            )

    dp_rank = mpu.get_data_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    cp_rank = mpu.get_context_parallel_rank()
    src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=0, cp_rank=cp_rank)
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    virtual_pp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
    mp_group = mpu.get_model_parallel_group()

    if torch.distributed.get_rank() == src_rank:
        assert mp_group.rank() == 0, f"mp_rank:[{mp_group.rank}] != 0 on rank #0"
        assert pp_rank == 0, f"pp_rank:[{pp_rank}] != 0 on rank #0"
        assert dp_rank == 0, f"dp_rank:[{dp_rank}] != 0 on rank #0"

    if not isinstance(wrapped_models, list | tuple):
        wrapped_models = list(wrapped_models)

    assert len(wrapped_models) == virtual_pp_size
    num_layers_per_model = config.num_hidden_layers // pp_size // virtual_pp_size
    assert num_layers_per_model * pp_size * virtual_pp_size == config.num_hidden_layers

    models = [None] * len(wrapped_models)

    for i, wrapped_model in enumerate(wrapped_models):
        models[i] = unwrap_model(wrapped_model, (torchDDP, LocalDDP, Float16Module))
        gpt_model_module = _get_gpt_model(models[i])
        assert len(gpt_model_module.decoder.layers) == num_layers_per_model

    def _broadcast_tensor(tensor, name) -> torch.Tensor:
        """broadcast tensor from rank0 across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        if torch.distributed.get_rank() == src_rank:
            if name in state_dict:
                weight = state_dict[name]
                tensor_shape = weight.shape
            else:
                tensor_shape = None
        else:
            weight = None
            tensor_shape = None

        obj_list = [tensor_shape]
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        tensor_shape = obj_list[0]

        if tensor_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tensor:[{name}] not in state_dict, skip load")
            return

        if tensor is None:
            tensor = torch.empty(
                tensor_shape,
                dtype=params_dtype,
                device=get_device_id(),
                requires_grad=False,
            )
        if torch.distributed.get_rank() == src_rank:
            tensor.data.copy_(weight)
        dist.broadcast(tensor, src=src_rank, group=mp_group)

    def _broadcast_tp_shard_tensor_vocab(tensor, name, chunk_dim=0, mutate_func=None) -> torch.Tensor:
        """broadcast tensor in tp shards across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        if torch.distributed.get_rank() == src_rank:
            if name in state_dict:
                full_weight = state_dict[name]

                if mutate_func is not None:
                    full_weight = mutate_func(full_weight)
                tensor_chunk = torch.chunk(full_weight, tp_size, dim=chunk_dim)
                chunk_shape = tensor_chunk[0].shape
            else:
                chunk_shape = None
        else:
            chunk_shape = None

        obj_list = [chunk_shape]
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        chunk_shape = obj_list[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{name}] not in state_dict, skip loading")
            return

        if tensor is None:
            sync_tensor = torch.empty(
                chunk_shape,
                dtype=params_dtype,
                device=get_device_id(),
                requires_grad=False,
            )
        else:
            assert tensor.shape == chunk_shape, (
                f"rank #{torch.distributed.get_rank()} tensor {name} shape {tensor.shape} != {chunk_shape}"
            )
            sync_tensor = torch.empty_like(tensor, device=get_device_id(), requires_grad=False)

        for i in range(tp_size):
            if torch.distributed.get_rank() == src_rank:
                sync_tensor.data.copy_(tensor_chunk[i])
            dist.broadcast(sync_tensor, src=src_rank, group=mp_group)
            if (i == tp_rank) and (tensor is not None):
                tensor.data.copy_(sync_tensor)

    def _broadcast_tp_shard_tensor(tensor, name, chunk_dim=0, mutate_func=None) -> torch.Tensor:
        """broadcast tensor in tp shards across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        if torch.distributed.get_rank() == src_rank:
            if name in state_dict:
                full_weight = state_dict[name]
                if mutate_func is not None:
                    full_weight = mutate_func(full_weight)
                tensor_chunk = torch.chunk(full_weight, tp_size, dim=chunk_dim)
                chunk_shape = tensor_chunk[0].shape
            else:
                chunk_shape = None
        else:
            chunk_shape = None

        obj_list = [chunk_shape]
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        chunk_shape = obj_list[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{name}] not in state_dict, skip loading")
            return

        if tensor is None:
            sync_tensor = torch.empty(
                chunk_shape,
                dtype=params_dtype,
                device=get_device_id(),
                requires_grad=False,
            )
        else:
            assert tensor.shape == chunk_shape, (
                f"rank #{torch.distributed.get_rank()} tensor {name} shape {tensor.shape} != {chunk_shape}"
            )
            sync_tensor = torch.empty_like(tensor, device=get_device_id(), requires_grad=False)

        for i in range(tp_size):
            if torch.distributed.get_rank() == src_rank:
                sync_tensor.data.copy_(tensor_chunk[i])
            dist.broadcast(sync_tensor, src=src_rank, group=mp_group)
            if (i == tp_rank) and (tensor is not None):
                tensor.data.copy_(sync_tensor)

    def _broadcast_tp_shard_tensor_gate_up(tensor, gate_name, up_name) -> torch.Tensor:
        """broadcast tensor in tp shards across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        if torch.distributed.get_rank() == src_rank:
            gate_weight = state_dict[gate_name]
            up_weight = state_dict[up_name]
            new_gate_up_weight = torch.empty(
                config.intermediate_size * 2, config.hidden_size, dtype=params_dtype, device=get_device_id()
            )
            for i in range(tp_size):
                intermediate_size_tp = config.intermediate_size // tp_size
                gate_weight_tp = gate_weight[i * intermediate_size_tp : (i + 1) * intermediate_size_tp]
                up_weight_tp = up_weight[i * intermediate_size_tp : (i + 1) * intermediate_size_tp]
                new_gate_up_weight[intermediate_size_tp * 2 * i : intermediate_size_tp * 2 * (i + 1)].copy_(
                    torch.cat([gate_weight_tp, up_weight_tp], dim=0)
                )

            tensor_chunk = torch.chunk(new_gate_up_weight, tp_size, dim=0)
            chunk_shape = tensor_chunk[0].shape
        else:
            chunk_shape = None

        obj_list = [chunk_shape]
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        chunk_shape = obj_list[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{gate_name, up_name}] not in state_dict, skip loading")
            return

        if tensor is None:
            sync_tensor = torch.empty(
                chunk_shape,
                dtype=params_dtype,
                device=get_device_id(),
                requires_grad=False,
            )
        else:
            assert tensor.shape == chunk_shape, (
                f"rank #{torch.distributed.get_rank() == src_rank:} tensor {gate_name, up_name} shape "
                f"{tensor.shape} != {chunk_shape}"
            )
            sync_tensor = torch.empty_like(tensor, device=get_device_id(), requires_grad=False)

        for i in range(tp_size):
            if torch.distributed.get_rank() == src_rank:
                sync_tensor.data.copy_(tensor_chunk[i])
            dist.broadcast(sync_tensor, src=src_rank, group=mp_group)
            if (i == tp_rank) and (tensor is not None):
                tensor.data.copy_(sync_tensor)

    def _broadcast_tp_shard_tensor_qkv(tensor, q_name, k_name, v_name, bias=False) -> torch.Tensor:
        """broadcast tensor in tp shards across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        if torch.distributed.get_rank() == src_rank:
            assert q_name in state_dict and k_name in state_dict and v_name in state_dict
            full_weight_q = state_dict[q_name]
            full_weight_k = state_dict[k_name]
            full_weight_v = state_dict[v_name]

            hidden_size_per_head = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

            if config.num_key_value_heads >= tp_size:
                q_size_tp = hidden_size_per_head * config.num_attention_heads // tp_size
                kv_size_tp = hidden_size_per_head * config.num_key_value_heads // tp_size
                total_size = q_size_tp + 2 * kv_size_tp
                sizes = [total_size * tp_size]
                if not bias:
                    sizes.append(config.hidden_size)
                new_weight_qkv = torch.empty(*sizes, dtype=params_dtype, device=get_device_id())
                for i in range(tp_size):
                    q_part = full_weight_q[i * q_size_tp : (i + 1) * q_size_tp]
                    k_part = full_weight_k[i * kv_size_tp : (i + 1) * kv_size_tp]
                    v_part = full_weight_v[i * kv_size_tp : (i + 1) * kv_size_tp]
                    num_query_groups_per_partition = models[0].config.num_query_groups // tp_size
                    new_weight_qkv_this_tp = new_weight_qkv[i * total_size : (i + 1) * total_size]
                    q_part_per_head = torch.chunk(q_part, num_query_groups_per_partition, dim=0)
                    k_part_per_head = torch.chunk(k_part, num_query_groups_per_partition, dim=0)
                    v_part_per_head = torch.chunk(v_part, num_query_groups_per_partition, dim=0)
                    total_size_per_head = total_size // num_query_groups_per_partition
                    for j in range(num_query_groups_per_partition):
                        new_weight_qkv_this_tp[j * total_size_per_head : (j + 1) * total_size_per_head].copy_(
                            torch.cat([q_part_per_head[j], k_part_per_head[j], v_part_per_head[j]], dim=0)
                        )

            else:
                q_size_tp = hidden_size_per_head * config.num_attention_heads // tp_size
                kv_size_tp = hidden_size_per_head
                total_size = q_size_tp + 2 * kv_size_tp
                sizes = [total_size * tp_size]
                if not bias:
                    sizes.append(config.hidden_size)
                new_weight_qkv = torch.empty(*sizes, dtype=params_dtype, device=get_device_id())
                for i in range(tp_size):
                    q_part = full_weight_q[i * q_size_tp : (i + 1) * q_size_tp]
                    start_idx = i * config.num_key_value_heads // tp_size * hidden_size_per_head
                    end_idx = (i * config.num_key_value_heads // tp_size + 1) * hidden_size_per_head
                    k_part = full_weight_k[start_idx:end_idx]
                    v_part = full_weight_v[start_idx:end_idx]
                    new_weight_qkv_this_tp = new_weight_qkv[i * total_size : (i + 1) * total_size]
                    q_part_per_head = torch.chunk(q_part, config.num_attention_heads, dim=0)
                    k_part_per_head = torch.chunk(k_part, config.num_attention_heads, dim=0)
                    v_part_per_head = torch.chunk(v_part, config.num_attention_heads, dim=0)
                    total_size_per_head = total_size // config.num_attention_heads
                    for j in range(config.num_attention_heads):
                        new_weight_qkv_this_tp[j * total_size_per_head : (j + 1) * total_size_per_head].copy_(
                            torch.cat([q_part_per_head[j], k_part_per_head[j], v_part_per_head[j]], dim=0)
                        )

            tensor_chunk = torch.chunk(new_weight_qkv, tp_size, dim=0)
            chunk_shape = tensor_chunk[0].shape
        else:
            chunk_shape = None

        obj_list = [chunk_shape]
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        chunk_shape = obj_list[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{q_name, k_name, v_name}] not in state_dict, skip loading")
            return

        if tensor is None:
            sync_tensor = torch.empty(
                chunk_shape,
                dtype=params_dtype,
                device=get_device_id(),
                requires_grad=False,
            )
        else:
            assert tensor.shape == chunk_shape, (
                f"rank #{torch.distributed.get_rank()} tensor {q_name} shape {tensor.shape} != {chunk_shape}"
            )
            sync_tensor = torch.empty_like(tensor, device=get_device_id(), requires_grad=False)

        for i in range(tp_size):
            if torch.distributed.get_rank() == src_rank:
                sync_tensor.data.copy_(tensor_chunk[i])
            dist.broadcast(sync_tensor, src=src_rank, group=mp_group)
            if (i == tp_rank) and (tensor is not None):
                tensor.data.copy_(sync_tensor)

    if dp_rank == 0:
        # Embeddings
        # -------------------
        print_rank_0("loading embeddings...")
        gpt_model_module = _get_gpt_model(models[0])
        embed_tokens_weight = None
        if pp_rank == 0:
            embed_tokens_weight = gpt_model_module.embedding.word_embeddings.weight
        _broadcast_tp_shard_tensor_vocab(embed_tokens_weight, "model.embed_tokens.weight")

        # Transformer layers
        # -------------------
        layer_map = _megatron_calc_layer_map(config)

        for layer in range(config.num_hidden_layers):
            layer_name = f"model.layers.{layer}"
            print_rank_0(f"loading layer #{layer}, with layer_name model.layers.{layer}...")
            dst_pp_rank, dst_virtual_pp_rank, dst_layer_idx = layer_map[layer]

            gpt_model_module = _get_gpt_model(models[dst_virtual_pp_rank])
            sync_layer = gpt_model_module.decoder.layers[dst_layer_idx]

            _broadcast_tensor(
                sync_layer.self_attention.linear_qkv.layer_norm_weight if dst_pp_rank == pp_rank else None,
                f"{layer_name}.input_layernorm.weight",
            )

            if f"{layer_name}.self_attn.q_norm.weight" in state_dict:
                _broadcast_tensor(
                    sync_layer.self_attention.q_layernorm.weight if dst_pp_rank == pp_rank else None,
                    f"{layer_name}.self_attn.q_norm.weight",
                )
                _broadcast_tensor(
                    sync_layer.self_attention.k_layernorm.weight if dst_pp_rank == pp_rank else None,
                    f"{layer_name}.self_attn.k_norm.weight",
                )

            _broadcast_tp_shard_tensor_qkv(
                sync_layer.self_attention.linear_qkv.weight if dst_pp_rank == pp_rank else None,
                f"{layer_name}.self_attn.q_proj.weight",
                f"{layer_name}.self_attn.k_proj.weight",
                f"{layer_name}.self_attn.v_proj.weight",
            )
            if f"{layer_name}.self_attn.q_proj.bias" in state_dict:
                _broadcast_tp_shard_tensor_qkv(
                    sync_layer.self_attention.linear_qkv.bias if dst_pp_rank == pp_rank else None,
                    f"{layer_name}.self_attn.q_proj.bias",
                    f"{layer_name}.self_attn.k_proj.bias",
                    f"{layer_name}.self_attn.v_proj.bias",
                    bias=True,
                )

            _broadcast_tp_shard_tensor(
                sync_layer.self_attention.linear_proj.weight if dst_pp_rank == pp_rank else None,
                f"{layer_name}.self_attn.o_proj.weight",
                chunk_dim=1,
            )
            _broadcast_tensor(
                sync_layer.mlp.linear_fc1.layer_norm_weight if dst_pp_rank == pp_rank else None,
                f"{layer_name}.post_attention_layernorm.weight",
            )

            _broadcast_tp_shard_tensor_gate_up(
                sync_layer.mlp.linear_fc1.weight if dst_pp_rank == pp_rank else None,
                f"{layer_name}.mlp.gate_proj.weight",
                f"{layer_name}.mlp.up_proj.weight",
            )

            _broadcast_tp_shard_tensor(
                sync_layer.mlp.linear_fc2.weight if dst_pp_rank == pp_rank else None,
                f"{layer_name}.mlp.down_proj.weight",
                chunk_dim=1,
            )
        # Final Layernorm
        # -------------------
        print_rank_0("loading final layernorm...")
        gpt_model_module = _get_gpt_model(models[-1])
        _broadcast_tensor(
            getattr(gpt_model_module.decoder.final_layernorm, "weight", None),
            "model.norm.weight",
        )

        print_rank_0("loading lm_head...")
        lm_head_weight = None
        if pp_rank + 1 == pp_size:
            lm_head_weight = gpt_model_module.output_layer.weight

        if is_value_model:
            # if torch.distributed.get_rank() == src_rank:
            if "lm_head.weight" in state_dict and state_dict["lm_head.weight"].shape[0] == 1:
                _broadcast_tensor(lm_head_weight, "lm_head.weight")
            elif "reward_head.weight" in state_dict and state_dict["reward_head.weight"].shape[0] == 1:
                _broadcast_tensor(lm_head_weight, "reward_head.weight")
                print_rank_0("load lm_head from value_head weight")
            elif "score.weight" in state_dict and state_dict["score.weight"].shape[0] == 1:
                _broadcast_tensor(lm_head_weight, "score.weight")
                print_rank_0("load lm_head from score weight")
            else:
                _broadcast_tensor(None, "lm_head.weight")
                print_rank_0("fail to match lm_head in value_model")
            # else:

            #     _broadcast_tensor(lm_head_weight, "lm_head.weight")

        else:
            _broadcast_tp_shard_tensor(lm_head_weight, "lm_head.weight")
    dist.barrier()
    # Broadcast weights inside data parallel groups
    for wrapped_model in wrapped_models:
        broadcast_params(wrapped_model)
    pass
    get_torch_device().empty_cache()
    print_rank_0(f"loading megatron ckpt done, time elapsed {time.time() - start_time}s")
