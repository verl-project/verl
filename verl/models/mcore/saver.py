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
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as LocalDDP
from megatron.core.transformer.module import Float16Module
from torch.nn.parallel import DistributedDataParallel as torchDDP

from verl.utils.device import get_device_id, get_torch_device
from verl.utils.logger import print_rank_0
from verl.utils.megatron_utils import unwrap_model


def _megatron_calc_global_rank(
    tp_rank: int = 0, dp_rank: int = 0, pp_rank: int = 0, cp_rank: int = 0, ep_rank: int = 0
):
    """Calculate global rank with support for CP/EP parallelism"""

    # Get parallel sizes for each dimension
    tp_size = mpu.get_tensor_model_parallel_world_size()
    dp_size = mpu.get_data_parallel_world_size()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    # ep_size = mpu.get_expert_model_parallel_world_size()

    # Verify total GPU count matches (must be consistent with parallel_state.py)
    total_size = tp_size * dp_size * pp_size * cp_size
    assert total_size == torch.distributed.get_world_size(), (
        f"{tp_size}x{dp_size}x{pp_size}x{cp_size} != {torch.distributed.get_world_size()}"
    )

    # Core calculation logic (corresponds to RankGenerator order parameter)
    # Assumes default order is "tp-cp-ep-dp-pp"
    return ((pp_rank * dp_size + dp_rank) * cp_size + cp_rank) * tp_size + tp_rank


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


def merge_megatron_ckpt_gptmodel(wrapped_models, config, dtype, is_value_model=False, tie_word_embeddings=False):
    """Merge sharded parameters of a Megatron module into a merged checkpoint.

    Args:
        wrapped_models (list of megatron.core.distributed.DistributedDataParallel):
            The local DDP wrapped megatron modules.
        config (str or None):
            HF config for model
        dtype: model params type
        is_value_model: if model is value model
        tie_word_embeddings: tie_word_embeddings
    Returns:
        state_dict (dict):
            The merged state_dict in rank 0, and an empty dictionary in other ranks.
    """
    start_time = time.time()

    def _get_gpt_model(model):
        return model

    dp_rank = mpu.get_data_parallel_rank()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    cp_rank = mpu.get_context_parallel_rank()
    virtual_pp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
    mp_group = mpu.get_model_parallel_group()

    if dist.get_rank() == 0:
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
        assert len(models[i].decoder.layers) == num_layers_per_model, (
            "len model layers {} not equal to num_layers_per_model {}".format(
                len(models[i].decoder.layers), num_layers_per_model
            )
        )

    state_dict = dict()

    def _get_cpu_tensor(tensor: torch.Tensor):
        if tensor is None:
            return None
        if tensor.device == torch.device("cpu"):
            return tensor.detach().clone()
        return tensor.detach().cpu()

    def _broadcast_tensor(tensor, name, src_pp_rank) -> torch.Tensor:
        """broadcast tensor across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank, cp_rank=cp_rank)

        if torch.distributed.get_rank() == src_rank:
            if tensor is None:
                weight = None
                tensor_shape = None
            else:
                weight = tensor
                tensor_shape = weight.shape
        else:
            weight = None
            tensor_shape = None

        obj_list = [tensor_shape]
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        tensor_shape = obj_list[0]

        if tensor_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tensor:[{name}] not exist, skip collect")
            return

        if weight is None:
            weight = torch.empty(
                tensor_shape,
                dtype=dtype,
                device=get_device_id(),
                requires_grad=False,
            )

        dist.broadcast(weight, src=src_rank, group=mp_group)

        if torch.distributed.get_rank() == 0:
            state_dict[name] = _get_cpu_tensor(weight)

    def _broadcast_tp_shard_tensor(tensor, name, src_pp_rank, concat_dim=0, mutate_func=None) -> torch.Tensor:
        """broadcast tensor in tp shards across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        # tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank, cp_rank=cp_rank)

        chunk_shape = tensor.shape if torch.distributed.get_rank() == src_rank else None

        obj_list = [chunk_shape]
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        chunk_shape = obj_list[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{name}] not exist, skip collecting")
            return

        buffer_tensor = torch.empty(
            chunk_shape,
            dtype=dtype,
            device=get_device_id(),
            requires_grad=False,
        )

        chunk_tensors = [None] * tp_size

        for i in range(tp_size):
            cur_src_rank = _megatron_calc_global_rank(tp_rank=i, dp_rank=0, pp_rank=src_pp_rank, cp_rank=cp_rank)
            sync_tensor = tensor if torch.distributed.get_rank() == cur_src_rank else buffer_tensor
            dist.broadcast(sync_tensor, src=cur_src_rank, group=mp_group)

            if torch.distributed.get_rank() == 0:
                chunk_tensors[i] = _get_cpu_tensor(sync_tensor)

        if torch.distributed.get_rank() == 0:
            full_tensor = torch.concat(chunk_tensors, dim=concat_dim)
            if mutate_func is not None:
                full_tensor = mutate_func(full_tensor)
            state_dict[name] = full_tensor

    def _broadcast_tp_shard_tensor_gate_up(tensor, gate_name, up_name, src_pp_rank) -> torch.Tensor:
        """broadcast tensor in tp shards across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        # tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank, cp_rank=cp_rank)

        chunk_shape = tensor.shape if torch.distributed.get_rank() == src_rank else None

        obj_list = [chunk_shape]
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        chunk_shape = obj_list[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{gate_name, up_name}] not exist, skip collecting")
            return

        buffer_tensor = torch.empty(
            chunk_shape,
            dtype=dtype,
            device=get_device_id(),
            requires_grad=False,
        )

        chunk_tensors = [None] * tp_size

        for i in range(tp_size):
            cur_src_rank = _megatron_calc_global_rank(tp_rank=i, dp_rank=0, pp_rank=src_pp_rank, cp_rank=cp_rank)
            sync_tensor = tensor if torch.distributed.get_rank() == cur_src_rank else buffer_tensor
            dist.broadcast(sync_tensor, src=cur_src_rank, group=mp_group)

            if torch.distributed.get_rank() == 0:
                chunk_tensors[i] = _get_cpu_tensor(sync_tensor)

        if torch.distributed.get_rank() == 0:
            full_tensor = torch.concat(chunk_tensors, dim=0)
            intermediate_size_tp = config.intermediate_size // tp_size
            gate_weight_list = []
            up_weight_list = []
            for i in range(tp_size):
                gate_up_weight_tp = full_tensor[intermediate_size_tp * 2 * i : intermediate_size_tp * 2 * (i + 1)]
                gate_weight_tp = gate_up_weight_tp[:intermediate_size_tp]
                up_weight_tp = gate_up_weight_tp[intermediate_size_tp:]
                gate_weight_list.append(gate_weight_tp)
                up_weight_list.append(up_weight_tp)

            state_dict[gate_name] = torch.cat(gate_weight_list, dim=0)
            state_dict[up_name] = torch.cat(up_weight_list, dim=0)

    def _broadcast_tp_shard_tensor_qkv(tensor, q_name, k_name, v_name, src_pp_rank):
        """broadcast tensor in tp shards across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        # tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank, cp_rank=cp_rank)

        chunk_shape = tensor.shape if torch.distributed.get_rank() == src_rank else None

        obj_list = [chunk_shape]
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        chunk_shape = obj_list[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{q_name}] not exist, skip collecting")
            return

        buffer_tensor = torch.empty(
            chunk_shape,
            dtype=dtype,
            device=get_device_id(),
            requires_grad=False,
        )

        chunk_tensors = [None] * tp_size

        for i in range(tp_size):
            cur_src_rank = _megatron_calc_global_rank(tp_rank=i, dp_rank=0, pp_rank=src_pp_rank, cp_rank=cp_rank)
            sync_tensor = tensor if torch.distributed.get_rank() == cur_src_rank else buffer_tensor
            dist.broadcast(sync_tensor, src=cur_src_rank, group=mp_group)

            if torch.distributed.get_rank() == 0:
                chunk_tensors[i] = _get_cpu_tensor(sync_tensor)

        if torch.distributed.get_rank() == 0:
            full_tensor = torch.concat(chunk_tensors, dim=0)
            q_weight_list = []
            k_weight_list = []
            v_weight_list = []
            hidden_size_per_head = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

            if config.num_key_value_heads >= tp_size:
                q_size_tp = hidden_size_per_head * config.num_attention_heads // tp_size
                kv_size_tp = hidden_size_per_head * config.num_key_value_heads // tp_size
                total_size = q_size_tp + 2 * kv_size_tp
                for i in range(tp_size):
                    num_query_groups_per_partition = wrapped_models[0].config.num_query_groups // tp_size
                    qkv_part = full_tensor[i * total_size : (i + 1) * total_size]
                    q_size_chunk = q_size_tp // num_query_groups_per_partition
                    kv_size_chunk = kv_size_tp // num_query_groups_per_partition
                    for qkv_part_chunk in qkv_part.chunk(num_query_groups_per_partition):
                        q_part = qkv_part_chunk[:q_size_chunk]
                        k_part = qkv_part_chunk[q_size_chunk : q_size_chunk + kv_size_chunk]
                        v_part = qkv_part_chunk[q_size_chunk + kv_size_chunk :]
                        q_weight_list.append(q_part)
                        k_weight_list.append(k_part)
                        v_weight_list.append(v_part)
            else:
                q_size_tp = hidden_size_per_head * config.num_attention_heads // tp_size
                kv_size_tp = hidden_size_per_head
                total_size = q_size_tp + 2 * kv_size_tp
                for i in range(tp_size):
                    num_query_groups_per_partition = wrapped_models[0].config.num_query_groups // tp_size
                    qkv_part = full_tensor[i * total_size : (i + 1) * total_size]
                    q_size_chunk = q_size_tp // num_query_groups_per_partition
                    kv_size_chunk = kv_size_tp // num_query_groups_per_partition
                    for qkv_part_chunk in qkv_part.chunk(num_query_groups_per_partition):
                        q_part = qkv_part_chunk[:q_size_chunk]
                        k_part = qkv_part_chunk[q_size_chunk : q_size_chunk + kv_size_chunk]
                        v_part = qkv_part_chunk[q_size_chunk + kv_size_chunk :]
                        q_weight_list.append(q_part)
                        if i * config.num_key_value_heads % tp_size == 0:
                            k_weight_list.append(k_part)
                            v_weight_list.append(v_part)

            state_dict[q_name] = torch.cat(q_weight_list, dim=0)
            state_dict[k_name] = torch.cat(k_weight_list, dim=0)
            state_dict[v_name] = torch.cat(v_weight_list, dim=0)

    # empty cache before collecting weights
    get_torch_device().empty_cache()
    # Embeddings
    # -------------------
    if dp_rank == 0 and cp_rank == 0:  # models are identical across cp ranks
        # Embeddings
        # -------------------
        print_rank_0("collecting embeddings...")
        gpt_model_module = _get_gpt_model(models[0])
        _broadcast_tp_shard_tensor(
            gpt_model_module.embedding.word_embeddings.weight if pp_rank == 0 else None,
            "model.embed_tokens.weight",
            src_pp_rank=0,
        )

        # Transformer layers
        # -------------------
        layer_map = _megatron_calc_layer_map(config)
        for layer in range(config.num_hidden_layers):
            print_rank_0(f"collecting layer #{layer}...")
            layer_name = f"model.layers.{layer}"
            src_pp_rank, src_virtual_pp_rank, src_layer_idx = layer_map[layer]

            gpt_model_module = _get_gpt_model(models[src_virtual_pp_rank])
            sync_layer = gpt_model_module.decoder.layers[src_layer_idx]

            _broadcast_tensor(
                sync_layer.self_attention.linear_qkv.layer_norm_weight,
                f"{layer_name}.input_layernorm.weight",
                src_pp_rank=src_pp_rank,
            )

            if gpt_model_module.config.qk_layernorm:
                _broadcast_tensor(
                    sync_layer.self_attention.q_layernorm.weight,
                    f"{layer_name}.self_attn.q_norm.weight",
                    src_pp_rank=src_pp_rank,
                )
                _broadcast_tensor(
                    sync_layer.self_attention.k_layernorm.weight,
                    f"{layer_name}.self_attn.k_norm.weight",
                    src_pp_rank=src_pp_rank,
                )

            _broadcast_tp_shard_tensor_qkv(
                sync_layer.self_attention.linear_qkv.weight,
                f"{layer_name}.self_attn.q_proj.weight",
                f"{layer_name}.self_attn.k_proj.weight",
                f"{layer_name}.self_attn.v_proj.weight",
                src_pp_rank=src_pp_rank,
            )

            if gpt_model_module.config.add_qkv_bias:
                _broadcast_tp_shard_tensor_qkv(
                    sync_layer.self_attention.linear_qkv.bias,
                    f"{layer_name}.self_attn.q_proj.bias",
                    f"{layer_name}.self_attn.k_proj.bias",
                    f"{layer_name}.self_attn.v_proj.bias",
                    src_pp_rank=src_pp_rank,
                )

            _broadcast_tp_shard_tensor(
                sync_layer.self_attention.linear_proj.weight,
                f"{layer_name}.self_attn.o_proj.weight",
                concat_dim=1,
                src_pp_rank=src_pp_rank,
            )

            _broadcast_tensor(
                sync_layer.mlp.linear_fc1.layer_norm_weight,
                f"{layer_name}.post_attention_layernorm.weight",
                src_pp_rank=src_pp_rank,
            )

            _broadcast_tp_shard_tensor_gate_up(
                sync_layer.mlp.linear_fc1.weight,
                f"{layer_name}.mlp.gate_proj.weight",
                f"{layer_name}.mlp.up_proj.weight",
                src_pp_rank=src_pp_rank,
            )

            _broadcast_tp_shard_tensor(
                sync_layer.mlp.linear_fc2.weight,
                f"{layer_name}.mlp.down_proj.weight",
                concat_dim=1,
                src_pp_rank=src_pp_rank,
            )

        # Final Layernorm
        # -------------------
        print_rank_0("collecting final layernorm...")
        gpt_model_module = _get_gpt_model(models[-1])
        _broadcast_tensor(
            getattr(gpt_model_module.decoder.final_layernorm, "weight", None),
            "model.norm.weight",
            src_pp_rank=pp_size - 1,
        )

        if tie_word_embeddings:
            print_rank_0("tie word embedding skip load lm_head...")
        else:
            print_rank_0("collecting lm_head...")

            if is_value_model:
                lm_head_weight = None
                if pp_rank == pp_size - 1:
                    lm_head_weight = getattr(gpt_model_module.output_layer, "weight", None)
                _broadcast_tensor(lm_head_weight, "lm_head.weight", src_pp_rank=pp_size - 1)

            else:
                _broadcast_tp_shard_tensor(
                    getattr(gpt_model_module.output_layer, "weight", None) if pp_rank == pp_size - 1 else None,
                    "lm_head.weight",
                    src_pp_rank=pp_size - 1,
                )

    dist.barrier()
    get_torch_device().empty_cache()
    if torch.distributed.get_rank() == 0:
        for k, v in state_dict.items():
            if dtype != v.dtype:
                state_dict[k] = v.to(dtype)

    print_rank_0(f"merge megatron ckpt done, time elapsed {time.time() - start_time}s")
    return state_dict


def merge_megatron_ckpt_gptmodel_qwen_moe(
    wrapped_models, config, dtype, is_value_model=False, tie_word_embeddings=False
):
    raise NotImplementedError("merge_megatron_ckpt_gptmodel_qwen_moe is not implemented")


def merge_megatron_ckpt_gptmodel_qwen2_5_vl(
    wrapped_models, config, dtype, is_value_model=False, tie_word_embeddings=False
):
    raise NotImplementedError("merge_megatron_ckpt_gptmodel_qwen2_5_vl is not implemented")


def merge_megatron_ckpt_gptmodel_dpskv3(wrapped_models, config, dtype, is_value_model=False, tie_word_embeddings=False):
    raise NotImplementedError("merge_megatron_ckpt_gptmodel_dpskv3 is not implemented")


def merge_megatron_ckpt_gptmodel_mixtral(
    wrapped_models, config, dtype, is_value_model=False, tie_word_embeddings=False
):
    raise NotImplementedError("merge_megatron_ckpt_gptmodel_mixtral is not implemented")
