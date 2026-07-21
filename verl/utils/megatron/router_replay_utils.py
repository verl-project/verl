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
Router Replay Utilities
Utilities for handling router replay functionality in Megatron models.
"""

import inspect
import warnings

import torch

try:
    from megatron.core.pipeline_parallel.utils import is_vp_first_stage, is_vp_last_stage
except ImportError:
    warnings.warn("NPU not support router replay for now.", stacklevel=2)
    pass

from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel.schedules import get_schedule_table
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region, scatter_to_sequence_parallel_region
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from verl.models.mcore.util import (
    postprocess_packed_seqs,
    postprocess_thd_engine,
    preprocess_packed_seqs,
    preprocess_thd_engine,
)
from verl.utils.device import get_device_name
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction

device_name = get_device_name()


def _context_parallel_layout(tf_config) -> str:
    if getattr(tf_config, "experimental_attention_variant", None) == "dsv4_hybrid":
        return "contiguous"
    return "zigzag"


def get_thd_sequence_parallel_padding_mask(input_ids: torch.Tensor) -> torch.Tensor | None:
    """Return this TP rank's THD alignment-only positions for router replay.

    The no-padding THD path strips these positions from exported
    ``routed_experts`` but recreates them as zeros during replay. Canonicalizing
    them before RECORD keeps the dispatcher topology identical across R2 passes.
    Context-parallel layouts use zig-zag packing and are intentionally left
    untouched until they receive their own audited mapping.
    """
    if not input_ids.is_nested or mpu.get_context_parallel_world_size() != 1:
        return None

    tp_size = mpu.get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return None

    seq_lens = input_ids.offsets().diff().tolist()
    padded_lens = [length + (-length % tp_size) for length in seq_lens]
    total_padded = sum(padded_lens)
    if total_padded == sum(seq_lens):
        return None
    if total_padded % tp_size:
        raise RuntimeError(f"THD padded token count must divide TP size: total={total_padded}, tp={tp_size}")

    global_padding_mask = torch.zeros(total_padded, dtype=torch.bool, device=input_ids.device)
    offset = 0
    for seq_len, padded_len in zip(seq_lens, padded_lens, strict=True):
        global_padding_mask[offset + seq_len : offset + padded_len] = True
        offset += padded_len

    local_tokens = total_padded // tp_size
    tp_rank = mpu.get_tensor_model_parallel_rank()
    return global_padding_mask.view(tp_size, local_tokens)[tp_rank].contiguous()


# from megatron.core.transformer.transformer_block import get_num_layers_to_build
def get_num_layers_to_build(config: TransformerConfig, vp_stage: int | None = None, pp_rank: int | None = None) -> int:
    """
    Determine the number of transformer layers to build for the current pipeline stage.
    Args:
        config (TransformerConfig): Configuration object containing transformer model parameters.
        vp_stage (Optional[int]): Virtual pipeline stage number.
        pp_rank (Optional[int]): Pipeline parallel rank.

    Returns:
        int: The number of layers to be built for the current pipeline stage.
    """
    # If we have a custom PP layout, straightforwardly
    # return the number of decoders in the layout array.
    if hasattr(config, "pipeline_model_parallel_layout") and config.pipeline_model_parallel_layout is not None:
        from megatron.core.transformer.enums import LayerType

        return config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.decoder, vp_stage=vp_stage
        )

    # Fallback for legacy tests.
    if pp_rank is None:
        pp_rank = mpu.get_pipeline_model_parallel_rank()

    is_first_pp_stage = pp_rank == 0
    is_last_pp_stage = pp_rank == config.pipeline_model_parallel_size - 1

    if config.num_layers_in_first_pipeline_stage is not None or config.num_layers_in_last_pipeline_stage is not None:
        assert not (config.account_for_embedding_in_pipeline_split or config.account_for_loss_in_pipeline_split), (
            " \
        Does not support standalone embedding stage and standalone loss stage with uneven pp"
        )
        # Number of layers to distribute over rest of pipeline stages
        layers_to_distribute = config.num_layers
        # Number of pipeline stages left for distributing transformer layers
        pipeline_stages_left = config.pipeline_model_parallel_size

        # If the uneven first (last) pipeline stage is enabled, remove the specified number
        # of layers to calculate the number of layers on each middle pipeline stage.
        if config.num_layers_in_first_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_first_pipeline_stage
            pipeline_stages_left -= 1

        if config.num_layers_in_last_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_last_pipeline_stage
            pipeline_stages_left -= 1

        # If pp_size <= 2, we do not have any intermediate pipeline stages, and we do not
        # need to check if the left over layers are divisible by the left over stages.
        if pipeline_stages_left > 0:
            assert layers_to_distribute % pipeline_stages_left == 0, (
                "With uneven pipelineing the left over layers must be divisible by left over stages"
            )
            num_layers_per_pipeline_rank = layers_to_distribute // pipeline_stages_left
        else:
            num_layers_per_pipeline_rank = 0

        # If the uneven first (last) pipeline stage is enabled, return the specified number
        # of layers for all virtual pipeline parallel stages within the first (last) pipeline
        # parallel stage.

        if is_first_pp_stage and config.num_layers_in_first_pipeline_stage is not None:
            num_layers_per_pipeline_rank = config.num_layers_in_first_pipeline_stage

        if is_last_pp_stage and config.num_layers_in_last_pipeline_stage is not None:
            num_layers_per_pipeline_rank = config.num_layers_in_last_pipeline_stage
    else:
        # Include the embedding layer and loss layer into pipeline parallelism partition
        num_layers = config.num_layers
        if config.account_for_embedding_in_pipeline_split:
            num_layers += 1

        if config.account_for_loss_in_pipeline_split:
            num_layers += 1

        assert num_layers % config.pipeline_model_parallel_size == 0, (
            "num_layers should be divisible by pipeline_model_parallel_size"
        )
        num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

    vp_size = config.virtual_pipeline_model_parallel_size
    if vp_size is not None and config.pipeline_model_parallel_size > 1:
        # Interleaved pipeline parallelism:
        # Number of layers in each model chunk is the number of layers in the stage,
        # divided by the number of model chunks in a stage.
        # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0]  [2]  [4]  [6]
        # Stage 1: [1]  [3]  [5]  [7]
        # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0, 1]  [4, 5]
        # Stage 1: [2, 3]  [6, 7]

        assert num_layers_per_pipeline_rank % vp_size == 0, (
            f"num_layers_per_pipeline_rank {num_layers_per_pipeline_rank} \
            should be divisible by vp_size {vp_size}"
        )
        num_layers_per_virtual_stage = num_layers_per_pipeline_rank // vp_size

        num_layers_to_build = num_layers_per_virtual_stage

    else:
        # Non-interleaved pipeline parallelism:
        # Each stage gets a contiguous set of layers.
        num_layers_to_build = num_layers_per_pipeline_rank

    # The embedding (or loss) layer cannot function as a standalone transformer layer
    # Reduce the number of layers to construct by 1 on the first (or last) stage if the
    # embedding (or loss) layer is included in the pipeline parallelism partition and placement.
    if config.account_for_embedding_in_pipeline_split:
        if is_vp_first_stage(vp_stage, vp_size) and is_first_pp_stage:
            num_layers_to_build -= 1
            assert num_layers_to_build >= 0, "Not enough layers in the first virtual pipeline stage"

    if config.account_for_loss_in_pipeline_split:
        if is_vp_last_stage(vp_stage, vp_size) and is_last_pp_stage:
            num_layers_to_build -= 1
            assert num_layers_to_build >= 0, "Not enough layers in the last virtual pipeline stage"

    return num_layers_to_build


def is_moe_layer(tf_config, layer_idx):
    moe_layer_freq = getattr(tf_config, "moe_layer_freq", None)

    if isinstance(moe_layer_freq, int):
        return layer_idx % moe_layer_freq == 0
    elif isinstance(moe_layer_freq, list):
        return moe_layer_freq[layer_idx] == 1
    else:
        raise ValueError(f"Unsupported moe_layer_freq type: {type(moe_layer_freq)}")


def get_moe_num_layers_to_build(
    config: TransformerConfig, vp_stage: int | None = None, pp_rank: int | None = None
) -> int:
    """Count the number of MoE layers assigned to the current rank.
    When ``moe_layer_freq`` is 1 or unset, every transformer layer is an MoE
    layer, so the count equals the total layer count. Otherwise only layers
    whose global index satisfies the frequency predicate are counted.
    Args:
        config: Megatron TransformerConfig providing layer layout information.
        vp_stage: Virtual-pipeline stage index (None defaults to current).
        pp_rank: Pipeline-parallel rank (None defaults to current).
    Returns:
        Number of MoE layers on the specified rank/stage.
    """
    total_layers = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)

    sig = inspect.signature(get_transformer_layer_offset)
    # core 0.12.1 is not support vp_stage and pp_rank as parameters
    if "vp_stage" in sig.parameters and "pp_rank" in sig.parameters:
        layer_offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
    elif "pp_rank" in sig.parameters:
        layer_offset = get_transformer_layer_offset(config, pp_rank=pp_rank)
    else:
        layer_offset = get_transformer_layer_offset(config)

    local_global_indices = range(layer_offset, layer_offset + total_layers)

    num_moe_layers = sum(1 for idx in local_global_indices if is_moe_layer(config, idx))

    return num_moe_layers


def _context_parallel_layout(tf_config) -> str:
    """Return the CP row layout used for THD packing, matching MegatronEngine.

    Router-replay tensors must be sharded exactly like the model's own token rows; otherwise the
    replayed routes are attached to the wrong tokens (or the shapes disagree outright, since the
    zigzag layout doubles the alignment). Keep this in sync with
    ``MegatronEngine._get_context_parallel_layout``.
    """
    if getattr(tf_config, "experimental_attention_variant", None) == "dsv4_hybrid":
        return "contiguous"
    return "zigzag"


def merge_router_topk_indices(
    attention_mask, input_ids, mini_layer_topk_idx_list, tf_config, vp_rank=None, local_cp_size=None
):
    """
    Merge recorded router top-k indices across sequence-parallel ranks for all router instances,
    then pack/unpack them to align with the original (batch, seq_len) layout and append the result.

    Args:
        attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_len]. Used to determine
            the valid token positions during pack/unpack.
        input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_len]. Used together with
            attention_mask for sequence packing/unpacking.
        mini_layer_topk_idx_list (list): A Python list to which the merged top-k indices tensor will be appended.
        tf_config: Megatron/Transformer engine configuration object. Used to locate router instances for
            the current micro-batch.
        vp_rank (Optional[int]): Virtual pipeline stage rank override. If None, the current VP rank from
            Megatron parallel state will be used.

    Returns:
        None: The function has side effects only; it appends a tensor of shape
        [1, dynamic_bs_all, layer_num, topk] to mini_layer_topk_idx_list.
    """
    with torch.no_grad():
        vp_rank = 0 if vp_rank is None else vp_rank
        local_router_instances = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        expected_router_count = len(local_router_instances)
        recorded_router_indices = [
            idx for idx, router in enumerate(RouterReplay.router_instances) if router.recorded_topk_idx is not None
        ]
        if len(recorded_router_indices) == expected_router_count:
            tf_config._verl_router_replay_local_router_indices = recorded_router_indices
            router_instances_list = [RouterReplay.router_instances[idx] for idx in recorded_router_indices]
        else:
            router_instances_list = local_router_instances

        missing = [idx for idx, router in enumerate(router_instances_list) if router.recorded_topk_idx is None]
        if missing:
            local_rank_info = get_current_rank_layer_info(tf_config, vp_rank)
            raise RuntimeError(
                "router replay RECORD did not capture all local routers: "
                f"missing_local_positions={missing}, selected={len(router_instances_list)}, "
                f"expected={expected_router_count}, registry={len(RouterReplay.router_instances)}, "
                f"recorded_global_indices={recorded_router_indices}, local_layer_range={local_rank_info}, "
                f"pp_rank={mpu.get_pipeline_model_parallel_rank()}, vp_rank={vp_rank}"
            )

        layers_topk_idx = []
        for router in router_instances_list:
            layers_topk_idx.append(router.recorded_topk_idx.to(torch.int16))  # dynamic_bs, topk

        # layer_num, dynamic_bs, topk  -> dynamic_bs, layer_num, topk
        layers_topk_idx = torch.stack(layers_topk_idx).permute(1, 0, 2).to(device_name)
        layers_topk_idx = layers_topk_idx.contiguous().view(torch.uint8)
        # dynamic_bs, layer_num, topk -> 1, dynamic_bs_all, layer_num, topk
        layers_topk_idx = (
            gather_from_sequence_parallel_region(layers_topk_idx, tensor_parallel_output_grad=False)
            .unsqueeze(0)
            .contiguous()
        )

        fp8 = tf_config.fp8
        use_fp8_padding = fp8 in ["e4m3", "hybrid"]
        cp_layout = _context_parallel_layout(tf_config)
        min_local_rows = (
            tf_config.csa_window_size
            if getattr(tf_config, "experimental_attention_variant", None) == "dsv4_hybrid"
            else None
        )

        cp_layout = _context_parallel_layout(tf_config)

        if input_ids.is_nested:
            batch_size = input_ids.shape[0]
            _, packed_seq_params, _ = preprocess_thd_engine(
                input_ids,
                pre_process=True,
                use_fp8_padding=use_fp8_padding,
                min_local_rows=min_local_rows,
                local_cp_size=local_cp_size,
                cp_layout=cp_layout,
            )
            layers_topk_idx = postprocess_thd_engine(
                layers_topk_idx,
                packed_seq_params,
                input_ids,
                batch_size,
                post_process=True,
                local_cp_size=local_cp_size,
                cp_layout=cp_layout,
            )
            # Undo the uint8 view. Nested tensors have no view(dtype), so reinterpret
            # the packed values buffer and re-wrap it on the same offsets.
            layers_topk_idx = torch.nested.nested_tensor_from_jagged(
                layers_topk_idx.values().contiguous().view(torch.int16), offsets=layers_topk_idx.offsets()
            )
        else:
            batch_size, seq_len = attention_mask.shape[:2]
            _, packed_seq_params = preprocess_packed_seqs(
                input_ids, attention_mask, pre_process=True, use_fp8_padding=use_fp8_padding
            )
            layers_topk_idx = postprocess_packed_seqs(
                layers_topk_idx, packed_seq_params, attention_mask, batch_size, seq_len, post_process=True
            )
            layers_topk_idx = layers_topk_idx.view(torch.int16)
        mini_layer_topk_idx_list.append(layers_topk_idx.cpu())


def build_r3_replay_mask(input_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Build a full-sequence replay mask for rollout-captured routes.

    Response logprobs are read from shifted model rows, but those rows attend to
    earlier hidden states whose MoE routing also affects the result. Replay every
    causal row that can influence response logprobs and skip only the final
    response row, whose logits are not used by ``no_padding_2_padding``.
    """
    total_lens = input_ids.offsets().diff()
    response_lens = response_mask.sum(dim=-1).to(device=total_lens.device, dtype=total_lens.dtype)
    bs = total_lens.size(0)
    values = torch.tensor([True, False], dtype=torch.bool, device=total_lens.device).repeat(bs)
    replay_lens = torch.where(response_lens > 0, total_lens - 1, torch.zeros_like(total_lens))
    suffix_lens = total_lens - replay_lens
    counts = torch.stack([replay_lens, suffix_lens], dim=1).flatten()
    mask_values = torch.repeat_interleave(values, counts)
    return torch.nested.nested_tensor_from_jagged(mask_values, offsets=input_ids.offsets())


def align_r3_router_replay_data(layers_topk_idx: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Align rollout routes with full training inputs one sequence at a time.

    Autoregressive rollout records the routes used to predict each generated
    token, so it has no route for the final generated token. The R3 replay mask
    leaves that final model row native; append one ignored placeholder per
    affected sequence so route and model tensors still have identical shapes.
    """
    if not layers_topk_idx.is_nested or not input_ids.is_nested:
        raise TypeError("R3 router replay requires jagged route targets and input_ids")

    route_parts = list(layers_topk_idx.unbind())
    input_lens = [int(length) for length in input_ids.offsets().diff().tolist()]
    if len(route_parts) != len(input_lens):
        raise RuntimeError(
            f"R3 router replay has {len(route_parts)} route sequences for {len(input_lens)} input sequences"
        )

    aligned_parts = []
    for sample_id, (routes, input_len) in enumerate(zip(route_parts, input_lens, strict=True)):
        route_len = routes.shape[0]
        if route_len == input_len:
            aligned_parts.append(routes)
        elif route_len == input_len - 1:
            placeholder = torch.zeros((1, *routes.shape[1:]), dtype=routes.dtype, device=routes.device)
            aligned_parts.append(torch.cat((routes, placeholder), dim=0))
        else:
            raise RuntimeError(
                f"R3 router replay sample {sample_id} has {route_len} route rows for {input_len} input tokens; "
                "expected equal lengths or exactly one missing final route"
            )

    return torch.nested.as_nested_tensor(aligned_parts, layout=torch.jagged)


def set_router_replay_data(
    layers_topk_idx,
    attention_mask,
    tf_config,
    vp_rank=None,
    replay_mask=None,
    local_cp_size=None,
    model=None,
):
    """
    Scatter the packed router top-k indices back to sequence-parallel ranks and update each local
    RouterReplay instance with target indices for replay mode.

    This function prepares the per-layer, per-sample top-k routing decisions (recorded during an earlier
    forward) so that subsequent replay passes can follow exactly the same routing.

    Args:
        layers_topk_idx (torch.Tensor): Router top-k indices with shape [bs, max_seq_len, layer_num, topk].
            This should be the merged output produced by merge_router_topk_indices.
        attention_mask (torch.Tensor): Attention mask [batch_size, seq_len] used for pack/unpack alignment.
        tf_config: Megatron/Transformer engine configuration object.
        vp_rank (Optional[int]): Virtual pipeline stage rank override. If None, the current VP rank from
            Megatron parallel state will be used.
        replay_mask (Optional[torch.Tensor]): Optional per-token mask. Masked tokens use replayed routes;
            unmasked tokens keep native Megatron routes.
        model: The forwarded model (or list of VPP chunks). When given, targets are written to its own
            routers instead of a positional slice of the global RouterReplay.router_instances list.

    Returns:
        None: The function updates internal RouterReplay instances in-place.
    """
    if layers_topk_idx is None:
        raise RuntimeError(
            "router_replay REPLAY requires routed_experts from the preceding RECORD forward."
        )

    with torch.no_grad():
        fp8 = tf_config.fp8
        use_fp8_padding = fp8 in ["e4m3", "hybrid"]
        cp_layout = _context_parallel_layout(tf_config)
        min_local_rows = (
            tf_config.csa_window_size
            if getattr(tf_config, "experimental_attention_variant", None) == "dsv4_hybrid"
            else None
        )

        cp_layout = _context_parallel_layout(tf_config)

        replay_mask_rmpad = None
        if layers_topk_idx.is_nested:
            layers_topk_idx_rmpad, _, _ = preprocess_thd_engine(
                layers_topk_idx,
                pre_process=True,
                use_fp8_padding=use_fp8_padding,
                min_local_rows=min_local_rows,
                local_cp_size=local_cp_size,
                cp_layout=cp_layout,
            )
            if replay_mask is not None:
                replay_mask_rmpad, _, _ = preprocess_thd_engine(
                    replay_mask,
                    pre_process=True,
                    use_fp8_padding=use_fp8_padding,
                    min_local_rows=min_local_rows,
                    local_cp_size=local_cp_size,
                    cp_layout=cp_layout,
                )
        else:
            layers_topk_idx_rmpad, _ = preprocess_packed_seqs(
                layers_topk_idx, attention_mask, pre_process=True, use_fp8_padding=use_fp8_padding
            )
        layers_topk_idx_rmpad = layers_topk_idx_rmpad.contiguous()  # 1, dynamic_bs_all, layer_num, topk

        # 1, dynamic_bs_split, layer_num, topk
        layers_topk_idx_rmpad_split = scatter_to_sequence_parallel_region(
            layers_topk_idx_rmpad.to(device_name).squeeze(dim=0)
        ).unsqueeze(dim=0)
        replay_mask_rmpad_split = None
        if replay_mask_rmpad is not None:
            replay_mask_rmpad_split = scatter_to_sequence_parallel_region(
                replay_mask_rmpad.to(device_name).squeeze(dim=0)
            )

        # dynamic_bs_split, layer_num, topk -> layer_num, dynamic_bs_split, topk
        layers_topk_idx_reshape = layers_topk_idx_rmpad_split.permute(0, 2, 1, 3).squeeze(
            dim=0
        )  # layer_num, dynamic_bs_all, topk
        # When dim-0 covers all layers (e.g. R3, or R2 with all-MoE models),
        # index by absolute layer_idx; otherwise (R2 with mixed dense/MoE),
        # dim-0 only contains MoE layers, index by MoE-layer ordinal.
        index_by_layer = len(layers_topk_idx_reshape) == tf_config.num_layers

        if model is not None:
            for layer_number, router in iter_model_routers(model):
                layer_idx = layer_number - 1
                idx = layer_idx if index_by_layer else sum(1 for i in range(layer_idx) if is_moe_layer(tf_config, i))
                if 0 <= idx < layers_topk_idx_reshape.shape[0]:
                    router.set_target_indices(
                        layers_topk_idx_reshape[idx].to(torch.int64),
                        replay_mask=replay_mask_rmpad_split,
                    )
            return

        local_rank_info = get_current_rank_layer_info(tf_config, vp_rank)
        offset, end = local_rank_info["start"], local_rank_info["end"]
        local_router_indices = getattr(tf_config, "_verl_router_replay_local_router_indices", None)
        if local_router_indices is not None:
            router_instances_list = [RouterReplay.router_instances[idx] for idx in local_router_indices]
        else:
            router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)

        # For R2: count MoE layers before `offset` as the starting position.
        moe_idx = sum(1 for i in range(offset) if is_moe_layer(tf_config, i))

        router_offset = 0
        for layer_idx in range(offset, end):
            if not is_moe_layer(tf_config, layer_idx):
                continue
            router = router_instances_list[router_offset]
            idx = layer_idx if index_by_layer else moe_idx
            router.set_target_indices(
                layers_topk_idx_reshape[idx].to(torch.int64),
                replay_mask=replay_mask_rmpad_split,
            )
            router_offset += 1
            moe_idx += 1


def iter_model_routers(model):
    """Yield ``(layer_number, RouterReplay)`` for every replay-enabled router owned by ``model``.

    ``model`` is a single module or the list of virtual-pipeline chunks. The process-global
    ``RouterReplay.router_instances`` list can hold more routers than this model has local layers —
    a model that rebuilds its decoder registers a set it then discards — so a positional slice of it
    can address the wrong objects. Walking the forwarded model reaches each layer's own router.
    """
    for chunk in model if isinstance(model, list | tuple) else [model]:
        # Scope to the decoder: MTP layers restart layer numbering at 1, so they alias decoder
        # layers, and the replay tensor carries no MTP rows. ``mtp`` is a sibling of ``decoder``
        # on both GPTModel and HybridModel, the only two classes that build one.
        for module in getattr(chunk, "decoder", chunk).modules():
            router = getattr(module, "router_replay", None)
            if isinstance(module, TopKRouter) and router is not None and module.layer_number is not None:
                yield module.layer_number, router


def set_model_router_replay_action(model, router_replay_action):
    """Set the router replay action on the forwarded model's own routers.

    Mirrors the module walk in ``set_router_replay_data``. Toggling through the positional slice can
    miss the real routers, leaving them stuck in REPLAY_FORWARD so the backward recompute reads a
    stale cross-micro-batch ``target_topk_idx`` instead of popping its own entry from
    ``replay_backward_list``.
    """
    for _, router in iter_model_routers(model):
        router.set_router_replay_action(router_replay_action)


def reorder_and_merge_vpp_layers(
    micro_batch_tensor_list,
    num_microbatches: int,
    vpp_size: int,
    microbatch_group_size_per_vp_stage: int,
) -> torch.Tensor:
    """
    Reorder and merge per-VPP layer blocks into a contiguous layer dimension.

    Given a tensor shaped as [bs*vpp_size, max_token_len, layer_num_per_vpp, topk], this function:
    1) Builds the schedule table for virtual microbatches and reorders the first dimension so that entries
       belonging to the same model chunk (VPP stage) become contiguous.
    2) Reshapes and merges the (vpp_size, layer_num_per_vpp) into a single layer dimension, producing
       [bs, max_token_len, layer_num, topk].

    Args:
        micro_batch_tensor_list : the list of Input tensor.
        num_microbatches (int): Number of microbatches per pipeline stage (bs).
        vpp_size (int): Virtual pipeline parallel size (number of model chunks).
        microbatch_group_size_per_vp_stage (int): Number of consecutive microbatches processed per VPP stage.

    Returns:
        torch.Tensor: Output tensor of shape [bs, max_token_len, layer_num, topk].

    Raises:
        ValueError: If input tensor dimensionality or expected sizes do not match.
        RuntimeError: If the computed output shape is unexpected or the schedule length mismatches.
    """
    # 1) Build schedule table: map each virtual_microbatch_id -> (microbatch_id, model_chunk_id)
    schedule_table = get_schedule_table(num_microbatches, vpp_size, microbatch_group_size_per_vp_stage)

    # 2) Group by model_chunk_id to build reorder indices so entries of the same chunk become contiguous along dim 0
    tensor_by_chunk = [[] for _ in range(vpp_size)]
    mini_tensor_list = []

    for vidx, (_mb, chunk_id) in enumerate(schedule_table):
        tensor_by_chunk[chunk_id].append(micro_batch_tensor_list[vidx])

    if micro_batch_tensor_list[0].is_nested:
        for chunk_id in range(vpp_size):
            mini_tensor_list.append(merge_nested_router_maps(tensor_by_chunk[chunk_id]))
    else:
        for chunk_id in range(vpp_size):
            mini_tensor_list.append(torch.cat(tensor_by_chunk[chunk_id], dim=0))

    out = torch.cat(mini_tensor_list, dim=2)

    return out


def merge_nested_router_maps(router_maps) -> torch.Tensor:
    """Flatten recorded micro-batch routes into one int16 jagged tensor."""
    tensors = [tensor.to(torch.int16) for router_map in router_maps for tensor in router_map.unbind()]
    return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)


def get_current_rank_layer_info(tf_config, vp_rank=None):
    # When vp_rank is None, default to the current VP rank (or 0 if VP is disabled).
    """Return the local layer range/count for the current process and the full assignment table.

    Args:
        tf_config: Configuration object used by compute_pipeline_layer_assignment.
        vp_rank (Optional[int]): Explicit virtual pipeline stage rank to query. If None, uses
            mpu.get_virtual_pipeline_model_parallel_rank() when VP is enabled; otherwise 0.

    Returns:
        Tuple[dict, dict]: A tuple of (local_assignment, all_assignments) where local_assignment contains
        keys {"start", "end", "count"} for the current (pp_rank, vp_stage).
    """
    if vp_rank is None:
        vp_rank = 0
    num_layers_to_build = get_num_layers_to_build(tf_config, vp_stage=vp_rank)

    sig = inspect.signature(get_transformer_layer_offset)

    if "vp_stage" in sig.parameters:
        offset = get_transformer_layer_offset(tf_config, vp_stage=vp_rank)
    else:
        offset = get_transformer_layer_offset(tf_config)
    local = {}
    local["start"] = offset
    local["end"] = offset + num_layers_to_build
    local["count"] = num_layers_to_build
    return local


def pp_gather(local_layers_router_map, tf_config):
    # TODO: Consider non-uniform layer allocation cases.
    """
    Gather local router maps from all PP ranks into a global router map.

    Args:
        local_layers_router_map (torch.Tensor): Local router map of shape
            [bs, max_seq_len, local_num_layers, topk].
        tf_config: Configuration providing pipeline_model_parallel_size.

    Returns:
        torch.Tensor: Global router map of shape [bs, max_seq_len, num_layers, topk] placed on CPU.
    """
    pp_size = tf_config.pipeline_model_parallel_size
    if pp_size <= 1:
        return local_layers_router_map

    pp_group = mpu.get_pipeline_model_parallel_group()
    world_size = torch.distributed.get_world_size(pp_group)
    if local_layers_router_map.is_nested:
        # all_gather_object preserves each sender's CUDA device, so normalize
        # jagged route maps before concatenating routes from local PP ranks.
        local_layers_router_map = local_layers_router_map.to("cpu")
        layers_topk_idx_global_list = [None] * world_size
        torch.distributed.all_gather_object(layers_topk_idx_global_list, local_layers_router_map, pp_group)
    else:
        # NCCL has no int16 datatype, so gather a bit-identical uint8 view and undo it
        # before the layer-dimension concat below. The nested branch above needs none
        # of this: all_gather_object pickles the tensor.
        payload = local_layers_router_map.to(device_name).contiguous().view(torch.uint8)
        layers_topk_idx_global_list = [torch.empty_like(payload) for _ in range(world_size)]
        torch.distributed.all_gather(
            tensor=payload,
            tensor_list=layers_topk_idx_global_list,
            group=pp_group,
            async_op=False,
        )
        layers_topk_idx_global_list = [t.view(torch.int16) for t in layers_topk_idx_global_list]
    vp_size = tf_config.virtual_pipeline_model_parallel_size
    if vp_size is not None:
        vpp_router_map_offset = [[] for _ in range(pp_size)]
        for pp_stage in range(pp_size):
            vpp_router_map_offset[pp_stage].append(0)
            for vp_stage in range(vp_size):
                num_layers_to_build = get_moe_num_layers_to_build(tf_config, vp_stage, pp_stage)
                vpp_router_map_offset[pp_stage].append(num_layers_to_build + vpp_router_map_offset[pp_stage][-1])
        layers_topk_idx_global = []
        for vp_stage in range(vp_size):
            for pp_stage in range(pp_size):
                piece = slice(vpp_router_map_offset[pp_stage][vp_stage], vpp_router_map_offset[pp_stage][vp_stage + 1])
                if layers_topk_idx_global_list[pp_stage].is_nested:
                    nested_item = layers_topk_idx_global_list[pp_stage]
                    sliced_tensors = [t[:, piece, :] for t in nested_item.unbind()]
                    sliced_nested = torch.nested.as_nested_tensor(sliced_tensors, layout=torch.jagged)
                    layers_topk_idx_global.append(sliced_nested)
                else:
                    layers_topk_idx_global.append(layers_topk_idx_global_list[pp_stage][:, :, piece, :])
        global_router_map = torch.cat(layers_topk_idx_global, dim=2).to("cpu")
    else:
        global_router_map = torch.cat(layers_topk_idx_global_list, dim=2).to("cpu")

    return global_router_map


class RouterReplayHelper:
    """Helper class to query router replay state and locate local RouterReplay instances."""

    @staticmethod
    def get_micro_batch_router_list(tf_config, vp_rank=None):
        """
        Return the list of RouterReplay instances corresponding to the current micro-batch and local
        (pp_rank, vp_stage) layer range.

        When virtual pipeline (VPP) is enabled, the local range for the PP rank is expanded to include
        all VP stages by multiplying the per-VP count by vp_size. The returned slice is taken from the
        global RouterReplay.router_instances list.

        Args:
            tf_config: Configuration object used to compute layer assignments.
            vp_rank (Optional[int]): Explicit virtual pipeline stage to query. If None, the current VP
                rank from Megatron parallel state is used when available.
        Returns:
            list: A contiguous sublist of RouterReplay.router_instances for the local layer range.
        """
        local_rank_info = get_current_rank_layer_info(tf_config, vp_rank)
        start, end = local_rank_info["start"], local_rank_info["end"]
        offset = sum(1 for layer_idx in range(start) if is_moe_layer(tf_config, layer_idx))
        num_layers_to_build = sum(1 for layer_idx in range(start, end) if is_moe_layer(tf_config, layer_idx))

        # Some older Megatron builds only register local router instances in each
        # PP worker, while Qwen3-Omni currently registers the full PP model in
        # this process. Support both layouts, but prefer the global layer offset
        # when the registry is large enough.
        if len(RouterReplay.router_instances) == num_layers_to_build:
            offset = 0

        router_instances_list = RouterReplay.router_instances[offset : offset + num_layers_to_build]
        return router_instances_list

    @staticmethod
    def is_r2_record_action(tf_config, vp_rank=None) -> bool:
        """Return True if the current router_replay_action is RECORD (R2) for the local router instances.

        This inspects the first local RouterReplay instance's router_replay_action and compares it to
        RouterReplayAction.RECORD.
        """
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return router_instances_list and router_instances_list[0].router_replay_action == RouterReplayAction.RECORD

    @staticmethod
    def is_replay_forward_action(tf_config, vp_rank=None) -> bool:
        """Return True if the current router_replay_action is REPLAY_FORWARD for the local router instances.

        This inspects the first local RouterReplay instance's router_replay_action and compares it to
        RouterReplayAction.REPLAY_FORWARD.
        """
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return (
            router_instances_list and router_instances_list[0].router_replay_action == RouterReplayAction.REPLAY_FORWARD
        )

    @staticmethod
    def is_replay_backward_action(tf_config, vp_rank=None) -> bool:
        """Return True if the current router_replay_action is REPLAY_BACKWARD for the local router instances.

        This inspects the first local RouterReplay instance's router_replay_action and compares it to
        RouterReplayAction.REPLAY_BACKWARD.
        """
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return (
            router_instances_list
            and router_instances_list[0].router_replay_action == RouterReplayAction.REPLAY_BACKWARD
        )
