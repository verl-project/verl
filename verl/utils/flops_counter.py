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

import inspect

import torch
from transformers import PretrainedConfig

from verl.utils.device import get_torch_device

_DEVICE_FLOPS = {
    "CPU": 448e9,
    "GB200": 2.5e15,
    "B200": 2.25e15,
    "MI300X": 1307e12,
    "MI350X": 2300e12,
    "MI355X": 2500e12,
    "H100": 989e12,
    "H800": 989e12,
    "H200": 989e12,
    "A100": 312e12,
    "A800": 312e12,
    "L40S": 362.05e12,
    "L40": 181.05e12,
    "A40": 149.7e12,
    "L20": 119.5e12,
    "H20": 148e12,
    "910B": 354e12,
    "A2G3": 354e12,
    "Ascend950DT": 432e12,
    "Ascend910": 354e12,
    "RTX 3070 Ti": 21.75e12,
}


def get_device_flops(unit="T", device_name=None):
    """Get the theoretical FLOPS (Floating Point Operations Per Second) capacity of the current device.

    Args:
        unit (str): The unit to return the FLOPS in. Supported values are:
            "B" - Billion (1e9)
            "K" - Thousand (1e3)
            "M" - Million (1e6)
            "G" - Giga (1e9)
            "T" - Tera (1e12, default)
            "P" - Peta (1e15)

    Returns:
        float: The theoretical FLOPS capacity of the current device in the specified unit.
        Returns float('inf') for unknown GPU types.
    """

    def unit_convert(number, level):
        units = ["B", "K", "M", "G", "T", "P"]
        if number <= 0:
            return number
        ptr = 0
        while ptr < len(units) and units[ptr] != level:
            number /= 1000
            ptr += 1
        return number

    # pass device_name is for testing purpose only
    if device_name is None:
        device = get_torch_device()
        if device == torch.cpu:
            device_name = "CPU"
        else:
            device_name = get_torch_device().get_device_name()

    flops = float("inf")  # INF flops for unkown gpu type

    for key, value in sorted(_DEVICE_FLOPS.items(), reverse=True):
        if key in device_name:
            flops = value
            break
    flops_unit = unit_convert(flops, unit)
    return flops_unit


def _estimate_qwen2_flops(config, tokens_sum, batch_seqlens, delta_time):
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # non-attn per layer parm
    # Qwen2/LLama use SwiGelu, gate, having up and down linear layer in mlp
    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer parm
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * dense_N * tokens_sum

    # attn all_layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    # all_layer & all_token fwd & bwd flops
    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_qwen3_vl_flops(config, tokens_sum, batch_seqlens, delta_time, **kargs):
    # qwen3_vl uses text_config and vision_config to distinguish configs of different parts.
    hidden_size = config.text_config.hidden_size
    vocab_size = config.text_config.vocab_size
    num_hidden_layers = config.text_config.num_hidden_layers
    num_key_value_heads = config.text_config.num_key_value_heads
    num_attention_heads = config.text_config.num_attention_heads
    intermediate_size = config.text_config.intermediate_size

    head_dim = hidden_size // num_attention_heads
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # non-attn per layer parm
    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer parm
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * dense_N * tokens_sum

    # qwen3_vl uses deepstack to merge visual embeds and text embeds, but it has no tensor operation.

    # attn all_layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    # vit flops
    images_seqlens = kargs.get("images_seqlens", None)
    if images_seqlens is not None:
        vit_flops = _estimate_qwen3_vit_flop(images_seqlens, config.vision_config)
    else:
        vit_flops = 0

    # all_layer & all_token fwd & bwd flops
    flops_all_token = dense_N_flops + attn_qkv_flops + vit_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_qwen3_vl_moe_flops(config, tokens_sum, batch_seqlens, delta_time, **kargs):
    # qwen3_vl uses text_config and vision_config to distinguish configs of different parts.
    hidden_size = config.text_config.hidden_size
    vocab_size = config.text_config.vocab_size
    num_hidden_layers = config.text_config.num_hidden_layers
    num_key_value_heads = config.text_config.num_key_value_heads
    num_attention_heads = config.text_config.num_attention_heads
    moe_intermediate_size = config.text_config.moe_intermediate_size
    moe_num_expert = config.text_config.num_experts
    moe_topk = config.text_config.num_experts_per_tok

    head_dim = getattr(
        config.text_config, "head_dim", config.text_config.hidden_size // config.text_config.num_attention_heads
    )
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # non-attn per layer parm
    moe_gata_N = hidden_size * moe_num_expert
    # moe has gate_proj, up_proj and down_proj using SwiGLU in ExpertMlp layer & shared experts
    moe_expertmlp_N = hidden_size * moe_intermediate_size * (moe_topk) * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer parm
    moe_N = (moe_gata_N + moe_expertmlp_N + attn_linear_N) * (num_hidden_layers) + emd_and_lm_head_N
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * moe_N * tokens_sum

    # attn all_layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    # vit flops
    images_seqlens = kargs.get("images_seqlens", None)
    if images_seqlens is not None:
        vit_flops = _estimate_qwen3_vit_flop(images_seqlens, config.vision_config)
    else:
        vit_flops = 0

    # all_layer & all_token fwd & bwd flops
    flops_all_token = dense_N_flops + attn_qkv_flops + vit_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_qwen3_vit_flop(images_seqlens, config):
    """
    Estimate the FLOPS of the vision encoder for Qwen3-VL
    """

    if config is None:
        return 0
    tokens_sum = sum(images_seqlens)

    num_heads = config.num_heads
    depth = config.depth

    dim = config.hidden_size
    mlp_hidden_dim = config.intermediate_size
    out_hidden_size = config.out_hidden_size

    spatial_merge_size = config.spatial_merge_size

    head_dim = dim // num_heads

    # every vision token's patch_embed comes from a conv of (C, T, H, W) -> (dim,)
    patch_embed_N = dim * config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size
    # Qwen3 VL vision mlp does not use GLU, thus 2.
    mlp_N = dim * mlp_hidden_dim * 2
    attn_linear_N = dim * (4 * dim)  # qkv and output proj
    merger_N = (out_hidden_size + (dim * (spatial_merge_size**2))) * (dim * (spatial_merge_size**2))

    # Qwen3 VL uses deep stack, one merger for every deepstack layer
    if getattr(config, "deepstack_visual_indexes", None) is not None:
        deepstack_merger_N = merger_N * len(config.deepstack_visual_indexes)
    else:
        deepstack_merger_N = 0
    # non-attn all_layer parm
    dense_N = patch_embed_N + (mlp_N + attn_linear_N) * depth + deepstack_merger_N + merger_N

    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * dense_N * tokens_sum

    # In Qwen3 VL, full attention is used in all vision layers.
    full_attn_layer_num = depth

    # full attn layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in images_seqlens:
        seqlen_square_sum += seqlen * seqlen
    attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_heads * full_attn_layer_num

    vit_flops = dense_N_flops + attn_qkv_flops

    return vit_flops


def _estimate_deepseek_v3_flops(config, tokens_sum, batch_seqlens, delta_time):
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    moe_intermediate_size = config.moe_intermediate_size
    num_hidden_layers = config.num_hidden_layers
    first_k_dense_replace = config.first_k_dense_replace
    num_query_heads = config.num_attention_heads
    moe_num_expert = config.n_routed_experts

    moe_topk = config.num_experts_per_tok
    share_expert_num = config.n_shared_experts

    # non-attn per layer parm
    moe_gata_N = hidden_size * moe_num_expert
    # moe has fc1_1, fc1_2 and fc2 using SwiGLU in ExpertMlp layer & shared experts
    moe_expertmlp_N = hidden_size * moe_intermediate_size * (moe_topk + share_expert_num) * 3
    # MLA attn
    attn_linear_N = 0
    q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    if config.q_lora_rank is None:
        attn_linear_N += hidden_size * num_query_heads * q_head_dim
    else:
        attn_linear_N += hidden_size * config.q_lora_rank
        attn_linear_N += num_query_heads * q_head_dim * config.q_lora_rank

    attn_linear_N += hidden_size * (config.kv_lora_rank + config.qk_rope_head_dim)
    attn_linear_N += num_query_heads * (q_head_dim - config.qk_rope_head_dim + config.v_head_dim) * config.kv_lora_rank
    attn_linear_N += num_query_heads * config.v_head_dim * hidden_size
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer parm
    moe_N = (
        (moe_gata_N + moe_expertmlp_N + attn_linear_N) * (num_hidden_layers - first_k_dense_replace)
        + (hidden_size * config.intermediate_size * 3 + attn_linear_N) * first_k_dense_replace
        + emd_and_lm_head_N
    )
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * moe_N * tokens_sum

    # attn all_layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen * num_hidden_layers

    # Core attention FLOPS for MLA with causal mask:
    # Q @ K^T: 3 * 2 * seq^2 * q_head_dim * num_heads / 2 (causal)
    # attn @ V: 3 * 2 * seq^2 * v_head_dim * num_heads / 2 (causal)
    attn_qkv_flops = 3 * seqlen_square_sum * (q_head_dim + config.v_head_dim) * num_query_heads
    # all_layer & all_token fwd & bwk flops
    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12

    return flops_achieved


def _estimate_qwen2_moe_flops(config, tokens_sum, batch_seqlens, delta_time):
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    moe_intermediate_size = config.moe_intermediate_size
    moe_topk = config.num_experts_per_tok
    num_experts = config.num_experts

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # non-attn per layer parm
    # gate + moe export
    moe_mlp_N = hidden_size * moe_topk * moe_intermediate_size * 3 + hidden_size * num_experts
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer parm
    dense_N = (moe_mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * dense_N * tokens_sum

    # attn all_layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    # all_layer & all_token fwd & bwd flops
    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_gemma3_flops(config, tokens_sum, batch_seqlens, delta_time):
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # non-attn per layer parm
    # Gemma3 uses GeGLU (gelu_pytorch_tanh), having 3 matrices in MLP (inherited from Gemma2MLP)
    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer parm
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * dense_N * tokens_sum

    # attn all_layer & all_token fwd & bwd flops
    # Gemma3 alternates between full and sliding window attention based on layer_types
    seqlen_square_sum = 0

    layer_types = getattr(config, "layer_types", None)
    sliding_window = getattr(config, "sliding_window", 1024)  # default 1024
    # default pattern: every 6th layer is full
    sliding_window_pattern = getattr(config, "sliding_window_pattern", 6)

    # If layer_types is not provided, generate it based on sliding_window_pattern
    if layer_types is None and sliding_window is not None and sliding_window_pattern is not None:
        layer_types = [
            "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
            for i in range(num_hidden_layers)
        ]

    if layer_types:
        # Calculate attention flops per layer based on attention type
        for layer_idx in range(num_hidden_layers):
            is_sliding = False
            if layer_types and layer_idx < len(layer_types):
                is_sliding = layer_types[layer_idx] == "sliding_attention"

            for seqlen in batch_seqlens:
                if is_sliding and sliding_window:
                    # Sliding window limits each token to attend to at most window_size tokens
                    effective_seqlen = min(seqlen, sliding_window)
                    seqlen_square_sum += seqlen * effective_seqlen
                else:
                    # Full attention
                    seqlen_square_sum += seqlen * seqlen
    else:
        # If no layer_types config, assume all layers use full attention
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        seqlen_square_sum *= num_hidden_layers

    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads

    # all_layer & all_token fwd & bwd flops
    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_apertus_flops(config, tokens_sum, batch_seqlens, delta_time):
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # Apertus MLP with XIELU activation uses only 2 linear layers (up_proj, down_proj)
    # No gate_proj for XIELU, unlike SwiGLU which has 3 layers
    mlp_N = hidden_size * intermediate_size * 2
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)

    # ApertusConfig has qk_norm defaulting to True.
    # This adds params for q_norm (on H) and k_norm (on num_kv_heads * head_dim)
    qk_norm_params_per_layer = hidden_size + num_key_value_heads * head_dim  # q_norm + k_norm

    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer params
    dense_N = (mlp_N + attn_linear_N + qk_norm_params_per_layer) * num_hidden_layers + emd_and_lm_head_N
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * dense_N * tokens_sum

    # attn all_layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    # all_layer & all_token fwd & bwd flops
    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_gpt_oss_flops(config, tokens_sum, batch_seqlens, delta_time):
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads

    # MoE params
    moe_intermediate_size = config.intermediate_size
    num_experts = config.num_local_experts
    num_experts_per_tok = config.num_experts_per_tok
    mlp_matrices = 3

    # Head dim
    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # 1. Attention Block (GQA)
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    # 2. MLP / MoE Block
    # Gate network
    moe_gate_N = hidden_size * num_experts
    # Expert forward calculation, Active parameters: mlp_matrices * H * I * num_experts_per_tok
    moe_expert_N = hidden_size * moe_intermediate_size * mlp_matrices * num_experts_per_tok

    moe_mlp_N = moe_gate_N + moe_expert_N

    emd_and_lm_head_N = vocab_size * hidden_size * 2

    # Total non-attn params per layer * layers + embeddings
    # (moe_mlp_N + attn_linear_N) * layers
    dense_N = (moe_mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N

    # FLOPs for dense part (fwd + bwd = 6 * N)
    dense_N_flops = 6 * dense_N * tokens_sum

    # 3. Attention Matrix FLOPs
    seqlen_square_sum = 0

    # Handle sliding window attention
    layer_types = getattr(config, "layer_types", None)
    sliding_window = getattr(config, "sliding_window", 128)

    if layer_types:
        for layer_type in layer_types:
            is_sliding = layer_type == "sliding_attention"

            for seqlen in batch_seqlens:
                if is_sliding and sliding_window:
                    # Sliding window limits each token to attend to at most window_size tokens
                    effective_seqlen = min(seqlen, sliding_window)
                    seqlen_square_sum += seqlen * effective_seqlen
                else:
                    # Full attention
                    seqlen_square_sum += seqlen * seqlen
    else:
        # Default to full attention for all layers
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        seqlen_square_sum *= num_hidden_layers

    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads

    # Total FLOPs
    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_qwen3_5_flops(config, tokens_sum, batch_seqlens, delta_time, **kargs):
    # Qwen3.5 (model_type "qwen3_5"): hybrid attention + dense SwiGLU MLP + 1 MTP layer + ViT.
    # Every `full_attention_interval`-th layer is standard softmax attention (O(L^2)); the rest
    # are GatedDeltaNet linear-attention layers (O(L), no seqlen-square term). Fields live under
    # config.text_config (top level also holds vision_config), matching qwen3_next in transformers.
    text_config = getattr(config, "text_config", config)

    hidden_size = text_config.hidden_size
    vocab_size = text_config.vocab_size
    num_hidden_layers = text_config.num_hidden_layers
    num_key_value_heads = text_config.num_key_value_heads
    num_attention_heads = text_config.num_attention_heads
    intermediate_size = text_config.intermediate_size
    head_dim = getattr(text_config, "head_dim", hidden_size // num_attention_heads)

    # Split layers into full vs linear. Prefer the explicit per-layer list; fall back to the
    # interval so a config that only sets full_attention_interval still works.
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is not None:
        num_full_layers = sum(1 for t in layer_types if t == "full_attention")
    else:
        interval = getattr(text_config, "full_attention_interval", 1)
        num_full_layers = num_hidden_layers // interval
    num_linear_layers = num_hidden_layers - num_full_layers

    # --- dense SwiGLU MLP (present in every layer) + embedding/lm_head (once) ---
    mlp_N = hidden_size * intermediate_size * 3
    emd_and_lm_head_N = vocab_size * hidden_size * 2

    # --- full-attention layers: q_proj is doubled when attn_output_gate is set (q + gate) ---
    q_mult = 2 if getattr(text_config, "attn_output_gate", False) else 1
    q_size = num_attention_heads * head_dim * q_mult
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim
    o_size = num_attention_heads * head_dim
    attn_linear_N_full = hidden_size * (q_size + k_size + v_size + o_size)

    # --- linear-attention (GatedDeltaNet) layers: in_proj(qkvz) + in_proj(ba) + conv + out_proj ---
    key_dim = text_config.linear_key_head_dim * text_config.linear_num_key_heads
    value_dim = text_config.linear_value_head_dim * text_config.linear_num_value_heads
    conv_kernel = text_config.linear_conv_kernel_dim
    in_proj_qkvz_N = hidden_size * (2 * key_dim + 2 * value_dim)
    in_proj_ba_N = hidden_size * (2 * text_config.linear_num_value_heads)
    conv_N = (2 * key_dim + value_dim) * conv_kernel  # depthwise conv1d
    out_proj_N = value_dim * hidden_size
    linear_attn_N = in_proj_qkvz_N + in_proj_ba_N + conv_N + out_proj_N

    # --- MTP head: approximate as one extra dense decoder layer (full attn linear + MLP) ---
    mtp_layers = getattr(text_config, "mtp_num_hidden_layers", 0)
    mtp_N = (attn_linear_N_full + mlp_N) * mtp_layers

    # non-attn parameter count across the whole model (weight-only linear algebra)
    dense_N = (
        mlp_N * num_hidden_layers
        + attn_linear_N_full * num_full_layers
        + linear_attn_N * num_linear_layers
        + emd_and_lm_head_N
        + mtp_N
    )
    dense_N_flops = 6 * dense_N * tokens_sum

    # --- softmax attention QK^T/AV: O(L^2), applied ONLY to full-attention layers ---
    seqlen_square_sum = 0
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_full_layers

    # --- GatedDeltaNet recurrence core: O(L) chunked delta rule. Approximated (chunk_size
    # implementation detail omitted); scales as tokens * per-head state (k_dim * v_dim). This
    # term is non-dominant vs the linear projections above (< 1% for the 9B config). ---
    linear_recurrence_flops = (
        6
        * tokens_sum
        * text_config.linear_num_value_heads
        * text_config.linear_key_head_dim
        * text_config.linear_value_head_dim
        * num_linear_layers
    )

    # --- ViT: only when images are actually in the batch (text-only training skips it) ---
    images_seqlens = kargs.get("images_seqlens", None)
    if images_seqlens is not None:
        vit_flops = _estimate_qwen3_vit_flop(images_seqlens, config.vision_config)
    else:
        vit_flops = 0

    flops_all_token = dense_N_flops + attn_qkv_flops + linear_recurrence_flops + vit_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_unknown_flops(config, tokens_sum, batch_seqlens, delta_time):
    return 0


ESTIMATE_FUNC = {
    "qwen2": _estimate_qwen2_flops,
    "llama": _estimate_qwen2_flops,
    "qwen2_moe": _estimate_qwen2_moe_flops,
    "qwen2_vl": _estimate_qwen3_vl_flops,
    "qwen2_5_vl": _estimate_qwen3_vl_flops,
    "qwen3": _estimate_qwen2_flops,
    "qwen3_moe": _estimate_qwen2_moe_flops,
    "qwen3_vl": _estimate_qwen3_vl_flops,
    "qwen3_vl_moe": _estimate_qwen3_vl_moe_flops,
    "qwen3_5": _estimate_qwen3_5_flops,
    "deepseek_v3": _estimate_deepseek_v3_flops,
    "minicpmv": _estimate_qwen2_flops,
    "minicpmo": _estimate_qwen2_flops,
    "mistral": _estimate_qwen2_flops,
    "gemma3_text": _estimate_gemma3_flops,
    "seed_oss": _estimate_qwen2_flops,
    "apertus": _estimate_apertus_flops,
    "glm4v": _estimate_qwen2_flops,
    "gpt_oss": _estimate_gpt_oss_flops,
    "mimo": _estimate_qwen2_flops,
}


class FlopsCounter:
    """
    Used to count mfu during training loop

    Example:
        flops_counter = FlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(tokens_list, delta_time)

    """

    def __init__(self, config: PretrainedConfig):
        VALID_CONFIG_TYPE = ESTIMATE_FUNC.keys()
        if config.model_type not in VALID_CONFIG_TYPE:
            print(
                f"Only support config type of {VALID_CONFIG_TYPE}, but got {config.model_type}. MFU will always be "
                f"zero."
            )

        self.config = config

    # TODO: actually we can make this a static method
    def estimate_flops(self, batch_seqlens, delta_time, **kargs):
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the
                current batch.
            delta_time (float): The time taken to process the batch, in seconds.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        tokens_sum = sum(batch_seqlens)
        func = ESTIMATE_FUNC.get(self.config.model_type, _estimate_unknown_flops)
        sig = inspect.signature(func)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            estimated_flops = func(self.config, tokens_sum, batch_seqlens, delta_time, **kargs)
        else:
            estimated_flops = func(self.config, tokens_sum, batch_seqlens, delta_time)
        promised_flops = get_device_flops()
        return estimated_flops, promised_flops
