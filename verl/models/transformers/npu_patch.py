# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team
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


import torch
import torch.nn.functional as F
import torch_npu
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3_moe import modeling_qwen3_moe
from transformers.models.qwen3_next import modeling_qwen3_next
from transformers.models.qwen3_vl import modeling_qwen3_vl
from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe
from transformers.utils import logging

logger = logging.get_logger(__name__)


def rms_norm_forward_npu(self, x):
    """NPU optimized implementation for RMSNorm."""
    if x.dtype != self.weight.dtype:
        x = x.to(self.weight.dtype)
    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]


def silu_forward_npu(self, hidden_state):
    """NPU optimized implementation for SiLU in `forward` func in MLP layer."""
    gate_up = torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1)
    return self.down_proj(torch_npu.npu_swiglu(gate_up, dim=-1))


def apply_rotary_pos_emb_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """NPU optimized implementation for RoPE."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def qwen3_next_rms_norm_forward_npu(self, x):
    return torch_npu.npu_rms_norm(x.float(), 1.0 + self.weight.float(), epsilon=self.eps)[0].type_as(x)


def qwen3_next_rms_norm_forward_gated_npu(self, hidden_states, gate=None):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    hidden_states = torch_npu.npu_rms_norm(hidden_states, self.weight.float(), epsilon=self.variance_epsilon)[0]
    hidden_states = hidden_states * F.silu(gate.to(torch.float32))
    return hidden_states.to(input_dtype)


def qwen3_next_apply_rotary_pos_emb_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = torch_npu.npu_rotary_mul(q_rot, cos, sin).to(q.dtype)
    k_embed = torch_npu.npu_rotary_mul(k_rot, cos, sin).to(k.dtype)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class NPUGmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, group_list, group_list_type=1):
        """
        Grouped Matmul(GMM) for Ascend NPU.

        Args:
            x (torch.Tensor): Input tensor, shape (tokens_num * top_k, hidden_size)
            weight (torch.Tensor): Expert weights, shape (n_experts, hidden_size, intermediate_size)
            group_list (torch.Tensor): Expert token counts, shape (n_experts,)
                - type 0: cumsum of tokens per expert
                - type 1: direct tokens per expert (default)
        """
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list
        ctx.group_list_type = group_list_type

        output = torch_npu.npu_grouped_matmul(
            [x], [weight], bias=None, group_list=group_list, split_item=2, group_type=0, group_list_type=group_list_type
        )[0]

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        group_list = ctx.group_list
        group_list_type = ctx.group_list_type

        dx = torch_npu.npu_grouped_matmul(
            [grad_output],
            [weight.transpose(1, 2)],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=group_list_type,
        )[0]

        dw = torch_npu.npu_grouped_matmul(
            [x.transpose(0, 1)],
            [grad_output],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=2,
            group_list_type=group_list_type,
        )[0]

        return dx, dw, None, None


def _qwen3_sparse_moe_routed_forward_npu(self, hidden_states: torch.Tensor):
    """
    Shared NPU routed-expert path for Qwen3Moe/Qwen3Next sparse MoE blocks.

    Returns:
        tuple: (flattened_input, routed_hidden_states, router_logits)
    """
    hidden_dim = hidden_states.shape[-1]
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    # Loop over all available experts in the model and perform the computation on each expert
    # Concat all weights
    input_dtype = hidden_states.dtype
    up_weight_list = [e.up_proj.weight for e in self.experts]
    gate_weight_list = [e.gate_proj.weight for e in self.experts]
    down_weight_list = [e.down_proj.weight for e in self.experts]
    w1 = torch.stack(up_weight_list).transpose(1, 2).to(input_dtype)
    w2 = torch.stack(gate_weight_list).transpose(1, 2).to(input_dtype)
    w3 = torch.stack(down_weight_list).transpose(1, 2).to(input_dtype)

    permuted_tokens, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states, selected_experts.to(torch.int32))
    tokens_per_expert = torch.histc(selected_experts, bins=self.num_experts, min=0, max=self.num_experts)

    up_res = NPUGmmFunction.apply(permuted_tokens, w1, tokens_per_expert)
    gate_res = NPUGmmFunction.apply(permuted_tokens, w2, tokens_per_expert)
    act_res = torch_npu.npu_swiglu(torch.cat([gate_res, up_res], dim=-1))
    down_res = NPUGmmFunction.apply(act_res, w3, tokens_per_expert)

    routed_hidden_states = torch_npu.npu_moe_token_unpermute(down_res, row_ids_map, probs=routing_weights)

    return hidden_states, routed_hidden_states, router_logits


def qwen3_moe_sparse_moe_block_forward_npu(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """NPU optimized implementation for `forward` in Qwen3MoeSparseMoeBlock."""
    output_shape = hidden_states.shape
    _, routed_hidden_states, router_logits = _qwen3_sparse_moe_routed_forward_npu(self, hidden_states)
    final_hidden_states = routed_hidden_states.reshape(output_shape)
    return final_hidden_states, router_logits


def qwen3_next_sparse_moe_block_forward_npu(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """NPU optimized implementation for `forward` in Qwen3NextSparseMoeBlock."""
    output_shape = hidden_states.shape
    hidden_states, routed_hidden_states, router_logits = _qwen3_sparse_moe_routed_forward_npu(self, hidden_states)

    shared_expert_output = self.shared_expert(hidden_states)
    shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

    final_hidden_states = (routed_hidden_states + shared_expert_output).reshape(output_shape)
    return final_hidden_states, router_logits


class NPUQwen3VLMoeTextExperts(nn.Module):
    """NPU optimized implementation for Qwen3VLMoeTextExperts."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        When training it is more efficient to just loop over the experts and compute the output for each expert
        as otherwise the memory would explode.

        For inference we can sacrifice some memory and compute the output for all experts at once.
        By repeating the inputs.

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            routing_weights (torch.Tensor): (batch_size * token_num, num_experts)
            router_indices (torch.Tensor): (batch_size * token_num, top_k)
        Returns:
            torch.Tensor
        """
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        if self.training:
            permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(
                hidden_states, router_indices.to(torch.int32)
            )
            tokens_per_expert = torch.histc(router_indices, bins=self.num_experts, min=0, max=self.num_experts)
            intermediate_hidden_states = NPUGmmFunction.apply(
                permuted_hidden_states, self.gate_up_proj, tokens_per_expert
            )
            intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
            output = NPUGmmFunction.apply(intermediate_activations, self.down_proj, tokens_per_expert)
            num_tokens = hidden_states.shape[0]
            top_k = router_indices.shape[1]
            batch_idx = torch.arange(num_tokens, device=routing_weights.device)
            batch_idx = batch_idx.unsqueeze(1).expand(-1, top_k)
            selected_probs = routing_weights[batch_idx, router_indices]
            next_states = torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=selected_probs)
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            hidden_states = hidden_states.repeat(self.num_experts, 1)
            hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
            gate_up = torch.bmm(hidden_states, self.gate_up_proj)
            gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
            next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)
            next_states = next_states.reshape(self.num_experts, batch_size, -1, self.hidden_size)
            next_states = (
                next_states * routing_weights.transpose(0, 1).view(self.num_experts, batch_size, -1)[..., None]
            )
            next_states = next_states.sum(dim=0)
        return next_states


class NPUQwen3VLMoeTextSparseMoeBlock(nn.Module):
    """NPU optimized implementation for Qwen3VLMoeTextSparseMoeBlock."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = NPUQwen3VLMoeTextExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
        if not self.training:
            routing_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)
        routed_out = self.experts(hidden_states, routing_weights, router_indices)
        return routed_out


# Patches for Qwen2 Model
modeling_qwen2.Qwen2RMSNorm.forward = rms_norm_forward_npu
modeling_qwen2.Qwen2MLP.forward = silu_forward_npu
modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb_npu

# Patches for Qwen2.5-VL Model
modeling_qwen2_5_vl.Qwen2RMSNorm.forward = rms_norm_forward_npu
modeling_qwen2_5_vl.Qwen2_5_VLMLP.forward = silu_forward_npu

# Patches for Qwen3 Model
modeling_qwen3.Qwen3RMSNorm.forward = rms_norm_forward_npu
modeling_qwen3.Qwen3MLP.forward = silu_forward_npu
modeling_qwen3.apply_rotary_pos_emb = apply_rotary_pos_emb_npu

# Patches for Qwen3 MoE Model
modeling_qwen3_moe.Qwen3MoeRMSNorm.forward = rms_norm_forward_npu
modeling_qwen3_moe.Qwen3MoeSparseMoeBlock.forward = qwen3_moe_sparse_moe_block_forward_npu
modeling_qwen3_moe.apply_rotary_pos_emb = apply_rotary_pos_emb_npu

# Patches for Qwen3 VL Model
modeling_qwen3_vl.Qwen3VLTextRMSNorm.forward = rms_norm_forward_npu
modeling_qwen3_vl.Qwen3VLTextMLP.forward = silu_forward_npu

# Patches for Qwen3-VL MoE Model
modeling_qwen3_vl_moe.Qwen3VLMoeTextSparseMoeBlock = NPUQwen3VLMoeTextSparseMoeBlock
modeling_qwen3_vl_moe.Qwen3VLMoeTextRMSNorm.forward = rms_norm_forward_npu
modeling_qwen3_vl_moe.apply_rotary_pos_emb = apply_rotary_pos_emb_npu

# Patches for Qwen3 Next Model
modeling_qwen3_next.Qwen3NextSparseMoeBlock.forward = qwen3_next_sparse_moe_block_forward_npu
modeling_qwen3_next.Qwen3NextRMSNormGated.forward = qwen3_next_rms_norm_forward_gated_npu
modeling_qwen3_next.Qwen3NextRMSNorm.forward = qwen3_next_rms_norm_forward_npu
modeling_qwen3_next.apply_rotary_pos_emb = qwen3_next_apply_rotary_pos_emb_npu
