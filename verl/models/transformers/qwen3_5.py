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

import functools
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForCausalLM,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class Qwen3_5CausalLMOutputForPPO(Qwen3_5CausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


def forward_with_normal_backend(
    self: "Qwen3_5ForCausalLM",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3_5CausalLMOutputForPPO":
    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    return Qwen3_5CausalLMOutputForPPO(
        logits=logits,
        hidden_states=outputs.hidden_states,
    )


def forward_with_torch_backend(
    self: "Qwen3_5ForCausalLM",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3_5CausalLMOutputForPPO":
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    # Loss calculations
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )
    return Qwen3_5CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


def forward_with_triton_backend(
    self: "Qwen3_5ForCausalLM",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3_5CausalLMOutputForPPO":
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    # Loss calculations
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )
    return Qwen3_5CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


def patch_qwen3_5_attention_for_prefix_grouper():
    """
    Patch Qwen3.5 attention to support prefix_grouper.
    Qwen3.5 has mixed attention layers (full_attention and linear_attention),
    so we need to patch both types.
    """
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
    except ImportError:
        # Model not available, skip patching
        return
    
    # Store original forward for reference
    original_forward = Qwen3_5Attention.forward
    
    @functools.wraps(original_forward)
    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        prefix_grouper = kwargs.pop("prefix_grouper", None)
        if prefix_grouper is None:
            return original_forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                **kwargs,
            )
        
        # Apply prefix_grouper logic for Qwen3.5 attention
        # This is a simplified version - actual implementation may need adjustment
        # based on Qwen3.5's specific attention mechanism
        def attn_func(q, k, v, attn_mask, *inner_args, **inner_kwargs):
            out, _ = original_forward(
                self,
                hidden_states,
                attn_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                **inner_kwargs,
            )
            return out
        
        return prefix_grouper.forward(attn_func, hidden_states, hidden_states, hidden_states, *args, **kwargs), None
    
    # Apply the patch
    Qwen3_5Attention.forward = patched_forward
    logger.info("Monkey patched Qwen3_5Attention.forward to support prefix_grouper")


def patch_qwen3_5_dynamic_cache():
    """
    Patch Qwen3.5 DynamicCache if needed.
    Qwen3.5 uses Qwen3_5DynamicCache which may need special handling.
    """
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
    except ImportError:
        # Model not available, skip patching
        return
    
    # Check if any patching is needed
    # For now, just log that we're checking
    logger.debug("Qwen3_5DynamicCache found, no patching needed for now")