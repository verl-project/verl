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

from dataclasses import dataclass
from typing import Optional, Union

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class CausalLMOutputForPPO(CausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    topk_log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


def forward_base_model(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> CausalLMOutputWithPast:
    r"""
    Copy paste LLaMa's forward
    https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/model/llama.py

    This function should be generic enough for all pure text models.
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    return outputs


def _scatter_indexed_output(selected: torch.Tensor, index: torch.Tensor, seq_len: int) -> torch.Tensor:
    full_shape = (*selected.shape[:-1], seq_len)
    return selected.new_zeros(full_shape).index_copy(-1, index, selected)


def _empty_indexed_outputs(
    hidden_states: torch.Tensor, vocab_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    zero_anchor = (hidden_states[..., :0, :].sum() + vocab_weights[:0, :].sum()) * 0.0
    out_shape = (*hidden_states.shape[:-1],)
    return (
        zero_anchor.to(torch.float32).expand(out_shape).clone(),
        zero_anchor.to(hidden_states.dtype).expand(out_shape).clone(),
    )


def _forward_fused_ppo_torch(
    fused_linear_for_ppo,
    hidden_states: torch.Tensor,
    vocab_weights: torch.Tensor,
    rolled_labels: torch.Tensor,
    temperature: float,
    logprob_index: Optional[torch.LongTensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logprob_index is None:
        return fused_linear_for_ppo.forward(
            hidden_states=hidden_states,
            vocab_weights=vocab_weights,
            input_ids=rolled_labels,
            temperature=temperature,
        )
    index = logprob_index.to(device=hidden_states.device, dtype=torch.long)
    if index.numel() == 0:
        return _empty_indexed_outputs(hidden_states, vocab_weights)
    seq_len = hidden_states.shape[-2]
    sel_lp, sel_ent = fused_linear_for_ppo.forward(
        hidden_states=hidden_states.index_select(-2, index),
        vocab_weights=vocab_weights,
        input_ids=rolled_labels.index_select(-1, index),
        temperature=temperature,
    )
    return (
        _scatter_indexed_output(sel_lp, index, seq_len),
        _scatter_indexed_output(sel_ent, index, seq_len),
    )


def _forward_fused_ppo_triton(
    linear_cross_entropy_fn,
    hidden_states: torch.Tensor,
    vocab_weights: torch.Tensor,
    rolled_labels: torch.Tensor,
    temperature: float,
    impl_backend: str,
    logprob_index: Optional[torch.LongTensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logprob_index is None:
        return linear_cross_entropy_fn(
            hidden_states,
            vocab_weights,
            rolled_labels,
            temperature,
            "none",
            impl_backend=impl_backend,
        )
    index = logprob_index.to(device=hidden_states.device, dtype=torch.long)
    if index.numel() == 0:
        zero_lp, zero_ent = _empty_indexed_outputs(hidden_states, vocab_weights)
        return zero_lp.reshape(-1), zero_ent.reshape(-1).to(torch.float32)
    seq_len = hidden_states.shape[-2]
    sel_lp, sel_ent = linear_cross_entropy_fn(
        hidden_states.index_select(-2, index),
        vocab_weights,
        rolled_labels.index_select(-1, index),
        temperature,
        "none",
        impl_backend=impl_backend,
    )
    return (
        _scatter_indexed_output(sel_lp, index, seq_len),
        _scatter_indexed_output(sel_ent, index, seq_len),
    )


def forward_with_torch_backend(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union["Cache", list[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: int | torch.Tensor = 0,
    temperature: float = 1.0,
    shift_labels: Optional[torch.LongTensor] = None,
    topk_labels: Optional[torch.LongTensor] = None,
    logprob_index: Optional[torch.LongTensor] = None,
    **loss_kwargs,
) -> tuple | CausalLMOutputForPPO:
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    outputs = forward_base_model(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]

    if not return_dict:
        raise NotImplementedError("forward_with_torch_backend has to return_dict")

    # Loss calculations.
    # When the engine has already prepared globally-rolled labels (e.g. the FSDP
    # path under Ulysses SP, see issue #6068), it passes them as `shift_labels`
    # so we don't redo `torch.roll` on a sequence-parallel-sliced shard.
    if shift_labels is not None:
        rolled_labels = shift_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    if topk_labels is not None:
        from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

        if topk_labels.shape[:-1] != rolled_labels.shape:
            raise ValueError("topk_labels must have rolled_labels.shape as its leading dimensions")
        kernel_labels = torch.cat((rolled_labels.unsqueeze(-1), topk_labels), dim=-1)
        all_log_probs, entropy = linear_cross_entropy(
            hidden_states,
            self.lm_head.weight,
            kernel_labels,
            temperature,
            "none",
            impl_backend=getattr(self, "_verl_fused_kernels_backend", "torch"),
        )
        return CausalLMOutputForPPO(
            log_probs=all_log_probs[..., 0],
            topk_log_probs=all_log_probs[..., 1:],
            entropy=entropy,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    fused_linear_for_ppo = FusedLinearForPPO(impl_backend=getattr(self, "_verl_fused_kernels_backend", "torch"))
    log_probs, entropy = _forward_fused_ppo_torch(
        fused_linear_for_ppo,
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        rolled_labels=rolled_labels,
        temperature=temperature,
        logprob_index=logprob_index,
    )

    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def forward_with_triton_backend(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union["Cache", list[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: int | torch.Tensor = 0,
    temperature: float = 1.0,
    shift_labels: Optional[torch.LongTensor] = None,
    topk_labels: Optional[torch.LongTensor] = None,
    logprob_index: Optional[torch.LongTensor] = None,
    **loss_kwargs,
) -> tuple | CausalLMOutputForPPO:
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    outputs = forward_base_model(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]

    if not return_dict:
        raise NotImplementedError("forward_with_triton_backend has to return_dict")

    # Loss calculations. See `forward_with_torch_backend` for why `shift_labels`
    # takes precedence over local `torch.roll` (issue #6068).
    if shift_labels is not None:
        rolled_labels = shift_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    impl_backend = getattr(self, "_verl_fused_kernels_backend", "triton")
    if topk_labels is not None:
        if topk_labels.shape[:-1] != rolled_labels.shape:
            raise ValueError("topk_labels must have rolled_labels.shape as its leading dimensions")
        kernel_labels = torch.cat((rolled_labels.unsqueeze(-1), topk_labels), dim=-1)
        all_log_probs, entropy = linear_cross_entropy(
            hidden_states,
            self.lm_head.weight,
            kernel_labels,
            temperature,
            "none",
            impl_backend=impl_backend,
        )
        log_probs = all_log_probs[..., 0]
        topk_log_probs = all_log_probs[..., 1:]
    else:
        log_probs, entropy = _forward_fused_ppo_triton(
            linear_cross_entropy,
            hidden_states,
            self.lm_head.weight,
            rolled_labels,
            temperature,
            impl_backend,
            logprob_index=logprob_index,
        )
        topk_log_probs = None

    return CausalLMOutputForPPO(
        log_probs=log_probs,
        topk_log_probs=topk_log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
