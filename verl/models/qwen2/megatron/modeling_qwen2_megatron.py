# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch Qwen2 model."""

from typing import Optional

import torch
import torch.utils.checkpoint
from megatron.core import ModelParallelConfig, mpu, parallel_state, tensor_parallel
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import CausalLMOutputWithPast

from verl.utils.device import get_device_name
from verl.utils.megatron import sequence_parallel as sp_utils
from verl.utils.megatron import tensor_parallel as tp_utils
from verl.utils.megatron_utils import TransformerConfig, convert_config

from .layers import ParallelQwen2DecoderLayer, ParallelQwen2DecoderLayerRmPad, ParallelQwen2RMSNorm

"""
TODO: 
1. Add weight initialization. Here we need to be careful on TP weight init.
2. Add sequence parallel
3. Load checkpoint from Qwen2 pretrained checkpoint
"""


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class ParallelQwen2Model(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        if megatron_config is not None:
            assert embedding_kwargs.get("config", False), "must have ModelParallelConfig"
            tp_utils.update_kwargs_with_config(embedding_kwargs, megatron_config)
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, **embedding_kwargs
        )

        self.layers = nn.ModuleList(
            [ParallelQwen2DecoderLayer(config, megatron_config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = ParallelQwen2RMSNorm(config, megatron_config)

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> tuple | BaseModelOutputWithPast:
        """

        Args:
            input_ids: input ids. shape (batch_size, seq_length)
            attention_mask: attention_mask. shape (batch_size, seq_length)
            position_ids: position ids. shape (batch_size, seq_length)

        Returns:

        """
        batch_size, seq_length = input_ids.shape
        inputs_embeds = self.embed_tokens(input_ids)
        # embed positions

        attention_mask = self._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds)

        hidden_states = inputs_embeds

        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states


class ParallelQwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.model = ParallelQwen2Model(config, megatron_config=megatron_config)
        self.vocab_size = config.vocab_size

        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if megatron_config is not None:
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)

        self.lm_head = tensor_parallel.ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        ```"""

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)[0]

        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)

        logits = logits.float()
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa: F401, E402


class ParallelQwen2ModelRmPad(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        self.megatron_config = megatron_config
        if megatron_config is not None:
            assert embedding_kwargs.get("config", False), "must have ModelParallelConfig"
            tp_utils.update_kwargs_with_config(embedding_kwargs, self.megatron_config)
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, **embedding_kwargs
        )

        self.layers = nn.ModuleList(
            [ParallelQwen2DecoderLayerRmPad(config, megatron_config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = ParallelQwen2RMSNorm(config, megatron_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        sequence_length: int = None,
        indices: torch.Tensor = None,
        cu_seqlens: int = None,
        max_seqlen_in_batch: int = None,
    ) -> tuple | BaseModelOutputWithPast:
        """

        Args:
            input_ids: input ids. shape (1, totol_nnz)
            position_ids: position ids. shape (batch_size, seq_length)

        Returns:

        """
        inputs_embeds = self.embed_tokens(input_ids)  # (1, total_nnz) -> (1, total_nnz, hidden_size)

        # (1, total_nnz, hidden_size) -> (total_nnz, 1, hidden_size) -> (total_nnz // sp, 1, hidden_size)
        inputs_embeds = inputs_embeds.transpose(0, 1)
        if self.megatron_config.sequence_parallel:
            inputs_embeds = tensor_parallel.scatter_to_sequence_parallel_region(inputs_embeds)

        hidden_states = inputs_embeds
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                sequence_length=sequence_length,
                indices=indices,
                cu_seqlens=cu_seqlens,
                max_seqlen_in_batch=max_seqlen_in_batch,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states


class ParallelQwen2ForCausalLMRmPad(nn.Module):
    def __init__(self, config: Qwen2Config, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.megatron_config = megatron_config
        self.model = ParallelQwen2ModelRmPad(config, megatron_config=megatron_config)
        self.vocab_size = config.vocab_size
        self._init_head(config)

    def _init_head(self, config: Qwen2Config):
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if self.megatron_config is not None:
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        self.lm_head = tensor_parallel.ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

    def _forward_head(self, hidden_states):
        # all_gather from sequence parallel region is performed inside lm_head
        logits = self.lm_head(hidden_states)[0]
        logits = logits.float()  # (total_nnz_padded, 1, vocab_size // tp)
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)  # (total_nnz_padded, 1, vocab_size)
        return logits

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        ```"""
        batch_size, sequence_length = input_ids.shape

        # remove padding here
        input_ids, indices, cu_seqlens, max_seqlen_in_batch, *_ = unpad_input(
            input_ids.unsqueeze(dim=-1), attention_mask
        )  # (total_nnz, 1)

        # pad input_ids to multiple of tp for all tp ranks
        # TODO: for better performance, the sp padding should be removed at each layer. Not sure the performance gap
        if self.megatron_config.sequence_parallel:
            input_ids = sp_utils.pad_to_sequence_parallel(input_ids)

        input_ids = input_ids.transpose(0, 1)  # (1, total_nnz+pad)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            sequence_length=sequence_length,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen_in_batch=max_seqlen_in_batch,
        )

        hidden_states = outputs

        logits = self._forward_head(hidden_states)

        # remove padding from sequence parallel
        if self.megatron_config.sequence_parallel:
            totol_nnz = cu_seqlens[-1]
            logits = logits[:totol_nnz]  # (total_nnz_padded)

        logits = torch.squeeze(logits, dim=1)  # remove the artificial batch dimension
        # add removed padding back
        logits = pad_input(
            logits, indices, batch_size, seqlen=sequence_length
        )  # (batch_size, sequence_length, vocab_size)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class ParallelQwen2ForValueRmPad(ParallelQwen2ForCausalLMRmPad):
    def _init_head(self, config):
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if self.megatron_config is not None:
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        self.lm_head = nn.Linear(in_features=config.hidden_size, out_features=1, bias=False)
        # lm_head is effectively the same as sequence parallel
        sp_utils.mark_parameter_as_sequence_parallel(self.lm_head.weight)

    def _forward_head(self, hidden_states):
        logits = self.lm_head(hidden_states)  # (total_nnz_padded // tp, 1, 1)
        logits = logits.float()
        if self.megatron_config.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> tuple | CausalLMOutputWithPast:
        output = super().forward(input_ids, attention_mask, position_ids)
        output.logits = torch.squeeze(output.logits, dim=-1)
        return output


"""
Support pipeline parallelism
"""


class ParallelQwen2ModelRmPadPP(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]
    This model definition supports pipeline parallelism. To support pp and vpp,
    - This model only contains layer in this pp stage and vpp chunk
    - When calling get_model in Megatron, this rank will instantiate all the vpp chunks in this pp.
    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config, megatron_config: ModelParallelConfig, pre_process, post_process):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pre_process = pre_process
        self.post_process = post_process
        self.megatron_config = megatron_config
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        if megatron_config is not None:
            assert embedding_kwargs.get("config", False), "must have ModelParallelConfig"
            tp_utils.update_kwargs_with_config(embedding_kwargs, self.megatron_config)
        if pre_process:
            self.embed_tokens = tensor_parallel.VocabParallelEmbedding(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, **embedding_kwargs
            )
        else:
            self.embed_tokens = None

        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = megatron_config.pipeline_model_parallel_size
        self.num_layer_per_pp = config.num_hidden_layers // pp_size
        vpp_size = megatron_config.virtual_pipeline_model_parallel_size
        vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()

        if vpp_size is not None:
            self.num_layer_vpp_chunk = self.num_layer_per_pp // vpp_size
            self.num_layer_this_model = self.num_layer_vpp_chunk
            offset = vpp_rank * (config.num_hidden_layers // vpp_size) + (pp_rank * self.num_layer_vpp_chunk)
        else:
            self.num_layer_this_model = self.num_layer_per_pp
            offset = pp_rank * self.num_layer_per_pp

        self.layers = nn.ModuleList()
        for i in range(self.num_layer_this_model):
            layer = ParallelQwen2DecoderLayerRmPad(config, megatron_config, layer_idx=i + offset)
            self.layers.add_module(f"{i}", layer)

        if post_process:
            self.norm = ParallelQwen2RMSNorm(config, megatron_config)
        else:
            self.norm = None

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        sequence_length: int = None,
        indices: torch.Tensor = None,
        cu_seqlens: int = None,
        max_seqlen_in_batch: int = None,
    ) -> tuple | BaseModelOutputWithPast:
        """

        Args:
            input_ids: input ids. shape (1, totol_nnz)
            position_ids: position ids. shape (batch_size, seq_length)

        Returns:

        """
        if self.pre_process:
            inputs_embeds = self.embed_tokens(input_ids)  # (1, total_nnz) -> (1, total_nnz, hidden_size)

            # vocab parallel embedding will not do sequence parallel reduce-scatter in open source megatron
            # so need to deal with it by handle here:
            # (1, total_nnz, hidden_size) -> (total_nnz, 1, hidden_size) -> (total_nnz // sp, 1, hidden_size)
            inputs_embeds = inputs_embeds.transpose(0, 1)
            if self.megatron_config.sequence_parallel:
                inputs_embeds = tensor_parallel.scatter_to_sequence_parallel_region(inputs_embeds)

            hidden_states = inputs_embeds
        else:
            # self.hidden_states should be passed by Megatron
            hidden_states = self.input_tensor

        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                sequence_length=sequence_length,
                indices=indices,
                cu_seqlens=cu_seqlens,
                max_seqlen_in_batch=max_seqlen_in_batch,
            )

            hidden_states = layer_outputs

        if self.post_process:
            hidden_states = self.norm(hidden_states)

        return hidden_states


class ParallelQwen2ForCausalLMRmPadPP(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        megatron_config: ModelParallelConfig,
        pre_process,
        post_process,
        share_embeddings_and_output_weights,
    ):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.megatron_config = megatron_config
        self.model = ParallelQwen2ModelRmPadPP(
            config, megatron_config=megatron_config, pre_process=pre_process, post_process=post_process
        )
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.vocab_size = config.vocab_size
        self.pre_process = pre_process
        self.post_process = post_process
        if post_process:
            self._init_head(config)
        if pre_process or post_process:
            self.setup_embeddings_and_output_layer()

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        assert len(input_tensor) == 1
        self.model.set_input_tensor(input_tensor[0])

    def _init_head(self, config):
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if self.megatron_config is not None:
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        self.lm_head = tensor_parallel.ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            skip_weight_param_allocation=self.pre_process and self.share_embeddings_and_output_weights,
            **column_kwargs,
        )

    def setup_embeddings_and_output_layer(self) -> None:
        """Sets up embedding layer in first stage and output layer in last stage.

        This function initializes word embeddings in the final stage when we are
        using pipeline parallelism and sharing word embeddings, and sets up param
        attributes on the embedding and output layers.
        """
        # Set `is_embedding_or_output_parameter` attribute.
        if self.pre_process:
            self.model.embed_tokens.weight.is_embedding_or_output_parameter = True
        if self.post_process and self.lm_head.weight is not None:
            self.lm_head.weight.is_embedding_or_output_parameter = True

        if not self.share_embeddings_and_output_weights:
            return

        if parallel_state.get_pipeline_model_parallel_world_size() == 1:
            # Zero out wgrad if sharing embeddings between two layers on same
            # pipeline stage to make sure grad accumulation into main_grad is
            # correct and does not include garbage values (e.g., from torch.empty).
            self.shared_embedding_or_output_weight().zero_out_wgrad = True
            return

        if parallel_state.is_pipeline_first_stage() and self.pre_process and not self.post_process:
            self.shared_embedding_or_output_weight().shared_embedding = True

        if self.post_process and not self.pre_process:
            assert not parallel_state.is_pipeline_first_stage()
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.lm_head.weight.data.fill_(0)
            self.lm_head.weight.shared = True
            self.lm_head.weight.shared_embedding = True

        if torch.distributed.is_initialized() and parallel_state.is_rank_in_embedding_group():
            weight = self.shared_embedding_or_output_weight()
            weight.data = weight.data.to(get_device_name())
            torch.distributed.all_reduce(weight.data, group=parallel_state.get_embedding_group())

    def shared_embedding_or_output_weight(self) -> torch.Tensor:
        if self.pre_process:
            return self.model.embed_tokens.weight
        elif self.post_process:
            return self.lm_head.weight
        return None

    def _forward_head(self, hidden_states):
        # all_gather from sequence parallel region is performed inside lm_head
        # print(f'logits shape before forward_head: {hidden_states.shape}, vocab_size = '
        # f'{self.config.vocab_size}') # [4, 32, 4096]
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits = self.lm_head(hidden_states, weight=output_weight)[0]
        # print(f'logits shape after forward_head: {logits.shape}') # [8, 32, 8]
        logits = logits.float()  # (total_nnz_padded, 1, vocab_size // tp)
        return logits

    def forward(
        self,
        # original input
        *,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        ```"""

        # Note that input_ids, attention_mask and position_ids should be passed to every pp layer.
        # In the first pp, input_ids will be used, in other pp layers hidden_states will be used inside self.model
        batch_size, sequence_length = input_ids.shape
        # remove padding here
        input_ids_rmpad, indices, cu_seqlens, max_seqlen_in_batch, *_ = unpad_input(
            input_ids.unsqueeze(dim=-1), attention_mask
        )  # (total_nnz, 1)

        # pad input_ids to multiple of tp for all tp ranks
        # TODO: for better performance, the sp padding should be removed at each layer. Not sure the performance gap
        if self.megatron_config.sequence_parallel:
            input_ids_rmpad = sp_utils.pad_to_sequence_parallel(input_ids_rmpad)

        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz+pad)

        outputs = self.model(
            input_ids=input_ids_rmpad,
            position_ids=position_ids,
            sequence_length=sequence_length,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen_in_batch=max_seqlen_in_batch,
        )

        if self.post_process:
            hidden_states = outputs
            logits = self._forward_head(hidden_states)
            logits = torch.squeeze(logits, dim=1)  # remove the artificial batch dimension # torch.Size([8, 32, 16])

            # remove padding from sequence parallel
            if self.megatron_config.sequence_parallel:
                totol_nnz = cu_seqlens[-1]
                logits = logits[:totol_nnz]  # (total_nnz_padded)
            # add removed padding back. If input is already rmpad, we let the caller pad_input
            logits = pad_input(
                logits, indices, batch_size, seqlen=sequence_length
            )  # (batch_size, sequence_length, vocab_size)

            return CausalLMOutputWithPast(
                loss=None,
                logits=logits,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )
        else:
            return outputs


class ParallelQwen2ForValueRmPadPP(ParallelQwen2ForCausalLMRmPadPP):
    def _init_head(self, config):
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if self.megatron_config is not None:
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        self.lm_head = nn.Linear(in_features=config.hidden_size, out_features=1, bias=False)
        # lm_head is effectively the same as sequence parallel
        sp_utils.mark_parameter_as_sequence_parallel(self.lm_head.weight)

    def _forward_head(self, hidden_states):
        logits = self.lm_head(hidden_states)  # (total_nnz_padded // tp, 1, 1)
        logits = logits.float()
        if self.megatron_config.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits

    def forward(
        self,
        *,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> tuple | CausalLMOutputWithPast:
        output = super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        if self.post_process:
            output.logits = torch.squeeze(output.logits, dim=-1)
            return output
        else:
            return output
