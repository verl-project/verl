# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from typing import Callable, Optional

import torch
from megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.utils import make_viewless_tensor
from torch import Tensor

from verl.models.mcore.util import preprocess_packed_seqs
from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
from verl.utils.megatron_utils import unwrap_model
from verl.utils.model import CausalLMOutputForPPO

from .util import postprocess_packed_seqs, postprocess_packed_seqs_for_dict_output


def gptmodel_forward_1f1b_overlap(
    model: GPTModel,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor = None,
    labels_mask: Tensor = None,
    multi_modal_inputs: Optional[dict] = None,
    logits_processor: Optional[Callable] = None,
    logits_processor_args: Optional[dict] = None,
    temperature: float = 1.0,
) -> TransformerModelChunkSchedulePlan:
    pre_process: bool = unwrap_model(model).pre_process
    post_process: bool = unwrap_model(model).post_process
    assert logits_processor is None, "only support fused kernel"
    batch_size, seq_len = attention_mask.shape[:2]
    input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
    input_ids_rmpad = input_ids_rmpad.contiguous()

    schedule_plan = model.build_schedule_plan(
        input_ids=input_ids_rmpad,
        attention_mask=attention_mask,
        labels=labels,
        position_ids=position_ids,
        packed_seq_params=packed_seq_params,
    )
    if post_process:
        attention_mask_out = attention_mask

        def _postprocess(
            self,
            hidden_states,
            input_ids,
            position_ids,
            labels,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            mtp_in_postprocess=None,
            loss_mask=None,
            decoder_input=None,
            attention_mask=None,
            inference_params=None,
            packed_seq_params=None,
            sequence_len_offset=None,
            runtime_gather_output=None,
            extra_block_kwargs=None,
            inference_context=None,
        ):
            """patched from https://github.com/NVIDIA/Megatron-LM/blob/core_r0.14.0/megatron/core/models/gpt/gpt_model.py#L412"""
            """Postprocesses decoder hidden states to generate logits or compute loss.

            Applies Multi-Token Prediction if enabled, generates output logits through
            the output layer, and computes language model loss when labels are provided.
            """
            from megatron.core import parallel_state
            from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

            in_inference_mode = inference_context is not None and not self.training
            if in_inference_mode:
                assert runtime_gather_output, "Inference must always gather TP logits"

            # logits and loss
            output_weight = None
            if self.share_embeddings_and_output_weights:
                output_weight = self.shared_embedding_or_output_weight()

            if mtp_in_postprocess:
                hidden_states = self.mtp(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    inference_params=inference_params,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                    embedding=self.embedding,
                    **(extra_block_kwargs or {}),
                )

            if not self.post_process:
                return hidden_states

            if self.mtp_process:
                from megatron.core.transformer.multi_token_prediction import (
                    MTPLossAutoScaler,
                    MTPLossLoggingHelper,
                    roll_tensor,
                )

                mtp_labels = labels.clone()
                hidden_states_list = torch.chunk(hidden_states, 1 + self.config.mtp_num_layers, dim=0)
                hidden_states = hidden_states_list[0]
                if loss_mask is None:
                    # if loss_mask is not provided, use all ones as loss_mask
                    loss_mask = torch.ones_like(mtp_labels)
                for mtp_layer_number in range(self.config.mtp_num_layers):
                    # output
                    mtp_logits, _ = self.output_layer(
                        hidden_states_list[mtp_layer_number + 1],
                        weight=output_weight,
                        runtime_gather_output=runtime_gather_output,
                    )
                    # Calc loss for the current Multi-Token Prediction (MTP) layers.
                    mtp_labels, _ = roll_tensor(mtp_labels, shifts=-1, dims=-1, cp_group=self.cp_group)
                    loss_mask, num_tokens = roll_tensor(loss_mask, shifts=-1, dims=-1, cp_group=self.cp_group)
                    mtp_loss = self.compute_language_model_loss(mtp_labels, mtp_logits)
                    mtp_loss = loss_mask * mtp_loss
                    if self.training:
                        # TODO(shifangx): remove the use of parallel_state here
                        # after moving loss logging to loss_func in pretrain_gpt.py
                        MTPLossLoggingHelper.save_loss_to_tracker(
                            torch.sum(mtp_loss) / num_tokens,
                            mtp_layer_number,
                            self.config.mtp_num_layers,
                            avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                        )
                    mtp_loss_scale = self.config.mtp_loss_scaling_factor / self.config.mtp_num_layers
                    if self.config.calculate_per_token_loss:
                        hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_scale * mtp_loss)
                    else:
                        hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_scale * mtp_loss / num_tokens)

            if logits_processor is not None:
                logits, _ = self.output_layer(
                    hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
                )
                output_orig = logits.transpose(0, 1).contiguous()
                args = {
                    k: preprocess_packed_seqs(v, attention_mask_out, pre_process=True)[0]
                    for k, v in logits_processor_args.items()
                }
                output_dict = logits_processor(output_orig, **args)
                output = {
                    k: postprocess_packed_seqs(
                        v, packed_seq_params, attention_mask_out, batch_size, seq_len, post_process=post_process
                    )
                    for k, v in output_dict.items()
                }
            else:
                # fused kernel

                labels_rmpad, _ = preprocess_packed_seqs(labels, attention_mask, pre_process=True)
                labels_mask_rmpad, _ = preprocess_packed_seqs(labels_mask, attention_mask, pre_process=True)
                labels_rmpad = labels_rmpad.contiguous()
                labels_mask_rmpad = labels_mask_rmpad.contiguous()

                output = CausalLMOutputForPPO(
                    loss=None,
                    logits=None,
                    past_key_values=None,
                    hidden_states=hidden_states,
                    attentions=None,
                )
                if self.config.sequence_parallel:
                    hidden_states = gather_from_sequence_parallel_region(hidden_states)
                logprobs, entropy = linear_cross_entropy(
                    hidden_states,
                    self.output_layer.weight,
                    labels_rmpad,
                    temperature,
                    "none",
                    parallel_state.get_tensor_model_parallel_group(),
                )
                output.entropy = entropy
                output.log_probs = logprobs

                output = postprocess_packed_seqs_for_dict_output(
                    labels_mask_rmpad,
                    output,
                    packed_seq_params,
                    attention_mask,
                    batch_size,
                    seq_len,
                    post_process=post_process,
                )
            output_ = [output["log_probs"]]
            # TODO NOW 1f1b overlap only support one tensor output
            # if "entropy" in output:
            #     output_.append(output["entropy"])
            output_ = tuple(output_)
            return output_

        def _custom_post_process_node_forward_impl(self, hidden_states):
            if self.gpt_model.decoder.final_layernorm and not self.gpt_model.mtp_process:
                hidden_states = self.gpt_model.decoder.final_layernorm(hidden_states)
                # TENorm produces a "viewed" tensor. This will result in schedule.py's
                # deallocate_output_tensor() throwing an error, so a viewless tensor is
                # created to prevent this.
                hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

            # Run GPTModel._postprocess
            output = self.gpt_model._postprocess(
                hidden_states=hidden_states,
                input_ids=self.chunk_state.input_ids,
                position_ids=self.chunk_state.position_ids,
                labels=self.chunk_state.labels,
                decoder_input=self.chunk_state.decoder_input,
                rotary_pos_emb=self.chunk_state.rotary_pos_emb,
                rotary_pos_cos=self.chunk_state.rotary_pos_cos,
                rotary_pos_sin=self.chunk_state.rotary_pos_sin,
                mtp_in_postprocess=False,
                loss_mask=self.chunk_state.loss_mask,
                attention_mask=self.chunk_state.attention_mask,
                packed_seq_params=self.chunk_state.packed_seq_params,
                sequence_len_offset=self.chunk_state.sequence_len_offset,
                runtime_gather_output=self.chunk_state.runtime_gather_output,
                extra_block_kwargs=self.chunk_state.extra_block_kwargs,
            )
            return output

        schedule_plan.post_process.forward_impl = _custom_post_process_node_forward_impl.__get__(
            schedule_plan.post_process, schedule_plan.post_process.__class__
        )
        unwrap_model(model)._postprocess = _postprocess.__get__(unwrap_model(model), unwrap_model(model).__class__)

    return schedule_plan
