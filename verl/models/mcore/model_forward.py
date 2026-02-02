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

import torch

from verl.utils.megatron_utils import unwrap_model
from verl.workers.config import MtpConfig

from .util import (
    postprocess_bshd,
    postprocess_bshd_no_padding,
    postprocess_packed_seqs,
    postprocess_thd_no_padding,
    preprocess_bshd,
    preprocess_bshd_no_padding,
    preprocess_packed_seqs,
    preprocess_thd_no_padding,
)


def model_forward_gen(vision_model: bool = False):
    def model_forward(
        model,
        input_ids,
        attention_mask,
        position_ids,
        multi_modal_inputs: dict,
        logits_processor=None,
        logits_processor_args: dict = None,
        value_model=False,
        data_format: str = "thd",
        mtp_config: MtpConfig = None,
    ):
        """Forward pass for models with sequence packing."""
        assert data_format in ["thd", "bshd"], "data_format must be 'thd' or 'bshd'"
        pre_process = (
            unwrap_model(model).pre_process if not vision_model else False
        )  # vision model does not need pre_process, because we pack the input_ids to thd in the forward function
        post_process = unwrap_model(model).post_process
        sp = unwrap_model(model).config.sequence_parallel
        fp8 = unwrap_model(model).config.fp8
        use_fp8_padding = fp8 in ["e4m3", "hybrid"]

        model_kwargs = {}
        if "pixel_values" in multi_modal_inputs:
            model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
        if "image_grid_thw" in multi_modal_inputs:
            model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)
        if "pixel_values_videos" in multi_modal_inputs:
            model_kwargs["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"].to(input_ids.device)
        if "video_grid_thw" in multi_modal_inputs:
            model_kwargs["video_grid_thw"] = multi_modal_inputs["video_grid_thw"].to(input_ids.device)

        batch_size, seq_len = attention_mask.shape[:2]
        if data_format == "thd":
            input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(
                input_ids, attention_mask, pre_process=pre_process or post_process, use_fp8_padding=use_fp8_padding
            )
            input_ids_rmpad = input_ids_rmpad.contiguous()

            # when pp > 1 and processor is not None, we need to pass the labels and loss_mask to the model
            if mtp_config and mtp_config.enable_train and post_process:
                args = {
                    k: preprocess_packed_seqs(v, attention_mask, pre_process=True, use_fp8_padding=use_fp8_padding)[0]
                    for k, v in logits_processor_args.items()
                }
                model_kwargs["labels"] = args["label"].contiguous()
                model_kwargs["loss_mask"] = args["label_mask"].contiguous()

            input_args = dict(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids if not vision_model else None,  # vision models will calculate position_ids
                packed_seq_params=packed_seq_params,
                **model_kwargs,
            )

            if vision_model:
                # workaround for supporting sequence packing with context parallelism
                # cp split with sequence packing will make model lose vision token information, so we need to keep
                # the original input_ids and pack them after vision embedding is calculated,
                # cooporate with mbridge
                input_args["input_ids"] = input_ids
                input_args["attention_mask"] = attention_mask

            output_orig = model(**input_args)

            if post_process and logits_processor is not None:
                args = {
                    k: preprocess_packed_seqs(v, attention_mask, pre_process=True, use_fp8_padding=use_fp8_padding)[0]
                    for k, v in logits_processor_args.items()
                }
                output_dict = logits_processor(output_orig, **args)
                output = {
                    k: postprocess_packed_seqs(
                        v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
                    )
                    for k, v in output_dict.items()
                }
            else:
                output = postprocess_packed_seqs(
                    output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
                )
        elif data_format == "bshd":
            """
            data_format: "thd" or "bshd", default is "thd",
            why we need this?
                for some new models, GPT-OSS, the thd format is not supported, so we need to use the bshd format.
            When using the bshd format, we have to add paddings to the input_ids to meet the longest sequence length, 
            so it is recommended to disable dynamic batch size and set batch size to 1
            """
            assert not vision_model, "vision model does not support bshd format"
            assert fp8 is None, "fp8 is not supported for bshd format yet"

            batch_size, sequence_length = attention_mask.shape[:2]
            new_input_ids, new_attention_mask, new_position_ids = preprocess_bshd(
                input_ids, attention_mask, position_ids, sequence_parallel=sp, pre_process=pre_process
            )
            output_orig = model(
                input_ids=new_input_ids,
                position_ids=new_position_ids,
                attention_mask=new_attention_mask,
                **model_kwargs,
            )
            if post_process and logits_processor is not None:
                args = {
                    k: preprocess_bshd(v, attention_mask, position_ids, sequence_parallel=sp, pre_process=True)[0]
                    for k, v in logits_processor_args.items()
                }
                output_dict = logits_processor(output_orig, **args)
                output = {
                    k: postprocess_bshd(
                        v, new_attention_mask, attention_mask, sequence_length, post_process=post_process
                    )
                    for k, v in output_dict.items()
                }
            else:
                output = postprocess_bshd(
                    output_orig, new_attention_mask, attention_mask, sequence_length, post_process=post_process
                )
        if value_model and post_process:
            output = output[..., 0]
        return output

    return model_forward


def gptmodel_forward_no_padding(
    model,
    input_ids,
    multi_modal_inputs: dict,
    logits_processor=None,
    logits_processor_args: dict = None,
    value_model=False,
    vision_model=False,
    pad_token_id=None,
    data_format: str = "thd",
    enable_mtp: bool = False,
):
    """Default forward pass for GPT models with optional sequence packing."""

    assert data_format in ["thd", "bshd"], "data_format must be 'thd' or 'bshd'"
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process

    model_kwargs = {}
    if "pixel_values" in multi_modal_inputs:
        model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
    if "image_grid_thw" in multi_modal_inputs:
        model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)
    if "pixel_values_videos" in multi_modal_inputs:
        model_kwargs["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"].to(input_ids.device)
    if "video_grid_thw" in multi_modal_inputs:
        model_kwargs["video_grid_thw"] = multi_modal_inputs["video_grid_thw"].to(input_ids.device)

    batch_size = input_ids.shape[0]
    if data_format == "thd":
        input_ids_rmpad, packed_seq_params = preprocess_thd_no_padding(input_ids, pre_process=pre_process)
        input_ids_rmpad = input_ids_rmpad.contiguous()

        if enable_mtp and post_process:
            args = {
                k: preprocess_thd_no_padding(v, pre_process=True, need_roll=(k == "label" or k == "loss_mask"))[0]
                for k, v in logits_processor_args.items()
            }
            model_kwargs["labels"] = args["label"].contiguous()
            model_kwargs["loss_mask"] = args["loss_mask"].contiguous()
        logits_processor_args.pop("loss_mask")

        # For VLM model, need to pass bshd format `input_ids` and `attention_mask`.
        attention_mask = None
        if vision_model:
            input_ids_rmpad = input_ids.to_padded_tensor(pad_token_id)
            seqlens_in_batch = input_ids.offsets().diff()
            attention_mask = torch.zeros_like(input_ids_rmpad, dtype=torch.bool)
            for i, seqlen in enumerate(seqlens_in_batch):
                attention_mask[i, :seqlen] = True

        output_orig = model(
            input_ids=input_ids_rmpad,
            attention_mask=attention_mask,
            position_ids=None,
            packed_seq_params=packed_seq_params,
            **model_kwargs,
        )

        if post_process and logits_processor is not None:
            args = {
                k: preprocess_thd_no_padding(v, pre_process=True, need_roll=(k == "label"))[0]
                for k, v in logits_processor_args.items()
            }
            output_dict = logits_processor(output_orig, **args)
            output = {
                k: postprocess_thd_no_padding(v, packed_seq_params, input_ids, batch_size, post_process=post_process)
                for k, v in output_dict.items()
            }
        else:
            output = postprocess_thd_no_padding(
                output_orig, packed_seq_params, input_ids, batch_size, post_process=post_process
            )
    else:
        """
        data_format: "thd" or "bshd", default is "thd",
        why we need this?
            for some new models, GPT-OSS, the thd format is not supported, so we need to use the bshd format.
        When using the bshd format, we have to add paddings to the input_ids to meet the longest sequence length, 
        so it is recommended to disable dynamic batch size and set batch size to 1
        """

        input_ids_bshd, attention_mask_bshd, position_ids_bshd = preprocess_bshd_no_padding(
            input_ids, pre_process=pre_process
        )

        if enable_mtp and post_process:
            args = {
                k: preprocess_bshd_no_padding(v, pre_process=True, need_roll=(k == "label" or k == "loss_mask"))[0]
                for k, v in logits_processor_args.items()
            }
            model_kwargs["labels"] = args["label"].contiguous()
            model_kwargs["loss_mask"] = args["loss_mask"].contiguous()
        logits_processor_args.pop("loss_mask")

        output_orig = model(
            input_ids=input_ids_bshd,
            attention_mask=attention_mask_bshd,
            position_ids=position_ids_bshd,
            **model_kwargs,
        )
        if post_process and logits_processor is not None:
            args = {
                k: preprocess_bshd_no_padding(v, pre_process=True, need_roll=(k == "label"))[0]
                for k, v in logits_processor_args.items()
            }
            output_dict = logits_processor(output_orig, **args)
            output = {
                k: postprocess_bshd_no_padding(v, attention_mask_bshd, post_process=post_process)
                for k, v in output_dict.items()
            }
        else:
            output = postprocess_bshd_no_padding(output_orig, attention_mask_bshd, post_process=post_process)

    if value_model and post_process:
        # output = output[..., 0]
        # while using nested tensor, the advanced indexing operation above will result in an error at backward, i.e.
        # ValueError: NestedTensor _nested_select_backward_default(grad_output: t, self: jt_all, dim: any, index: any)
        # so we use `squeeze` to remove the last dimension
        output = output.squeeze(-1)

    return output
