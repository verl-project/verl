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

import inspect
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import megatron.core as mcore
import torch
from megatron.core import parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.utils import deprecate_inference_params
from packaging import version
from torch import Tensor

from verl.models.mcore.util import preprocess_packed_seqs, preprocess_thd_engine
from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
from verl.utils.megatron_utils import unwrap_model
from verl.utils.model import CausalLMOutputForPPO

from .util import postprocess_packed_seqs_for_dict_output, postprocess_thd_engine

_FUSED_FORWARD_MODE_ATTR = "_verl_fused_forward_mode"
_HOOK_MODE = "hook"
_LEGACY_MODE = "legacy"
_AUTO_MODE = "auto"


def _requested_fused_forward_mode() -> str:
    """Parse VERL_FUSED_USE_OP_HOOK once at model-build time."""
    raw_mode = os.environ.get("VERL_FUSED_USE_OP_HOOK", _AUTO_MODE).strip().lower()
    if raw_mode in ("", _AUTO_MODE):
        return _AUTO_MODE
    if raw_mode in ("1", "true", "yes", _HOOK_MODE):
        return _HOOK_MODE
    if raw_mode in ("0", "false", "no", _LEGACY_MODE):
        return _LEGACY_MODE
    raise ValueError("VERL_FUSED_USE_OP_HOOK must be one of: auto, hook, legacy, 1/true/yes, or 0/false/no.")


def _supports_output_processor_hook(patching_model: torch.nn.Module) -> bool:
    parameters = inspect.signature(patching_model.forward).parameters
    return {"output_processor", "output_processor_context"}.issubset(parameters)


def _resolve_fused_forward_mode(patching_model: torch.nn.Module) -> str:
    requested_mode = _requested_fused_forward_mode()
    if requested_mode == _LEGACY_MODE:
        return _LEGACY_MODE

    if _supports_output_processor_hook(patching_model):
        return _HOOK_MODE

    if requested_mode == _HOOK_MODE:
        raise RuntimeError(
            "VERL_FUSED_USE_OP_HOOK=hook requires the model forward to accept "
            "`output_processor` and `output_processor_context` (Megatron-LM PR #4686, "
            "megatron-core>=0.18.0). Use VERL_FUSED_USE_OP_HOOK=auto to fall back "
            "automatically, or VERL_FUSED_USE_OP_HOOK=legacy to force the legacy path."
        )

    return _LEGACY_MODE


def _get_fused_forward_mode(model: torch.nn.Module) -> str:
    model = unwrap_model(model)
    mode = getattr(model, _FUSED_FORWARD_MODE_ATTR, None)
    if mode is None and hasattr(model, "language_model"):
        mode = getattr(model.language_model, _FUSED_FORWARD_MODE_ATTR, None)
    return mode if mode in (_HOOK_MODE, _LEGACY_MODE) else _LEGACY_MODE


def _use_output_processor_hook(model: torch.nn.Module) -> bool:
    return _get_fused_forward_mode(model) == _HOOK_MODE


@dataclass
class FusedOutputProcessorContext:
    """Carries training-loop-owned semantics that Megatron itself should not know about
    (currently only `temperature`). Passed through GPTModel.forward(output_processor_context=...)
    and handed to the callback by Megatron's _postprocess as context=<this object>."""

    temperature: float


def fused_output_processor(
    *,
    hidden_states,  # decoder output [S,B,H]; under SP this is the sequence shard, not yet gathered
    output_layer,  # ColumnParallelLinear (may be tied)
    output_weight,  # tied -> shared weight; NOT tied -> None (opposite of the legacy patch; see below)
    labels,
    context,  # keyword MUST be `context` (Megatron _postprocess forwards context=output_processor_context)
    config,
    **_ignored,  # loss_mask/input_ids/position_ids/packed_seq_params/runtime_gather_output/...: absorbed
):
    """output_processor callback: run the fused logprob/entropy kernel at the _postprocess boundary.

    Mirrors the legacy _fused_GPTModel_forward tail: build CausalLMOutputForPPO -> SP gather ->
    resolve weight -> linear_cross_entropy -> fill log_probs/entropy. The return value is passed
    back out unchanged by Megatron."""
    output = CausalLMOutputForPPO(
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=hidden_states,
        attentions=None,
    )

    # The hook fires before the output layer, so under sequence parallelism gather first.
    if config.sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(hidden_states)

    # output_weight semantics are inverted vs the legacy patch: the hook passes None when the
    # embedding is NOT tied (real weight lives in output_layer.weight), and the shared weight
    # when it is tied.
    weight = output_weight if output_weight is not None else output_layer.weight

    temperature = context.temperature
    logprobs, entropy = linear_cross_entropy(
        hidden_states,
        weight,
        labels,
        temperature,
        "none",
        parallel_state.get_tensor_model_parallel_group(),
    )

    if has_config_logger_enabled(config):
        payload = OrderedDict(
            {
                "logprobs": logprobs,
                "entropy": entropy,
            }
        )
        log_config_to_disk(config, payload, prefix="input_and_logits")

    output.entropy = entropy
    output.log_probs = logprobs
    return output


def _get_patching_model(model: torch.nn.Module):
    model = unwrap_model(model)
    if isinstance(model, GPTModel):
        return model

    if not (hasattr(model, "language_model") and isinstance(model.language_model, GPTModel)):
        print(f"Model {model.__class__.__name__} is not a supported for fused forward")
        return None

    return model.language_model


def patch_fused_forward(model: torch.nn.Module):
    model = _get_patching_model(model)
    if model is None:
        return

    mode = getattr(model, _FUSED_FORWARD_MODE_ATTR, None)
    if mode is None:
        mode = _resolve_fused_forward_mode(model)
        setattr(model, _FUSED_FORWARD_MODE_ATTR, mode)

    if mode == _HOOK_MODE:
        return

    assert version.parse(mcore.__version__) >= version.parse("0.13.0"), (
        "Fused forward patching requires mecore >= 0.13.0"
    )
    if not hasattr(model, "forward_backup"):
        model.forward_backup = model.forward
        model.forward = _fused_GPTModel_forward.__get__(model, model.__class__)


def unpatch_fused_forward(model: torch.nn.Module):
    model = _get_patching_model(model)
    if model is None or _get_fused_forward_mode(model) == _HOOK_MODE:
        return
    if hasattr(model, "forward_backup"):
        model.forward = model.forward_backup
        delattr(model, "forward_backup")


def fused_forward_model_gen(vision_model: bool = False):
    def fused_forward_model(
        model,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        labels_mask: Tensor,
        temperature: float,
        multi_modal_inputs: dict,
    ):
        pre_process: bool = (
            unwrap_model(model).pre_process if not vision_model else False
        )  # vision model does not need pre_process, because we pack the input_ids to thd in the forward function
        post_process: bool = unwrap_model(model).post_process

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
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
        input_ids_rmpad = input_ids_rmpad.contiguous()
        labels_rmpad, _ = preprocess_packed_seqs(labels, attention_mask, pre_process=True)
        labels_mask_rmpad, _ = preprocess_packed_seqs(labels_mask, attention_mask, pre_process=True)
        labels_rmpad = labels_rmpad.contiguous()
        labels_mask_rmpad = labels_mask_rmpad.contiguous()

        input_args = dict(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids if not vision_model else None,  # vision models will calculate position_ids
            packed_seq_params=packed_seq_params,
            labels=labels_rmpad,
            temperature=temperature,
            **model_kwargs,
        )

        if vision_model:
            # workaround for supporting sequence packing with context parallelism
            # cp split with sequence packing will make model lose vision token information, so we need to keep
            # the original input_ids and pack them after vision embedding is calculated,
            # cooporate with mbridge
            input_args["input_ids"] = input_ids
            input_args["attention_mask"] = attention_mask

        if _use_output_processor_hook(model):
            # Hook path: call the native forward (no `temperature` param); temperature rides on
            # output_processor_context and the callback runs the fused kernel at _postprocess.
            # Note the keyword here is the forward param name output_processor_context= (the
            # callback receives it as context=).
            input_args.pop("temperature", None)
            output_orig: CausalLMOutputForPPO = model(
                **input_args,
                output_processor=fused_output_processor,
                output_processor_context=FusedOutputProcessorContext(temperature=temperature),
            )
        else:
            output_orig: CausalLMOutputForPPO = model(**input_args)

        if post_process:
            # output_orig is in type of CausalLMOutputForPPO
            output = postprocess_packed_seqs_for_dict_output(
                labels_mask_rmpad,
                output_orig,
                packed_seq_params,
                attention_mask,
                batch_size,
                seq_len,
                post_process=post_process,
            )
        else:
            output = output_orig
        return output

    return fused_forward_model


def fused_forward_model_engine(vision_model: bool = False):
    def fused_forward_model_engine_inner(
        model,
        input_ids: Tensor,
        labels: Tensor,
        multi_modal_inputs: dict,
        temperature: float,
        calculate_entropy: bool,
        pad_token_id: int,
    ):
        pre_process = unwrap_model(model).pre_process
        post_process = unwrap_model(model).post_process

        fp8 = unwrap_model(model).config.fp8
        use_fp8_padding = fp8 in ["e4m3", "hybrid"]
        config = unwrap_model(model).config
        min_local_rows = (
            config.csa_window_size if getattr(config, "experimental_attention_variant", None) == "dsv4_hybrid" else None
        )

        input_ids_rmpad, packed_seq_params, _ = preprocess_thd_engine(
            input_ids,
            pre_process=pre_process,
            use_fp8_padding=use_fp8_padding,
            min_local_rows=min_local_rows,
        )
        input_ids_rmpad = input_ids_rmpad.contiguous()

        model_kwargs = {}
        if "pixel_values" in multi_modal_inputs:
            model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
        if "image_grid_thw" in multi_modal_inputs:
            model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)
        if "pixel_values_videos" in multi_modal_inputs:
            model_kwargs["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"].to(input_ids.device)
        if "video_grid_thw" in multi_modal_inputs:
            model_kwargs["video_grid_thw"] = multi_modal_inputs["video_grid_thw"].to(input_ids.device)

        attention_mask = None
        if vision_model:
            input_ids_rmpad = input_ids.to_padded_tensor(pad_token_id)
            seqlens_in_batch = input_ids.offsets().diff().to(input_ids.device)
            max_seq_len = input_ids_rmpad.shape[1]
            attention_mask = torch.arange(max_seq_len, device=input_ids.device).unsqueeze(
                0
            ) < seqlens_in_batch.unsqueeze(1)

        labels_rmpad, _, _ = preprocess_thd_engine(
            labels,
            pre_process=True,
            need_roll=True,
            use_fp8_padding=use_fp8_padding,
            min_local_rows=min_local_rows,
        )
        labels_rmpad = labels_rmpad.contiguous()
        forward_kwargs = dict(
            input_ids=input_ids_rmpad,
            attention_mask=attention_mask,
            position_ids=None,
            packed_seq_params=packed_seq_params,
            labels=labels_rmpad,
            **model_kwargs,
        )
        if _use_output_processor_hook(model):
            # Hook path: same swap as the classic caller -- native forward has no `temperature`
            # param; temperature rides on output_processor_context and the callback runs the
            # fused kernel at _postprocess. This is the single live Megatron fused path in the
            # engine backend (patch_fused_forward is a no-op under the hook).
            output_orig: CausalLMOutputForPPO = model(
                **forward_kwargs,
                output_processor=fused_output_processor,
                output_processor_context=FusedOutputProcessorContext(temperature=temperature),
            )
        else:
            output_orig: CausalLMOutputForPPO = model(temperature=temperature, **forward_kwargs)

        if not post_process:
            return output_orig

        log_probs = output_orig.log_probs
        if log_probs.dim() == 1:
            log_probs = log_probs.unsqueeze(0)
        log_probs = postprocess_thd_engine(
            log_probs, packed_seq_params, input_ids, input_ids.shape[0], post_process=post_process
        )

        output = {"log_probs": log_probs}

        if calculate_entropy:
            entropy = output_orig.entropy
            if entropy.dim() == 1:
                entropy = entropy.unsqueeze(0)
            entropy = postprocess_thd_engine(
                entropy, packed_seq_params, input_ids, input_ids.shape[0], post_process=post_process
            )
            output["entropy"] = entropy

        return output

    return fused_forward_model_engine_inner


def _fused_GPTModel_forward(
    model,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    decoder_input: Tensor = None,
    labels: Tensor = None,
    inference_context: BaseInferenceContext = None,
    packed_seq_params: PackedSeqParams = None,
    extra_block_kwargs: dict = None,
    runtime_gather_output: Optional[bool] = None,
    *,
    inference_params: Optional[BaseInferenceContext] = None,
    loss_mask: Optional[Tensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> CausalLMOutputForPPO:
    """
    Patch self._postprocess in forward for GPT models to enable fused kernel support.
    https://github.com/NVIDIA/Megatron-LM/blob/core_v0.13.0/megatron/core/models/gpt/gpt_model.py

    TODO: Currently we still need to patch `forward` because we need to pass `temperature`
    explicitly to `self._postprocess` when calling, maybe there can be a better way to handle this?
    """

    inference_context = deprecate_inference_params(inference_context, inference_params)

    preproc_output = model._preprocess(
        input_ids=input_ids,
        position_ids=position_ids,
        decoder_input=decoder_input,
        inference_context=inference_context,
        packed_seq_params=packed_seq_params,
    )

    (decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset) = preproc_output[:5]

    decoder_extra_block_kwargs = extra_block_kwargs or {}
    if getattr(model.config, "moe_n_hash_layers", 0) > 0 and input_ids is not None:
        decoder_extra_block_kwargs["input_ids"] = input_ids

    # Run decoder.
    decoder_output = model.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        inference_context=inference_context,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        packed_seq_params=packed_seq_params,
        sequence_len_offset=sequence_len_offset,
        **decoder_extra_block_kwargs,
        **kwargs,
    )
    hidden_states = decoder_output[0] if isinstance(decoder_output, tuple) else decoder_output

    if not model.post_process:
        return hidden_states

    output = CausalLMOutputForPPO(
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=hidden_states,
        attentions=None,
    )

    if model.config.sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(hidden_states)

    # Get the output weight - use embedding weight if output_layer is None or weight is shared
    if hasattr(model, "output_layer") and model.output_layer is not None and model.output_layer.weight is not None:
        output_weight = model.output_layer.weight
    else:
        # When embeddings are tied, use the embedding weight
        output_weight = model.embedding.word_embeddings.weight

    logprobs, entropy = linear_cross_entropy(
        hidden_states,
        output_weight,
        labels,
        temperature,
        "none",
        parallel_state.get_tensor_model_parallel_group(),
    )

    if has_config_logger_enabled(model.config):
        payload = OrderedDict(
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "decoder_input": decoder_input,
                "logprobs": logprobs,
                "entropy": entropy,
            }
        )
        log_config_to_disk(model.config, payload, prefix="input_and_logits")

    output.entropy = entropy
    output.log_probs = logprobs

    return output
