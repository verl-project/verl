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

from verl.models.mcore.util import preprocess_thd_engine, preprocess_vlm_thd_engine
from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
from verl.utils.megatron_utils import unwrap_model
from verl.utils.model import CausalLMOutputForPPO

from .util import postprocess_thd_engine

_FUSED_FORWARD_MODE_ATTR = "_verl_fused_forward_mode"
_FUSED_IMPL_BACKEND_ATTR = "_verl_fused_impl_backend"
_HOOK_MODE = "hook"
_LEGACY_MODE = "legacy"


def _supports_output_processor_hook(patching_model: torch.nn.Module) -> bool:
    """Check whether the model supports Megatron's native output-processor hook.

    The ``output_processor`` / ``output_processor_context`` contract was
    introduced in Megatron Core 0.18.0. Models without this contract fall back
    to the legacy ``forward`` monkey patch.

    TODO: Remove this check and the legacy patch once all supported Megatron
    stacks require Megatron Core 0.18.0 or newer.
    """
    parameters = inspect.signature(patching_model.forward).parameters
    return {"output_processor", "output_processor_context"}.issubset(parameters)


def _resolve_fused_forward_mode(patching_model: torch.nn.Module) -> str:
    return _HOOK_MODE if _supports_output_processor_hook(patching_model) else _LEGACY_MODE


def _get_fused_forward_mode(model: torch.nn.Module) -> str:
    model = unwrap_model(model)
    mode = getattr(model, _FUSED_FORWARD_MODE_ATTR, None)
    if mode is None and hasattr(model, "language_model"):
        mode = getattr(model.language_model, _FUSED_FORWARD_MODE_ATTR, None)
    return mode if mode in (_HOOK_MODE, _LEGACY_MODE) else _LEGACY_MODE


def _use_output_processor_hook(model: torch.nn.Module) -> bool:
    return _get_fused_forward_mode(model) == _HOOK_MODE


def get_fused_impl_backend(model: torch.nn.Module) -> str:
    model = unwrap_model(model)
    if hasattr(model, "language_model"):
        model = model.language_model
    return getattr(model, _FUSED_IMPL_BACKEND_ATTR, "triton")


def _gather_fused_hidden_states(hidden_states: Tensor, sequence_parallel: bool, impl_backend: str) -> Tensor:
    if not sequence_parallel:
        return hidden_states

    # Liger TP-FLSCE already reduces dHidden across vocabulary shards. Its
    # sequence-parallel gather must only split that gradient in backward.
    tensor_parallel_output_grad = impl_backend.lower() != "liger"
    return gather_from_sequence_parallel_region(
        hidden_states,
        tensor_parallel_output_grad=tensor_parallel_output_grad,
    )


@dataclass
class FusedOutputProcessorContext:
    """Context passed through Megatron's native output-processor hook."""

    temperature: float
    impl_backend: str = "triton"


def fused_output_processor(
    *,
    hidden_states,
    output_layer,
    output_weight,
    labels,
    context,
    config,
    **_ignored,
):
    """Compute fused log probabilities and entropy at Megatron's postprocess boundary."""
    output = CausalLMOutputForPPO(
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=hidden_states,
        attentions=None,
    )

    # Megatron passes the shared embedding as output_weight for tied models. For
    # untied models the weight lives on output_layer.
    weight = output_weight if output_weight is not None else output_layer.weight

    temperature = context.temperature
    hidden_states = _gather_fused_hidden_states(hidden_states, config.sequence_parallel, context.impl_backend)
    logprobs, entropy = linear_cross_entropy(
        hidden_states,
        weight,
        labels,
        temperature,
        "none",
        parallel_state.get_tensor_model_parallel_group(),
        impl_backend=context.impl_backend,
    )

    if has_config_logger_enabled(config):
        payload = OrderedDict(
            {
                "input_ids": _ignored.get("input_ids"),
                "position_ids": _ignored.get("position_ids"),
                "attention_mask": _ignored.get("attention_mask"),
                "decoder_input": _ignored.get("decoder_input"),
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


def _validate_liger_moe_runtime(model: GPTModel) -> None:
    config = model.config
    if not getattr(config, "num_moe_experts", None):
        return
    legacy_deepep = getattr(config, "moe_enable_deepep", False) or (
        getattr(config, "moe_token_dispatcher_type", None) == "flex"
        and getattr(config, "moe_flex_dispatcher_backend", None) == "deepep"
    )
    if not legacy_deepep:
        return

    from deep_ep import Buffer

    group = parallel_state.get_expert_tensor_and_model_parallel_group()
    size = torch.distributed.get_world_size(group)
    # Match the legacy dispatcher's public buffer-size hints without creating
    # its lazy NVSHMEM-owning buffer after Liger has initialized the runtime.
    hidden_bytes = config.hidden_size * 2
    for options in (Buffer.get_dispatch_config(size), Buffer.get_combine_config(size)):
        if options.get_rdma_buffer_size_hint(hidden_bytes, size) > 0:
            raise RuntimeError(
                "Native Liger and DeepEP V1 RDMA cannot share NVSHMEM in one process. "
                "Use the alltoall dispatcher, a supported DeepEP V2 dispatcher, "
                "or disable the Liger fused output head."
            )


def _configure_liger_runtime(model: GPTModel, engine_config) -> bool:
    from verl.utils.kernel.linear_cross_entropy import configure_liger_flsce

    token_limits = [
        value
        for value in (engine_config.max_token_len_per_gpu, engine_config.infer_max_token_len_per_gpu)
        if value is not None
    ]
    reservation = getattr(engine_config, "_liger_flsce_capacity", None)
    if not token_limits and reservation is None:
        raise RuntimeError("Liger TP-FLSCE requires an existing max-token limit in the Megatron engine config")

    process_group = parallel_state.get_tensor_model_parallel_group()
    tp_size = torch.distributed.get_world_size(process_group)
    # GPTModel keeps the padded vocabulary and hidden dimensions on every PP
    # stage, including virtual chunks without an embedding or output weight.
    if model.vocab_size % tp_size:
        raise ValueError("Liger TP-FLSCE requires the model vocabulary to be divisible by TP size")
    max_tokens = max(token_limits, default=0) * engine_config.context_parallel_size
    min_tp_size = tp_size
    if reservation is not None:
        max_tokens = max(max_tokens, reservation[0])
        min_tp_size = min(min_tp_size, reservation[1])
    configured = configure_liger_flsce(
        max_tokens=max_tokens,
        hidden_size=model.config.hidden_size,
        local_vocab_size=(model.vocab_size + min_tp_size - 1) // min_tp_size,
        process_group=process_group,
        device=next(model.parameters()).device,
    )
    if configured:
        _validate_liger_moe_runtime(model)
    return configured


def patch_fused_forward(
    model: torch.nn.Module,
    model_config=None,
    *,
    engine_config=None,
    impl_backend: str = "triton",
):
    model = _get_patching_model(model)
    if model is None:
        return
    if model_config is not None:
        impl_backend = "liger" if model_config.use_liger else "triton"
    if impl_backend == "liger" and engine_config is not None and model.config.params_dtype == torch.bfloat16:
        _configure_liger_runtime(model, engine_config)
    setattr(model, _FUSED_IMPL_BACKEND_ATTR, impl_backend)

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


def fused_forward_model_engine(vision_model: bool = False):
    def fused_forward_model_engine_inner(
        model,
        input_ids: Tensor,
        labels: Tensor,
        multi_modal_inputs: dict,
        temperature: float,
        calculate_entropy: bool,
        pad_token_id: int,
        cp_layout: str = "zigzag",
        local_cp_size: int | None = None,
        router_padding_mask: Tensor | None = None,
        pad_to_length_bucket: int | None = None,
        position_ids: Tensor | None = None,
    ):
        pre_process = unwrap_model(model).pre_process
        post_process = unwrap_model(model).post_process

        fp8 = unwrap_model(model).config.fp8
        use_fp8_padding = fp8 in ["e4m3", "hybrid"]
        config = unwrap_model(model).config
        min_local_rows = (
            config.csa_window_size if getattr(config, "experimental_attention_variant", None) == "dsv4_hybrid" else None
        )

        thd_kwargs = dict(
            use_fp8_padding=use_fp8_padding,
            min_local_rows=min_local_rows,
            pad_to_length_bucket=pad_to_length_bucket,
            cp_layout=cp_layout,
            local_cp_size=local_cp_size,
        )
        input_ids_rmpad, packed_seq_params, _ = preprocess_thd_engine(
            input_ids, pre_process=pre_process or vision_model, **thd_kwargs
        )
        attention_mask = None
        position_ids_rmpad = None
        if vision_model:
            input_ids_rmpad, attention_mask, position_ids_rmpad = preprocess_vlm_thd_engine(
                model, input_ids, input_ids_rmpad, packed_seq_params, position_ids, pad_token_id, **thd_kwargs
            )
        input_ids_rmpad = input_ids_rmpad.contiguous()

        model_kwargs = {}
        if router_padding_mask is not None:
            model_kwargs["padding_mask"] = router_padding_mask
        if "pixel_values" in multi_modal_inputs:
            model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
        if "image_grid_thw" in multi_modal_inputs:
            model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)
        if "pixel_values_videos" in multi_modal_inputs:
            model_kwargs["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"].to(input_ids.device)
        if "video_grid_thw" in multi_modal_inputs:
            model_kwargs["video_grid_thw"] = multi_modal_inputs["video_grid_thw"].to(input_ids.device)

        labels_rmpad, _, _ = preprocess_thd_engine(
            labels,
            pre_process=True,
            need_roll=True,
            use_fp8_padding=use_fp8_padding,
            min_local_rows=min_local_rows,
            pad_to_length_bucket=pad_to_length_bucket,
            cp_layout=cp_layout,
            local_cp_size=local_cp_size,
        )
        labels_rmpad = labels_rmpad.contiguous()
        forward_kwargs = dict(
            input_ids=input_ids_rmpad,
            attention_mask=attention_mask,
            position_ids=position_ids_rmpad,
            packed_seq_params=packed_seq_params,
            labels=labels_rmpad,
            **model_kwargs,
        )
        if _use_output_processor_hook(model):
            impl_backend = get_fused_impl_backend(model)
            output_orig: CausalLMOutputForPPO = model(
                **forward_kwargs,
                output_processor=fused_output_processor,
                output_processor_context=FusedOutputProcessorContext(
                    temperature=temperature,
                    impl_backend=impl_backend,
                ),
            )
        else:
            impl_backend = get_fused_impl_backend(model)
            output_orig: CausalLMOutputForPPO = model(
                temperature=temperature,
                impl_backend=impl_backend,
                **forward_kwargs,
            )

        if not post_process:
            return output_orig

        log_probs = output_orig.log_probs
        if log_probs.dim() == 1:
            log_probs = log_probs.unsqueeze(0)
        log_probs = postprocess_thd_engine(
            log_probs,
            packed_seq_params,
            input_ids,
            input_ids.shape[0],
            post_process=post_process,
            cp_layout=cp_layout,
            local_cp_size=local_cp_size,
        )

        output = {"log_probs": log_probs}

        if calculate_entropy:
            entropy = output_orig.entropy
            if entropy.dim() == 1:
                entropy = entropy.unsqueeze(0)
            entropy = postprocess_thd_engine(
                entropy,
                packed_seq_params,
                input_ids,
                input_ids.shape[0],
                post_process=post_process,
                cp_layout=cp_layout,
                local_cp_size=local_cp_size,
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
    impl_backend: str = "triton",
    padding_mask: Tensor | None = None,
    **kwargs,
) -> CausalLMOutputForPPO:
    """
    Patch self._postprocess in forward for GPT models to enable fused kernel support.
    https://github.com/NVIDIA/Megatron-LM/blob/core_v0.13.0/megatron/core/models/gpt/gpt_model.py

    TODO: Currently we still need to patch `forward` because we need to pass `temperature`
    explicitly to `self._postprocess` when calling, maybe there can be a better way to handle this?
    """

    inference_context = deprecate_inference_params(inference_context, inference_params)

    preprocess_kwargs = {}
    if padding_mask is not None:
        # Only forward the kwarg when set: older Megatron-Core _preprocess
        # signatures (without MoE router padding support) stay compatible.
        preprocess_kwargs["padding_mask"] = padding_mask
    preproc_output = model._preprocess(
        input_ids=input_ids,
        position_ids=position_ids,
        decoder_input=decoder_input,
        inference_context=inference_context,
        packed_seq_params=packed_seq_params,
        **preprocess_kwargs,
    )

    (decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset) = preproc_output[:5]

    decoder_extra_block_kwargs = extra_block_kwargs or {}
    if padding_mask is not None:
        # _preprocess scatters the mask across sequence-parallel ranks.
        decoder_extra_block_kwargs["padding_mask"] = preproc_output[5]
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

    hidden_states = _gather_fused_hidden_states(hidden_states, model.config.sequence_parallel, impl_backend)

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
        impl_backend=impl_backend,
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
