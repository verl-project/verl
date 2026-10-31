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

"""Fused PPO outputs at Megatron's native output-processor boundary."""

from collections import OrderedDict
from dataclasses import dataclass

import torch
from megatron.core import parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from torch import Tensor

from verl.models.mcore.compat import ensure_output_weight_ready, has_output_processor, supports_fused_readiness
from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
from verl.utils.megatron.native import get_language_model

from .model_forward import gptmodel_forward_model_engine

_FUSED_IMPL_BACKEND_ATTR = "_verl_fused_impl_backend"


def get_fused_impl_backend(model: torch.nn.Module) -> str:
    """Return the backend selected during engine initialization."""
    return getattr(get_language_model(model), _FUSED_IMPL_BACKEND_ATTR, "triton")


def _gather_fused_hidden_states(hidden_states, sequence_parallel, impl_backend, tp_group=None):
    if not sequence_parallel:
        return hidden_states
    return gather_from_sequence_parallel_region(
        hidden_states, tensor_parallel_output_grad=impl_backend.lower() != "liger", group=tp_group
    )


@dataclass(frozen=True)
class FusedOutputProcessorContext:
    """Immutable per-forward inputs; packed labels are supplied by the layout adapter."""

    temperature: float
    impl_backend: str = "triton"
    labels: Tensor | None = None


def fused_output_processor(*, hidden_states, output_layer, output_weight, labels, context, config, **kwargs):
    """Read synchronized output weights and compute vocabulary-sharded PPO statistics."""
    weight = output_weight if output_weight is not None else output_layer.weight
    ensure_output_weight_ready(weight)
    tp_group = output_layer.tp_group
    hidden_states = _gather_fused_hidden_states(
        hidden_states, config.sequence_parallel, context.impl_backend, tp_group=tp_group
    )
    labels = context.labels if context.labels is not None else labels
    hidden_states = hidden_states.transpose(0, 1).contiguous()
    log_probs, entropy = linear_cross_entropy(
        hidden_states, weight, labels, context.temperature, "none", tp_group, impl_backend=context.impl_backend
    )
    log_probs, entropy = log_probs.reshape(labels.shape), entropy.reshape(labels.shape)
    if has_config_logger_enabled(config):
        payload = OrderedDict(
            (key, kwargs.get(key)) for key in ("input_ids", "position_ids", "attention_mask", "decoder_input")
        )
        payload.update(logprobs=log_probs, entropy=entropy)
        log_config_to_disk(config, payload, prefix="input_and_logits")
    return {"log_probs": log_probs, "entropy": entropy}


def _validate_liger_moe_runtime(model: torch.nn.Module) -> None:
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


def _configure_liger_runtime(model: torch.nn.Module, engine_config) -> bool:
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


def configure_fused_forward(model, model_config=None, *, engine_config=None, impl_backend="triton"):
    """Configure native callbacks without replacing model methods.

    Providers without callbacks or safe overlapped parameter readiness use the
    existing eager engine path instead of a copied GPT forward.
    """
    if not has_output_processor(model) or not supports_fused_readiness(model):
        return False
    language_model = get_language_model(model)
    if model_config is not None:
        impl_backend = "liger" if model_config.use_liger else "triton"
    if impl_backend == "liger" and engine_config is not None and language_model.config.params_dtype == torch.bfloat16:
        _configure_liger_runtime(language_model, engine_config)
    setattr(language_model, _FUSED_IMPL_BACKEND_ATTR, impl_backend)
    return True


# Keep the setup name used by existing callers; it no longer patches forward.
patch_fused_forward = configure_fused_forward


def fused_forward_model_engine(vision_model: bool = False):
    """Use the shared THD/BSHD layout adapter for fused outputs, including MTP."""

    def forward(
        model,
        input_ids,
        labels,
        multi_modal_inputs,
        temperature,
        calculate_entropy,
        pad_token_id,
        *,
        loss_mask=None,
        response_attention_mask=None,
        mtp_enable_train=False,
        data_format="thd",
        forced_max_seqlen=None,
        mtp_loss_normalization_factor=None,
        **kwargs,
    ):
        args = {"label": labels}
        if mtp_enable_train:
            args.update(loss_mask=loss_mask, response_attention_mask=response_attention_mask)
        output = gptmodel_forward_model_engine(
            model,
            input_ids,
            multi_modal_inputs,
            logits_processor_args=args,
            vision_model=vision_model,
            pad_token_id=pad_token_id,
            data_format=data_format,
            forced_max_seqlen=forced_max_seqlen,
            mtp_enable_train=mtp_enable_train,
            mtp_loss_normalization_factor=mtp_loss_normalization_factor,
            output_processor=fused_output_processor,
            output_processor_context=FusedOutputProcessorContext(temperature, get_fused_impl_backend(model)),
            **kwargs,
        )
        if not calculate_entropy and isinstance(output, dict):
            output.pop("entropy", None)
        return output

    return forward
