# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Megatron engine adapter for MCore's combined 1F1B EP-overlap schedule.

MCore owns the pipeline scheduler and layer-level overlap. This module only
turns verl BSHD/THD microbatches into schedule plans and restores RL outputs.
"""

import inspect
from collections.abc import Callable

import torch
from megatron.core import parallel_state
from megatron.core.package_info import __version__ as mcore_version
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

from verl.models.mcore.util import (
    build_vlm_attn_mask_bshd,
    postprocess_bshd_engine,
    postprocess_thd_engine,
    preprocess_bshd_engine,
    preprocess_thd_engine,
    preprocess_vlm_thd_engine,
)
from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
from verl.utils.megatron_utils import unwrap_model


def build_schedule_plan(
    model,
    input_ids: torch.Tensor,
    *,
    data_format: str,
    multi_modal_inputs: dict,
    logits_processor: Callable,
    label: torch.Tensor,
    temperature: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    forced_max_seqlen: int | None = None,
    pad_to_length_bucket: int | None = None,
    cp_layout: str = "zigzag",
    router_padding_mask: torch.Tensor | None = None,
    pad_token_id: int | None = None,
    calculate_entropy: bool = False,
    use_fused_kernels: bool = False,
):
    """Build a BSHD or packed THD plan and adapt its tensor output for verl."""
    if data_format not in ("bshd", "thd"):
        raise ValueError(f"Unsupported EP-overlap format: {data_format}")

    unwrapped_model = unwrap_model(model)
    if not hasattr(unwrapped_model, "build_schedule_plan"):
        raise NotImplementedError(f"{type(unwrapped_model).__name__} has no Megatron build_schedule_plan")
    if getattr(unwrapped_model.config, "mtp_num_layers", 0):
        raise NotImplementedError("EP all-to-all overlap with MTP is not supported by the verl Megatron engine")

    use_fp8_padding = unwrapped_model.config.fp8 in ("e4m3", "hybrid")
    # A delegated bound method retains the child module as its owner. Native
    # schedule builders stay on the direct path, including future VLM builders.
    schedule_model = getattr(unwrapped_model.build_schedule_plan, "__self__", unwrapped_model)
    vision_model = schedule_model is not unwrapped_model
    if data_format == "bshd":
        model_input_ids, attention_mask, model_position_ids = preprocess_bshd_engine(
            input_ids,
            pre_process=unwrapped_model.pre_process,
            use_fp8_padding=use_fp8_padding,
            forced_max_seqlen=forced_max_seqlen,
        )
        packed_seq_params = None
        if vision_model:
            model_input_ids, attention_mask = build_vlm_attn_mask_bshd(
                input_ids, input_ids.shape[0], pad_token_id, forced_max_seqlen=forced_max_seqlen
            )
            model_position_ids = None
        label_input = preprocess_bshd_engine(
            label,
            pre_process=True,
            need_roll=True,
            use_fp8_padding=use_fp8_padding,
            forced_max_seqlen=forced_max_seqlen,
        )[0]
        temperature_input = preprocess_bshd_engine(
            temperature,
            pre_process=True,
            use_fp8_padding=use_fp8_padding,
            forced_max_seqlen=forced_max_seqlen,
        )[0]

        def restore_output(value):
            return postprocess_bshd_engine(value, attention_mask, post_process=True)

    else:
        thd_kwargs = dict(
            use_fp8_padding=use_fp8_padding,
            pad_to_length_bucket=pad_to_length_bucket,
            cp_layout=cp_layout,
        )
        model_input_ids, packed_seq_params, model_position_ids = preprocess_thd_engine(
            input_ids, pre_process=True, **thd_kwargs
        )
        attention_mask = None
        if vision_model:
            model_input_ids, attention_mask, model_position_ids = preprocess_vlm_thd_engine(
                model,
                input_ids,
                model_input_ids,
                packed_seq_params,
                position_ids,
                pad_token_id,
                **thd_kwargs,
            )
        model_input_ids = model_input_ids.contiguous()
        label_input = preprocess_thd_engine(label, pre_process=True, need_roll=True, **thd_kwargs)[0]
        temperature_input = preprocess_thd_engine(temperature, pre_process=True, **thd_kwargs)[0]

        def restore_output(value):
            return postprocess_thd_engine(
                value,
                packed_seq_params,
                input_ids,
                input_ids.shape[0],
                post_process=True,
                cp_layout=cp_layout,
            )

    # A full FP32 logits tensor can exceed the memory left after a MoE forward.
    # MCore's output_processor runs before Float16Module casts the schedule
    # output, so return only per-token statistics from that boundary.
    use_fused_output = use_fused_kernels and unwrapped_model.post_process
    temperature_value = None
    if use_fused_output:
        if "output_processor" not in inspect.signature(unwrapped_model.build_schedule_plan).parameters:
            raise NotImplementedError(
                f"Fused output with EP overlap requires the build_schedule_plan(output_processor=...) hook "
                f"introduced in Megatron-Core 0.18.0; detected Megatron-Core {mcore_version}, but "
                f"{type(schedule_model).__name__}.build_schedule_plan does not expose this hook. "
                "Use a compatible Megatron-Core/model implementation or set use_fused_kernels=False."
            )
        temperature_value = _uniform_positive_temperature(temperature_input)
        if temperature_value is None:
            raise NotImplementedError("Fused EP overlap requires a uniform positive temperature")
    output_processor = None
    if use_fused_output:

        def output_processor(*, hidden_states, output_layer, output_weight, config, **_ignored):
            if config.sequence_parallel:
                hidden_states = gather_from_sequence_parallel_region(hidden_states)
            weight = output_weight if output_weight is not None else output_layer.weight
            log_probs, entropy = linear_cross_entropy(
                hidden_states.contiguous(),
                weight,
                label_input.transpose(0, 1).contiguous(),
                temperature_value,
                "none",
                parallel_state.get_tensor_model_parallel_group(),
            )
            # The schedule accepts one tensor. Packing both statistics also
            # supplies a zero entropy gradient when only log_probs is used.
            return torch.stack((log_probs, entropy))

    if vision_model:
        if router_padding_mask is not None:
            raise NotImplementedError("VLM EP overlap does not support router padding masks")
        plan = _build_vlm_schedule_plan(
            unwrapped_model,
            schedule_model,
            model_input_ids,
            attention_mask,
            model_position_ids,
            packed_seq_params,
            multi_modal_inputs,
            output_processor,
        )
    else:
        plan_kwargs = dict(
            input_ids=model_input_ids,
            attention_mask=attention_mask,
            position_ids=model_position_ids,
        )
        if packed_seq_params is not None:
            plan_kwargs["packed_seq_params"] = packed_seq_params
        if router_padding_mask is not None:
            plan_kwargs["padding_mask"] = router_padding_mask
        if output_processor is not None:
            plan_kwargs["output_processor"] = output_processor
        plan = unwrapped_model.build_schedule_plan(**plan_kwargs)

    if not unwrapped_model.post_process:
        return plan, lambda output: output

    if use_fused_output:

        def adapt_output(statistics: torch.Tensor):
            def to_batch_first(value):
                return value.reshape(label_input.shape[1], label_input.shape[0]).transpose(0, 1).contiguous()

            log_probs = restore_output(to_batch_first(statistics[0]))
            result = {"log_probs": log_probs}
            if calculate_entropy:
                result["entropy"] = restore_output(to_batch_first(statistics[1]))
            return result

        return plan, adapt_output

    def adapt_output(logits: torch.Tensor):
        processed = logits_processor(logits, label=label_input, temperature=temperature_input)
        return {key: restore_output(value) for key, value in processed.items()}

    return plan, adapt_output


def _uniform_positive_temperature(temperature: torch.Tensor) -> float | None:
    """Return a scalar temperature, ignoring nonpositive padding positions."""
    values = temperature.detach().reshape(-1)
    if values.numel() == 0 or not torch.isfinite(values).all().item():
        return None
    positive = values[values > 0]
    if positive.numel() == 0 or not torch.all(positive == positive[0]).item():
        return None
    return float(positive[0].item())


def _build_vlm_schedule_plan(
    model,
    language_model,
    input_ids,
    attention_mask,
    position_ids,
    packed_seq_params,
    multi_modal_inputs,
    output_processor,
):
    """Preserve a Bridge VLM's vision/position path while scheduling its inner GPTModel."""
    original_forward = language_model.forward
    language_model.rotary_pos_emb.is_thd_format = False

    def build_inner_plan(*args, **kwargs):
        if args:
            raise TypeError("VLM language forward must pass schedule inputs by keyword")
        visual_pos_masks = kwargs.pop("visual_pos_masks", None)
        deepstack_visual_embeds = kwargs.pop("deepstack_visual_embeds", None)
        if output_processor is not None:
            kwargs["output_processor"] = output_processor
        plan = language_model.build_schedule_plan(**kwargs)
        if deepstack_visual_embeds:
            if visual_pos_masks is None:
                raise ValueError("VLM DeepStack embeddings require visual position masks")
            for layer_idx, visual_embeds in enumerate(deepstack_visual_embeds):
                layer_plan = plan.get_layer(layer_idx)
                last_node = layer_plan.moe_combine
                if not hasattr(last_node, "submodule"):
                    last_node = layer_plan.mlp
                if not hasattr(last_node, "submodule"):
                    raise NotImplementedError("MCore EP overlap layer has no DeepStack-compatible schedule node")
                original_submodule = last_node.submodule

                def with_deepstack(node, *node_args, _original=original_submodule, _embeds=visual_embeds):
                    output = _original(node, *node_args)
                    return language_model.decoder._deepstack_process(output, visual_pos_masks, _embeds)

                last_node.submodule = with_deepstack
        return plan

    model_kwargs = {
        key: multi_modal_inputs[key].to(input_ids.device)
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
        if key in multi_modal_inputs
    }
    language_model.forward = build_inner_plan
    try:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
            **model_kwargs,
        )
    finally:
        language_model.forward = original_forward


def expose_schedule_plan(model) -> None:
    """Expose a unique child's schedule API while retaining the outer model's forward.

    MCore only traverses ``.module`` wrappers when looking up this API. Delegate
    the bound method so its ``__self__`` identifies the schedule owner without
    additional model flags or assumptions about the child's attribute name.
    """
    unwrapped_model = unwrap_model(model)
    if callable(getattr(unwrapped_model, "build_schedule_plan", None)):
        return
    candidates = [
        (name, child)
        for name, child in unwrapped_model.named_modules()
        if child is not unwrapped_model and callable(getattr(child, "build_schedule_plan", None))
    ]
    if len(candidates) != 1:
        names = ", ".join(name for name, _ in candidates) or "none"
        raise NotImplementedError(
            f"{type(unwrapped_model).__name__} requires exactly one submodule with build_schedule_plan; found: {names}"
        )
    unwrapped_model.build_schedule_plan = candidates[0][1].build_schedule_plan
