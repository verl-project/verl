# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Explicit full-model R3 arguments for the controlled Kimi K3.1 HF view.

The multimodal and residual forward bodies retain the native FSDPTurbo structure;
only route propagation differs. Components reuse existing parameter objects,
so checkpoint names and FSDP/EP ownership stay unchanged. Route tensors are
forward arguments, including during activation recomputation.
"""

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from fsdp_turbo.models.kimi.kimi_k31.moe import FSDPTurboKimiSparseMoeBlock
from fsdp_turbo.models.kimi.kimi_k31.dsa import (
    KimiK31ForConditionalGeneration as NativeKimiK31ForConditionalGeneration,
)
from fsdp_turbo.models.kimi.kimi_k31.modeling_kimi_k31 import (
    LlavaCausalLMOutputWithPast,
)
from fsdp_turbo.models.kimi.kimi_k31.modeling_kimi_linear import (
    KimiDecoderLayer, KimiLinearForCausalLM, KimiMoEGate, KimiPreTrainedModel,
)


def _align_replay_to_model_rows(
    routes: torch.Tensor, weights: torch.Tensor, source_mask: torch.Tensor | None,
    model_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Place executed causal rows in the expanded, possibly padded HF input.

    The final sampled token has no rollout forward pass. It and HF padding
    therefore use the ordinary gate; their placeholders are never replayed.
    All scored causal rows retain the exact captured IDs and weight bytes.
    """
    valid = model_mask.bool()
    source_mask = (torch.ones(routes.shape[:2], dtype=torch.bool, device=routes.device)
                   if source_mask is None else source_mask.bool())
    if weights.shape != routes.shape or source_mask.shape != routes.shape[:2]:
        raise ValueError("Kimi R3 IDs, weights and source mask must share one row layout")
    counts = source_mask.sum(dim=1)
    expected = valid.sum(dim=1) - 1
    if not torch.equal(counts, expected):
        raise ValueError(
            "Kimi R3 executed causal row count differs after image expansion: "
            f"captured={counts.tolist()}, expected={expected.tolist()}"
        )
    ordinal = valid.long().cumsum(dim=1) - 1
    replay = valid & (ordinal < counts[:, None])
    indices = ordinal.clamp(min=0, max=routes.shape[1] - 1)
    indices = indices[:, :, None, None].expand(-1, -1, routes.shape[2], routes.shape[3])
    return routes.gather(1, indices), weights.gather(1, indices), replay


class KimiK31ReplayGate(KimiMoEGate):
    """Execute captured FP32 weights with gradients through the selected scores."""

    def __init__(self, original: KimiMoEGate) -> None:
        nn.Module.__init__(self)
        for name in ("config", "top_k", "num_experts", "routed_scaling_factor",
                     "moe_router_activation_func", "num_expert_group", "topk_group",
                     "moe_renormalize", "gating_dim"):
            setattr(self, name, getattr(original, name))
        self.weight = original.weight
        self.e_score_correction_bias = original.e_score_correction_bias

    def forward(
        self, hidden_states: torch.Tensor, routed_experts: torch.Tensor | None = None,
        routed_expert_weights: torch.Tensor | None = None, router_replay_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        natural_ids, natural_weights = super().forward(hidden_states)
        if routed_experts is None:
            return natural_ids, natural_weights
        if routed_expert_weights is None:
            raise ValueError("Full Kimi R3 requires captured expert weights")
        replay_ids = routed_experts.reshape(-1, self.top_k).long()
        mask = (torch.ones(replay_ids.shape[0], dtype=torch.bool, device=hidden_states.device)
                if router_replay_mask is None else router_replay_mask.reshape(-1).bool())
        ids = torch.where(mask[:, None], replay_ids, natural_ids)
        logits = F.linear(hidden_states.reshape(-1, self.gating_dim).float(), self.weight.float())
        scores = logits.sigmoid() if self.moe_router_activation_func == "sigmoid" else logits.softmax(dim=-1)
        selected = scores.gather(1, ids)
        if self.top_k > 1 and self.moe_renormalize:
            selected = selected / (selected.sum(dim=-1, keepdim=True) + 1e-20)
        selected = selected * self.routed_scaling_factor
        # The forward value is exactly the captured execution value; the zero
        # difference carries gradients through current scores at replayed IDs.
        weights = routed_expert_weights.reshape_as(selected).to(selected.dtype)
        weights = weights + (selected - selected.detach())
        result = torch.where(mask[:, None], weights, natural_weights)
        import os
        audit_dir = os.environ.get("KIMI_R3_AUDIT_DIR")
        visit = getattr(self, "_r3_train_audit_visits", 0)
        if audit_dir and self.training and visit < 2:
            from pathlib import Path
            import json
            with torch.no_grad():
                captured = routed_expert_weights.reshape_as(selected).to(selected.dtype)
                exact = torch.equal(result[mask], captured[mask]) and torch.equal(ids[mask], replay_ids[mask])
                if not exact:
                    raise AssertionError("Kimi R3 executed actor routes/weights differ from captured inputs")
                directory = Path(audit_dir)
                directory.mkdir(parents=True, exist_ok=True)
                stem = f"{os.getpid()}-{id(self)}-{visit}"
                torch.save({"ids": ids.detach().cpu(), "weights": result.detach().cpu(), "mask": mask.detach().cpu()}, directory / (stem + ".pt"))
                (directory / (stem + ".json")).write_text(json.dumps({"pid": os.getpid(), "visit": visit, "dtype": str(result.dtype), "rows": int(mask.sum()), "exact": exact}))
                self._r3_train_audit_visits = visit + 1
        return ids, result


class KimiK31ReplayMoE(FSDPTurboKimiSparseMoeBlock):
    """Packed expert structure with an explicit per-layer replay input."""

    def __init__(self, original: FSDPTurboKimiSparseMoeBlock) -> None:
        nn.Module.__init__(self)
        self.config = original.config
        self.use_latent_moe = original.use_latent_moe
        self.moe_hidden_size = original.moe_hidden_size
        self.latent_moe_use_norm = original.latent_moe_use_norm
        for name, child in original.named_children():
            self.add_module(name, KimiK31ReplayGate(child) if name == "gate" else child)

    def forward(
        self, hidden_states: torch.Tensor, routed_experts: torch.Tensor | None = None,
        routed_expert_weights: torch.Tensor | None = None, router_replay_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states, routed_experts, routed_expert_weights, router_replay_mask)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        if self.use_latent_moe:
            hidden_states = self.routed_expert_down_proj(hidden_states)

        y = self.experts(hidden_states, topk_idx, topk_weight)

        if self.use_latent_moe:
            if self.latent_moe_use_norm:
                y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)

        y = y.view(*orig_shape)

        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y


class KimiK31ReplayDecoderLayer(KimiDecoderLayer):
    """Keep replay arguments in the recomputed decoder call."""

    def __init__(self, original: KimiDecoderLayer) -> None:
        nn.Module.__init__(self)
        for name in ("config", "hidden_size", "layer_idx", "is_linear_attn", "use_attn_residuals"):
            setattr(self, name, getattr(original, name))
        if self.use_attn_residuals:
            self.attn_res_block_size = original.attn_res_block_size
            self.attn_res_fn = original.attn_res_fn
        for name, child in original.named_children():
            self.add_module(name, KimiK31ReplayMoE(child) if name == "block_sparse_moe" else child)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        block_residual: torch.Tensor | None = None,
        routed_experts: torch.Tensor | None = None,
        routed_expert_weights: torch.Tensor | None = None,
        router_replay_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        if self.use_attn_residuals:
            return self._forward_attn_residual(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                block_residual,
                routed_experts=routed_experts,
                routed_expert_weights=routed_expert_weights,
                router_replay_mask=router_replay_mask,
                **kwargs,
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        if self.is_linear_attn is False:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache_params=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            hidden_states = self.block_sparse_moe(
                hidden_states,
                None if routed_experts is None else routed_experts[:, :, self.layer_idx, :],
                None if routed_expert_weights is None else routed_expert_weights[:, :, self.layer_idx, :],
                router_replay_mask,
            )
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def _forward_attn_residual(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        block_residual: torch.Tensor | None = None,
        routed_experts: torch.Tensor | None = None,
        routed_expert_weights: torch.Tensor | None = None,
        router_replay_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        batch_size, seq_len, hidden_size = hidden_states.shape
        prefix_sum = hidden_states

        if block_residual is not None and block_residual.shape[1] > 0:
            hidden_states = self.attn_res_fn(
                prefix_sum.view(-1, hidden_size),
                block_residual,
                self.self_attention_res_proj.weight,
                self.self_attention_res_norm.weight,
                self.self_attention_res_norm.variance_epsilon,
            ).view(batch_size, seq_len, hidden_size)

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual = torch.cat([block_residual, prefix_sum.view(-1, hidden_size).unsqueeze(1)], dim=1)
            prefix_sum = None

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        if self.is_linear_attn is False:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache_params=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

        if prefix_sum is not None:
            prefix_sum = prefix_sum + hidden_states
        else:
            prefix_sum = hidden_states

        hidden_states = self.attn_res_fn(
            prefix_sum.view(-1, hidden_size),
            block_residual,
            self.mlp_res_proj.weight,
            self.mlp_res_norm.weight,
            self.mlp_res_norm.variance_epsilon,
        ).view(batch_size, seq_len, hidden_size)

        hidden_states = self.post_attention_layernorm(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            hidden_states = self.block_sparse_moe(
                hidden_states,
                None if routed_experts is None else routed_experts[:, :, self.layer_idx, :],
                None if routed_expert_weights is None else routed_expert_weights[:, :, self.layer_idx, :],
                router_replay_mask,
            )
        else:
            hidden_states = self.mlp(hidden_states)

        if prefix_sum is None:
            prefix_sum = hidden_states
        else:
            prefix_sum = prefix_sum + hidden_states

        return prefix_sum, block_residual


class KimiK31ReplayForCausalLM(KimiLinearForCausalLM):
    """Preserve language-model kwargs through to every decoder layer."""

    def __init__(self, original: KimiLinearForCausalLM) -> None:
        KimiPreTrainedModel.__init__(self, original.config)
        self.model = original.model
        self.lm_head = original.lm_head
        self.vocab_size = original.vocab_size
        self.model.layers = nn.ModuleList(KimiK31ReplayDecoderLayer(layer) for layer in self.model.layers)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        generation_mode: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        """Preserve the base forward contract while propagating explicit R3 tensors."""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            **kwargs,
        )

        # Modification start: return hidden_states to the top-level model for
        # unified logits/loss computation, enabling MindSpeed-MM chunk loss.
        # The VLM wrapper (KimiK3ForConditionalGeneration) will apply lm_head.
        return CausalLMOutputWithPast(
            past_key_values=outputs.past_key_values,
            hidden_states=outputs[0],
            attentions=outputs.attentions,
        )


class KimiK31ReplayForConditionalGeneration(NativeKimiK31ForConditionalGeneration):
    """Forward captured full model rows through multimodal expansion."""

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | list[torch.FloatTensor] | None = None,
        grid_thws: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        routed_experts: torch.Tensor | None = None,
        routed_expert_weights: torch.Tensor | None = None,
        router_replay_mask: torch.Tensor | None = None,
        r3_full_model_rows: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> tuple | LlavaCausalLMOutputWithPast:
        """Preserve dense R3 and delegate non-replay packing to the native VLM.

        The engine discovers packing support from the explicit cu_seqlens
        argument. Its optional CPU copy is accepted for engine compatibility;
        the native multimodal forward owns expansion of the device boundaries.
        """
        if cu_seqlens is not None:
            if (routed_experts is not None or routed_expert_weights is not None
                    or router_replay_mask is not None or r3_full_model_rows):
                raise NotImplementedError("Packed Kimi forward does not support R3 replay")
            return super().forward(
                input_ids=input_ids, pixel_values=pixel_values, grid_thws=grid_thws,
                attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                use_cache=use_cache, output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, return_dict=return_dict,
                cu_seqlens=cu_seqlens,
            )
        assert self.vision_tower is not None, "vision_tower is not loaded"
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Chunk loss needs the full hidden-state sequence; force it on if enabled.
        enable_chunk_loss = getattr(self, "enable_chunk_loss", False)
        if enable_chunk_loss:
            output_hidden_states = True

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and len(pixel_values) > 0 and input_ids.shape[1] != 1:
                image_features = self._extract_image_features(pixel_values, grid_thws)
                if self.mm_projector:
                    image_features = self.mm_projector(image_features)

                inputs_embeds = inputs_embeds.to(image_features[0].dtype)  # num_tokens, embed_dim
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    labels,
                )

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        if routed_experts is not None:
            if not r3_full_model_rows:
                raise ValueError("Kimi R3 requires expanded multimodal model rows")
            model_mask = (torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
                          if attention_mask is None else attention_mask)
            routed_experts, routed_expert_weights, router_replay_mask = _align_replay_to_model_rows(
                routed_experts, routed_expert_weights, router_replay_mask, model_mask,
            )
        outputs = self.language_model(
            routed_experts=routed_experts,
            routed_expert_weights=routed_expert_weights,
            router_replay_mask=router_replay_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Modification start: chunk loss adaptation, following Kimi-K2.5 FSDP.
        # The language model returns hidden_states; the VLM top-level applies
        # lm_head so that MindSpeed-MM's chunk loss patch can be used.
        hidden_states = outputs.hidden_states

        if enable_chunk_loss:
            logits = None
            chunk_size = getattr(self, "chunk_size", 1024)
            if labels is None:
                raise ValueError("labels must be provided when chunk loss is enabled")

            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if attention_mask is not None:
                shift_labels = shift_labels.masked_fill(
                    attention_mask[..., 1:].to(shift_labels.device) == 0,
                    -100,
                )

            # Normalize every chunk by the total number of valid tokens so that
            # their accumulated loss matches the regular mean cross-entropy.
            num_valid_tokens = shift_labels.ne(-100).sum().clamp_min(1)
            lm_head = self.language_model.lm_head
            loss = sum(
                F.cross_entropy(
                    lm_head(hidden_chunk).float().reshape(-1, self.config.text_config.vocab_size),
                    label_chunk.reshape(-1), reduction="sum", ignore_index=-100,
                )
                for hidden_chunk, label_chunk in zip(
                    torch.split(shift_hidden_states, chunk_size, dim=1),
                    torch.split(shift_labels, chunk_size, dim=1),
                )
            ) / num_valid_tokens

        else:
            logits = self.language_model.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                if attention_mask is not None:
                    shift_attention_mask = attention_mask[..., 1:]
                    shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                    shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
                else:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1).to(shift_logits.device),
                )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
