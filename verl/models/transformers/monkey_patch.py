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
"""
Apply monkey-patch function to models
"""

import sys
from types import SimpleNamespace
from typing import Optional

import torch
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_utils import PreTrainedModel

from verl.utils.import_utils import is_trl_available
from verl.utils.transformers_compat import is_transformers_version_in_range
from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)

_PREFIX_GROUPER_PATCHED = False
_PREFIX_GROUPER_SUPPORTED_ATTENTIONS = {"flash_attention_2", "flash_attention_3", "sdpa", "flex_attention", "eager"}


def _create_prefix_grouper_wrapper(original_fn):
    """Wrap attention function to support prefix_grouper in kwargs."""

    def wrapped(module, query, key, value, attention_mask, *args, **kwargs):
        prefix_grouper = kwargs.pop("prefix_grouper", None)
        if prefix_grouper is None:
            return original_fn(module, query, key, value, attention_mask, *args, **kwargs)

        def attn_func(q, k, v, attn_mask, *inner_args, **inner_kwargs):
            out, _ = original_fn(module, q, k, v, attn_mask, *inner_args, **inner_kwargs)
            return out

        return prefix_grouper.forward(attn_func, query, key, value, *args, **kwargs), None

    return wrapped


def apply_prefix_grouper_patch():
    """Patch ALL_ATTENTION_FUNCTIONS to support prefix_grouper parameter."""
    global _PREFIX_GROUPER_PATCHED
    if _PREFIX_GROUPER_PATCHED:
        return

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    patched = []
    for name in list(ALL_ATTENTION_FUNCTIONS.keys()):
        if name in _PREFIX_GROUPER_SUPPORTED_ATTENTIONS:
            ALL_ATTENTION_FUNCTIONS[name] = _create_prefix_grouper_wrapper(ALL_ATTENTION_FUNCTIONS[name])
            patched.append(name)

    _PREFIX_GROUPER_PATCHED = True
    print(f"[PrefixGrouper] Patched: {patched}")


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=2, repeats=n_rep). The hidden states go from (batch,
    seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def _ulysses_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    *args,
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Insert all-to-all before and after flash attention.
    DeepSpeed-Ulysses: https://arxiv.org/pdf/2309.14509

    For transformers>=4.55, the flash attention api has changed,
    we need to pass the query_length after doing ulysses all2all.
    See https://github.com/huggingface/transformers/issues/40399

    Args:
        query_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads, head_dim)
        key_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        value_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        position_ids (torch.Tensor, optional): (batch_size, seqlen/sp_size)

    Returns:
        torch.Tensor: (batch_size, seqlen/sp_size, nheads, head_dim)

    """
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    ########## AlltoAll for Ulysses ##########
    # TODO: Disable sp for ViT, there's no elegent way to determine whether it's ViT or not.
    # Use `position_ids` as condition since ViT doesn't pass it to flash attention.
    if ulysses_sp_size > 1 and position_ids is not None:
        # NOTE: repeat kv heads to be divided by sequence parallel. Instead of repeating nheads_q//nheads_k,
        # we choose to repeat sp_size//nheads_k, since flash_attention supports MQA/GQA.
        # For example:
        # - nheads_k=4, sp=8, repeats=2
        # - nheads_k=8, sp=8, repeats=1
        # - nheads_k=16, sp=8, repeats=1
        repeats = max(ulysses_sp_size // key_states.size(2), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)

        # (bsz, seq_len/n, n_head, head_dim) -> (bsz, seq_len, n_head/n, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)

        # TODO: all_gather position_ids because `prepare_fa2_from_position_ids` needs it, we can eliminate
        # this all_gather by passing cu_seq_lens_q, cu_seq_lens_k, max_length_k, max_length_q explicitly.
        # https://github.com/huggingface/transformers/pull/33932

        # (bsz, seq_len/n) -> (bsz, seq_len)
        position_ids_list = [torch.empty_like(position_ids) for _ in range(ulysses_sp_size)]
        torch.distributed.all_gather(position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group())
        position_ids = torch.concat(position_ids_list, dim=-1)

    # (bsz, seq_len, n_head/n, head_dim)
    query_length = query_states.size(1)
    attn_output = _flash_attention_forward(
        query_states, key_states, value_states, attention_mask, query_length, *args, position_ids=position_ids, **kwargs
    )

    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1 and position_ids is not None:
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)

    return attn_output


def patch_vlm_for_ulysses_input_slicing(model_class: type):
    """
    Applies a monkey patch to the forward method of a given model class
    to enable Ulysses sequence parallelism input slicing.
    """

    def _create_ulysses_wrapped_decoder_forward(original_forward):
        def ulysses_wrapped_decoder_forward(self, *args, **kwargs):
            inputs_embeds = kwargs.get("inputs_embeds")
            position_ids = kwargs.get("position_ids")
            visual_pos_masks = kwargs.get("visual_pos_masks")
            deepstack_visual_embeds = kwargs.get("deepstack_visual_embeds")
            call_kwargs = kwargs.copy()

            current_ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

            slice_now = (
                inputs_embeds is not None
                and current_ulysses_sp_size > 1
                and getattr(self, "_needs_initial_slice", True)
            )
            if slice_now:
                call_kwargs["inputs_embeds"] = slice_input_tensor(inputs_embeds, dim=1, padding=False)
                call_kwargs["position_ids"] = slice_input_tensor(position_ids, dim=-1, padding=False)
                # Also slice visual_pos_masks and deepstack_visual_embeds for Qwen3 VL models
                if visual_pos_masks is not None:
                    original_visual_mask = visual_pos_masks
                    sliced_visual_mask = slice_input_tensor(visual_pos_masks, dim=1, padding=False)
                    call_kwargs["visual_pos_masks"] = sliced_visual_mask

                    if deepstack_visual_embeds is not None:
                        sliced_embeds = []

                        num_visual_before = original_visual_mask.sum().item()
                        num_visual_in_shard = sliced_visual_mask.sum().item()

                        if num_visual_in_shard > 0 and num_visual_before > 0:
                            # Calculate which visual embeddings belong to this shard
                            # We need to find the offset of visual tokens in this shard
                            from verl.utils.ulysses import get_ulysses_sequence_parallel_rank

                            rank = get_ulysses_sequence_parallel_rank()
                            seq_len = original_visual_mask.shape[1]
                            local_seq_len = seq_len // current_ulysses_sp_size
                            start_idx = rank * local_seq_len
                            end_idx = start_idx + local_seq_len

                            # Get total visual tokens before and up to the end of the shard's sequence slice
                            # This correctly handles batches by summing across all samples
                            visual_start = original_visual_mask[:, :start_idx].sum().item() if start_idx > 0 else 0
                            visual_end = original_visual_mask[:, :end_idx].sum().item()

                            # Slice each tensor in deepstack_visual_embeds
                            for embed in deepstack_visual_embeds:
                                sliced_embeds.append(embed[visual_start:visual_end])
                        else:
                            # No visual tokens in this shard, create empty tensors to maintain gradient flow
                            for embed in deepstack_visual_embeds:
                                sliced_embeds.append(embed[:0])
                        call_kwargs["deepstack_visual_embeds"] = sliced_embeds

                self._needs_initial_slice = False
            try:
                return original_forward(self, *args, **call_kwargs)
            finally:
                if slice_now:
                    self._needs_initial_slice = True

        return ulysses_wrapped_decoder_forward

    original_forward = model_class.forward
    wrapped_forward = _create_ulysses_wrapped_decoder_forward(original_forward)
    model_class.forward = wrapped_forward
    print(f"Monkey patch {model_class.__name__}.forward for Ulysses SP input slicing.")


def patch_forward_with_backends(
    model: PreTrainedModel,
    use_fused_kernels: bool = False,
    fused_kernels_backend: str = None,
):
    """
    Choose the forward function based on the model and backend.
    Args:
        model (PreTrainedModel): The model to apply the monkey patch.
        use_fused_kernels (bool): Whether to use fused kernels.
        fused_kernels_backend (str): The backend to use for fused kernels.
    """
    if not use_fused_kernels or fused_kernels_backend not in ["triton", "torch"]:
        print(
            f"Skipping monkey patch for {model.__class__.__name__} as use_fused_kernels is "
            f"{use_fused_kernels} or fused_kernels_backend is {fused_kernels_backend}"
        )
        return

    forward_with_torch_backend_function = model.__class__.forward
    forward_with_triton_backend_function = model.__class__.forward
    if model.config.model_type in ["qwen2_5_vl", "qwen2_vl"]:
        from verl.models.transformers.qwen2_vl import forward_with_torch_backend, forward_with_triton_backend

        forward_with_torch_backend_function = forward_with_torch_backend
        forward_with_triton_backend_function = forward_with_triton_backend
    elif model.config.model_type in ["qwen3_vl", "qwen3_vl_moe"]:
        from verl.models.transformers.qwen3_vl import forward_with_torch_backend, forward_with_triton_backend

        forward_with_torch_backend_function = forward_with_torch_backend
        forward_with_triton_backend_function = forward_with_triton_backend
    elif model.config.model_type == "glm4v":
        from verl.models.transformers.glm4v import forward_with_torch_backend, forward_with_triton_backend

        forward_with_torch_backend_function = forward_with_torch_backend
        forward_with_triton_backend_function = forward_with_triton_backend
    else:
        from verl.models.transformers.dense_common import forward_with_torch_backend, forward_with_triton_backend

        forward_with_torch_backend_function = forward_with_torch_backend
        forward_with_triton_backend_function = forward_with_triton_backend

    if fused_kernels_backend == "triton":
        model.__class__.forward = forward_with_triton_backend_function
        print(f"Using Triton backend for fused kernels in {model.__class__.__name__}")
    elif fused_kernels_backend == "torch":
        model.__class__.forward = forward_with_torch_backend_function
        print(f"Using Torch backend for fused kernels in {model.__class__.__name__}")
    else:
        raise ValueError(f"Unsupported fused_kernels_backend: {fused_kernels_backend}. Choose 'triton' or 'torch'.")


def apply_monkey_patch(
    model: PreTrainedModel,
    ulysses_sp_size: int = 1,
    use_remove_padding: bool = True,
    use_fused_kernels: bool = False,
    fused_kernels_backend: str = None,
    use_prefix_grouper: bool = False,
    use_tiled_mlp: bool = False,
    tiled_mlp_shards: int = 4,
):
    """
    Apply monkey patch to the models for ulysses sequence parallel, fused kernel, tiled MLP and prefix grouper.

    In the end of this function forward function of the model is patched for fused kernel.
    If the model is not supported with fused kernel, please return after patch.

    Args:
        model: The model to apply the monkey patch.
        ulysses_sp_size: The size of ulysses sequence parallel.
        use_remove_padding: Whether to use remove padding.
        use_fused_kernels: Whether to use fused kernels.
        fused_kernels_backend: The backend to use for fused kernels.
        use_tiled_mlp: Whether to use TiledMLP for memory-efficient MLP computation.
        tiled_mlp_shards: Number of shards for TiledMLP (higher = lower memory, slightly slower).
    """

    # Apply TiledMLP monkey patch for memory-efficient MLP computation
    if use_tiled_mlp:
        from verl.models.transformers.tiled_mlp import apply_tiled_mlp_monkey_patch

        model_type = getattr(model.config, "model_type", None)
        apply_tiled_mlp_monkey_patch(num_shards=tiled_mlp_shards, model_type=model_type)
    # Apply PrefixGrouper patch if enabled
    if use_prefix_grouper:
        apply_prefix_grouper_patch()

    """Replace _flash_attention_forward to _ulysses_flash_attention_forward"""
    module = sys.modules[model.__module__]

    try:
        num_attention_heads, num_key_value_heads = model.config.num_attention_heads, model.config.num_key_value_heads
    except AttributeError:
        num_attention_heads, num_key_value_heads = (
            model.config.text_config.num_attention_heads,
            model.config.text_config.num_key_value_heads,
        )

    assert num_attention_heads % ulysses_sp_size == 0, (
        f"num_attention_heads {num_attention_heads} must be divisible by ulysses_sp_size {ulysses_sp_size}"
    )
    assert num_key_value_heads % ulysses_sp_size == 0 or ulysses_sp_size % num_key_value_heads == 0, (
        f"num_key_value_heads {num_key_value_heads} must be divisible by ulysses_sp_size "
        f"{ulysses_sp_size}or vise versa. Upon ulysses_sp_size % num_key_value_heads == 0,"
        f"kv heads are repeated to ensure correctness."
    )

    if is_trl_available():
        from trl import AutoModelForCausalLMWithValueHead  # type: ignore

        def state_dict(self, *args, **kwargs):
            return torch.nn.Module.state_dict(self, *args, **kwargs)

        AutoModelForCausalLMWithValueHead.state_dict = state_dict
        print("Monkey patch state_dict in AutoModelForCausalLMWithValueHead. ")

    # TODO: VLM models only, unify monkey patch to LLM models.
    if model.config.model_type in ["qwen2_5_vl", "qwen2_vl"]:
        # Step 1: patch model to support image-text mixed data
        if is_transformers_version_in_range(min_version="4.52.0"):
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLForConditionalGeneration,
                Qwen2_5_VLModel,
                Qwen2_5_VLTextModel,
            )
            from transformers.models.qwen2_vl.modeling_qwen2_vl import (
                Qwen2VLForConditionalGeneration,
                Qwen2VLModel,
                Qwen2VLTextModel,
            )
        else:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel as Qwen2_5_VLTextModel
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel as Qwen2VLTextModel

            Qwen2_5_VLModel = SimpleNamespace(forward=None)
            Qwen2VLModel = SimpleNamespace(forward=None)

        from verl.models.transformers.qwen2_vl import forward_with_normal_backend, qwen2_vl_base_forward

        Qwen2_5_VLModel.forward = qwen2_vl_base_forward
        Qwen2VLModel.forward = qwen2_vl_base_forward
        Qwen2_5_VLForConditionalGeneration.forward = forward_with_normal_backend
        Qwen2VLForConditionalGeneration.forward = forward_with_normal_backend
        print(f"Monkey patch {model.__class__.__name__} model forward")

        # Step 2: patch attention to support ulysses parallelism
        if is_transformers_version_in_range(min_version="4.54.0"):
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention
        elif is_transformers_version_in_range(min_version="4.53.0"):
            raise RuntimeError("Transformers 4.53.* is bugged. Use transformers 4.54.0 or later.")
        else:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLFlashAttention2 as Qwen2_5_VLAttention,
            )
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLFlashAttention2 as Qwen2VLAttention

        if use_remove_padding or ulysses_sp_size > 1:
            from verl.models.transformers.qwen2_vl import qwen2_vl_attn_forward

            Qwen2_5_VLAttention.forward = qwen2_vl_attn_forward
            Qwen2VLAttention.forward = qwen2_vl_attn_forward
            print(f"Monkey patch {model.__class__.__name__} attention layer")

        # Step 3: patch input for multimodal sequence parallelism
        if ulysses_sp_size > 1:
            patch_vlm_for_ulysses_input_slicing(Qwen2_5_VLTextModel)
            patch_vlm_for_ulysses_input_slicing(Qwen2VLTextModel)

    elif model.config.model_type in ["qwen3_vl", "qwen3_vl_moe"]:
        # Step 1: patch model to support image-text mixed data
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
            Qwen3VLModel,
            Qwen3VLTextModel,
        )
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeForConditionalGeneration,
            Qwen3VLMoeModel,
            Qwen3VLMoeTextModel,
        )

        from verl.models.transformers.qwen3_vl import (
            forward_with_normal_backend,
            patch_qwen3_vl_moe_sparse_moe_block_forward,
            qwen3_vl_base_forward,
        )

        Qwen3VLModel.forward = qwen3_vl_base_forward
        Qwen3VLMoeModel.forward = qwen3_vl_base_forward
        Qwen3VLForConditionalGeneration.forward = forward_with_normal_backend
        Qwen3VLMoeForConditionalGeneration.forward = forward_with_normal_backend
        print(f"Monkey patch {model.__class__.__name__} model forward")

        # Step 1.5: patch Qwen3VLMoeTextSparseMoeBlock to fix transformers 4.57.3 bug
        if model.config.model_type == "qwen3_vl_moe" and is_transformers_version_in_range(max_version="4.57.3"):
            patch_qwen3_vl_moe_sparse_moe_block_forward()

        # Step 2: patch input for multimodal sequence parallelism
        if ulysses_sp_size > 1:
            patch_vlm_for_ulysses_input_slicing(Qwen3VLTextModel)
            patch_vlm_for_ulysses_input_slicing(Qwen3VLMoeTextModel)

    elif model.config.model_type == "glm4v":
        # Step 1: patch model to support image-text mixed data

        from transformers.models.glm4v.modeling_glm4v import (
            Glm4vForConditionalGeneration,
            Glm4vModel,
            Glm4vTextAttention,
            Glm4vTextModel,
        )

        from verl.models.transformers.glm4v import forward_with_normal_backend, glm4v_base_forward

        Glm4vModel.forward = glm4v_base_forward
        Glm4vForConditionalGeneration.forward = forward_with_normal_backend
        print(f"Monkey patch {model.__class__.__name__} model forward")

        # Step 2: patch attention to support ulysses parallelism
        if use_remove_padding or ulysses_sp_size > 1:
            from verl.models.transformers.glm4v import glm4v_attn_forward

            Glm4vTextAttention.forward = glm4v_attn_forward
            print(f"Monkey patch {model.__class__.__name__} attention layer")

        # Step 3: patch input for multimodal sequence parallelism
        if ulysses_sp_size > 1:
            patch_vlm_for_ulysses_input_slicing(Glm4vTextModel)

    elif model.config.model_type == "kimi_vl":
        if use_remove_padding or ulysses_sp_size > 1:
            # TODO: Changes need to be made when transformers are adapted.
            from verl.models.transformers.kimi_vl import _ulysses_flash_attn_forward

            module.DeepseekV3FlashAttention2.forward = _ulysses_flash_attn_forward
            print("Monkey patch FlashAttention2.forward in KimiVL")

        if ulysses_sp_size > 1:
            patch_vlm_for_ulysses_input_slicing(module.DeepseekV3ForCausalLM)

        if use_fused_kernels:
            print("Not support fused kernels for KimiVL")

        return

    if use_remove_padding or ulysses_sp_size > 1:
        if hasattr(module, "_flash_attention_forward"):  # transformers <= 4.47.1 or legacy models
            module._flash_attention_forward = _ulysses_flash_attention_forward
            print(f"Monkey patch _flash_attention_forward in {model.__module__}")
        else:
            from transformers.integrations import flash_attention

            flash_attention._flash_attention_forward = _ulysses_flash_attention_forward
            print(f"Monkey patch _flash_attention_forward in {flash_attention.__name__}")

    patch_forward_with_backends(model, use_fused_kernels=use_fused_kernels, fused_kernels_backend=fused_kernels_backend)
