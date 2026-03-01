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

import os
from typing import Optional
import torch

# Environment variable to enable TP resharding fix for async rollout
# Set VERL_ENABLE_TP_RESHARD=1 to enable automatic weight resharding when
# training TP size differs from inference TP size
ENABLE_TP_RESHARD = os.environ.get("VERL_ENABLE_TP_RESHARD", "0") == "1"

# To support different vLLM versions, we add the model into SUPPORTED_MOE_MODELS separately to avoid triggering
# unsupported issues.
SUPPORTED_MOE_MODELS = []

try:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM

    SUPPORTED_MOE_MODELS.append(DeepseekV2ForCausalLM)
    SUPPORTED_MOE_MODELS.append(DeepseekV3ForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.mixtral import MixtralForCausalLM

    SUPPORTED_MOE_MODELS.append(MixtralForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen2MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeLLMForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3NextForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.kimi_vl import KimiVLForConditionalGeneration

    SUPPORTED_MOE_MODELS.append(KimiVLForConditionalGeneration)
except ImportError:
    pass


def patch_vllm_moe_model_weight_loader(model):
    """
    Patch vLLM model weight loaders for proper weight synchronization.

    This function does two things:
    1. Patches MoE expert weights (w13_weight, w2_weight) to use the experts' weight_loader
       (workaround for vLLM 0.8.2+ bug where MoE weights don't have weight_loaders)
    2. Patches ALL parameters without weight_loaders to handle TP resharding when training
       and inference use different tensor parallelism configurations (e.g., Megatron TP=8
       exports full HF weights, but vLLM uses TP=16)

    Parameters with existing weight_loaders are not modified.

    Args:
        model: The vLLM model to patch
    """
    original_model_type = type(model)
    if hasattr(model, "runnable") and "ACLGraphWrapper" in str(original_model_type):
        model = model.runnable
        original_model_type = type(model)

    # Define MLP attribute mapping for different model types
    MLP_ATTR_MAPPING = {}
    try:
        from vllm.model_executor.models.mixtral import MixtralForCausalLM

        MLP_ATTR_MAPPING[MixtralForCausalLM] = "block_sparse_moe"
    except ImportError:
        pass

    DEFAULT_MLP_ATTR = "mlp"

    # Get inner model (either model.model or model.language_model)
    inner_model = getattr(model, "model", None) or getattr(model, "language_model", None)
    if inner_model is None:
        # Can't patch if we can't find the inner model
        return

    # For nested models like Qwen3-vl
    if type(inner_model).__name__ == "Qwen3MoeLLMForCausalLM":
        inner_model = inner_model.model

    # Check if this is a supported MoE model for the expert weight patching
    is_moe_model = SUPPORTED_MOE_MODELS and (
        isinstance(model, tuple(SUPPORTED_MOE_MODELS)) or isinstance(inner_model, tuple(SUPPORTED_MOE_MODELS))
    )

    # Step 1: Patch MoE expert weights (only for supported MoE models)
    if is_moe_model and hasattr(inner_model, "layers"):
        for layer_idx, layer in enumerate(inner_model.layers):
            mlp_attr = MLP_ATTR_MAPPING.get(original_model_type, DEFAULT_MLP_ATTR)

            mlp = getattr(layer, mlp_attr, None)
            if not mlp:
                continue

            experts = getattr(mlp, "experts", None)
            if not experts or not hasattr(experts, "weight_loader"):
                continue

            # Patch MoE expert weight loaders
            for name, param in mlp.named_parameters():
                if "w13_weight" in name or "w2_weight" in name:
                    param.weight_loader = experts.weight_loader

    # Step 2: Patch ALL parameters without weight_loaders for TP resharding
    # This handles the case where training exports full HF weights but vLLM uses different TP
    # Only enabled when VERL_ENABLE_TP_RESHARD=1
    if ENABLE_TP_RESHARD:
        _, tp_size = _get_tp_rank_and_size()
        if tp_size > 1:
            for name, param in inner_model.named_parameters():
                if not hasattr(param, "weight_loader"):
                    param.weight_loader = _create_tp_aware_weight_loader(name)


def _get_tp_rank_and_size():
    """Get tensor parallel rank and world size from vLLM's distributed state."""
    try:
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

        return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    except ImportError:
        # Fallback for older vLLM versions
        try:
            from vllm.model_executor.parallel_utils.parallel_state import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        except ImportError:
            return 0, 1


def _create_tp_aware_weight_loader(param_name: str):
    """
    Create a weight_loader that can handle full (unsharded) weights from HuggingFace format
    and shard them according to vLLM's tensor parallelism configuration.

    This is needed when training uses a different TP size than inference (e.g., Megatron TP=8
    exports full HF weights, but vLLM uses TP=16).

    Args:
        param_name: The parameter name (used for error messages)

    Returns:
        A weight_loader function that handles TP resharding
    """

    def tp_aware_weight_loader(
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: Optional[str] = None,
    ):
        """
        Weight loader that handles TP resharding for full HuggingFace weights.

        If loaded_weight matches param size, uses default loading.
        If loaded_weight is larger, shards it according to TP rank.
        """
        tp_rank, tp_size = _get_tp_rank_and_size()

        # If sizes match, no resharding needed
        if loaded_weight.shape == param.shape:
            param.data.copy_(loaded_weight)
            return

        # Determine which dimension to shard
        # Compare loaded_weight shape with param shape to find the sharded dimension
        shard_dim = None
        for dim in range(loaded_weight.dim()):
            if loaded_weight.shape[dim] != param.shape[dim]:
                if loaded_weight.shape[dim] == param.shape[dim] * tp_size:
                    shard_dim = dim
                    break

        if shard_dim is None:
            # Can't determine sharding, fall back to assertion (will fail with clear error)
            assert param.shape == loaded_weight.shape, (
                f"Cannot determine sharding strategy for {param_name}. "
                f"Loaded weight shape {loaded_weight.shape} does not match "
                f"parameter shape {param.shape} and is not a simple TP multiple."
            )
            param.data.copy_(loaded_weight)
            return

        # Calculate shard size and extract the correct shard
        shard_size = loaded_weight.shape[shard_dim] // tp_size

        # Build slice for extracting this rank's shard
        slices = [slice(None)] * loaded_weight.dim()
        slices[shard_dim] = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)

        shard = loaded_weight[tuple(slices)]

        # Verify shape matches
        assert shard.shape == param.shape, (
            f"Sharded weight shape {shard.shape} does not match "
            f"parameter shape {param.shape} for {param_name}"
        )

        param.data.copy_(shard)

    return tp_aware_weight_loader
