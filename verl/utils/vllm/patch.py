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

import re
from collections.abc import Iterable
from functools import wraps

import torch

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

try:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3_5MoeForCausalLM)
except ImportError:
    pass


def patch_vllm_moe_model_weight_loader(model):
    # this is a work around to load the weight of vllm fused moe model
    # it is from a bug from vllm 0.8.2
    # all the weights are supposed to have a weight_loader, but the moe weights
    # do not have a weight_loader, so we need to patch it
    # (True, 'model.embed_tokens.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.bias')
    # (True, 'model.layers.0.self_attn.o_proj.weight')
    # (True, 'model.layers.0.mlp.gate.weight')
    # (True, 'model.layers.0.mlp.shared_expert.gate_up_proj.weight')
    # (True, 'model.layers.0.mlp.shared_expert.down_proj.weight')
    # (False, 'model.layers.0.mlp.shared_expert_gate.weight')   use default
    # (False, 'model.layers.0.input_layernorm.weight')          use default
    # (False, 'model.layers.0.post_attention_layernorm.weight') use default
    # (False, 'model.layers.0.mlp.experts.w13_weight')          use mlp.experts.weight_loader
    # (False, 'model.layers.0.mlp.experts.w2_weight')          use mlp.experts.weight_loader

    # Early return if no MOE models are supported
    if not SUPPORTED_MOE_MODELS:
        return

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
        raise ValueError("The provided model does not have a valid 'model' or 'language_model' attribute.")

    if not isinstance(model, tuple(SUPPORTED_MOE_MODELS)) and not isinstance(inner_model, tuple(SUPPORTED_MOE_MODELS)):
        return

    # TODO(@leisuzz): class Qwen3MoeLLMForCausalLM is not available if VLLM version < 0.11.0,
    # will update the 'if statement' with 'isinstance' when verl commonly use VLLM version >= 0.11.0
    if type(inner_model).__name__ in ("Qwen3MoeLLMForCausalLM", "Qwen3_5MoeForCausalLM"):
        inner_model = inner_model.model  # Reassign inner_model in Qwen3-vl

    for layer_idx, layer in enumerate(inner_model.layers):
        mlp_attr = MLP_ATTR_MAPPING.get(original_model_type, DEFAULT_MLP_ATTR)

        mlp = getattr(layer, mlp_attr, None)
        if not mlp:
            continue

        experts = getattr(mlp, "experts", None)
        if not experts or not hasattr(experts, "weight_loader"):
            continue

        # Patch the weight loaders
        for name, param in mlp.named_parameters():
            if "w13_weight" in name or "w2_weight" in name:
                param.weight_loader = experts.weight_loader


_QWEN3_OMNI_PACKED_EXPERT_RE = re.compile(
    r"^(?P<prefix>(?:(?:thinker|language_model)\.)?model\.layers\.\d+\.mlp\.experts)\."
    r"(?P<kind>gate_up_proj|down_proj)(?:\.weight)?$"
)
_QWEN3_OMNI_THINKER_MAPPER_PATCHED = False


def _expand_qwen3_omni_packed_moe_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        match = _QWEN3_OMNI_PACKED_EXPERT_RE.match(name)
        if match is None or tensor.dim() != 3:
            yield name, tensor
            continue

        prefix = match.group("prefix")
        if match.group("kind") == "gate_up_proj":
            gate_proj, up_proj = tensor.chunk(2, dim=1)
            for expert_idx in range(tensor.shape[0]):
                yield f"{prefix}.{expert_idx}.gate_proj.weight", gate_proj[expert_idx]
                yield f"{prefix}.{expert_idx}.up_proj.weight", up_proj[expert_idx]
        else:
            for expert_idx in range(tensor.shape[0]):
                yield f"{prefix}.{expert_idx}.down_proj.weight", tensor[expert_idx]


def _patch_qwen3_omni_thinker_load_weights(model_cls: type) -> None:
    if getattr(model_cls, "_verl_packed_moe_load_weights_patched", False):
        return

    original_load_weights = model_cls.load_weights

    @wraps(original_load_weights)
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return original_load_weights(self, _expand_qwen3_omni_packed_moe_weights(weights))

    model_cls.load_weights = load_weights
    model_cls._verl_packed_moe_load_weights_patched = True


def apply_qwen3_omni_thinker_patches() -> None:
    # The training-side standalone thinker state_dict emits bare keys
    # (``audio_tower.*``, ``model.*``, ``lm_head.*``), while vLLM's mapper
    # only rewrites ``thinker.*`` prefixes. It also emits packed HF expert
    # tensors that vLLM's generic Omni loader does not unpack itself. Patch
    # both surfaces before rollout weight sync calls ``load_weights``.
    global _QWEN3_OMNI_THINKER_MAPPER_PATCHED

    try:
        from vllm.model_executor.models.qwen3_omni_moe_thinker import (
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )
    except ImportError:
        return

    _patch_qwen3_omni_thinker_load_weights(Qwen3OmniMoeThinkerForConditionalGeneration)

    if _QWEN3_OMNI_THINKER_MAPPER_PATCHED:
        return

    mapper = getattr(Qwen3OmniMoeThinkerForConditionalGeneration, "hf_to_vllm_mapper", None)
    if mapper is None or not hasattr(mapper, "orig_to_new_prefix"):
        return

    extra = {
        "lm_head.": "language_model.lm_head.",
        "model.": "language_model.model.",
    }
    # Existing ``thinker.*`` rules run first (dict iteration order preserved);
    # for bare keys those rules no-op and the new prefixes apply.
    mapper.orig_to_new_prefix = {**mapper.orig_to_new_prefix, **extra}
    _QWEN3_OMNI_THINKER_MAPPER_PATCHED = True
