# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Megatron setup for routed-expert NVFP4 training: configure, finish and attest the model."""

import re
from typing import Any, Iterable

import torch

from verl.utils.megatron.low_precision import bf16_layer_indices, num_decoder_layers

from .config import (
    is_routed_expert_module,
    real_nvfp4_bf16_layers,
    real_nvfp4_quant_recipe_config,
    validate_real_nvfp4_model_contract,
    validate_real_nvfp4_parallelism,
    validate_real_nvfp4_te_recipe,
)

_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def apply_real_nvfp4_overrides(
    override_transformer_config: dict,
    hf_config: Any,
    *,
    pipeline_size: int,
    virtual_pipeline_size: int | None,
) -> None:
    """Limit ``override_transformer_config.fp4`` to routed-expert GEMMs with a per-module recipe.

    The rollout's ``NVFP4_PER_TOKEN`` preset quantizes routed experts only, so attention,
    dense MLPs and shared experts stay BF16 in training too.
    """

    from megatron.core.quantization.quant_config import RecipeConfig

    start, end = real_nvfp4_bf16_layers(override_transformer_config)
    validate_real_nvfp4_model_contract(hf_config)
    # Layer-index recipes and the post-build attestation use global layer
    # indices, which only holds without pipeline splitting.
    validate_real_nvfp4_parallelism(pipeline_size=pipeline_size, virtual_pipeline_size=virtual_pipeline_size)
    if override_transformer_config.get("fp8") not in (None, False):
        raise ValueError("override_transformer_config.fp4 cannot be combined with fp8")
    if override_transformer_config.get("fp4_param"):
        raise ValueError("NVFP4 training keeps BF16 parameters for Adam and refit; set fp4_param=False")
    if override_transformer_config.get("quant_recipe") is not None:
        raise ValueError("NVFP4 training sets override_transformer_config.quant_recipe itself")
    override_transformer_config["quant_recipe"] = RecipeConfig.from_config_dict(
        real_nvfp4_quant_recipe_config(num_decoder_layers(hf_config), start, end)
    )


def finish_real_nvfp4_module_init(models: Iterable[torch.nn.Module]) -> None:
    """Re-apply the per-module recipe to every GPTModel of the built (unwrapped) model chunks.

    GPTModel assigns it at the end of its constructor, but a subclass may rebuild modules
    afterwards: Megatron-Bridge's Qwen3-VL text model replaces its whole decoder, which
    would leave every decoder layer in the global FP4 autocast. The call is idempotent.
    """

    from megatron.core.models.gpt import GPTModel
    from megatron.core.quantization.utils import get_quant_config_or_none

    for model in models:
        for gpt_model in model.modules():
            if not isinstance(gpt_model, GPTModel):
                continue
            for name, submodule in gpt_model.named_modules():
                if hasattr(submodule, "finish_init"):
                    submodule.finish_init(get_quant_config_or_none(name, gpt_model.config.quant_recipe))


def attest_real_nvfp4_runtime(
    models: Iterable[torch.nn.Module],
    tf_config: Any,
    hf_config: Any,
    num_layers_at_start_in_bf16: int,
    num_layers_at_end_in_bf16: int,
) -> None:
    """Prove that the built (unwrapped) Megatron model chunks will enter TE's NVFP4 context."""

    fp4 = tf_config.fp4
    fp4_recipe = getattr(tf_config.fp4_recipe, "value", tf_config.fp4_recipe)
    if fp4 != "e2m1" or fp4_recipe != "nvfp4":
        raise RuntimeError(f"Megatron finalized a different FP4 config: fp4={fp4!r}, fp4_recipe={fp4_recipe!r}")

    from megatron.core.fp4_utils import get_fp4_recipe

    recipe = get_fp4_recipe(tf_config)
    if type(recipe).__name__ != "NVFP4BlockScaling":
        raise RuntimeError(f"Expected Transformer Engine NVFP4BlockScaling, got {type(recipe).__name__}")
    validate_real_nvfp4_te_recipe(recipe)

    # Check the effective precision of every TE linear module: routed-expert
    # GEMMs outside the carve-out in NVFP4, everything else in BF16.
    carved = set(
        bf16_layer_indices(num_decoder_layers(hf_config), num_layers_at_start_in_bf16, num_layers_at_end_in_bf16)
    )

    def recipe_name(training_recipe, field):
        value = getattr(training_recipe, field, None)
        return getattr(value, "value", value)

    nvfp4_layers, mismatches = set(), []
    for model in models:
        for name, submodule in model.named_modules():
            if not hasattr(submodule, "te_quant_params"):
                continue
            quant_params = submodule.te_quant_params
            if quant_params is None:
                # Without per-module params a TE module inherits the outer autocast. That is
                # BF16 only for a submodel built without FP4, such as a VLM vision tower.
                if getattr(getattr(submodule, "config", None), "fp4", None):
                    raise RuntimeError(f"real_nvfp4 per-module recipe did not match {name}")
                actual = fp8 = None
            else:
                actual = recipe_name(quant_params.training_recipe, "fp4_quantization_recipe")
                fp8 = recipe_name(quant_params.training_recipe, "fp8_quantization_recipe")
                if actual is not None and not quant_params.training_recipe.override_nonquantized_autocast:
                    # Would fall back to BF16 in a block that opens no FP4 autocast.
                    actual = "nvfp4 without autocast override"
            layer = _LAYER_INDEX.search(name)
            nvfp4 = is_routed_expert_module(name) and layer is not None and int(layer.group(1)) not in carved
            if fp8 is not None or actual != ("nvfp4" if nvfp4 else None):
                mismatches.append((name, actual, fp8))
            elif nvfp4:
                nvfp4_layers.add(int(layer.group(1)))

    if mismatches:
        raise RuntimeError(f"real_nvfp4 applied an unexpected precision to {len(mismatches)} modules: {mismatches[:8]}")
    if not nvfp4_layers:
        raise RuntimeError("real_nvfp4 found no routed-expert GEMM to train in NVFP4")
