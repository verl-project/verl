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

"""Validation for routed-expert, per-token NVFP4 training and rollout."""

import fnmatch
from typing import Any, Mapping

from verl.utils.megatron.low_precision import bf16_layer_indices, hf_config_get, text_config

# Megatron module paths of routed-expert GEMMs (grouped or sequential experts). Dense MLPs
# (``mlp.linear_fc1``) and shared experts (``mlp.shared_experts.linear_fc1``) do not match.
ROUTED_EXPERT_PATTERNS = ("*.mlp.experts.*linear_fc1", "*.mlp.experts.*linear_fc2")


def is_routed_expert_module(module_path: str) -> bool:
    return any(fnmatch.fnmatch(module_path, pattern) for pattern in ROUTED_EXPERT_PATTERNS)


def real_nvfp4_enabled(override_transformer_config: Mapping[str, Any]) -> bool:
    """Whether ``override_transformer_config.fp4`` trains the routed experts in NVFP4.

    NVFP4 training is switched on by ``fp4``, as FP8 training is by ``fp8``; it applies
    to routed-expert GEMMs only.
    """

    fp4 = override_transformer_config.get("fp4")
    if not fp4:
        return False
    fp4_recipe = override_transformer_config.get("fp4_recipe") or "nvfp4"
    if fp4 != "e2m1" or fp4_recipe != "nvfp4":
        raise ValueError(f"FP4 training supports fp4='e2m1' with fp4_recipe='nvfp4', got {fp4!r}/{fp4_recipe!r}")
    return True


def real_nvfp4_quant_recipe_config(num_layers: int, num_layers_at_start: int, num_layers_at_end: int) -> dict:
    """Megatron-Core per-module recipe: routed-expert GEMMs NVFP4, everything else BF16.

    The result is passed to ``RecipeConfig.from_config_dict``. MCore takes the
    first matching matcher, so the carved-out layers precede the expert
    patterns, and a final catch-all keeps attention, dense MLPs and shared
    experts in BF16: a TE module no matcher selects would inherit the layer's
    FP4 autocast. Layer indices in module paths are global only with PP=1,
    which :func:`validate_real_nvfp4_parallelism` enforces.
    """

    def glob(config: str, pattern: str) -> dict:
        return {"config": config, "type": "glob", "pattern": pattern, "enabled": True}

    matchers = {
        f"layer_{index}_bf16": glob("bf16", f"*.layers.{index}.*")
        for index in bf16_layer_indices(num_layers, num_layers_at_start, num_layers_at_end)
    }
    matchers["experts_fc1_nvfp4"] = glob("nvfp4", ROUTED_EXPERT_PATTERNS[0])
    matchers["experts_fc2_nvfp4"] = glob("nvfp4", ROUTED_EXPERT_PATTERNS[1])
    matchers["other_bf16"] = glob("bf16", "*")
    return {
        "configs": {
            "bf16": {"transformer_engine_config_type": "TEQuantizationParams", "training_recipe": {}},
            "nvfp4": {
                "transformer_engine_config_type": "TEQuantizationParams",
                # Enter NVFP4 even outside an FP4 autocast: some decoder blocks, such as
                # Megatron-Bridge's Qwen3-VL block, only open FP8 contexts.
                "training_recipe": {"fp4_quantization_recipe": "nvfp4", "override_nonquantized_autocast": True},
            },
        },
        "matchers": matchers,
    }


def validate_real_nvfp4_model_contract(hf_config: Any) -> None:
    """real_nvfp4 trains routed experts in NVFP4, so the model must have them."""

    decoder = text_config(hf_config)
    num_experts = int(hf_config_get(decoder, "num_experts") or hf_config_get(decoder, "n_routed_experts") or 0)
    if num_experts <= 0:
        raise ValueError(
            "real_nvfp4 requires a routed-expert MoE model; this config declares no experts "
            "(looked for num_experts / n_routed_experts)"
        )


def validate_real_nvfp4_parallelism(*, pipeline_size: int, virtual_pipeline_size: int | None) -> None:
    """Keep layer-index recipes and rank-local attestation within the PP=1 scope."""

    if pipeline_size != 1 or virtual_pipeline_size not in (None, 1):
        raise ValueError(
            "real_nvfp4 currently requires pipeline_model_parallel_size=1 and no virtual pipeline splitting; "
            "layer precision and local attestation use global layer indices, got pp="
            f"{pipeline_size} vpp={virtual_pipeline_size}"
        )


BACKWARD_OVERRIDES = ("dequantized", "high_precision")


def validate_real_nvfp4_te_recipe(recipe: Any) -> None:
    """Validate the effective Transformer Engine NVFP4 recipe.

    TE reads these fields from ``NVTE_*`` environment variables when the recipe
    is constructed, so this catches a worker started without them. Backward
    GEMMs may use the dequantized forward operands or the high-precision ones
    (``NVTE_BACKWARD_OVERRIDE``); every other field is fixed.
    """

    expected = {
        "row_scaled_activation": True,
        "disable_rht": True,
        "disable_stochastic_rounding": True,
        "disable_2d_quantization": True,
        # The per-token NVFP4 rollout uses standard NVFP4, not TE's
        # adaptive 4-over-6 representation. Keep training aligned.
        "nvfp4_4over6": "none",
        "nvfp4_4over6_e4m3_use_256": "all",
        "nvfp4_4over6_err_mode": "MAE",
    }
    mismatches = {
        name: {"expected": wanted, "actual": getattr(recipe, name, None)}
        for name, wanted in expected.items()
        if getattr(recipe, name, None) != wanted
    }
    for group_name in ("fp4_quant_fwd_inp", "fp4_quant_fwd_weight", "fp4_quant_bwd_grad"):
        group = getattr(recipe, group_name, None)
        for field_name in ("random_hadamard_transform", "stochastic_rounding", "fp4_2d_quantization"):
            actual = getattr(group, field_name, None)
            if actual is not False:
                mismatches[f"{group_name}.{field_name}"] = {"expected": False, "actual": actual}
    backward_override = getattr(recipe, "backward_override", None)
    if backward_override not in BACKWARD_OVERRIDES:
        mismatches["backward_override"] = {"expected": BACKWARD_OVERRIDES, "actual": backward_override}
    if mismatches:
        raise RuntimeError(f"real_nvfp4 Transformer Engine recipe drifted: {mismatches}")


def check_nvfp4_per_token_rollout(
    rollout_config: Any, quantization_config: Any, compilation_config: dict, engine_kwargs: dict
) -> None:
    """Keep a per-token NVFP4 rollout to the validated layout: no all2all expert
    parallelism, colocated IPC refit and full CUDA graphs for decode only.

    A no-op for other quantization configs, from ``rollout.quantization`` or the checkpoint.
    """
    from verl.utils.qat.compressed_tensors import is_nvfp4_per_token_config

    if not is_nvfp4_per_token_config(quantization_config):
        return
    dp = rollout_config.data_parallel_size
    # verl runs per-token NVFP4 on the monolithic FlashInfer TRT-LLM MoE kernel,
    # which has no all2all expert dispatch.
    if dp != 1 or engine_kwargs.get("enable_eplb", False):
        raise ValueError(f"Per-token NVFP4 rollout requires data_parallel_size=1 and no EPLB, got dp={dp}")
    # DCP cannot run FULL CUDA graphs, and PCP turns expert parallelism into all2all.
    context_parallel = {
        key: engine_kwargs.get(key, 1) or 1 for key in ("decode_context_parallel_size", "prefill_context_parallel_size")
    }
    if any(size > 1 for size in context_parallel.values()):
        raise ValueError(f"Per-token NVFP4 rollout does not support context parallelism, got {context_parallel}")
    mtp = rollout_config.mtp
    if mtp is not None and mtp.enable and mtp.enable_rollout:
        raise ValueError("Per-token NVFP4 rollout does not support speculative decoding/MTP")
    checkpoint_backend = getattr(rollout_config.checkpoint_engine, "backend", None)
    if checkpoint_backend != "naive":
        raise ValueError(
            f"Per-token NVFP4 refits through the colocated naive IPC checkpoint backend, got {checkpoint_backend!r}"
        )
    compilation_config.setdefault("cudagraph_mode", "FULL_DECODE_ONLY")
    if compilation_config["cudagraph_mode"] != "FULL_DECODE_ONLY":
        raise ValueError("Per-token NVFP4 rollout requires compilation_config.cudagraph_mode=FULL_DECODE_ONLY")
