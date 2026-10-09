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

from typing import Any


def _hf_get(hf_config: Any, name: str, default: Any = None) -> Any:
    if isinstance(hf_config, dict):
        return hf_config.get(name, default)
    return getattr(hf_config, name, default)


def real_nvfp4_moe_layer_indices(hf_config: Any) -> list[int]:
    """Indices of the decoder layers that carry routed experts.

    Derived from the config rather than from the architecture name, following
    the HF placement rule shared by the Qwen MoE family: layer ``i`` is sparse
    when it is not listed in ``mlp_only_layers`` and ``(i + 1)`` is a multiple
    of ``decoder_sparse_step``. Models that omit those keys are all-MoE.
    """

    num_layers = int(_hf_get(hf_config, "num_hidden_layers") or 0)
    sparse_step = int(_hf_get(hf_config, "decoder_sparse_step") or 1)
    dense_layers = set(_hf_get(hf_config, "mlp_only_layers") or [])
    if sparse_step <= 0:
        raise ValueError(f"real_nvfp4 needs a positive decoder_sparse_step, got {sparse_step}")
    return [i for i in range(num_layers) if i not in dense_layers and (i + 1) % sparse_step == 0]


def real_nvfp4_bf16_layer_indices(num_layers: int, num_layers_at_start: int, num_layers_at_end: int) -> list[int]:
    """Decoder layers kept in BF16 by the first/last carve-out."""

    if num_layers_at_start < 0 or num_layers_at_end < 0:
        raise ValueError(
            f"real_nvfp4 BF16 layer carve-out must be non-negative, got {num_layers_at_start}/{num_layers_at_end}"
        )
    if num_layers_at_start + num_layers_at_end >= num_layers:
        raise ValueError(
            f"real_nvfp4 BF16 layer carve-out {num_layers_at_start}/{num_layers_at_end} "
            f"leaves no quantized layer of {num_layers}"
        )
    return list(range(num_layers_at_start)) + list(range(num_layers - num_layers_at_end, num_layers))


def real_nvfp4_moe_layer_partition(
    hf_config: Any,
    *,
    num_layers_at_start_in_bf16: int = 0,
    num_layers_at_end_in_bf16: int = 0,
) -> tuple[list[int], list[int]]:
    """Return routed-expert layer indices as ``(quantized, bf16)``."""

    num_layers = int(_hf_get(hf_config, "num_hidden_layers") or 0)
    carved = set(real_nvfp4_bf16_layer_indices(num_layers, num_layers_at_start_in_bf16, num_layers_at_end_in_bf16))
    moe_layers = real_nvfp4_moe_layer_indices(hf_config)
    return [i for i in moe_layers if i not in carved], [i for i in moe_layers if i in carved]


def real_nvfp4_vllm_ignore_layers(
    hf_config: Any,
    *,
    num_layers_at_start_in_bf16: int = 0,
    num_layers_at_end_in_bf16: int = 0,
) -> list[str]:
    """Exact vLLM module names excluded from online NVFP4."""

    _, bf16_moe_layers = real_nvfp4_moe_layer_partition(
        hf_config,
        num_layers_at_start_in_bf16=num_layers_at_start_in_bf16,
        num_layers_at_end_in_bf16=num_layers_at_end_in_bf16,
    )
    return [f"model.layers.{index}.mlp.experts" for index in bf16_moe_layers]


def real_nvfp4_quant_recipe_config(num_layers: int, num_layers_at_start: int, num_layers_at_end: int) -> dict:
    """Megatron-Core per-module recipe: attention BF16, routed-expert MLP NVFP4.

    The result is passed to ``RecipeConfig.from_config_dict``. MCore takes the
    first matching matcher, so the carved-out layers precede the general MLP
    patterns. Layer indices in module paths are global only with PP=1, which
    :func:`validate_real_nvfp4_parallelism` enforces.
    """

    def glob(config: str, pattern: str) -> dict:
        return {"config": config, "type": "glob", "pattern": pattern, "enabled": True}

    matchers = {
        f"layer_{index}_bf16": glob("bf16", f"*.layers.{index}.*")
        for index in real_nvfp4_bf16_layer_indices(num_layers, num_layers_at_start, num_layers_at_end)
    }
    matchers["attn_qkv_bf16"] = glob("bf16", "*.linear_qkv")
    matchers["attn_proj_bf16"] = glob("bf16", "*.linear_proj")
    matchers["mlp_fc1_nvfp4"] = glob("nvfp4", "*.linear_fc1")
    matchers["mlp_fc2_nvfp4"] = glob("nvfp4", "*.linear_fc2")
    return {
        "configs": {
            "bf16": {"transformer_engine_config_type": "TEQuantizationParams", "training_recipe": {}},
            "nvfp4": {
                "transformer_engine_config_type": "TEQuantizationParams",
                "training_recipe": {"fp4_quantization_recipe": "nvfp4"},
            },
        },
        "matchers": matchers,
    }


def validate_real_nvfp4_model_contract(hf_config: Any) -> None:
    """Restrict the current recipe to all-MoE decoders without shared experts.

    The training recipe's fc1/fc2 patterns would also quantize dense MLPs,
    whereas rollout quantizes routed experts only, so mixed dense/MoE layouts
    stay unsupported until non-expert precision is configured on both sides.
    """

    num_experts = int(_hf_get(hf_config, "num_experts") or _hf_get(hf_config, "n_routed_experts") or 0)
    if num_experts <= 0:
        raise ValueError(
            "real_nvfp4 requires a routed-expert MoE model; this config declares no experts "
            "(looked for num_experts / n_routed_experts)"
        )
    num_layers = int(_hf_get(hf_config, "num_hidden_layers") or 0)
    if num_layers == 0 or len(real_nvfp4_moe_layer_indices(hf_config)) != num_layers:
        raise ValueError(
            "real_nvfp4 currently requires every decoder layer to be MoE; mixed dense/MoE layouts "
            "would quantize dense training MLPs that remain BF16 in rollout"
        )
    # Shared experts would be extra per-layer weights on the refit stream, so the
    # expert-weight attestation would be wrong rather than merely conservative.
    for key in ("n_shared_experts", "shared_expert_intermediate_size", "num_shared_experts"):
        if int(_hf_get(hf_config, key) or 0) > 0:
            raise ValueError(
                f"real_nvfp4 does not model shared experts yet ({key}="
                f"{_hf_get(hf_config, key)}); the refit expert-weight count would not match"
            )


def validate_real_nvfp4_parallelism(*, pipeline_size: int, virtual_pipeline_size: int | None) -> None:
    """Keep layer-index recipes and rank-local attestation within the PP=1 scope."""

    if pipeline_size != 1 or virtual_pipeline_size not in (None, 1):
        raise ValueError(
            "real_nvfp4 currently requires pipeline_model_parallel_size=1 and no virtual pipeline splitting; "
            "layer precision and local attestation use global layer indices, got pp="
            f"{pipeline_size} vpp={virtual_pipeline_size}"
        )


def validate_real_nvfp4_te_recipe(recipe: Any, *, backward_override: str) -> None:
    """Validate the effective Transformer Engine NVFP4 recipe.

    TE reads these fields from ``NVTE_*`` environment variables when the recipe
    is constructed, so this catches a worker started without them.
    """

    expected = {
        "backward_override": backward_override,
        "row_scaled_activation": True,
        "disable_rht": True,
        "disable_stochastic_rounding": True,
        "disable_2d_quantization": True,
        # Native nvfp4_per_token rollout uses standard NVFP4, not
        # TE's adaptive 4-over-6 representation. Keep training aligned.
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
    if mismatches:
        raise RuntimeError(f"real_nvfp4 Transformer Engine recipe drifted: {mismatches}")
