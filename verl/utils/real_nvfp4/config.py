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

from importlib.metadata import version
from typing import Any

REAL_NVFP4_TE_VERSION = "2.18.0"


def _hf_get(hf_config: Any, name: str, default: Any = None) -> Any:
    if isinstance(hf_config, dict):
        return hf_config.get(name, default)
    return getattr(hf_config, name, default)


def real_nvfp4_moe_layer_indices(hf_config: Any) -> list[int]:
    """Indices of the decoder layers that carry routed experts.

    Derived from the config rather than from the architecture name, following
    the HF placement rule shared by the Qwen MoE family: layer ``i`` is sparse
    when it is not listed in ``mlp_only_layers`` and ``(i + 1)`` is a multiple
    of ``decoder_sparse_step``. Models that omit those keys are all-MoE, which
    is the case this recipe was first validated on.
    """

    num_layers = int(_hf_get(hf_config, "num_hidden_layers") or 0)
    sparse_step = int(_hf_get(hf_config, "decoder_sparse_step") or 1)
    dense_layers = set(_hf_get(hf_config, "mlp_only_layers") or [])
    if sparse_step <= 0:
        raise ValueError(f"real_nvfp4 needs a positive decoder_sparse_step, got {sparse_step}")
    return [i for i in range(num_layers) if i not in dense_layers and (i + 1) % sparse_step == 0]


def real_nvfp4_rollout_layer_partition(
    hf_config: Any,
    *,
    num_layers_at_start_in_bf16: int = 0,
    num_layers_at_end_in_bf16: int = 0,
) -> tuple[list[int], list[int]]:
    """Return routed-expert layer indices as ``(quantized, bf16)`` for rollout.

    Training's first/last carve-out is defined over decoder layers, while only
    sparse decoder layers own a vLLM ``RoutedExperts`` container. Intersect the
    carve-out with the structural MoE layout so interleaved/dense-prefix models
    are counted correctly too.
    """

    num_layers = int(_hf_get(hf_config, "num_hidden_layers") or 0)
    start = int(num_layers_at_start_in_bf16)
    end = int(num_layers_at_end_in_bf16)
    if start < 0 or end < 0:
        raise ValueError(f"real_nvfp4 BF16 layer carve-out must be non-negative, got {start}/{end}")
    if start + end >= num_layers:
        raise ValueError(f"real_nvfp4 BF16 layer carve-out {start}/{end} leaves no quantized layer of {num_layers}")

    all_moe_layers = real_nvfp4_moe_layer_indices(hf_config)
    carved_decoder_layers = set(range(start)) | set(range(num_layers - end, num_layers))
    bf16_moe_layers = [index for index in all_moe_layers if index in carved_decoder_layers]
    quantized_moe_layers = [index for index in all_moe_layers if index not in carved_decoder_layers]
    if not quantized_moe_layers:
        raise ValueError("real_nvfp4 rollout carve-out leaves no routed-expert layer quantized")
    return quantized_moe_layers, bf16_moe_layers


def real_nvfp4_vllm_ignore_layers(
    hf_config: Any,
    *,
    num_layers_at_start_in_bf16: int = 0,
    num_layers_at_end_in_bf16: int = 0,
) -> list[str]:
    """Exact vLLM 0.26 module names excluded from online NVFP4."""

    _, bf16_moe_layers = real_nvfp4_rollout_layer_partition(
        hf_config,
        num_layers_at_start_in_bf16=num_layers_at_start_in_bf16,
        num_layers_at_end_in_bf16=num_layers_at_end_in_bf16,
    )
    return [f"model.layers.{index}.mlp.experts" for index in bf16_moe_layers]


def validate_real_nvfp4_model_contract(hf_config: Any) -> None:
    """Fail closed on layouts whose refit counts this recipe cannot predict.

    This used to be an allowlist of one architecture name
    (``Qwen3MoeForCausalLM`` with ``decoder_sparse_step=1`` and no
    ``mlp_only_layers``), which rejected every other MoE model even when the
    layout was one it handles perfectly well. The properties that actually
    matter are structural, so check those instead:

    * routed experts exist and their count is known, since the refit
      attestation is an exact expert-weight count;
    * at least one layer is sparse under the placement rule above;
    * there are no shared experts, because those add per-layer weights the
      expert-count arithmetic below does not model.

    Anything else is still refused rather than quietly mis-counted.
    """

    num_experts = int(_hf_get(hf_config, "num_experts") or _hf_get(hf_config, "n_routed_experts") or 0)
    if num_experts <= 0:
        raise ValueError(
            "real_nvfp4 requires a routed-expert MoE model; this config declares no experts "
            "(looked for num_experts / n_routed_experts)"
        )
    if not real_nvfp4_moe_layer_indices(hf_config):
        raise ValueError(
            "real_nvfp4 found no sparse decoder layer under decoder_sparse_step="
            f"{_hf_get(hf_config, 'decoder_sparse_step') or 1} with mlp_only_layers="
            f"{_hf_get(hf_config, 'mlp_only_layers') or []}"
        )
    # Shared experts would be extra per-layer weights on the refit stream, so the
    # expert-weight attestation would be wrong rather than merely conservative.
    for key in ("n_shared_experts", "shared_expert_intermediate_size", "num_shared_experts"):
        if int(_hf_get(hf_config, key) or 0) > 0:
            raise ValueError(
                f"real_nvfp4 does not model shared experts yet ({key}="
                f"{_hf_get(hf_config, key)}); the refit expert-weight count would not match"
            )


def real_nvfp4_expected_counts(hf_config: Any) -> tuple[int, int]:
    """Return exact ``(expert_weights, quantized_groups)`` for the refit."""

    validate_real_nvfp4_model_contract(hf_config)

    num_experts = int(_hf_get(hf_config, "num_experts") or _hf_get(hf_config, "n_routed_experts") or 0)
    moe_layers = len(real_nvfp4_moe_layer_indices(hf_config))
    if moe_layers <= 0 or num_experts <= 0:
        raise ValueError("real_nvfp4 model config requires positive MoE layer and expert counts")
    # Three projections per expert on the wire (gate/up/down); two quantized
    # groups per expert once gate/up are fused into w13.
    expert_weights = moe_layers * num_experts * 3
    quantized_groups = moe_layers * num_experts * 2
    return expert_weights, quantized_groups


def validate_real_nvfp4_te_recipe(recipe: Any, *, backward_override: str) -> None:
    """Fail closed unless TE implements the exact audited training semantics.

    TE 2.18 release packages, also used by NeMo RL PR #3566, contain both the
    GroupedLinear packed-wgrad lifetime fix (TE PR #3049) and the fix that
    preserves quantized/dequantized forward operands for a dequantized backward
    (TE PR #3141). Check all three distribution versions to reject mixed
    Python/core/extension installations, and still validate the recipe payload.
    """

    for package in ("transformer-engine", "transformer-engine-cu13", "transformer-engine-torch"):
        actual_version = version(package)
        if actual_version != REAL_NVFP4_TE_VERSION:
            raise RuntimeError(
                "real_nvfp4 requires matched Transformer Engine release packages "
                f"at {REAL_NVFP4_TE_VERSION}, got {package}=={actual_version}"
            )

    expected = {
        "backward_override": backward_override,
        "row_scaled_activation": True,
        "disable_rht": True,
        "disable_stochastic_rounding": True,
        "disable_2d_quantization": True,
        # vLLM 0.26's native nvfp4_per_token rollout uses standard NVFP4, not
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
    if mismatches:
        raise RuntimeError(f"real_nvfp4 Transformer Engine recipe drifted: {mismatches}")

    qparam_contract = {
        "fp4_quant_fwd_inp": {
            "random_hadamard_transform": False,
            "stochastic_rounding": False,
            "fp4_2d_quantization": False,
        },
        "fp4_quant_fwd_weight": {
            "random_hadamard_transform": False,
            "stochastic_rounding": False,
            "fp4_2d_quantization": False,
        },
        "fp4_quant_bwd_grad": {
            "random_hadamard_transform": False,
            "stochastic_rounding": False,
            "fp4_2d_quantization": False,
        },
    }
    qparam_mismatches = {}
    for group_name, group_expected in qparam_contract.items():
        group = getattr(recipe, group_name, None)
        for field_name, wanted in group_expected.items():
            actual = getattr(group, field_name, None)
            if actual != wanted:
                qparam_mismatches[f"{group_name}.{field_name}"] = {
                    "expected": wanted,
                    "actual": actual,
                }
    if qparam_mismatches:
        raise RuntimeError(f"real_nvfp4 Transformer Engine quantizer contract drifted: {qparam_mismatches}")
