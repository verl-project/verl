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

REAL_NVFP4_TE_COMMIT = "e7c550c5f80636cf841a8204b1d6f85a5f3f28b7"
REAL_NVFP4_TE_VERSION = "2.18.0+e7c550c5"


def validate_real_nvfp4_model_contract(hf_config: Any) -> None:
    """Limit real W4A4 to the Qwen3 all-MoE layout validated end to end."""

    architectures = getattr(hf_config, "architectures", None)
    if architectures is None and isinstance(hf_config, dict):
        architectures = hf_config.get("architectures")
    architectures = architectures or []

    def _get(name: str):
        if isinstance(hf_config, dict):
            return hf_config.get(name)
        return getattr(hf_config, name, None)

    if (
        "Qwen3MoeForCausalLM" not in architectures
        or _get("decoder_sparse_step") != 1
        or (_get("mlp_only_layers") or [])
    ):
        raise ValueError(
            "real_nvfp4 currently supports only the validated Qwen3 all-MoE "
            "layout (Qwen3MoeForCausalLM, decoder_sparse_step=1, no "
            "mlp_only_layers)"
        )


def real_nvfp4_expected_counts(hf_config: Any) -> tuple[int, int]:
    """Return exact ``(expert_weights, quantized_groups)`` for all-MLP refit."""

    validate_real_nvfp4_model_contract(hf_config)

    def _get(name: str):
        if isinstance(hf_config, dict):
            return hf_config.get(name)
        return getattr(hf_config, name, None)

    num_layers = int(_get("num_hidden_layers") or 0)
    num_experts = int(_get("num_experts") or 0)
    if num_layers <= 0 or num_experts <= 0:
        raise ValueError("real_nvfp4 model config requires positive num_hidden_layers and num_experts")
    expert_weights = num_layers * num_experts * 3
    quantized_groups = num_layers * num_experts * 2
    return expert_weights, quantized_groups


def validate_real_nvfp4_te_recipe(recipe: Any, *, backward_override: str) -> None:
    """Fail closed unless TE implements the exact audited training semantics.

    TE ``e7c550c5`` is the current NeMo RL PR #3566 pin. It contains both the
    GroupedLinear packed-wgrad lifetime fix (TE PR #3049) and the fix that
    preserves quantized/dequantized forward operands for a dequantized backward
    (TE PR #3141). A release-only version check is insufficient because the
    latter changes gradients without necessarily crashing.
    """

    actual_version = version("transformer-engine")
    if actual_version != REAL_NVFP4_TE_VERSION:
        raise RuntimeError(
            "real_nvfp4 requires the audited Transformer Engine source pin "
            f"{REAL_NVFP4_TE_COMMIT} ({REAL_NVFP4_TE_VERSION}), got {actual_version}"
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
