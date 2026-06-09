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

from typing import Any

import torch


def assert_weight_sync_equal(
    expected_state_dict: dict[str, torch.Tensor],
    actual_state_dict: dict[str, torch.Tensor],
    num_hidden_layers: int,
    tie_word_embeddings: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    max_mismatches: int = 10,
) -> dict[str, Any]:
    """Compare source HF weights against backend-loaded weights.

    The comparison handles common vLLM fused weights for QKV and SwiGLU gate/up
    projections. It raises AssertionError with missing, unexpected, and
    mismatched samples when the loaded weights do not match the source weights.
    """

    expected_names = set(expected_state_dict)
    actual_names = set(actual_state_dict)
    compared_expected_names = set()
    compared_actual_names = set()
    mismatched = []

    def check_tensor(name: str, expected_weight: torch.Tensor, actual_weight: torch.Tensor) -> None:
        actual_weight = actual_weight.detach()
        if expected_weight.shape != actual_weight.shape:
            mismatched.append(
                f"{name}: shape expected={tuple(expected_weight.shape)} actual={tuple(actual_weight.shape)}"
            )
            return

        expected_weight = expected_weight.to(device=actual_weight.device, dtype=actual_weight.dtype)
        if not torch.allclose(expected_weight, actual_weight, rtol=rtol, atol=atol):
            max_abs_diff = (expected_weight - actual_weight).abs().max().item()
            mismatched.append(f"{name}: max_abs_diff={max_abs_diff}")

    for name, expected_weight in expected_state_dict.items():
        if name not in actual_state_dict:
            continue
        check_tensor(name, expected_weight, actual_state_dict[name])
        compared_expected_names.add(name)
        compared_actual_names.add(name)

    def check_fused_tensor(actual_name: str, expected_component_names: list[str]) -> None:
        if actual_name not in actual_state_dict:
            return
        if any(name not in expected_state_dict for name in expected_component_names):
            return
        expected_weight = torch.cat([expected_state_dict[name] for name in expected_component_names], dim=0)
        check_tensor(actual_name, expected_weight, actual_state_dict[actual_name])
        compared_expected_names.update(expected_component_names)
        compared_actual_names.add(actual_name)

    for layer_idx in range(num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"
        check_fused_tensor(
            actual_name=f"{prefix}.self_attn.qkv_proj.weight",
            expected_component_names=[
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ],
        )
        check_fused_tensor(
            actual_name=f"{prefix}.self_attn.qkv_proj.bias",
            expected_component_names=[
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.v_proj.bias",
            ],
        )
        check_fused_tensor(
            actual_name=f"{prefix}.mlp.gate_up_proj.weight",
            expected_component_names=[
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
            ],
        )

    if (
        tie_word_embeddings
        and "lm_head.weight" in expected_names
        and "lm_head.weight" not in actual_names
        and "model.embed_tokens.weight" in compared_expected_names
    ):
        compared_expected_names.add("lm_head.weight")

    missing = sorted(expected_names - compared_expected_names)
    unexpected = sorted(actual_names - compared_actual_names)

    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing[:max_mismatches]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:max_mismatches]}")
        if mismatched:
            details.append(f"mismatched={mismatched[:max_mismatches]}")
        raise AssertionError("; ".join(details))

    return {
        "checked": len(compared_expected_names),
        "missing": 0,
        "unexpected": 0,
        "mismatched": 0,
    }
