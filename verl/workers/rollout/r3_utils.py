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
"""Exact Kimi full-R3 rollout payload helpers.

Current Ascend execution supplies FP32 routing weights. The integer transport
contains IDs[K] followed by four raw byte lanes[K] per FP32 weight. Scheduler
uint8/uint16 storage therefore preserves every executed bit. Legacy two-lane
BF16 payloads remain readable; FSDP receives explicit IDs and FP32 weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

KIMI_FULL_R3_PACK_FACTOR = 3  # legacy BF16 transport
KIMI_FULL_R3_FP32_PACK_FACTOR = 5
KIMI_FULL_MODEL_ROUTE_SEMANTICS = "kimi_natural_rollout_full_model_rows_v1"


def is_kimi_full_r3_config(hf_config: Any) -> bool:
    """Whether the model uses Kimi K3's IDs+executed-weights R3 schema."""
    return getattr(hf_config, "model_type", None) in {"kimi_k3", "kimi_k31"}


def get_kimi_full_r3_topk(hf_config: Any) -> int:
    """Return Kimi K3's real (unpacked) experts-per-token width."""
    if not is_kimi_full_r3_config(hf_config):
        raise ValueError("full routing replay with gate weights is implemented only for Kimi K3")
    text_config = getattr(hf_config, "text_config", hf_config)
    topk = int(text_config.num_experts_per_token)
    if topk <= 0:
        raise ValueError("Kimi full R3 requires a positive num_experts_per_token in the HF config")
    return topk


def validate_kimi_full_model_routes(
    routes: torch.Tensor,
    route_weights: torch.Tensor,
    hf_config: Any,
) -> None:
    """Validate complete Kimi route rows once before agent-loop transport."""
    if routes.dim() != 3 or route_weights.shape != routes.shape:
        raise ValueError(
            f"Kimi full-model R3 shape mismatch: ids={tuple(routes.shape)}, weights={tuple(route_weights.shape)}"
        )
    if not route_weights.is_floating_point():
        raise TypeError(f"Kimi full-model route weights must be floating point, got {route_weights.dtype}")
    text_config = getattr(hf_config, "text_config", hf_config)
    expected_layers = int(getattr(text_config, "num_hidden_layers", 0) or 0)
    num_experts = int(getattr(text_config, "num_experts", 0) or 0)
    expected_topk = get_kimi_full_r3_topk(hf_config)
    first_dense = int(getattr(text_config, "first_k_dense_replace", 0) or 0)
    moe_frequency = int(getattr(text_config, "moe_layer_freq", 1) or 1)
    if expected_layers <= 0 or num_experts <= 0:
        raise ValueError("Kimi R3 requires positive num_hidden_layers and num_experts")
    if routes.shape[1:] != (expected_layers, expected_topk):
        raise ValueError(
            "Kimi route capture shape does not match the model config: "
            f"routes={tuple(routes.shape)}, layers={expected_layers}, "
            f"topk={expected_topk}"
        )
    if routes.shape[0] == 0:
        return

    moe_layers = [
        layer_idx for layer_idx in range(expected_layers) if layer_idx >= first_dense and layer_idx % moe_frequency == 0
    ]
    if not moe_layers:
        return
    moe_routes = routes[:, moe_layers, :].to(dtype=torch.long)
    moe_weights = route_weights[:, moe_layers, :].to(dtype=torch.float32)

    bad_range = (moe_routes < 0) | (moe_routes >= num_experts)
    if bool(bad_range.any()):
        bad = torch.nonzero(bad_range, as_tuple=False)[0]
        raise ValueError(
            "Kimi route capture contains an out-of-range expert ID: "
            f"row={int(bad[0])}, layer={moe_layers[int(bad[1])]}, "
            f"topk_slot={int(bad[2])}, "
            f"value={int(moe_routes[tuple(bad.tolist())])}, "
            f"num_experts={num_experts}"
        )

    ordered = torch.sort(moe_routes, dim=-1).values
    duplicate = ordered[..., 1:] == ordered[..., :-1]
    if bool(duplicate.any()):
        bad = torch.nonzero(duplicate, as_tuple=False)[0]
        row, layer = int(bad[0]), int(bad[1])
        raise ValueError(
            "Kimi route capture contains duplicate top-k expert IDs: "
            f"row={row}, layer={moe_layers[layer]}, "
            f"routes={moe_routes[row, layer].tolist()}"
        )

    finite = torch.isfinite(moe_weights)
    if not bool(finite.all()):
        bad = torch.nonzero(~finite, as_tuple=False)[0]
        raise ValueError(
            "Kimi full R3 route weight is NaN/Inf: "
            f"row={int(bad[0])}, layer={moe_layers[int(bad[1])]}, "
            f"topk_slot={int(bad[2])}"
        )
    if bool((moe_weights < 0).any()):
        bad = torch.nonzero(moe_weights < 0, as_tuple=False)[0]
        raise ValueError(
            "Kimi full R3 route weight is negative: "
            f"row={int(bad[0])}, layer={moe_layers[int(bad[1])]}, "
            f"topk_slot={int(bad[2])}, "
            f"value={float(moe_weights[tuple(bad.tolist())])}"
        )

    weight_sums = moe_weights.sum(dim=-1)
    if bool((weight_sums <= 0).any()):
        bad = torch.nonzero(weight_sums <= 0, as_tuple=False)[0]
        raise ValueError(
            f"Kimi full R3 route weights sum to zero: row={int(bad[0])}, layer={moe_layers[int(bad[1])]}"
        )
    if not bool(getattr(text_config, "moe_renormalize", False)):
        return

    expected_sum = float(getattr(text_config, "routed_scaling_factor", 1.0))
    tolerance = max(abs(expected_sum) * 0.02, 0.02)
    close = torch.isclose(
        weight_sums,
        torch.full_like(weight_sums, expected_sum),
        rtol=0.0,
        atol=tolerance,
    )
    if not bool(close.all()):
        bad = torch.nonzero(~close, as_tuple=False)[0]
        raise ValueError(
            "Kimi full R3 route weights violate the normalized sum contract: "
            f"row={int(bad[0])}, layer={moe_layers[int(bad[1])]}, "
            f"actual={float(weight_sums[tuple(bad.tolist())])}, "
            f"expected={expected_sum}, tolerance={tolerance}"
        )


def decode_kimi_full_r3_payload(
    packed: Any,
    *,
    expected_topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode exact FP32 (five-lane) or legacy BF16 (three-lane) weights."""
    payload = np.asarray(packed)
    if payload.ndim != 3:
        raise ValueError(f"Kimi full R3 packed payload must be [model_rows, layers, 5*topk] (legacy BF16: 3*topk), got {payload.shape!r}")
    if expected_topk <= 0:
        raise ValueError(f"expected_topk must be positive, got {expected_topk}")
    width = payload.shape[-1]
    if width not in (3 * expected_topk, 5 * expected_topk):
        raise ValueError(f"Kimi full R3 payload width proves that rollout weight capture is not active or is incompatible: got={width}, expected={5 * expected_topk} (legacy BF16={3 * expected_topk})")
    if not np.issubdtype(payload.dtype, np.integer):
        raise TypeError(f"Kimi full R3 payload must be integer, got {payload.dtype}")
    expert_ids = np.array(payload[..., :expected_topk], dtype=np.int32, copy=True)
    lanes = payload[..., expected_topk:]
    if lanes.size and (int(lanes.min()) < 0 or int(lanes.max()) > 255):
        raise ValueError("Kimi full R3 weight byte lanes are out of range")
    bits = np.zeros(payload.shape[:-1] + (expected_topk,), dtype=np.uint32)
    count = width // expected_topk - 1
    for i in range(count):
        bits |= lanes[..., i * expected_topk:(i + 1) * expected_topk].astype(np.uint32) << (8 * i)
    if count == 2:
        bits <<= 16
    weights = np.array(bits.view(np.float32), copy=True)
    if not np.isfinite(weights).all():
        raise ValueError("Kimi full R3 decoded router weights contain NaN or Inf")
    if (weights < 0).any():
        raise ValueError("Kimi full R3 decoded router weights contain negative values")
    return expert_ids, weights


def pack_kimi_full_r3_for_megatron(ids, weights):
    """Convert shared split routes into Megatron's exact BF16 byte payload.

    Keep expanded model rows and sequence boundaries; raw token lengths do not
    describe multimodal routes. Only the Megatron engine uses this wire layout.
    """
    if weights is None:
        raise ValueError("Kimi full R3 requires both expert IDs and weights")
    if not ids.is_nested or not weights.is_nested:
        raise TypeError("Kimi full R3 requires jagged IDs and weights")
    if not torch.equal(ids.offsets(), weights.offsets()):
        raise ValueError("Kimi full R3 IDs and weights have different sequence boundaries")
    id_values, weight_values = ids.values(), weights.values()
    if id_values.shape != weight_values.shape:
        raise ValueError("Kimi full R3 IDs and weights have different shapes")
    bf16_weights = weight_values.to(torch.bfloat16)
    if not torch.equal(bf16_weights.float(), weight_values.float()):
        raise ValueError("Kimi full R3 weights must be finite exact BF16 values")
    bits = bf16_weights.contiguous().view(torch.uint8).reshape(*weight_values.shape, 2)
    packed = torch.cat((id_values, bits[..., 0], bits[..., 1]), dim=-1)
    return torch.nested.nested_tensor_from_jagged(packed, offsets=ids.offsets())
