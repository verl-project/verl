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

"""Utilities shared by agent-loop and policy-training trajectory paths."""

from __future__ import annotations

import math
from typing import Any

import torch

LOSS_WEIGHT_KEY = "loss_weight"
"""Canonical ``AgentLoopOutput`` field used to weight policy-gradient samples."""


def validate_loss_weight(value: Any, *, source: str = LOSS_WEIGHT_KEY) -> float:
    """Validate and convert a trajectory loss weight to a Python float.

    Loss weights are metadata and must not participate in autograd. Requiring a
    strictly positive, finite value prevents malformed agent-loop output from
    silently inverting or disabling a policy-gradient sample.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{source} must be a scalar, got shape {tuple(value.shape)}")
        value = value.item()

    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be a finite positive number, got {value!r}") from exc

    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(f"{source} must be a finite positive number, got {weight!r}")
    return weight


def resolve_agent_loop_loss_weight(output: Any) -> float:
    """Resolve the canonical loss weight from an agent-loop output.

    ``loss_weight`` is a first-class field on ``AgentLoopOutput``. Reading the
    same key from ``extra_fields`` keeps custom outputs that were written against
    the extensible metadata interface source-compatible.

    A missing weight resolves to neutral ``1.0``. Callers must not substitute a
    different implicit default (such as ``1 / N`` for an N-segment trajectory):
    the scale that keeps a split trajectory equivalent to an unsplit one depends
    on ``actor.loss_agg_mode``, so only the agent loop -- which knows the training
    configuration -- may choose a non-neutral weight.
    """
    weight = getattr(output, LOSS_WEIGHT_KEY, None)
    if weight is None:
        extra_fields = getattr(output, "extra_fields", None)
        if isinstance(extra_fields, dict):
            weight = extra_fields.get(LOSS_WEIGHT_KEY)
    if weight is None:
        weight = 1.0
    return validate_loss_weight(weight, source=LOSS_WEIGHT_KEY)


def validate_loss_weights(weights: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Validate and prepare per-sample loss weights.

    loss_weight is an explicit policy-gradient multiplier. The values are
    intentionally not renormalized here: the absolute scale chosen by the
    producer must reach the configured loss aggregation unchanged.

    When valid_mask is supplied, invalid samples are zeroed rather than set to
    one. These are synthetic rows appended to reach a data-parallel divisor;
    their response_mask is all-zero, so token-normalized aggregation modes drop
    them either way, but an explicit 0.0 also keeps them out of any aggregation
    that does not re-apply the mask.
    """
    if not isinstance(weights, torch.Tensor):
        raise TypeError(f"weights must be a torch.Tensor, got {type(weights)}")
    if weights.ndim != 1:
        raise ValueError(f"weights must have shape [batch_size], got {tuple(weights.shape)}")
    if weights.numel() == 0:
        return weights.to(dtype=torch.float32)

    if valid_mask is not None:
        if not isinstance(valid_mask, torch.Tensor):
            raise TypeError(f"valid_mask must be a torch.Tensor, got {type(valid_mask)}")
        if valid_mask.ndim != 1 or valid_mask.shape != weights.shape:
            raise ValueError(
                "valid_mask must have shape [batch_size] matching weights, "
                f"got {tuple(valid_mask.shape)} for weights {tuple(weights.shape)}"
            )
        valid_mask = valid_mask.to(device=weights.device, dtype=torch.bool)

    weights = weights.detach().to(dtype=torch.float32)
    if not torch.isfinite(weights).all():
        raise ValueError("loss weights must contain only finite values")
    if (weights <= 0).any():
        raise ValueError("loss weights must contain only positive values")

    prepared_weights = weights
    if valid_mask is not None:
        prepared_weights = prepared_weights.masked_fill(~valid_mask, 0.0)
    return prepared_weights


def apply_loss_weight_to_advantages(advantages: torch.Tensor, loss_weight: torch.Tensor | None) -> torch.Tensor:
    """Apply per-sample policy-gradient weights to an advantage tensor.

    Args:
        advantages: Advantage tensor of shape ``[batch_size, response_length]``
            (optionally with trailing dimensions).
        loss_weight: Optional tensor with shape ``[batch_size]`` or
            ``[batch_size, 1]``.

    Returns:
        The advantages multiplied by detached, validated sample weights.

    Raises:
        TypeError: If ``loss_weight`` is not a tensor.
        ValueError: If the advantage rank, weight shape, or values are invalid.

    This helper is shared by PPO and policy-gradient distillation so every
    policy-gradient entry point applies the same trajectory-weight contract.
    """
    if loss_weight is None:
        return advantages
    if not isinstance(loss_weight, torch.Tensor):
        raise TypeError(f"{LOSS_WEIGHT_KEY} must be a torch.Tensor, got {type(loss_weight)}")
    if advantages.ndim < 2:
        # A 1-D [batch_size] advantage would broadcast against the [batch_size, 1]
        # weight into a bogus [batch_size, batch_size] tensor instead of failing,
        # so require the token dimension explicitly.
        raise ValueError(f"advantages must have shape [batch_size, response_length], got {tuple(advantages.shape)}")

    if loss_weight.ndim == 1:
        flat_loss_weight = loss_weight
    elif loss_weight.ndim == 2 and loss_weight.shape[1] == 1:
        flat_loss_weight = loss_weight.squeeze(-1)
    else:
        raise ValueError(
            f"{LOSS_WEIGHT_KEY} must have shape [batch_size] or [batch_size, 1], got {tuple(loss_weight.shape)}"
        )

    if flat_loss_weight.shape[0] != advantages.shape[0]:
        raise ValueError(
            f"{LOSS_WEIGHT_KEY} batch dimension {flat_loss_weight.shape[0]} does not match "
            f"advantages batch dimension {advantages.shape[0]}"
        )

    # Values were already range-checked by validate_loss_weights() when the batch was
    # assembled, which also zeroes padding rows. Only re-check for finiteness and
    # non-negativity here: requiring strict positivity would reject those zeroed rows.
    flat_loss_weight = flat_loss_weight.detach()
    if not torch.isfinite(flat_loss_weight).all():
        raise ValueError(f"{LOSS_WEIGHT_KEY} must contain only finite values")
    if (flat_loss_weight < 0).any():
        raise ValueError(f"{LOSS_WEIGHT_KEY} must contain only non-negative values")

    return advantages * flat_loss_weight.to(device=advantages.device, dtype=advantages.dtype).unsqueeze(-1)
