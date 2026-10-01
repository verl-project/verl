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
    different implicit default (such as ``1 / N`` for an N-row trajectory): the
    right weight depends both on the objective (session-equal ``1 / N`` vs
    partition-preserving ``T_j / mean(T)``) and on ``actor.loss_agg_mode``, so
    only the agent loop -- which knows the training configuration -- may choose
    a non-neutral weight.
    """
    weight = getattr(output, LOSS_WEIGHT_KEY, None)
    if weight is None:
        extra_fields = getattr(output, "extra_fields", None)
        if isinstance(extra_fields, dict):
            weight = extra_fields.get(LOSS_WEIGHT_KEY)
    if weight is None:
        weight = 1.0
    return validate_loss_weight(weight, source=LOSS_WEIGHT_KEY)


def final_row_per_session(keys: list[str]) -> dict[str, int]:
    """Map each logical session to the position of its final stored row.

    V1 agent-loop keys are ``{uid}_{session}_{index}``; the highest ``index`` is
    the row the GRPO advantage is computed from and the one that carries the
    trajectory's reward. Keys without that structure are their own session.
    """
    final: dict[str, tuple[int, int]] = {}
    for pos, key in enumerate(keys):
        parts = key.rsplit("_", 2)
        if len(parts) == 3:
            session, index = "_".join(parts[:2]), int(parts[2])
        else:
            session, index = key, 0
        if session not in final or index > final[session][0]:
            final[session] = (index, pos)
    return {session: pos for session, (_, pos) in final.items()}


def validate_loss_weights(weights: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Validate and prepare per-sample loss weights.

    loss_weight is an explicit per-sample loss multiplier. This function only
    validates and masks; it does not rescale. The single mean-1.0 rescaling
    over the global batch happens afterwards in the trainer via
    :func:`normalize_loss_weight_global`, so that ratios between samples reach
    the loss aggregation unchanged while the total magnitude is conserved.

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


def apply_loss_weight_to_loss_mat(loss_mat: torch.Tensor, loss_weight: torch.Tensor | None) -> torch.Tensor:
    """Broadcast per-sample weights onto a ``(batch_size, response_length)`` matrix.

    ``agg_loss`` has no per-sample weight argument, but every one of its
    aggregation modes is linear in ``loss_mat``, so pre-multiplying row ``i`` by
    ``w_i`` is identical to scaling that sample's aggregated contribution by
    ``w_i``. This is how the entropy and KL terms receive the same trajectory
    weight as the policy-gradient term.

    ``None`` means neutral weights and returns the input unchanged, so the
    unweighted path stays bit-identical.
    """
    if loss_weight is None:
        return loss_mat
    if loss_weight.ndim != 1:
        raise ValueError(f"{LOSS_WEIGHT_KEY} must have shape [batch_size], got {tuple(loss_weight.shape)}")
    if loss_mat.ndim != 2:
        raise ValueError(f"loss_mat must have shape [batch_size, response_length], got {tuple(loss_mat.shape)}")
    if loss_weight.shape[0] != loss_mat.shape[0]:
        raise ValueError(
            f"{LOSS_WEIGHT_KEY} batch dimension {loss_weight.shape[0]} does not match "
            f"loss_mat batch dimension {loss_mat.shape[0]}"
        )
    return loss_mat * loss_weight.to(device=loss_mat.device, dtype=loss_mat.dtype).unsqueeze(-1)


def normalize_loss_weight_global(
    loss_weight: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rescale weights to mean 1.0 over the GLOBAL batch. Call this exactly once.

    WHY NORMALIZE (this is the whole point of this function)
    -------------------------------------------------------
    Multiplying advantages by ``w_i`` fixes the RELATIVE share each trajectory
    contributes, but it also shrinks the ABSOLUTE loss, because every
    ``loss_agg_mode`` divides by an UNWEIGHTED denominator:

        token-mean = sum(adv * w * mask) / sum(mask)
                                            ^^^^^^^^ no w here

    MEASURED on the batch shape this feature exists for -- one 10-segment
    trajectory (w=0.1 each) plus one 1-segment trajectory (w=1.0):

        relative share of the long trajectory   90.9%  ->  50.0%   (correct)
        total loss magnitude                     1.000 ->   0.182  (5.5x smaller)

    A 5.5x smaller loss at a fixed learning rate is a silent 5.5x learning-rate
    cut, and it DRIFTS: the factor is sum(w)/N, which changes with whatever mix
    of long and short trajectories a step happens to draw. Two steps with the
    same lr then take different effective step sizes.

    Trainers that weight the per-sample loss *after* aggregation (e.g. ScaleCUA's
    ``loss * loss_weight[i] / batch_weight``) avoid this by dividing by the
    weighted denominator. Normalizing to ``mean_rows(w) == 1`` here is the
    equivalent for verl's aggregation-inside-the-loss-function structure: it
    preserves every ratio ``w_i / w_j`` and removes the drifting ``mean(w)``
    factor in every mode.

    WHAT IT GUARANTEES, PRECISELY. For aggregation modes that normalize by row
    count (``seq-mean-token-mean``) the total magnitude -- and therefore the
    effective learning rate -- is unchanged exactly. For token-normalized modes
    (``token-mean``) the effective scale is the *token-weighted* mean of ``w``,
    which equals 1 only when ``w`` is uncorrelated with row length: a batch of
    {100 tokens, w=2} + {1000 tokens, w=0.5} has mean_rows(w) = 1 but scales the
    token-mean loss by 0.51. The rescaling is packing- and mix-invariant in every
    mode; the magnitude promise is limited to row-normalized aggregation, which
    is also the only mode in which a per-row weight is needed to make splitting
    neutral in the first place.

    MUST BE CALLED ON THE GLOBAL BATCH, NOT A MICRO-BATCH. Rescaling within a
    micro-batch would silently erase the reweighting: a micro-batch holding only
    same-weight samples normalizes every one of them to 1.0. MEASURED on 10
    deep segments plus 1 shallow trajectory, where the deep one must land at
    50% of the gradient:

        all deep in one micro-batch  ->  90.9%   (weight fully erased)
        interleaved                  ->  83.5%

    i.e. the objective would depend on the batcher's packing. Doing it once over
    the global batch is packing-invariant.

    ``valid_mask`` marks real samples; padding rows carry weight 0 and are
    excluded from the mean so they neither dilute it nor get resurrected.

    Args:
        loss_weight: ``[batch_size]`` non-negative weights, already range
            checked by :func:`validate_loss_weights`.
        valid_mask: optional ``[batch_size]`` bool mask of real samples.

    Returns:
        ``[batch_size]`` weights whose mean over the valid rows is 1.0. Returned
        unchanged when already neutral or when no valid row carries weight.
    """
    if not isinstance(loss_weight, torch.Tensor):
        raise TypeError(f"{LOSS_WEIGHT_KEY} must be a torch.Tensor, got {type(loss_weight)}")
    if loss_weight.ndim != 1:
        raise ValueError(f"{LOSS_WEIGHT_KEY} must have shape [batch_size], got {tuple(loss_weight.shape)}")

    weights = loss_weight.detach()
    if not torch.isfinite(weights).all():
        raise ValueError(f"{LOSS_WEIGHT_KEY} must contain only finite values")
    if (weights < 0).any():
        raise ValueError(f"{LOSS_WEIGHT_KEY} must contain only non-negative values")

    if valid_mask is None:
        valid_mask = torch.ones_like(weights, dtype=torch.bool)
    else:
        if valid_mask.shape != weights.shape:
            raise ValueError(f"valid_mask must have shape {tuple(weights.shape)}, got {tuple(valid_mask.shape)}")
        valid_mask = valid_mask.to(device=weights.device, dtype=torch.bool)

    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        return weights

    total = weights[valid_mask].sum()
    if total <= 0:
        # Every valid row has weight 0. Any scale is arbitrary and dividing
        # would be a NaN; leave the batch untouched.
        return weights

    scale = n_valid / total
    # Bit-identical no-op for the common neutral case, so single-output agent
    # loops and existing runs are unaffected.
    if torch.isclose(scale, torch.ones_like(scale)):
        return weights

    return weights * scale
