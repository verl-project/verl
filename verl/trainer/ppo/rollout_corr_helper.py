# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""
Rollout Correction Helper Module

This module provides a complete pipeline to address **off-policy issues** in RL training,
including:
1. Policy mismatch between rollout and training implementations (e.g., vLLM BFloat16 vs FSDP FP32)
2. Model update staleness (training on trajectories from older checkpoints)
3. General distribution shifts between data collection and training

Its core capabilities include computing importance sampling (IS) weights,
filtering outlier samples via rejection sampling (RS), and
tracking metrics to diagnose and correct off-policy issues.

## Core Capabilities
1. **Multi-Granularity Aggregation**:
   - Importance Sampling (IS):
        Token-level
        Sequence-level
   - Rejection Sampling (RS):
        Divergence-based filters (token_k*, seq_sum_k*, seq_mean_k*, seq_max_k*)
2. **Memory-Efficient Design**:
   - Log-space computations to avoid numerical overflow/underflow.
   - Fixed safety bounds (exp(±20)) for stable exponentiation.
   - Metrics calculated without large intermediate tensors (prevents CUDA OOM).
3. **Comprehensive Metrics Tracking**:
   - IS/RS statistics (mean/max/min, effective sample size ESS, rejection rate).
   - Off-policy diagnostics (KL divergence, perplexity PPL, log PPL difference, χ² divergence).
   - Sequence-level breakdowns (deviation from ideal weights, outlier fraction).

## Key Interfaces & Usage
- compute_rollout_correction_and_rejection_mask(): compute IS weights + rejection mask.
- compute_rollout_correction_weights(): only compute truncated IS weights (for variance
  reduction, no outlier rejection).
- compute_rollout_rejection_mask(): only filter outliers (for sample cleaning, no IS weight
  computation).
- compute_offpolicy_metrics(): called by core functions to calculate off-policy diagnostics
  (KL/PPL/χ²) — no direct external calls needed.

### Integration Notes
- Used in `ray_trainer.py` via `compute_rollout_correction_and_add_to_batch()` (batch training pipeline).
- Used in `dp_actor.py` for distributed worker computations (distributed training scenarios).
- All functions support batch inputs and valid token masking (via `response_mask`).


## References
- "When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch": https://richardli.xyz/rl-collapse
- Off-policy RL (theoretical basis for IS): https://fengyao.notion.site/off-policy-rl
"""

import math
from typing import Any, Optional

import torch

import verl.utils.torch_functional as verl_F
from verl.protocol import DataProto
from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.workers.config.actor import PolicyLossConfig

# Safety bound to prevent numerical overflow/underflow when exponentiating
# exp(20) ≈ 485 million (upper limit for stable weights), exp(-20) ≈ 2e-9 (lower limit)
SAFETY_BOUND = 20.0

SUPPORTED_ROLLOUT_RS_OPTIONS: set[str] = {
    "token_k1",
    "token_k2",
    "token_k3",
    "seq_sum_k1",
    "seq_sum_k2",
    "seq_sum_k3",
    "seq_mean_k1",
    "seq_mean_k2",
    "seq_mean_k3",
    "seq_max_k2",
    "seq_max_k3",
}
TOKEN_LEVEL_ROLLOUT_RS_OPTIONS: set[str] = {"token_k1", "token_k2", "token_k3"}


def _parse_rollout_rs_thresholds(
    options: list[str], threshold_spec: Optional[str | float]
) -> dict[str, dict[str, Optional[float]]]:
    if threshold_spec is None:
        raise ValueError("rollout_rs_threshold must be provided for rejection sampling.")

    if isinstance(threshold_spec, int | float):
        raw_specs: list[str] = [str(threshold_spec)]
    elif isinstance(threshold_spec, str):
        raw_specs = [part.strip() for part in threshold_spec.split(",") if part.strip()]
    else:
        raise TypeError("rollout_rs_threshold must be a string or numeric value specifying per-option thresholds.")

    if not raw_specs:
        raise ValueError("rollout_rs_threshold must contain at least one threshold value.")

    if len(raw_specs) not in (1, len(options)):
        raise ValueError(
            f"rollout_rs_threshold expects either one threshold shared by all options or exactly "
            f"{len(options)} thresholds to match the provided rollout_rs options."
        )

    if len(raw_specs) == 1 and len(options) > 1:
        raw_specs = raw_specs * len(options)

    thresholds: dict[str, dict[str, Optional[float]]] = {}
    for option, spec in zip(options, raw_specs, strict=False):
        if option.endswith("k1"):
            if "_" in spec:
                lower_str, upper_str = spec.split("_", 1)
            else:
                upper_str = spec
                lower_str = str(1.0 / float(upper_str))
            try:
                lower = float(lower_str)
                upper = float(upper_str)
            except ValueError as exc:
                raise ValueError(f"Invalid numeric threshold '{spec}' for option '{option}'.") from exc
            if lower <= 0 or upper <= 0:
                raise ValueError(f"Thresholds for option '{option}' must be positive, got {spec}.")
            thresholds[option] = {
                "lower": lower,
                "upper": upper,
            }
        else:
            if "_" in spec:
                raise ValueError(
                    f"rollout_rs_threshold for option '{option}' must provide a single upper bound "
                    f"without '_'. Received '{spec}'."
                )
            try:
                upper = float(spec)
            except ValueError as exc:
                raise ValueError(f"Invalid numeric threshold '{spec}' for option '{option}'.") from exc
            if upper <= 0:
                raise ValueError(f"Threshold for option '{option}' must be positive, got {spec}.")
            thresholds[option] = {
                "lower": None,
                "upper": upper,
            }
    return thresholds


def compute_rollout_rejection_mask(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_rs: str = "token_k1",
    rollout_rs_threshold: Optional[str | float] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute hard trust region mask using divergence estimators.

    This function enforces a hard trust region constraint by masking tokens/sequences
    where the estimated divergence (between training and rollout policies) exceeds
    a threshold. Unlike PPO's soft clipping, this provides a hard boundary.

    Multiple rejection criteria can be supplied via a comma separated `rollout_rs` string.
    All requested options must pass for a token/sequence to remain valid.

    Supported KL divergence-based modes (ideal = 0.0 unless noted):
    - "token_k{1,2,3}": Token-level divergences.
    - "seq_sum_k{1,2,3}": Sum of token divergences per sequence.
    - "seq_mean_k{1,2,3}": Mean of token divergences per sequence.
    - "seq_max_k{2,3}": Maximum token divergence per sequence.

    Args:
        log_ratio: Log ratio of training policy probability to rollout policy probability,
            shape (batch_size, seq_length).
        response_mask: Binary mask for valid tokens (1=valid, 0=padding),
            shape (batch_size, seq_length).
        rollout_rs: Comma separated rejection sampling options (e.g. "token_k1,seq_sum_k3").
        rollout_rs_threshold: Threshold specification string (required). Provide one entry per
            rollout_rs option separated by commas. Each entry must be a positive number.
            For K1-style options (``*k1``), specify ``lower_upper`` (e.g. ``"0.1_1.2"``)
            to denote lower/upper ratio bounds; other options accept a single upper bound.

    Returns:
        Tuple containing:
            modified_response_mask: Response mask with trust region violations masked (0=rejected),
                shape (batch_size, seq_length).
            metrics: Dictionary of trust region metrics (all scalars).
    """
    if rollout_rs is None or not isinstance(rollout_rs, str):
        raise ValueError("rollout_rs must be a non-empty string (comma separated for multiple options).")
    if rollout_rs_threshold is None:
        raise ValueError("rollout_rs_threshold must be provided for rejection sampling.")

    if log_ratio.shape[0] == 0:
        return response_mask, {}

    # rollout_rs supports chained criteria via comma separation (e.g. "token_k1,seq_mean_k3").
    #            Every listed option must pass; combined_mask aggregates them via logical AND.
    option_modes = [opt.strip() for opt in rollout_rs.split(",") if opt.strip()]
    if not option_modes:
        raise ValueError("rollout_rs must contain at least one valid option.")

    normalized_options: list[str] = []
    seen: set[str] = set()
    for opt in option_modes:
        if opt not in SUPPORTED_ROLLOUT_RS_OPTIONS:
            raise ValueError(
                f"Invalid rollout_rs option: {opt}. Must be one of {sorted(SUPPORTED_ROLLOUT_RS_OPTIONS)}."
            )
        if opt not in seen:
            normalized_options.append(opt)
            seen.add(opt)

    threshold_specs = _parse_rollout_rs_thresholds(normalized_options, rollout_rs_threshold)

    log_ratio_safe: torch.Tensor = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
    token_k1: torch.Tensor = -log_ratio_safe
    token_k2: torch.Tensor = 0.5 * log_ratio_safe**2
    token_k3: torch.Tensor = torch.exp(log_ratio_safe) - 1.0 - log_ratio_safe

    response_mask_bool: torch.Tensor = response_mask.bool()
    seq_valid_mask: torch.Tensor = response_mask.sum(dim=-1) > 0
    # combined_mask accumulates per-option passes; any failure flips tokens to 0.
    combined_mask: torch.Tensor = torch.ones_like(response_mask, dtype=log_ratio.dtype)
    metrics: dict[str, float] = {}

    def _sequence_sum(values: torch.Tensor) -> torch.Tensor:
        return verl_F.masked_sum(values, response_mask, axis=-1)

    def _sequence_mean(values: torch.Tensor) -> torch.Tensor:
        return verl_F.masked_mean(values, response_mask, axis=-1)

    def _sequence_max(values: torch.Tensor) -> torch.Tensor:
        mask_bool = response_mask.bool()
        neg_inf = torch.tensor(float("-inf"), device=values.device, dtype=values.dtype)
        masked_values = values.masked_fill(~mask_bool, neg_inf)
        max_values = masked_values.max(dim=-1).values
        return torch.where(max_values == neg_inf, torch.zeros_like(max_values), max_values)

    for option_name in normalized_options:
        thresholds_info = threshold_specs[option_name]
        is_k1_option = option_name.endswith("k1")
        upper_value = thresholds_info["upper"]
        lower_value = thresholds_info["lower"]
        apply_lower_threshold = is_k1_option
        lower_log: Optional[float] = None
        upper_log: Optional[float] = None

        if is_k1_option:
            if lower_value is None or upper_value is None:
                raise ValueError(
                    f"rollout_rs_threshold for option '{option_name}' must specify both lower and upper bounds."
                )
            lower_log = math.log(lower_value)
            upper_log = math.log(upper_value)
        else:
            if upper_value is None:
                raise ValueError(f"rollout_rs_threshold for option '{option_name}' must specify an upper bound.")

        level = "sequence" if option_name not in TOKEN_LEVEL_ROLLOUT_RS_OPTIONS else "token"

        per_token_stat: torch.Tensor
        per_sequence_stat: Optional[torch.Tensor] = None
        token_keep_bool: torch.Tensor

        if option_name == "token_k1":
            if lower_log is None:
                raise ValueError("Threshold specification for token_k1 must include lower and upper bounds.")
            per_token_stat = token_k1
            token_keep_bool = (per_token_stat >= lower_log) & (per_token_stat <= upper_log)
        elif option_name == "token_k2":
            per_token_stat = token_k2
            token_keep_bool = per_token_stat <= upper_value
        elif option_name == "token_k3":
            per_token_stat = token_k3
            token_keep_bool = per_token_stat <= upper_value
        elif option_name.startswith("seq_sum"):
            if option_name.endswith("k1"):
                if lower_log is None:
                    raise ValueError(
                        f"Threshold specification for option '{option_name}' must include lower and upper bounds."
                    )
                seq_stat = _sequence_sum(token_k1)
                seq_keep_bool_direct = (seq_stat >= lower_log) & (seq_stat <= upper_log)
            elif option_name.endswith("k2"):
                seq_stat = _sequence_sum(token_k2)
                seq_keep_bool_direct = seq_stat <= upper_value
            elif option_name.endswith("k3"):
                seq_stat = _sequence_sum(token_k3)
                seq_keep_bool_direct = seq_stat <= upper_value
            else:
                raise ValueError(f"Unsupported rollout_rs option: {option_name}.")
            per_sequence_stat = seq_stat
            token_keep_bool = seq_keep_bool_direct.unsqueeze(-1).expand_as(response_mask_bool)
            per_token_stat = seq_stat.unsqueeze(-1).expand_as(response_mask)
        elif option_name.startswith("seq_mean"):
            if option_name.endswith("k1"):
                if lower_log is None:
                    raise ValueError(
                        f"Threshold specification for option '{option_name}' must include lower and upper bounds."
                    )
                seq_stat = _sequence_mean(token_k1)
                seq_keep_bool_direct = (seq_stat >= lower_log) & (seq_stat <= upper_log)
            elif option_name.endswith("k2"):
                seq_stat = _sequence_mean(token_k2)
                seq_keep_bool_direct = seq_stat <= upper_value
            elif option_name.endswith("k3"):
                seq_stat = _sequence_mean(token_k3)
                seq_keep_bool_direct = seq_stat <= upper_value
            else:
                raise ValueError(f"Unsupported rollout_rs option: {option_name}.")
            per_sequence_stat = seq_stat
            token_keep_bool = seq_keep_bool_direct.unsqueeze(-1).expand_as(response_mask_bool)
            per_token_stat = seq_stat.unsqueeze(-1).expand_as(response_mask)
        elif option_name.startswith("seq_max"):
            if option_name.endswith("k2"):
                seq_stat = _sequence_max(token_k2)
                seq_keep_bool_direct = seq_stat <= upper_value
            elif option_name.endswith("k3"):
                seq_stat = _sequence_max(token_k3)
                seq_keep_bool_direct = seq_stat <= upper_value
            else:
                raise ValueError(f"Unsupported rollout_rs option: {option_name}.")
            per_sequence_stat = seq_stat
            token_keep_bool = seq_keep_bool_direct.unsqueeze(-1).expand_as(response_mask_bool)
            per_token_stat = seq_stat.unsqueeze(-1).expand_as(response_mask)
        else:
            raise ValueError(f"Unsupported rollout_rs option: {option_name}.")

        metrics_upper_threshold = upper_log if is_k1_option else upper_value
        metrics_lower_threshold = lower_log if (is_k1_option and lower_log is not None) else 0.0

        token_keep_mask = token_keep_bool.to(dtype=log_ratio.dtype)
        combined_mask = combined_mask * token_keep_mask
        seq_keep_bool_tensor = (~((~token_keep_bool) & response_mask_bool)).all(dim=-1)

        option_metrics = compute_rs_metrics(
            option_name=option_name,
            rs_statistic=per_token_stat,
            response_mask=response_mask,
            seq_valid_mask=seq_valid_mask,
            level=level,
            per_sequence_values=per_sequence_stat,
            rollout_rs_threshold=metrics_upper_threshold,
            rollout_rs_threshold_lower=metrics_lower_threshold,
            apply_lower_threshold=apply_lower_threshold,
        )
        metrics.update(option_metrics)

        token_masked_fraction = verl_F.masked_mean(1 - token_keep_mask, response_mask).item()
        seq_valid_float = seq_valid_mask.float()
        if seq_valid_float.sum() > 0:
            seq_keep_float = seq_keep_bool_tensor.to(dtype=log_ratio.dtype)
            seq_masked_fraction = (((1.0 - seq_keep_float) * seq_valid_float).sum() / seq_valid_float.sum()).item()
        else:
            seq_masked_fraction = 0.0
        metrics[f"rollout_rs_{option_name}_masked_fraction"] = token_masked_fraction
        metrics[f"rollout_rs_{option_name}_seq_masked_fraction"] = seq_masked_fraction

    final_mask = combined_mask
    metrics["rollout_rs_masked_fraction"] = verl_F.masked_mean(1 - final_mask, response_mask).item()
    final_keep_bool = (final_mask > 0.5) & response_mask_bool
    seq_has_masked: torch.Tensor = (~final_keep_bool & response_mask_bool).any(dim=-1)
    metrics["rollout_rs_seq_masked_fraction"] = seq_has_masked.float().mean().item()

    modified_response_mask: torch.Tensor = (response_mask * final_mask).to(dtype=response_mask.dtype)
    return modified_response_mask, metrics


def compute_rs_metrics(
    option_name: str,
    rs_statistic: torch.Tensor,
    response_mask: torch.Tensor,
    seq_valid_mask: torch.Tensor,
    *,
    level: str,
    per_sequence_values: Optional[torch.Tensor],
    rollout_rs_threshold: float,
    rollout_rs_threshold_lower: float,
    apply_lower_threshold: bool,
) -> dict[str, float]:
    """Compute metrics for hard trust region enforcement (per-option).

    Args:
        option_name: Original option string supplied by the user.
        rs_statistic: Trust region statistic (per token) used for thresholding.
        response_mask: Binary mask for valid tokens (1=valid, 0=padding).
        seq_valid_mask: Boolean mask indicating sequences with at least one valid token.
        level: "token" or "sequence" describing aggregation level.
        per_sequence_values: Optional per-sequence statistic (same semantics as rs_statistic).
        rollout_rs_threshold: Upper threshold.
        rollout_rs_threshold_lower: Lower threshold (ignored if ``apply_lower_threshold`` is False).
        apply_lower_threshold: Whether to mask/log metrics for values below the lower threshold.
    """
    if not response_mask.any():
        raise ValueError("response_mask must contain at least one valid token (1).")

    metrics: dict[str, float] = {}
    prefix = f"rollout_rs_{option_name}"
    mask_bool: torch.Tensor = response_mask.bool()

    # Compute sequence statistics (used by several metrics).
    if per_sequence_values is not None:
        seq_values = per_sequence_values
    else:
        seq_values = verl_F.masked_mean(rs_statistic, response_mask, axis=-1)
    if seq_values.dim() > 1:
        seq_values = seq_values.squeeze(-1)
    seq_values_valid = seq_values[seq_valid_mask]

    # Mean of the statistic (always reported).
    metrics[f"{prefix}_mean"] = verl_F.masked_mean(rs_statistic, response_mask).item()

    # Max/min values.
    if level == "sequence" and seq_values_valid.numel() > 0:
        metrics[f"{prefix}_max"] = seq_values_valid.max().item()
        metrics[f"{prefix}_min"] = seq_values_valid.min().item()
    else:
        metrics[f"{prefix}_max"] = rs_statistic.masked_fill(~mask_bool, float("-inf")).max().item()
        metrics[f"{prefix}_min"] = rs_statistic.masked_fill(~mask_bool, float("inf")).min().item()

    # Fractions above/below the thresholds.
    if level == "sequence" and seq_values_valid.numel() > 0:
        fraction_high = (seq_values_valid > rollout_rs_threshold).float().mean().item()
        fraction_low = (
            (seq_values_valid < rollout_rs_threshold_lower).float().mean().item() if apply_lower_threshold else 0.0
        )
    else:
        fraction_high = verl_F.masked_mean((rs_statistic > rollout_rs_threshold).float(), response_mask).item()
        fraction_low = (
            verl_F.masked_mean((rs_statistic < rollout_rs_threshold_lower).float(), response_mask).item()
            if apply_lower_threshold
            else 0.0
        )
    metrics[f"{prefix}_fraction_high"] = fraction_high
    metrics[f"{prefix}_fraction_low"] = fraction_low

    # Standard deviation (clamped for stability).
    mask_count: torch.Tensor = response_mask.sum()
    if mask_count > 1:
        if apply_lower_threshold:
            clamp_min = rollout_rs_threshold_lower
        else:
            clamp_min = 0.0
        stat_for_std: torch.Tensor = rs_statistic.clamp(min=clamp_min, max=rollout_rs_threshold)
        mean_clamped: torch.Tensor = verl_F.masked_mean(stat_for_std, response_mask)
        stat_var: torch.Tensor = verl_F.masked_mean(stat_for_std.square(), response_mask) - mean_clamped.square()
        metrics[f"{prefix}_std"] = torch.sqrt(torch.clamp(stat_var, min=0.0)).item()
    else:
        metrics[f"{prefix}_std"] = 0.0

    # Sequence-level summary metrics.
    if seq_values_valid.numel() > 0:
        metrics[f"{prefix}_seq_mean"] = seq_values_valid.mean().item()
        metrics[f"{prefix}_seq_std"] = seq_values_valid.std().item() if seq_values_valid.numel() > 1 else 0.0
        metrics[f"{prefix}_seq_max"] = seq_values_valid.max().item()
        metrics[f"{prefix}_seq_min"] = seq_values_valid.min().item()
        metrics[f"{prefix}_seq_max_deviation"] = (seq_values_valid - 0.0).abs().max().item()
        metrics[f"{prefix}_seq_fraction_high"] = (seq_values_valid > rollout_rs_threshold).float().mean().item()
        if apply_lower_threshold:
            metrics[f"{prefix}_seq_fraction_low"] = (
                (seq_values_valid < rollout_rs_threshold_lower).float().mean().item()
            )
    else:
        metrics[f"{prefix}_seq_mean"] = 0.0
        metrics[f"{prefix}_seq_std"] = 0.0
        metrics[f"{prefix}_seq_max"] = 0.0
        metrics[f"{prefix}_seq_min"] = 0.0
        metrics[f"{prefix}_seq_max_deviation"] = 0.0
        metrics[f"{prefix}_seq_fraction_high"] = 0.0
        metrics[f"{prefix}_seq_fraction_low"] = 0.0

    return metrics


def compute_rollout_correction_weights(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: str = "token",
    rollout_is_threshold: float = 2.0,
    rollout_is_batch_normalize: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute importance sampling weights to correct for off-policy distribution shifts.

    This function calculates IS weights (π_train / π_rollout) using log ratios for numerical stability.
    It supports multiple aggregation levels and truncates extreme weights to prevent training instability.

    Key design:
    - Log-space computations to avoid overflow
    - Truncation of extreme weights (TIS: Truncated Importance Sampling)
    - Optional batch normalization (normalize to mean=1.0)
    - Metrics tracking for weight distribution analysis

    Args:
        log_ratio: Log ratio of training policy probability to rollout policy probability,
            shape (batch_size, seq_length).
        response_mask: Binary mask for valid tokens (1=valid, 0=padding),
            shape (batch_size, seq_length).
        rollout_is: IS weight aggregation level, must be one of:
            - "token": Per-token weights (biased, low variance)
            - "sequence": Per-sequence weight (product of tokens; unbiased, high variance)
        rollout_is_threshold: Upper threshold for truncating extreme weights (e.g., 2.0),
            default 2.0.
        rollout_is_batch_normalize: Whether to normalize IS weights to have mean=1.0 per batch,
            default False.

    Returns:
        Tuple containing:
            rollout_is_weights: Truncated IS weights (masked to zero for padding tokens),
                shape (batch_size, seq_length). If batch_normalize=True, normalized to mean=1.0.
            metrics: Dictionary of IS weight metrics (all scalars), including:
                - rollout_is_mean/max/min: Statistic of weights (before batch normalization)
                - rollout_is_eff_sample_size: Effective sample size (ESS)
                - rollout_is_seq_*: Sequence-level weight statistics
                - rollout_is_batch_norm_factor: Normalization factor (only if batch_normalize=True)
    """
    # Validate input parameters
    valid_is_levels = {"token", "sequence"}
    if rollout_is not in valid_is_levels:
        raise ValueError(f"Invalid rollout_is: {rollout_is}. Must be one of {valid_is_levels}.")
    if rollout_is_threshold <= 0:
        raise ValueError(f"rollout_is_threshold must be positive, got {rollout_is_threshold}.")

    # Compute IS weights from log ratio (handles different aggregation levels)
    if rollout_is == "token":
        # Per-token IS weight: exp(log(π_train/π_rollout)) with safety clamp
        log_ratio_for_metrics: torch.Tensor = log_ratio
        log_ratio_safe: torch.Tensor = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights: torch.Tensor = torch.exp(log_ratio_safe)

    elif rollout_is == "sequence":
        # Sequence-level IS weight: product of token ratios (exp(sum(log ratios)))
        log_ratio_sum: torch.Tensor = verl_F.masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(
            -1
        )  # Shape: (batch_size, 1)
        log_ratio_for_metrics = log_ratio_sum

        log_ratio_sum_safe: torch.Tensor = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights = torch.exp(log_ratio_sum_safe).expand_as(log_ratio)  # Broadcast to sequence length

    else:
        raise ValueError(f"Unsupported rollout_is: {rollout_is}")

    # Zero out weights for padding tokens using response mask
    rollout_is_weights = rollout_is_weights * response_mask

    # Compute IS weight metrics (BEFORE truncation to get accurate fraction_high/low)
    metrics: dict[str, float] = compute_is_metrics(
        rollout_is_weights=rollout_is_weights,
        log_ratio_for_metrics=log_ratio_for_metrics,
        response_mask=response_mask,
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
    )

    # Truncate extreme weights (TIS: Truncated Importance Sampling)
    rollout_is_weights = rollout_is_weights.clamp(max=rollout_is_threshold)

    # Detach weights to prevent gradient flow (mathematically required by IS theory)
    # IS weights change the measure, not the objective. See §3.2.2 in docs/algo/rollout_corr_math.md
    rollout_is_weights = rollout_is_weights.detach()

    # Apply batch normalization if requested
    if rollout_is_batch_normalize:
        # Compute mean based on aggregation level
        mask_float = response_mask.to(dtype=rollout_is_weights.dtype)
        if rollout_is == "token":
            # Token-level: normalize over all token weights
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                weights_mean = verl_F.distributed_masked_mean(rollout_is_weights, mask_float)
            else:
                weights_mean = verl_F.masked_mean(rollout_is_weights, response_mask)
        elif rollout_is == "sequence":
            # Sequence-level: normalize over sequence weights (one weight per sequence)
            # For each sequence, compute mean over valid tokens (they all have the same weight)
            # then average across sequences
            seq_weights = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)  # (batch_size,)
            seq_mask = (response_mask.sum(dim=-1) > 0).to(dtype=rollout_is_weights.dtype)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                weights_mean = verl_F.distributed_masked_mean(seq_weights, seq_mask)
            else:
                weights_mean = (seq_weights * seq_mask).sum() / seq_mask.sum().clamp_min(1e-8)
        else:
            raise ValueError(f"Unsupported rollout_is: {rollout_is}")

        # Normalize to mean=1.0 (avoid division by zero)
        if weights_mean > 1e-8:
            rollout_is_weights = rollout_is_weights / weights_mean
            metrics["rollout_is_batch_norm_factor"] = weights_mean.item()
        else:
            metrics["rollout_is_batch_norm_factor"] = 1.0

    return rollout_is_weights, metrics


def compute_is_metrics(
    rollout_is_weights: torch.Tensor,
    log_ratio_for_metrics: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: str,
    rollout_is_threshold: float,
) -> dict[str, float]:
    """Compute comprehensive metrics for truncated importance sampling weights.

    This function calculates statistics for truncated IS weights (TIS), using log-space
    for accurate threshold checks and clamped weights for stable mean/std calculations.

    Args:
        rollout_is_weights: Truncated IS weights (π_train / π_rollout),
            shape (batch_size, seq_length).
        log_ratio_for_metrics: Log ratio of training to rollout probabilities (unclamped),
            shape varies by aggregation level.
        response_mask: Binary mask for valid tokens (1=valid, 0=padding),
            shape (batch_size, seq_length).
        rollout_is: IS weight aggregation level (matches compute_rollout_correction_weights).
        rollout_is_threshold: Upper threshold for truncated IS weights.

    Returns:
        Dictionary of IS weight metrics (all scalars).
    """
    if not response_mask.any():
        raise ValueError("response_mask must contain at least one valid token (1).")

    metrics: dict[str, float] = {}
    device: torch.device = rollout_is_weights.device
    # Default lower threshold (reciprocal of upper threshold)
    rollout_is_threshold_lower: float = 1.0 / rollout_is_threshold

    # Precompute log thresholds for accurate checks
    log_threshold_upper: torch.Tensor = torch.log(torch.tensor(rollout_is_threshold, device=device))
    log_threshold_lower: torch.Tensor = torch.log(torch.tensor(rollout_is_threshold_lower, device=device))

    # Compute metrics based on aggregation level
    if rollout_is == "sequence":
        # Sequence-level aggregation: use log-space for unclamped stats
        log_max: torch.Tensor = log_ratio_for_metrics.max()
        log_min: torch.Tensor = log_ratio_for_metrics.min()
        metrics["rollout_is_max"] = torch.exp(torch.clamp(log_max, max=SAFETY_BOUND)).item()
        metrics["rollout_is_min"] = torch.exp(log_min).item()

        # Mean uses truncated weights to avoid overflow
        metrics["rollout_is_mean"] = verl_F.masked_mean(rollout_is_weights, response_mask).item()

        # Fraction of weights exceeding thresholds (log-space for accuracy)
        exceeds_upper: torch.Tensor = log_ratio_for_metrics > log_threshold_upper
        below_lower: torch.Tensor = log_ratio_for_metrics < log_threshold_lower
        metrics["rollout_is_ratio_fraction_high"] = exceeds_upper.float().mean().item()
        metrics["rollout_is_ratio_fraction_low"] = below_lower.float().mean().item()

    else:  # token-level
        # Token-level aggregation: compute directly from truncated weights
        metrics["rollout_is_mean"] = verl_F.masked_mean(rollout_is_weights, response_mask).item()

        # Fraction of tokens exceeding thresholds
        rollout_is_above_threshold: torch.Tensor = rollout_is_weights > rollout_is_threshold
        rollout_is_below_threshold: torch.Tensor = rollout_is_weights < rollout_is_threshold_lower
        metrics["rollout_is_ratio_fraction_high"] = verl_F.masked_mean(
            rollout_is_above_threshold.float(), response_mask
        ).item()
        metrics["rollout_is_ratio_fraction_low"] = verl_F.masked_mean(
            rollout_is_below_threshold.float(), response_mask
        ).item()

        # Max/min (mask out padding tokens)
        mask_bool: torch.Tensor = response_mask.bool()
        metrics["rollout_is_max"] = rollout_is_weights.masked_fill(~mask_bool, float("-inf")).max().item()
        metrics["rollout_is_min"] = rollout_is_weights.masked_fill(~mask_bool, float("inf")).min().item()

    # Compute standard deviation (using clamped weights for stability)
    mask_count: torch.Tensor = response_mask.sum()
    if mask_count > 1:
        weights_for_std: torch.Tensor = rollout_is_weights.clamp(
            min=rollout_is_threshold_lower, max=rollout_is_threshold
        )
        mean_clamped: torch.Tensor = verl_F.masked_mean(weights_for_std, response_mask)
        rollout_is_var: torch.Tensor = (
            verl_F.masked_mean(weights_for_std.square(), response_mask) - mean_clamped.square()
        )
        metrics["rollout_is_std"] = torch.sqrt(torch.clamp(rollout_is_var, min=0.0)).item()
    else:
        metrics["rollout_is_std"] = 0.0

    # Compute Effective Sample Size (ESS) for truncated weights
    weights_for_ess: torch.Tensor = rollout_is_weights.clamp(min=rollout_is_threshold_lower, max=rollout_is_threshold)
    mean_for_ess: torch.Tensor = verl_F.masked_mean(weights_for_ess, response_mask)
    is_weights_normalized: torch.Tensor = weights_for_ess / (mean_for_ess + 1e-8)  # Avoid division by zero
    metrics["rollout_is_eff_sample_size"] = (
        1.0 / verl_F.masked_mean(is_weights_normalized.square(), response_mask).item()
    )

    # Add sequence-level metrics if weights have batch dimension
    if rollout_is_weights.dim() > 1:
        seq_mean_weights: torch.Tensor = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)

        metrics["rollout_is_seq_mean"] = seq_mean_weights.mean().item()
        metrics["rollout_is_seq_std"] = seq_mean_weights.std().item() if seq_mean_weights.numel() > 1 else 0.0
        metrics["rollout_is_seq_max"] = seq_mean_weights.max().item()
        metrics["rollout_is_seq_min"] = seq_mean_weights.min().item()

        # Sequence deviation from ideal weight (1.0)
        seq_deviation: torch.Tensor = (seq_mean_weights - 1.0).abs()
        metrics["rollout_is_seq_max_deviation"] = seq_deviation.max().item()

        # Fraction of sequences with extreme weights
        metrics["rollout_is_seq_fraction_high"] = (seq_mean_weights > rollout_is_threshold).float().mean().item()
        metrics["rollout_is_seq_fraction_low"] = (seq_mean_weights < rollout_is_threshold_lower).float().mean().item()

    return metrics


def compute_rollout_correction_and_rejection_mask(
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: Optional[str] = None,
    rollout_is_threshold: Optional[float] = 2.0,
    rollout_is_batch_normalize: bool = False,
    rollout_rs: Optional[str] = None,
    rollout_rs_threshold: Optional[str | float] = None,
) -> tuple[Optional[DataProto], torch.Tensor, dict[str, float]]:
    """Unified interface for computing IS weights and rejection masks.

    This function combines IS weight calculation (truncated) and rejection sampling (masked)
    into a single pipeline.

    Key design:
    - Separation of IS weights (for variance reduction) and rejection masks (for sample filtering)
    - Comprehensive metrics tracking for mismatch diagnosis

    Args:
        old_log_prob: Log probabilities from the training policy (e.g., FSDP FP32),
            shape (batch_size, seq_length).
        rollout_log_prob: Log probabilities from the rollout policy (e.g., vLLM BF16),
            shape (batch_size, seq_length).
        response_mask: Binary mask for valid tokens (1=valid, 0=padding),
            shape (batch_size, seq_length).
        rollout_is: IS weight aggregation level (see compute_rollout_correction_weights for options).
            Set to None to disable IS weight computation.
        rollout_is_threshold: Upper threshold for truncated IS weights (used if rollout_is is set),
            default 2.0.
        rollout_rs: Rejection sampling aggregation modes as a comma separated string
            (see compute_rollout_rejection_mask for the full list). Set to None to disable
            rejection sampling.
        rollout_rs_threshold: Threshold specification string (see compute_rollout_rejection_mask for details).
            Provide one threshold per option (comma separated). For K1-style options, specify
            ``lower_upper`` to denote the lower/upper ratio bounds.
        rollout_is_batch_normalize: Whether to normalize IS weights to have mean=1.0 per batch.
            Default: False.

    Returns:
        Tuple containing:
            rollout_is_weights_proto: DataProto with IS weights (None if rollout_is is None),
                key "rollout_is_weights", shape (batch_size, seq_length).
            modified_response_mask: Response mask with rejection sampling applied,
                shape (batch_size, seq_length).
            metrics: Dictionary of all metrics (prefixed with "rollout_corr/"), including:
                - IS weight statistics
                - Rejection sampling rates
                - Policy mismatch metrics (KL, PPL, etc.)
    """
    # Validate input masks
    if not response_mask.any():
        raise ValueError("response_mask must contain at least one valid token (1).")
    if old_log_prob.shape != rollout_log_prob.shape:
        raise ValueError(
            f"old_log_prob shape {old_log_prob.shape} does not match rollout_log_prob shape {rollout_log_prob.shape}."
        )
    if old_log_prob.shape != response_mask.shape:
        raise ValueError(
            f"log_prob shape {old_log_prob.shape} does not match response_mask shape {response_mask.shape}."
        )

    # Step 1: Compute log ratio (log(π_train / π_rollout))
    log_ratio: torch.Tensor = old_log_prob - rollout_log_prob
    metrics: dict[str, float] = {}

    # Step 2: Compute IS weights (if enabled)
    rollout_is_weights: Optional[torch.Tensor] = None
    if rollout_is is not None and rollout_is_threshold is not None:
        rollout_is_weights, is_metrics = compute_rollout_correction_weights(
            log_ratio=log_ratio,
            response_mask=response_mask,
            rollout_is=rollout_is,
            rollout_is_threshold=rollout_is_threshold,
            rollout_is_batch_normalize=rollout_is_batch_normalize,
        )
        metrics.update(is_metrics)

    # Step 3: Compute rejection mask (if enabled)
    modified_response_mask: torch.Tensor = response_mask.clone()
    if rollout_rs is not None:
        if rollout_rs_threshold is None:
            raise ValueError(
                "rollout_rs_threshold must be explicitly provided when rollout_rs is enabled. "
                "Set rollout_rs_threshold to the desired threshold value."
            )
        modified_response_mask, rs_metrics = compute_rollout_rejection_mask(
            log_ratio=log_ratio,
            response_mask=response_mask,
            rollout_rs=rollout_rs,
            rollout_rs_threshold=rollout_rs_threshold,
        )
        metrics.update(rs_metrics)

    # Step 4: Compute off-policy metrics (KL, PPL, χ², etc.)
    offpolicy_metrics: dict[str, float] = compute_offpolicy_metrics(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
    )
    metrics.update(offpolicy_metrics)

    # Step 6: Add "rollout_corr/" prefix to all metrics for logging consistency
    metrics_scalar: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            metrics_scalar[f"rollout_corr/{key}"] = value.item()
        else:
            metrics_scalar[f"rollout_corr/{key}"] = value

    # Step 7: Wrap IS weights in DataProto for consistency with API
    rollout_is_weights_proto: Optional[DataProto] = None
    if rollout_is_weights is not None:
        rollout_is_weights_proto = DataProto.from_dict(tensors={"rollout_is_weights": rollout_is_weights})

    return rollout_is_weights_proto, modified_response_mask, metrics_scalar


def compute_offpolicy_metrics(
    old_log_prob: torch.Tensor,
    rollout_log_prob: Optional[torch.Tensor],
    response_mask: torch.Tensor,
) -> dict[str, Any]:
    """Compute off-policy diagnostic metrics (helper function).

    This helper function operates on raw tensors and is used internally by:
    - compute_rollout_correction_and_rejection_mask() in this module (automatically included)
    - Tests (test_rollout_corr.py, test_rollout_corr_integration.py)

    These metrics help diagnose the off-policy gap between rollout and training policies,
    which can arise from:
    - Policy mismatch (e.g., vLLM BF16 vs FSDP FP32)
    - Model staleness (training on trajectories from older checkpoints)
    - General distribution shifts

    Key metrics:
    - kl: Direct KL divergence estimator KL(π_rollout || π_training)
    - k3_kl: K3 KL estimator for stability (more stable for small KL)
    - training_ppl: Perplexity of training policy
    - rollout_ppl: Perplexity of rollout policy
    - log_ppl_diff: Difference in log perplexities
    - ppl_ratio: Ratio of training PPL to rollout PPL
    - chi2_token: Token-level χ² divergence E[ρ²] - 1
    - chi2_seq: Sequence-level χ² divergence E[(∏ρ_t)²] - 1

    Args:
        old_log_prob: Log probabilities from training policy, shape (batch_size, seq_length)
        rollout_log_prob: Log probabilities from rollout policy, shape (batch_size, seq_length)
        response_mask: Mask for valid tokens, shape (batch_size, seq_length)

    Returns:
        Dictionary of off-policy metrics (without prefix)
    """
    # Validate that we have at least one valid token
    assert response_mask.any(), "Expected at least one valid token in response_mask"

    metrics = {}

    # 1. Training policy perplexity (always available)
    # Formula: exp(-1/|T| * Σ log π_training(y_t|y_<t))
    # where |T| is the number of tokens generated by the model
    mean_log_prob_training = verl_F.masked_mean(old_log_prob, response_mask, axis=-1)  # (batch_size,)
    training_ppl = torch.exp(-mean_log_prob_training).mean()  # Batch mean of per-sequence PPL
    metrics["training_ppl"] = training_ppl.detach().item()

    # Also log log-ppl for easier analysis (avoids exponential scale)
    metrics["training_log_ppl"] = (-mean_log_prob_training).mean().detach().item()

    # 2. Compute rollout off-policy metrics (only if rollout_log_probs available)
    if rollout_log_prob is not None:
        # 2a. kl: Direct estimator for KL(π_rollout || π_training)
        # This is the standard KL divergence: E[log(π_rollout) - log(π_training)]
        # Positive value means rollout policy is more confident than training policy
        metrics["kl"] = verl_F.masked_mean(rollout_log_prob - old_log_prob, response_mask).detach().item()

        # 2b. k3_kl: K3 estimator for KL(π_rollout || π_training)
        # More stable for small KL values using: E[exp(log_ratio) - log_ratio - 1]
        # Formula: KL ≈ E[r - log(r) - 1] where r = π_training/π_rollout
        log_ratio = old_log_prob - rollout_log_prob
        k3_kl_matrix = torch.exp(log_ratio) - log_ratio - 1
        metrics["k3_kl"] = verl_F.masked_mean(k3_kl_matrix, response_mask).detach().item()

        # 2c. Rollout policy perplexity
        mean_log_prob_rollout = verl_F.masked_mean(rollout_log_prob, response_mask, axis=-1)  # (batch_size,)
        rollout_ppl = torch.exp(-mean_log_prob_rollout).mean()  # Batch mean of per-sequence PPL
        metrics["rollout_ppl"] = rollout_ppl.detach().item()
        metrics["rollout_log_ppl"] = (-mean_log_prob_rollout).mean().detach().item()

        # 2d. Log PPL difference (sequence-level perplexity difference)
        # log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
        # Since ppl = exp(-log_prob), we have:
        #   log(ppl_ratio) = log(training_ppl/rollout_ppl) = log_ppl_diff
        # Positive value means training assigns lower probability (higher PPL) than rollout
        log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
        metrics["log_ppl_diff"] = log_ppl_diff.mean().detach().item()
        metrics["log_ppl_abs_diff"] = log_ppl_diff.abs().mean().detach().item()
        metrics["log_ppl_diff_max"] = log_ppl_diff.max().detach().item()
        metrics["log_ppl_diff_min"] = log_ppl_diff.min().detach().item()

        # 2e. PPL ratio (how much higher is training PPL vs rollout PPL)
        # Compute per-sequence ratio first, then average
        # For numerical stability, compute in log space using log_ppl_diff
        # Note: log_ppl_diff = log(ppl_ratio), so ppl_ratio = exp(log_ppl_diff)
        # This is the inverse of geometric IS: ppl_ratio_i = 1 / geometric_is_i for each sequence
        ppl_ratio = torch.exp(log_ppl_diff).mean()  # mean(exp(log_ppl_diff)) = mean(ppl_ratio_i)
        metrics["ppl_ratio"] = ppl_ratio.detach().item()

        # 2f. Chi-squared divergence: χ²(π_training || π_rollout) = E_μ[ρ²] - 1
        # where ρ = π_training / π_rollout and μ = π_rollout (rollout distribution)
        # This measures the variance of importance sampling weights
        # Token-level: E_token[ρ²] - 1 (averaged over all tokens)
        log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rho_token = torch.exp(log_ratio_safe)  # ρ = π_training / π_rollout (token-level)
        rho_squared_token = rho_token.square()
        chi2_token = verl_F.masked_mean(rho_squared_token, response_mask) - 1.0
        metrics["chi2_token"] = chi2_token.detach().item()

        # Sequence-level: E_seq[(Π ρ_t)²] - 1 = E_seq[exp(2 * Σ log ρ_t)] - 1
        log_ratio_sum = verl_F.masked_sum(log_ratio, response_mask, axis=-1)  # Σ log ρ_t per sequence
        log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rho_squared_seq = torch.exp(2.0 * log_ratio_sum_safe)  # (Π ρ_t)²
        chi2_seq = rho_squared_seq.mean() - 1.0
        metrics["chi2_seq"] = chi2_seq.detach().item()

    return metrics


def compute_rollout_correction_and_add_to_batch(
    batch: DataProto, rollout_corr_config: RolloutCorrectionConfig
) -> tuple[DataProto, dict]:
    """Compute rollout correction weights and apply rejection sampling.

    Computes importance sampling weights to correct for off-policy issues between
    rollout and training policies. Applies rejection sampling by modifying response_mask.
    Always updates response_mask; conditionally adds IS weights.

    Key behavior:
    - response_mask: ALWAYS updated with rejection (RS exclusions removed from training)
    - rollout_is_weights: Added to batch ONLY if rollout_is parameter is set

    This separation ensures:
    - Rejection works independently of IS weight application
    - Metrics can be monitored before enabling IS weight correction

    Args:
        batch: DataProto with old_log_probs, rollout_log_probs, response_mask

    Returns:
        Tuple of (updated_batch, metrics):
            updated_batch: Batch with modified response_mask (always) and rollout_is_weights (if enabled)
            metrics: Dict of IS and off-policy metrics, all with "rollout_corr/" prefix

    Note:
        The implementation is copied from szrlee <szrlee@gmail.com>.
    """
    # Get new API parameters directly from config
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)

    # Compute IS weights and get modified response_mask
    rollout_is_weights, modified_response_mask, rollout_corr_metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=batch.batch["old_log_probs"],
        rollout_log_prob=batch.batch["rollout_log_probs"],
        response_mask=batch.batch["response_mask"],
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
        rollout_is_batch_normalize=rollout_is_batch_normalize,
        rollout_rs=rollout_rs,
        rollout_rs_threshold=rollout_rs_threshold,
    )

    # ALWAYS update response_mask with rejection applied
    batch.batch["response_mask"] = modified_response_mask

    # Add IS weights to batch if computed
    if rollout_is_weights is not None:
        batch = batch.union(rollout_is_weights)

    return batch, rollout_corr_metrics


def compute_rollout_corr_metrics_from_logprobs(
    log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute rollout correction metrics from log probabilities during training.

    This function is used in the actor to compute metrics using the CURRENT policy
    log probabilities versus rollout log probabilities, allowing tracking of the
    off-policy gap as training progresses.

    It computes off-policy diagnostic metrics (KL, PPL, χ²) from log probabilities.

    Args:
        log_prob: Current policy log probabilities, shape (batch_size, seq_length)
        rollout_log_prob: Rollout policy log probabilities, shape (batch_size, seq_length)
        response_mask: Valid token mask, shape (batch_size, seq_length)

    Returns:
        Dictionary of metrics with "rollout_corr/" prefix
    """
    # Compute off-policy diagnostic metrics
    offpolicy_metrics = compute_offpolicy_metrics(
        old_log_prob=log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
    )

    # Add rollout_corr/ prefix to all metrics
    metrics_with_prefix = {}
    for key, value in offpolicy_metrics.items():
        if isinstance(value, torch.Tensor):
            metrics_with_prefix[f"rollout_corr/{key}"] = value.item()
        else:
            metrics_with_prefix[f"rollout_corr/{key}"] = value

    return metrics_with_prefix


def apply_bypass_mode(
    batch: DataProto,
    rollout_corr_config: Optional[RolloutCorrectionConfig] = None,
    policy_loss_config: PolicyLossConfig = None,
) -> None:
    """
    Setup bypass mode: Use rollout_log_probs as old_log_probs.

    Bypass mode skips expensive actor forward pass for old_log_prob computation
    by setting old_log_probs = rollout_log_probs (2 policies instead of 3).

    Uses compute_policy_loss_bypass_mode() which supports:
    - loss_type="ppo_clip" (default): PPO clipped objective (IS handled by ratio)
    - loss_type="reinforce": REINFORCE with explicit IS weights

    Both loss types benefit from rejection sampling (RS) which masks out-of-distribution samples.

    Note:
        The implementation is copied from szrlee <szrlee@gmail.com>.
    """
    from omegaconf import open_dict

    if "rollout_log_probs" not in batch.batch:
        raise ValueError(
            "bypass_mode=True requires rollout_log_probs in batch. "
            "Ensure rollout worker is configured to calculate_log_probs=true."
        )

    # Use rollout log probs as old log probs (zero-cost substitution)
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]

    with open_dict(policy_loss_config):
        # Pass rollout_correction config to actor for loss computation and metrics
        policy_loss_config["rollout_correction"] = rollout_corr_config
        # Always use bypass_mode loss function which handles both loss_types
        policy_loss_config["loss_mode"] = "bypass_mode"
