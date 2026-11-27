# Copyright (c) Meta Platforms, Inc. and affiliates.
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
"""Per-sequence rollout correction metrics skip rows without valid tokens.

Some training backends pad a micro-batch with empty rows (all-zero ``response_mask``) so that
every micro-batch has the same shape. Such rows must not change the logged metrics, and the
metrics of batches without empty rows must stay what they were.
"""

import pytest
import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.rollout_corr_helper import (
    compute_is_metrics,
    compute_offpolicy_metrics,
    compute_rollout_corr_metrics_from_logprobs,
    compute_rollout_correction_and_rejection_mask,
    compute_rollout_rejection_mask,
    compute_rs_metrics,
)

CORRECTION_CONFIGS = [
    pytest.param(dict(rollout_is="token", rollout_is_threshold=2.0), id="token_is"),
    pytest.param(dict(rollout_is="sequence", rollout_is_threshold=2.0), id="sequence_is"),
    pytest.param(
        dict(rollout_is="token", rollout_is_threshold="0.5_2.0", rollout_rs="token_k1", rollout_rs_threshold="0.7_1.4"),
        id="icepop_token_rs",
    ),
    pytest.param(
        dict(
            rollout_is="sequence",
            rollout_is_threshold=3.0,
            rollout_rs="seq_mean_k1,seq_max_k2",
            rollout_rs_threshold="0.9_1.1,0.01",
        ),
        id="sequence_is_seq_rs",
    ),
    pytest.param(dict(rollout_rs="seq_sum_k3,token_k2", rollout_rs_threshold="0.5,0.02"), id="rs_only"),
]


def _make_batch(seed: int, batch_size: int = 12, seq_len: int = 20) -> tuple[torch.Tensor, ...]:
    """Create a seeded ``(old_log_prob, rollout_log_prob, response_mask)`` batch without empty rows.

    Every row has 1 to ``seq_len`` valid tokens, and the rollout log-probs are the training
    log-probs plus Gaussian noise.
    """
    gen = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, seq_len + 1, (batch_size,), generator=gen)
    response_mask = (torch.arange(seq_len)[None, :] < lengths[:, None]).float()
    old_log_prob = -3.0 * torch.rand(batch_size, seq_len, generator=gen)
    rollout_log_prob = old_log_prob + 0.3 * torch.randn(batch_size, seq_len, generator=gen)
    return old_log_prob, rollout_log_prob, response_mask


def _add_empty_rows(seed: int, num_empty: int, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Interleave ``num_empty`` rows with arbitrary log-probs and an all-zero response mask."""
    old_log_prob, rollout_log_prob, response_mask = tensors
    gen = torch.Generator().manual_seed(seed + 1000)
    seq_len = response_mask.shape[1]
    old_log_prob = torch.cat([old_log_prob, -5.0 * torch.rand(num_empty, seq_len, generator=gen)])
    rollout_log_prob = torch.cat([rollout_log_prob, -5.0 * torch.rand(num_empty, seq_len, generator=gen)])
    response_mask = torch.cat([response_mask, torch.zeros(num_empty, seq_len)])
    perm = torch.randperm(response_mask.shape[0], generator=gen)
    return old_log_prob[perm], rollout_log_prob[perm], response_mask[perm]


def _assert_metrics_match(actual: dict[str, float], expected: dict[str, float]) -> None:
    """Assert that both dicts have the same keys and values.

    Maxima, minima and fractions (counts over counts) must be equal. Means and standard deviations
    must agree within 1e-5, relative or absolute: inserted rows change the order of the sums.
    """
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if key.endswith(("_max", "_min", "_max_deviation", "_oob_ratio")) or "fraction" in key:
            assert actual[key] == value, key
        else:
            assert actual[key] == pytest.approx(value, rel=1e-5, abs=1e-5), key


def test_offpolicy_metrics_unchanged_without_empty_rows():
    """Without empty rows, the per-sequence off-policy metrics equal the plain reductions over all rows exactly."""
    old_log_prob, rollout_log_prob, response_mask = _make_batch(seed=0)
    metrics = compute_offpolicy_metrics(old_log_prob, rollout_log_prob, response_mask)

    mean_log_prob_training = verl_F.masked_mean(old_log_prob, response_mask, axis=-1)
    mean_log_prob_rollout = verl_F.masked_mean(rollout_log_prob, response_mask, axis=-1)
    log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
    log_ratio_sum = verl_F.masked_sum(old_log_prob - rollout_log_prob, response_mask, axis=-1)
    expected = {
        "training_ppl": torch.exp(-mean_log_prob_training).mean().item(),
        "training_log_ppl": (-mean_log_prob_training).mean().item(),
        "rollout_ppl": torch.exp(-mean_log_prob_rollout).mean().item(),
        "rollout_log_ppl": (-mean_log_prob_rollout).mean().item(),
        "log_ppl_diff": log_ppl_diff.mean().item(),
        "log_ppl_abs_diff": log_ppl_diff.abs().mean().item(),
        "log_ppl_diff_max": log_ppl_diff.max().item(),
        "log_ppl_diff_min": log_ppl_diff.min().item(),
        "ppl_ratio": torch.exp(log_ppl_diff).mean().item(),
        "chi2_seq": (torch.exp(2.0 * log_ratio_sum.clamp(-20.0, 20.0)).mean() - 1.0).item(),
    }
    for key, value in expected.items():
        assert metrics[key] == value, key


@pytest.mark.parametrize("rollout_is", ["token", "sequence"])
def test_is_metrics_unchanged_without_empty_rows(rollout_is):
    """Without empty rows, the per-sequence IS metrics equal the plain reductions over all rows.

    In sequence mode this includes ``rollout_is_max`` / ``rollout_is_min`` and ``rollout_is_ratio_fraction_*``.
    Every value is equal, except ``rollout_is_seq_std``: ``verl_F.masked_var`` and ``Tensor.std`` round differently.
    """
    old_log_prob, rollout_log_prob, response_mask = _make_batch(seed=1)
    threshold = 1.2
    log_ratio = old_log_prob - rollout_log_prob
    if rollout_is == "token":
        log_ratio_for_metrics = log_ratio
        raw_weights = torch.exp(log_ratio.clamp(-20.0, 20.0)) * response_mask
    else:
        log_ratio_for_metrics = verl_F.masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(-1)
        raw_weights = torch.exp(log_ratio_for_metrics.clamp(-20.0, 20.0)).expand_as(log_ratio) * response_mask
    weights = raw_weights.clamp(max=threshold)

    metrics = compute_is_metrics(
        rollout_is_weights=weights,
        raw_rollout_is_weights=raw_weights,
        log_ratio_for_metrics=log_ratio_for_metrics,
        response_mask=response_mask,
        rollout_is=rollout_is,
        rollout_is_threshold=threshold,
    )

    seq_mean = verl_F.masked_mean(weights, response_mask, axis=-1)
    raw_seq_mean = verl_F.masked_mean(raw_weights, response_mask, axis=-1)
    expected = {
        "rollout_is_seq_mean": seq_mean.mean().item(),
        "rollout_is_seq_max": seq_mean.max().item(),
        "rollout_is_seq_min": seq_mean.min().item(),
        "rollout_is_seq_max_deviation": (seq_mean - 1.0).abs().max().item(),
        "rollout_is_seq_fraction_high": (raw_seq_mean > threshold).float().mean().item(),
        "rollout_is_seq_fraction_low": (raw_seq_mean < 1.0 / threshold).float().mean().item(),
    }
    if rollout_is == "sequence":
        log_upper = torch.log(torch.tensor(threshold))
        log_lower = torch.log(torch.tensor(1.0 / threshold))
        expected["rollout_is_max"] = torch.exp(log_ratio_for_metrics.max().clamp(max=20.0)).item()
        expected["rollout_is_min"] = torch.exp(log_ratio_for_metrics.min()).item()
        expected["rollout_is_ratio_fraction_high"] = (log_ratio_for_metrics > log_upper).float().mean().item()
        expected["rollout_is_ratio_fraction_low"] = (log_ratio_for_metrics < log_lower).float().mean().item()
    for key, value in expected.items():
        assert metrics[key] == value, key
    assert metrics["rollout_is_seq_std"] == pytest.approx(seq_mean.std().item(), rel=1e-6)


@pytest.mark.parametrize("apply_lower_threshold", [False, True])
@pytest.mark.parametrize("level", ["token", "sequence"])
def test_rs_metrics_unchanged_without_empty_rows(level, apply_lower_threshold):
    """Without empty rows, the per-option RS metrics equal the plain reductions over all rows.

    Every value is equal, except ``*_seq_std``: ``verl_F.masked_var`` and ``Tensor.std`` round differently.
    """
    _, _, response_mask = _make_batch(seed=2)
    gen = torch.Generator().manual_seed(2)
    if level == "sequence":
        seq_values = torch.randn(response_mask.shape[0], generator=gen)
        per_sequence_values = seq_values
        rs_statistic = seq_values.unsqueeze(-1).expand_as(response_mask)
    else:
        rs_statistic = torch.randn(response_mask.shape, generator=gen)
        per_sequence_values = None
        seq_values = verl_F.masked_mean(rs_statistic, response_mask, axis=-1)
    upper, lower = 0.3, -0.3

    metrics = compute_rs_metrics(
        option_name="opt",
        rs_statistic=rs_statistic,
        response_mask=response_mask,
        seq_valid_mask=response_mask.sum(dim=-1) > 0,
        level=level,
        per_sequence_values=per_sequence_values,
        rollout_rs_threshold=upper,
        rollout_rs_threshold_lower=lower,
        apply_lower_threshold=apply_lower_threshold,
    )

    seq_fraction_high = (seq_values > upper).float().mean().item()
    seq_fraction_low = (seq_values < lower).float().mean().item()
    assert 0.0 < seq_fraction_high < 1.0 and 0.0 < seq_fraction_low < 1.0
    expected = {
        "rollout_rs_opt_seq_mean": seq_values.mean().item(),
        "rollout_rs_opt_seq_max": seq_values.max().item(),
        "rollout_rs_opt_seq_min": seq_values.min().item(),
        "rollout_rs_opt_seq_max_deviation": seq_values.abs().max().item(),
        "rollout_rs_opt_seq_fraction_high": seq_fraction_high,
    }
    if apply_lower_threshold:
        expected["rollout_rs_opt_seq_fraction_low"] = seq_fraction_low
    if level == "sequence":
        expected["rollout_rs_opt_max"] = seq_values.max().item()
        expected["rollout_rs_opt_min"] = seq_values.min().item()
        expected["rollout_rs_opt_fraction_high"] = seq_fraction_high
        expected["rollout_rs_opt_fraction_low"] = seq_fraction_low if apply_lower_threshold else 0.0
    for key, value in expected.items():
        assert metrics[key] == value, key
    assert metrics["rollout_rs_opt_seq_std"] == pytest.approx(seq_values.std().item(), rel=1e-6)


@pytest.mark.parametrize(
    "rollout_rs, rollout_rs_threshold", [("token_k1", "0.8_1.25"), ("seq_mean_k1", "0.95_1.05"), ("seq_max_k2", "0.1")]
)
def test_rs_seq_masked_fractions_unchanged_without_empty_rows(rollout_rs, rollout_rs_threshold):
    """Without empty rows, the overall and the per-option ``seq_masked_fraction`` equal the plain mean over all rows.

    With a single option both are the fraction of rows with a rejected token, checked to lie strictly
    between 0 and 1, so the comparison is not trivial.
    """
    old_log_prob, rollout_log_prob, response_mask = _make_batch(seed=2)
    _, modified_mask, metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob, rollout_log_prob, response_mask, rollout_rs=rollout_rs, rollout_rs_threshold=rollout_rs_threshold
    )
    expected = ((modified_mask == 0) & response_mask.bool()).any(dim=-1).float().mean().item()
    assert 0.0 < expected < 1.0
    assert metrics["rollout_corr/rollout_rs_seq_masked_fraction"] == expected
    assert metrics[f"rollout_corr/rollout_rs_{rollout_rs}_seq_masked_fraction"] == expected


@pytest.mark.parametrize("config", CORRECTION_CONFIGS)
@pytest.mark.parametrize("seed", range(3))
def test_empty_rows_do_not_change_correction_metrics(config, seed):
    """Empty rows inserted at random positions leave every rollout-correction metric unchanged.

    Checked for each ``CORRECTION_CONFIGS`` case. The inserted rows also stay masked out, and their
    IS weights, when computed, are zero.
    """
    batch = _make_batch(seed)
    padded = _add_empty_rows(seed, 5, *batch)

    weights, _, metrics = compute_rollout_correction_and_rejection_mask(*batch, **config)
    padded_weights, padded_mask, padded_metrics = compute_rollout_correction_and_rejection_mask(*padded, **config)

    _assert_metrics_match(padded_metrics, metrics)
    # The empty rows stay empty and get zero weights.
    padded_rows_valid = padded[2].sum(dim=-1) > 0
    assert not padded_mask[~padded_rows_valid].any()
    if weights is not None:
        assert not padded_weights.batch["rollout_is_weights"][~padded_rows_valid].any()


@pytest.mark.parametrize("config", CORRECTION_CONFIGS)
def test_one_valid_row_among_empty_rows(config):
    """A batch of one valid row and seven empty rows has the metrics of the valid row alone.

    The standard deviations across sequences are then 0, as for a one-row batch.
    """
    batch = tuple(t[:1] for t in _make_batch(seed=5))
    padded = _add_empty_rows(5, 7, *batch)

    _, _, metrics = compute_rollout_correction_and_rejection_mask(*batch, **config)
    _, _, padded_metrics = compute_rollout_correction_and_rejection_mask(*padded, **config)

    _assert_metrics_match(padded_metrics, metrics)
    for key, value in padded_metrics.items():
        if key.endswith("_seq_std"):
            assert value == 0.0, key


def test_batch_without_valid_tokens_is_rejected():
    """A batch whose rows are all empty is rejected as before, so no reduction runs over zero sequences."""
    old_log_prob, rollout_log_prob, response_mask = _make_batch(seed=6)
    empty = torch.zeros_like(response_mask)
    log_ratio = old_log_prob - rollout_log_prob

    with pytest.raises(ValueError, match="at least one valid token"):
        compute_rollout_correction_and_rejection_mask(old_log_prob, rollout_log_prob, empty, rollout_is="sequence")
    with pytest.raises(ValueError, match="at least one valid token"):
        compute_rollout_rejection_mask(log_ratio, empty, rollout_rs="seq_mean_k1", rollout_rs_threshold="0.9_1.1")
    with pytest.raises(ValueError, match="at least one valid token"):
        compute_is_metrics(log_ratio.exp(), log_ratio.exp(), log_ratio, empty, "token", rollout_is_threshold=2.0)
    with pytest.raises(AssertionError, match="at least one valid token"):
        compute_offpolicy_metrics(old_log_prob, rollout_log_prob, empty)


@pytest.mark.parametrize("level", ["token", "sequence"])
def test_rs_metrics_without_selected_sequences_keep_their_fallbacks(level):
    """With no sequence selected, ``compute_rs_metrics`` reports what it reported before.

    The ``seq_*`` metrics are 0 and the max, min and fractions come from the token-level statistic,
    so the infinities that mask out unselected sequences never reach a metric.
    """
    _, _, response_mask = _make_batch(seed=7)
    rs_statistic = torch.randn(response_mask.shape, generator=torch.Generator().manual_seed(7))
    no_sequence = torch.zeros(response_mask.shape[0], dtype=torch.bool)
    metrics = compute_rs_metrics(
        option_name="opt",
        rs_statistic=rs_statistic,
        response_mask=response_mask,
        seq_valid_mask=no_sequence,
        level=level,
        per_sequence_values=None,
        rollout_rs_threshold=0.3,
        rollout_rs_threshold_lower=-0.3,
        apply_lower_threshold=True,
    )

    mask_bool = response_mask.bool()
    token_fraction_high = verl_F.masked_mean((rs_statistic > 0.3).float(), response_mask).item()
    token_fraction_low = verl_F.masked_mean((rs_statistic < -0.3).float(), response_mask).item()
    assert metrics["rollout_rs_opt_max"] == rs_statistic.masked_fill(~mask_bool, float("-inf")).max().item()
    assert metrics["rollout_rs_opt_min"] == rs_statistic.masked_fill(~mask_bool, float("inf")).min().item()
    assert metrics["rollout_rs_opt_fraction_high"] == token_fraction_high
    assert metrics["rollout_rs_opt_fraction_low"] == token_fraction_low
    for name in ("mean", "std", "max", "min", "max_deviation", "fraction_high", "fraction_low"):
        assert metrics[f"rollout_rs_opt_seq_{name}"] == 0.0, name


@pytest.mark.parametrize("seed", range(3))
def test_empty_rows_do_not_change_training_offpolicy_metrics(seed):
    """Empty rows leave the off-policy metrics that the actor computes during training unchanged.

    ``compute_rollout_corr_metrics_from_logprobs`` takes them from ``compute_offpolicy_metrics``,
    which excludes such rows since #7785.
    """
    batch = _make_batch(seed)
    padded = _add_empty_rows(seed, 7, *batch)
    _assert_metrics_match(
        compute_rollout_corr_metrics_from_logprobs(*padded), compute_rollout_corr_metrics_from_logprobs(*batch)
    )
