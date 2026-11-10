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
metrics of batches without empty rows must stay exactly what they were.
"""

import pytest
import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.rollout_corr_helper import (
    _seq_masked_max,
    _seq_masked_mean,
    _seq_masked_min,
    _seq_masked_std,
    compute_is_metrics,
    compute_offpolicy_metrics,
    compute_rollout_corr_metrics_from_logprobs,
    compute_rollout_correction_and_rejection_mask,
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


def _assert_metrics_close(actual: dict[str, float], expected: dict[str, float]) -> None:
    """Assert that both dicts have the same keys and that their values agree within 1e-5, relative or absolute.

    The tolerance covers floating-point rounding, since inserted rows change the order of the sums.
    """
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, rel=1e-5, abs=1e-5), key


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("numel", [1, 2, 3, 7, 41, 49, 128])
def test_seq_masked_helpers_equal_plain_reductions_when_all_rows_valid(numel, dtype):
    """With every row selected, the masked helpers return exactly ``Tensor.mean`` / ``max`` / ``min`` / ``std``.

    ``std`` is only compared for two or more rows; the callers of ``_seq_masked_std`` handle a single row.
    """
    values = torch.randn(numel, generator=torch.Generator().manual_seed(numel)).to(dtype)
    all_valid = torch.ones(numel, dtype=torch.bool)

    assert torch.equal(_seq_masked_mean(values, all_valid), values.mean())
    assert torch.equal(_seq_masked_max(values, all_valid), values.max())
    assert torch.equal(_seq_masked_min(values, all_valid), values.min())
    if numel > 1:
        assert torch.equal(_seq_masked_std(values, all_valid), values.std())


@pytest.mark.parametrize("seed", range(5))
def test_seq_masked_helpers_ignore_unselected_rows(seed):
    """With a random subset of rows selected, the masked helpers match the plain reductions of that subset."""
    gen = torch.Generator().manual_seed(seed)
    values = torch.randn(32, generator=gen, dtype=torch.float64)
    seq_mask = torch.rand(32, generator=gen) < 0.6
    seq_mask[seed] = True
    selected = values[seq_mask]

    torch.testing.assert_close(_seq_masked_mean(values, seq_mask), selected.mean())
    torch.testing.assert_close(_seq_masked_max(values, seq_mask), selected.max())
    torch.testing.assert_close(_seq_masked_min(values, seq_mask), selected.min())
    expected_std = selected.std() if selected.numel() > 1 else torch.zeros((), dtype=values.dtype)
    torch.testing.assert_close(_seq_masked_std(values, seq_mask), expected_std)


def test_seq_masked_std_is_zero_for_a_single_selected_row():
    """With one selected row, ``_seq_masked_std`` returns 0, as ``rollout_is_seq_std`` does for a one-row batch."""
    values = torch.tensor([0.3, 1.7, -2.0, 5.0])
    seq_mask = torch.tensor([False, True, False, False])
    assert _seq_masked_std(values, seq_mask).item() == 0.0


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
    """Without empty rows, the per-sequence IS metrics equal the plain reductions over all rows exactly.

    In sequence mode this includes ``rollout_is_max`` / ``rollout_is_min`` and ``rollout_is_ratio_fraction_*``.
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
        "rollout_is_seq_std": seq_mean.std().item(),
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


def test_rs_seq_masked_fraction_unchanged_without_empty_rows():
    """Without empty rows, ``rollout_rs_seq_masked_fraction`` equals the plain mean over all rows exactly.

    The fraction is checked to lie strictly between 0 and 1, so the comparison is not trivial.
    """
    old_log_prob, rollout_log_prob, response_mask = _make_batch(seed=2)
    _, modified_mask, metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob, rollout_log_prob, response_mask, rollout_rs="token_k1", rollout_rs_threshold="0.8_1.25"
    )
    seq_has_masked = ((modified_mask == 0) & response_mask.bool()).any(dim=-1)
    assert 0.0 < metrics["rollout_corr/rollout_rs_seq_masked_fraction"] < 1.0
    assert metrics["rollout_corr/rollout_rs_seq_masked_fraction"] == seq_has_masked.float().mean().item()


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

    _assert_metrics_close(padded_metrics, metrics)
    # The empty rows stay empty and get zero weights.
    padded_rows_valid = padded[2].sum(dim=-1) > 0
    assert not padded_mask[~padded_rows_valid].any()
    if weights is not None:
        assert not padded_weights.batch["rollout_is_weights"][~padded_rows_valid].any()


@pytest.mark.parametrize("seed", range(3))
def test_empty_rows_do_not_change_training_offpolicy_metrics(seed):
    """Empty rows leave the off-policy metrics that the actor computes during training unchanged.

    ``compute_rollout_corr_metrics_from_logprobs`` takes them from ``compute_offpolicy_metrics``,
    which excludes such rows since #7785.
    """
    batch = _make_batch(seed)
    padded = _add_empty_rows(seed, 7, *batch)
    _assert_metrics_close(
        compute_rollout_corr_metrics_from_logprobs(*padded), compute_rollout_corr_metrics_from_logprobs(*batch)
    )
