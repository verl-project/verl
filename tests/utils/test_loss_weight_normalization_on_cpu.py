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
"""Loss-weight normalization: relative shares change, magnitude does not.

WHY THIS EXISTS
---------------
Weighting advantages by ``w_i`` fixes which trajectory dominates the gradient,
but every ``loss_agg_mode`` divides by an UNWEIGHTED denominator, so the total
loss also shrinks by ``mean(w)``. On the batch shape this feature exists for --
one 10-segment trajectory at ``w=0.1`` plus one 1-segment trajectory at
``w=1.0`` -- that is a 5.5x loss cut, i.e. a silent 5.5x learning-rate cut that
DRIFTS with the long/short mix of each step.

ScaleCUA (68.7% OSWorld) does not have this problem because it divides by the
weighted denominator (``megatron_worker.py:43``:
``loss * loss_weight[i] / batch_weight``). ``normalize_loss_weight`` is the
equivalent for verl, whose aggregation happens inside the loss function:
rescaling to ``mean(w) == 1`` keeps every ratio ``w_i / w_j`` while leaving the
magnitude alone.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

_TRAJ = Path(__file__).resolve().parents[2] / "verl" / "utils" / "trajectory.py"
_spec = importlib.util.spec_from_file_location("_verl_trajectory", _TRAJ)
trajectory = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trajectory)

normalize_loss_weight_global = trajectory.normalize_loss_weight_global

# The batch shape the feature targets: a 10-segment trajectory (1/N = 0.1 each)
# alongside a 1-segment trajectory (1/N = 1.0). Both are ONE logical trajectory,
# so each must end up with 50% of the gradient.
DEEP_PLUS_SHALLOW = [0.1] * 10 + [1.0]
RESPONSE_LEN = 100


def _token_mean(loss_mat, mask):
    return ((loss_mat * mask).sum() / mask.sum()).item()


def _seq_mean_token_mean(loss_mat, mask):
    per_seq = (loss_mat * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    return per_seq.mean().item()


def _weighted(loss_mat, weights):
    if weights is None:
        return loss_mat
    return loss_mat * weights.unsqueeze(-1)


def test_mean_is_one_and_ratios_are_preserved():
    raw = torch.tensor(DEEP_PLUS_SHALLOW)
    w = normalize_loss_weight_global(raw)

    assert w is not None
    assert w.mean().item() == pytest.approx(1.0)
    # Every pairwise ratio must survive; that is what makes it a reweighting
    # rather than a different objective.
    assert (w[0] / w[10]).item() == pytest.approx((raw[0] / raw[10]).item())


@pytest.mark.parametrize("agg", [_token_mean, _seq_mean_token_mean])
def test_magnitude_is_conserved(agg):
    """The whole point: normalized weights must not act as a hidden lr change."""
    n = len(DEEP_PLUS_SHALLOW)
    loss_mat = torch.ones(n, RESPONSE_LEN)
    mask = torch.ones(n, RESPONSE_LEN)

    raw = torch.tensor(DEEP_PLUS_SHALLOW)
    w = normalize_loss_weight_global(raw)

    unweighted = agg(loss_mat, mask)
    with_raw = agg(_weighted(loss_mat, raw), mask)
    with_norm = agg(_weighted(loss_mat, w), mask)

    # Documents the bug being fixed: raw 1/N weights shrink the loss.
    assert with_raw < 0.25 * unweighted
    # And the fix: magnitude restored.
    assert with_norm == pytest.approx(unweighted, rel=1e-5)


@pytest.mark.parametrize("agg", [_token_mean, _seq_mean_token_mean])
def test_relative_share_is_still_corrected(agg):
    """Normalizing must not undo the reweighting it is protecting."""
    n = len(DEEP_PLUS_SHALLOW)
    loss_mat = torch.ones(n, RESPONSE_LEN)
    mask = torch.ones(n, RESPONSE_LEN)
    w = normalize_loss_weight_global(torch.tensor(DEEP_PLUS_SHALLOW))

    weighted = _weighted(loss_mat, w)
    deep = (weighted[:10] * mask[:10]).sum()
    total = (weighted * mask).sum()

    # Without any weight the deep trajectory would take 10/11 = 90.9%.
    assert (deep / total).item() == pytest.approx(0.5, abs=1e-4)


def test_neutral_weights_are_unchanged():
    """A no-op must be a literal no-op so unweighted runs stay bit-identical."""
    assert torch.equal(normalize_loss_weight_global(torch.ones(4)), torch.ones(4))


def test_all_zero_weights_stay_neutral():
    """An all-padding batch must not divide by zero."""
    assert torch.equal(normalize_loss_weight_global(torch.zeros(4)), torch.zeros(4))


def test_padding_rows_keep_zero_weight():
    """Zeroed padding rows must not be resurrected by normalization."""
    raw = torch.tensor([1.0, 1.0, 0.0, 0.0])
    w = normalize_loss_weight_global(raw)
    assert w is not None
    assert w[2].item() == 0.0 and w[3].item() == 0.0
    # The two real samples absorb the full scale.
    assert w[0].item() == pytest.approx(2.0)


def test_rejects_column_vector_shape():
    with pytest.raises(ValueError):
        normalize_loss_weight_global(torch.tensor([[0.5], [1.5]]))


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([float("nan"), 1.0]),
        torch.tensor([float("inf"), 1.0]),
        torch.tensor([-1.0, 1.0]),
    ],
)
def test_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        normalize_loss_weight_global(bad)


def test_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        normalize_loss_weight_global(torch.tensor([[1.0], [2.0]]))


@pytest.mark.parametrize(
    "packing",
    [
        [list(range(10)), [10]],  # all deep segments land in one micro-batch
        [[0, 10], [1, 2], [3, 4], [5, 6], [7, 8], [9]],  # interleaved
        [[i] for i in range(11)],  # one sample per micro-batch
    ],
)
def test_normalization_is_packing_invariant(packing):
    """Regression test for a real bug: normalizing per MICRO-batch erases the weight.

    ``normalize_loss_weight_global`` must run once on the whole batch. If it ran
    inside the loss function -- which only ever sees a micro-batch -- then a
    micro-batch holding only same-weight samples would rescale all of them to
    1.0 and the reweighting would vanish. MEASURED with the broken ordering, on
    a batch where the deep trajectory must take 50%:

        all deep in one micro-batch  ->  90.9%
        interleaved                  ->  83.5%

    Normalizing globally first makes the result identical for every packing.
    """
    weights = normalize_loss_weight_global(torch.tensor(DEEP_PLUS_SHALLOW))

    deep = 0.0
    total = 0.0
    for micro_batch in packing:
        # Each micro-batch consumes the ALREADY normalized weights, unchanged.
        for i in micro_batch:
            total += weights[i].item()
            if i < 10:
                deep += weights[i].item()

    assert deep / total == pytest.approx(0.5, abs=1e-4)


def test_padding_rows_are_excluded_from_the_mean():
    """Padding must not dilute the scale computed for the real samples."""
    raw = torch.tensor([0.5, 1.5, 0.0, 0.0])
    valid = torch.tensor([True, True, False, False])
    w = normalize_loss_weight_global(raw, valid)

    # mean over the two VALID rows must be 1.0, ignoring the padding.
    assert w[valid].mean().item() == pytest.approx(1.0)
    assert w[2].item() == 0.0 and w[3].item() == 0.0


def test_composes_with_validate_loss_weights():
    """The trainer calls validate -> normalize; that exact order must work.

    ``validate_loss_weights`` requires strictly positive input and is what
    zeroes the padding rows, so ``normalize_loss_weight_global`` has to tolerate
    the zeros its predecessor introduces rather than reject them.
    """
    validate_loss_weights = trajectory.validate_loss_weights

    raw = torch.tensor([0.1] * 10 + [1.0] + [1.0, 1.0])
    valid = torch.tensor([True] * 11 + [False, False])

    validated = validate_loss_weights(raw.clone(), valid_mask=valid)
    assert validated[-1].item() == 0.0, "validate is what zeroes padding"

    normalized = normalize_loss_weight_global(validated, valid)

    assert normalized[valid].mean().item() == pytest.approx(1.0)
    assert normalized[~valid].abs().sum().item() == 0.0
    assert (normalized[0] / normalized[10]).item() == pytest.approx(0.1)


def test_rejects_valid_mask_shape_mismatch():
    with pytest.raises(ValueError):
        normalize_loss_weight_global(torch.tensor([1.0, 2.0]), torch.tensor([True]))


def test_weights_do_not_carry_gradient():
    """Weights are metadata; they must never participate in autograd."""
    raw = torch.tensor([0.5, 1.5], requires_grad=True)
    w = normalize_loss_weight_global(raw)
    assert w is not None
    assert not w.requires_grad


# --------------------------------------------------------------------------- #
# Boundaries of the magnitude guarantee, pinned so nobody over-reads it.
# --------------------------------------------------------------------------- #


def _unequal_rows():
    """{100 tokens, w=2} + {1000 tokens, w=0.5}: mean_rows(w) == 1.25 before rescaling."""
    loss_mat = torch.ones(2, 1000)
    mask = torch.zeros(2, 1000)
    mask[0, :100] = 1
    mask[1, :] = 1
    raw = torch.tensor([2.0, 0.5])
    return loss_mat, mask, raw


def test_magnitude_conserved_under_row_normalized_agg_with_unequal_lengths():
    loss_mat, mask, raw = _unequal_rows()
    w = normalize_loss_weight_global(raw)
    assert w.mean().item() == pytest.approx(1.0)
    assert _seq_mean_token_mean(_weighted(loss_mat, w), mask) == pytest.approx(_seq_mean_token_mean(loss_mat, mask))


def test_magnitude_not_conserved_under_token_mean_when_weight_correlates_with_length():
    """Documents the limit: token-mean's scale is the token-weighted mean of w, not the row mean."""
    loss_mat, mask, raw = _unequal_rows()
    w = normalize_loss_weight_global(raw)
    token_weighted_mean = (w * mask.sum(-1)).sum() / mask.sum()
    assert token_weighted_mean.item() == pytest.approx(0.5091, abs=1e-3)
    assert _token_mean(_weighted(loss_mat, w), mask) == pytest.approx(
        token_weighted_mean.item() * _token_mean(loss_mat, mask)
    )


# --------------------------------------------------------------------------- #
# Two objectives for a multi-row trajectory under seq-mean-token-mean.
# --------------------------------------------------------------------------- #


def _split_trajectory():
    """A 1000-token trajectory cut 100 / 900 with per-token loss 0.2 / 0.8."""
    unsplit_loss = torch.zeros(1, 1000)
    unsplit_loss[0, :100] = 0.2
    unsplit_loss[0, 100:] = 0.8
    unsplit_mask = torch.ones(1, 1000)

    split_loss = torch.zeros(2, 1000)
    split_loss[0, :100] = 0.2
    split_loss[1, :900] = 0.8
    split_mask = torch.zeros(2, 1000)
    split_mask[0, :100] = 1
    split_mask[1, :900] = 1
    return unsplit_loss, unsplit_mask, split_loss, split_mask


def test_one_over_n_is_session_equal_not_partition_preserving():
    unsplit_loss, unsplit_mask, split_loss, split_mask = _split_trajectory()
    target = _seq_mean_token_mean(unsplit_loss, unsplit_mask)  # 0.74

    w = normalize_loss_weight_global(torch.tensor([0.5, 0.5]))
    one_over_n = _seq_mean_token_mean(_weighted(split_loss, w), split_mask)
    assert one_over_n == pytest.approx(0.5)
    assert one_over_n != pytest.approx(target)


def test_tokens_over_mean_tokens_is_partition_preserving():
    unsplit_loss, unsplit_mask, split_loss, split_mask = _split_trajectory()
    target = _seq_mean_token_mean(unsplit_loss, unsplit_mask)

    tokens = split_mask.sum(-1)
    w = normalize_loss_weight_global(tokens / tokens.mean())
    assert _seq_mean_token_mean(_weighted(split_loss, w), split_mask) == pytest.approx(target)
