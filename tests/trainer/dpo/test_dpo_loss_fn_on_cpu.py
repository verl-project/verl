# Copyright 2026 Amazon.com Inc and/or its affiliates
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
Tests for the DPO loss passed to TrainingWorker (verl.workers.utils.losses.dpo_loss) and for
sum_logps_per_sequence.

The scaling tests simulate data-parallel ranks and micro-batches the way the engine combines them:
the reported loss is the sum over a rank's micro-batches, averaged over ranks, and the parameter
gradient is averaged over ranks. Both must equal a single-device step over the whole batch.
"""

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.dpo.dpo_losses import DPOLossConfig
from verl.trainer.dpo.utils import sum_logps_per_sequence
from verl.utils import tensordict_utils as tu
from verl.workers.utils.losses import dpo_loss

# ─── Helpers ─────────────────────────────────────────────────────────────


def _nested(tensors):
    return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)


def _make_pairs(num_pairs, seed=0):
    """Per-sequence (log_probs leaf, loss_mask) for 2 * num_pairs interleaved sequences, plus ref log-probs."""
    gen = torch.Generator().manual_seed(seed)
    seqs = []
    for _ in range(2 * num_pairs):
        prompt_len = int(torch.randint(2, 6, (1,), generator=gen))
        response_len = int(torch.randint(1, 8, (1,), generator=gen))
        mask = torch.cat([torch.zeros(prompt_len), torch.ones(response_len)]).long()
        log_probs = (-torch.rand(prompt_len + response_len, generator=gen) * 3).requires_grad_(True)
        seqs.append((log_probs, mask))
    ref_logps = -torch.rand(2 * num_pairs, generator=gen) * 10
    return seqs, ref_logps


def _micro_batch(seqs, ref_logps, rows, dp_size, global_pairs):
    """Build (model_output, data) for the interleaved rows `rows` (even = chosen, odd = rejected)."""
    lengths = torch.tensor([seqs[i][0].shape[0] for i in rows])
    values = torch.cat([seqs[i][0] for i in rows])
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
    log_probs = torch.nested.nested_tensor_from_jagged(values, offsets)
    data = TensorDict(
        {"loss_mask": _nested([seqs[i][1] for i in rows]), "ref_logps": ref_logps[rows]},
        batch_size=[len(rows)],
    )
    tu.assign_non_tensor(data, pad_mode="no_padding", dp_size=dp_size, global_batch_size=global_pairs)
    return {"log_probs": log_probs}, data


def _pair_rows(pairs):
    return [row for p in pairs for row in (2 * p, 2 * p + 1)]


CONFIG = DPOLossConfig(loss_type="sigmoid", beta=0.1)


# ─── sum_logps_per_sequence ──────────────────────────────────────────────


class TestSumLogpsPerSequence:
    def test_exact_values_with_shift(self):
        # log_probs[t] scores token t + 1, so a token with mask 1 at position t is read from t - 1.
        log_probs = _nested([torch.tensor([-1.0, -2.0, -3.0, -4.0]), torch.tensor([-5.0, -6.0, -7.0])])
        loss_mask = _nested([torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 1])])
        result = sum_logps_per_sequence(log_probs, loss_mask)
        assert result.tolist() == [-2.0 + -3.0, -5.0 + -6.0]

    def test_last_position_never_counted(self):
        # The second sequence starts with a response token (e.g. after left truncation). Its mask must not
        # leak onto the first sequence's last position, which predicts past the end of that sequence.
        log_probs = _nested([torch.tensor([-1.0, -2.0, -100.0]), torch.tensor([-3.0, -4.0])])
        loss_mask = _nested([torch.tensor([0, 1, 1]), torch.tensor([1, 1])])
        result = sum_logps_per_sequence(log_probs, loss_mask)
        assert result.tolist() == [-1.0 + -2.0, -3.0]

    def test_all_masked_is_zero(self):
        log_probs = _nested([torch.tensor([-1.0, -2.0]), torch.tensor([-3.0])])
        loss_mask = _nested([torch.tensor([0, 0]), torch.tensor([0])])
        assert sum_logps_per_sequence(log_probs, loss_mask).tolist() == [0.0, 0.0]

    def test_gradient_reaches_only_scored_positions(self):
        values = torch.tensor([-1.0, -2.0, -3.0, -4.0], requires_grad=True)
        log_probs = torch.nested.nested_tensor_from_jagged(values, torch.tensor([0, 4]))
        sum_logps_per_sequence(log_probs, _nested([torch.tensor([0, 0, 1, 1])])).sum().backward()
        assert values.grad.tolist() == [0.0, 1.0, 1.0, 0.0]

    def test_padded_input_rejected(self):
        with pytest.raises(NotImplementedError):
            sum_logps_per_sequence(torch.zeros(2, 3), torch.ones(2, 3))


# ─── dpo_loss scaling ────────────────────────────────────────────────────


def _single_device(seqs, ref_logps, num_pairs):
    model_output, data = _micro_batch(seqs, ref_logps, _pair_rows(range(num_pairs)), dp_size=1, global_pairs=num_pairs)
    return dpo_loss(CONFIG, model_output, data)


def _simulated_parallel(seqs, ref_logps, rank_micro_batches, num_pairs):
    """rank_micro_batches: per rank, a list of micro-batches, each a list of pair indices."""
    dp_size = len(rank_micro_batches)
    rank_losses, rank_metrics, total = [], [], 0.0
    for micro_batches in rank_micro_batches:
        losses, metrics = [], []
        for pairs in micro_batches:
            model_output, data = _micro_batch(seqs, ref_logps, _pair_rows(pairs), dp_size, num_pairs)
            loss, m = dpo_loss(CONFIG, model_output, data)
            losses.append(loss)
            metrics.append(m)
        rank_losses.append(sum(loss.item() for loss in losses))
        rank_metrics.append({k: sum(m[k] for m in metrics) for k in metrics[0]})
        # Each rank back-propagates its own micro-batch losses; FSDP then averages gradients over ranks.
        total = total + sum(losses) / dp_size
    total.backward()
    reported_loss = sum(rank_losses) / dp_size
    reported_metrics = {k: sum(m[k] for m in rank_metrics) / dp_size for k in rank_metrics[0]}
    return reported_loss, reported_metrics


@pytest.mark.parametrize(
    "rank_micro_batches",
    [
        [[[0, 1, 2, 3, 4, 5, 6, 7]]],
        [[[0, 1], [2, 3]], [[4], [5, 6, 7]]],
        [[[0]], [[1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]]],
        [[[3, 0, 6], [5]], [[7, 2], [1, 4]]],
    ],
    ids=["one-device", "2-ranks-uneven-micro-batches", "8-ranks", "2-ranks-reordered"],
)
def test_loss_metrics_and_gradient_match_single_device(rank_micro_batches):
    num_pairs = 8
    seqs, ref_logps = _make_pairs(num_pairs)
    loss, metrics = _single_device(seqs, ref_logps, num_pairs)
    loss.backward()
    expected_grads = [lp.grad.clone() for lp, _ in seqs]
    for lp, _ in seqs:
        lp.grad = None

    reported_loss, reported_metrics = _simulated_parallel(seqs, ref_logps, rank_micro_batches, num_pairs)

    assert reported_loss == pytest.approx(loss.item(), rel=1e-6)
    for key, value in metrics.items():
        assert reported_metrics[key] == pytest.approx(value, rel=1e-5, abs=1e-7), key
    for (lp, _), expected in zip(seqs, expected_grads, strict=True):
        torch.testing.assert_close(lp.grad, expected)


def test_single_device_loss_is_plain_mean():
    num_pairs = 4
    seqs, ref_logps = _make_pairs(num_pairs, seed=1)
    loss, metrics = _single_device(seqs, ref_logps, num_pairs)
    assert metrics["dpo/loss"] == pytest.approx(loss.item())
    assert 0.0 <= metrics["dpo/accuracy"] <= 1.0


# ─── dpo_loss inputs ─────────────────────────────────────────────────────


def test_log_probs_removed_from_model_output():
    seqs, ref_logps = _make_pairs(2)
    model_output, data = _micro_batch(seqs, ref_logps, _pair_rows([0, 1]), dp_size=1, global_pairs=2)
    dpo_loss(CONFIG, model_output, data)
    assert "log_probs" not in model_output


def test_reference_free_without_ref_logps():
    seqs, ref_logps = _make_pairs(2)
    model_output, data = _micro_batch(seqs, ref_logps, _pair_rows([0, 1]), dp_size=1, global_pairs=2)
    del data["ref_logps"]
    loss, _ = dpo_loss(DPOLossConfig(reference_free=True), model_output, data)
    assert torch.isfinite(loss)


def test_odd_row_count_rejected():
    seqs, ref_logps = _make_pairs(2)
    model_output, data = _micro_batch(seqs, ref_logps, [0, 1, 2], dp_size=1, global_pairs=2)
    with pytest.raises(ValueError, match="whole"):
        dpo_loss(CONFIG, model_output, data)


def test_missing_global_batch_size_rejected():
    seqs, ref_logps = _make_pairs(1)
    model_output, data = _micro_batch(seqs, ref_logps, [0, 1], dp_size=1, global_pairs=1)
    tu.assign_non_tensor(data, global_batch_size=None)
    with pytest.raises(ValueError, match="global_batch_size"):
        dpo_loss(CONFIG, model_output, data)


def test_padded_mode_rejected():
    seqs, ref_logps = _make_pairs(1)
    model_output, data = _micro_batch(seqs, ref_logps, [0, 1], dp_size=1, global_pairs=1)
    tu.assign_non_tensor(data, pad_mode="right")
    with pytest.raises(NotImplementedError):
        dpo_loss(CONFIG, model_output, data)
