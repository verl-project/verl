# Copyright 2026 verl contributors
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
"""Off-policy sequence diagnostics must ignore all-masked padding rows."""

import math

import pytest
import torch

from verl.protocol import DataProto
from verl.trainer.ppo.padding_utils import construct_minimal_padding_template
from verl.trainer.ppo.rollout_corr_helper import (
    compute_offpolicy_metrics,
    compute_rollout_correction_and_add_to_batch,
    compute_rollout_correction_and_rejection_mask,
)


def _inputs(mask_dtype=torch.bool, gap=0.2):
    # Unequal valid lengths, including an internal tool-observation mask hole.
    old = torch.tensor([[-1.0, -7.0, -2.0, -7.0], [-3.0, -2.0, -1.0, -7.0]], dtype=torch.float64)
    rollout = old + gap
    mask = torch.tensor([[1, 0, 1, 0], [1, 1, 1, 0]], dtype=mask_dtype)
    return old, rollout, mask


def _expected(old, rollout, mask):
    # Independent scalar reference over selected tokens, then over nonempty rows.
    rows = [(a[m.bool()], b[m.bool()]) for a, b, m in zip(old, rollout, mask, strict=True) if m.any()]
    actor_means = [a.mean().item() for a, _ in rows]
    rollout_means = [b.mean().item() for _, b in rows]
    differences = [b - a for a, b in zip(actor_means, rollout_means, strict=True)]
    ratios = torch.cat([a - b for a, b in rows])
    avg = lambda values: sum(values) / len(values)
    return {
        "training_ppl": avg([math.exp(-x) for x in actor_means]),
        "training_log_ppl": -avg(actor_means),
        "rollout_ppl": avg([math.exp(-x) for x in rollout_means]),
        "rollout_log_ppl": -avg(rollout_means),
        "log_ppl_diff": avg(differences),
        "log_ppl_abs_diff": avg([abs(x) for x in differences]),
        "log_ppl_diff_max": max(differences),
        "log_ppl_diff_min": min(differences),
        "ppl_ratio": avg([math.exp(x) for x in differences]),
        "chi2_seq": avg([math.exp(2 * (a - b).sum().item()) for a, b in rows]) - 1,
        "kl": -ratios.mean().item(),
        "k3_kl": (ratios.exp() - ratios - 1).mean().item(),
        "chi2_token": ratios.mul(2).exp().mean().item() - 1,
    }


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.int64, torch.float32])
@pytest.mark.parametrize("gap", [0.2, -0.3])
@pytest.mark.parametrize("padding_count", [0, 1, 3])
def test_metrics_match_valid_sequence_reference(mask_dtype, gap, padding_count):
    old, rollout, mask = _inputs(mask_dtype, gap)
    expected = _expected(old, rollout, mask)
    old = torch.cat([old, torch.full((padding_count, 4), -3.0, dtype=old.dtype)])
    rollout = torch.cat([rollout, torch.full((padding_count, 4), -4.0, dtype=rollout.dtype)])
    mask = torch.cat([mask, torch.zeros((padding_count, 4), dtype=mask_dtype)])
    # Interleave padding with real rows rather than assuming it is a suffix.
    order = torch.tensor([0, *range(2, len(old)), 1])
    actual = compute_offpolicy_metrics(old[order], rollout[order], mask[order])
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)


def test_training_only_metrics_ignore_padding():
    old = torch.tensor([[-2.0, -2.0], [-9.0, -9.0]])
    mask = torch.tensor([[1, 1], [0, 0]])
    actual = compute_offpolicy_metrics(old, None, mask)
    assert actual == pytest.approx({"training_ppl": math.exp(2), "training_log_ppl": 2})


def test_v1_padding_template_through_batch_wrapper():
    source = {"response_mask": torch.ones(2, dtype=torch.int64)}
    padding, tag = construct_minimal_padding_template(source, {}, eos_token_id=2)
    assert tag["is_padding"] is True
    mask = torch.nn.utils.rnn.pad_sequence([source["response_mask"], padding["response_mask"]], batch_first=True)
    old = torch.full(mask.shape, -2.0)
    rollout = torch.full(mask.shape, -1.0)
    batch = DataProto.from_dict(tensors={"old_log_probs": old, "rollout_log_probs": rollout, "response_mask": mask})
    updated, metrics = compute_rollout_correction_and_add_to_batch(batch, {})
    assert metrics["rollout_corr/log_ppl_abs_diff"] == pytest.approx(1.0)
    assert metrics["rollout_corr/training_ppl"] == pytest.approx(math.exp(2))
    assert torch.equal(updated.batch["response_mask"], mask)
    assert "rollout_is_weights" not in updated.batch


def test_diagnostics_still_use_pre_rejection_mask_and_preserve_is_weights():
    old = torch.tensor([[-2.0, -2.0], [-1.0, -1.0], [-3.0, -3.0]])
    rollout = torch.full_like(old, -1.0)
    mask = torch.tensor([[1, 1], [1, 1], [0, 0]])
    before = [x.clone() for x in (old, rollout, mask)]
    weights, modified_mask, metrics = compute_rollout_correction_and_rejection_mask(
        old,
        rollout,
        mask,
        rollout_is="token",
        rollout_is_threshold=2.0,
        rollout_rs="token_k1",
        rollout_rs_threshold="0.5_2.0",
    )
    assert torch.equal(modified_mask, torch.tensor([[0, 0], [1, 1], [0, 0]]))
    assert metrics["rollout_corr/log_ppl_abs_diff"] == pytest.approx(0.5)
    expected_weights = torch.exp(old - rollout).clamp(max=2.0) * mask
    assert torch.allclose(weights.batch["rollout_is_weights"], expected_weights)
    for actual, original in zip((old, rollout, mask), before, strict=True):
        assert torch.equal(actual, original)


def test_all_masked_batch_keeps_existing_error_contract():
    old = torch.zeros(2, 3)
    mask = torch.zeros_like(old)
    with pytest.raises(AssertionError, match="Expected at least one valid token"):
        compute_offpolicy_metrics(old, old, mask)
    with pytest.raises(ValueError, match="at least one valid token"):
        compute_rollout_correction_and_rejection_mask(old, old, mask)
