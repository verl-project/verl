#!/usr/bin/env python3
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
Unit tests for the ``binary_kl`` (KPop) rejection sampling option.

KPop applies a hard trust region using the bidirectional Bernoulli KL divergence
between the training policy and the rollout policy: a token is kept only when
``max(KL(train||rollout), KL(rollout||train)) <= phi``.

This covers:
- ``compute_binary_kl_divergence`` numerics (self-KL = 0, no NaN at the p=1 boundary,
  agreement with a manual reference value).
- ``binary_kl`` registration as a token-level rejection option.
- ``compute_rollout_rejection_mask`` directly: high-divergence tokens are rejected,
  matched tokens are kept, and missing log-probs raise a clear error.
- The unified ``compute_rollout_correction_and_rejection_mask`` entry point and the
  ``RolloutCorrectionConfig.decoupled_token_kpop`` preset.

Usage:
    python test_binary_kl_rejection.py
"""

import math

import pytest
import torch

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.trainer.ppo.rollout_corr_helper import (
    SUPPORTED_ROLLOUT_RS_OPTIONS,
    TOKEN_LEVEL_ROLLOUT_RS_OPTIONS,
    compute_binary_kl_divergence,
    compute_rollout_correction_and_rejection_mask,
    compute_rollout_rejection_mask,
)


def test_binary_kl_self_divergence_is_zero():
    """KL(P||P) must be exactly zero for identical distributions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_p = torch.log(torch.tensor([0.1, 0.5, 0.9, 0.99], device=device))
    kl = compute_binary_kl_divergence(log_p, log_p)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_binary_kl_no_nan_at_probability_one_boundary():
    """log_q == 0 (q == 1.0) must not produce NaN/Inf.

    Without upcasting to float32 and clamping with eps, ``1 - q`` rounds to exactly
    0.0 in float32 and the KL term ``log((1 - p) / (1 - q))`` becomes NaN. This is the
    exact failure mode flagged in review for the original implementation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_p = torch.tensor([-0.1, -2.0, 0.0, -5.0], device=device)
    log_q = torch.tensor([0.0, -0.1, 0.0, -0.2], device=device)  # includes q == 1.0
    kl_fwd = compute_binary_kl_divergence(log_p, log_q)
    kl_rev = compute_binary_kl_divergence(log_q, log_p)
    for kl in (kl_fwd, kl_rev):
        assert not torch.isnan(kl).any(), "binary KL produced NaN at the p=1 boundary"
        assert not torch.isinf(kl).any(), "binary KL produced Inf at the p=1 boundary"
        assert (kl >= 0).all(), "Bernoulli KL must be non-negative"


def test_binary_kl_matches_reference_value():
    """Spot-check against a hand-computed Bernoulli KL value."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p, q = 0.8, 0.5
    expected = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
    log_p = torch.log(torch.tensor([p], device=device))
    log_q = torch.log(torch.tensor([q], device=device))
    kl = compute_binary_kl_divergence(log_p, log_q)
    assert kl.item() == pytest.approx(expected, abs=1e-5)


def test_binary_kl_dtype_preserved():
    """Output dtype matches the input dtype even though math runs in float32."""
    log_p = torch.log(torch.tensor([0.6, 0.4], dtype=torch.bfloat16))
    log_q = torch.log(torch.tensor([0.5, 0.5], dtype=torch.bfloat16))
    kl = compute_binary_kl_divergence(log_p, log_q)
    assert kl.dtype == torch.bfloat16


def test_binary_kl_registered_as_token_level_option():
    """binary_kl must be a recognized, token-level rejection option."""
    assert "binary_kl" in SUPPORTED_ROLLOUT_RS_OPTIONS
    assert "binary_kl" in TOKEN_LEVEL_ROLLOUT_RS_OPTIONS


def test_binary_kl_rejects_high_divergence_tokens():
    """A token whose bidirectional KL exceeds phi is masked; a matched token is kept."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Token 0: identical policies -> KL = 0 -> kept.
    # Token 1: p=0.99 vs q=0.5 -> max(KL_fwd, KL_rev) ~= 1.61 -> rejected at phi=1.0.
    old_log_prob = torch.log(torch.tensor([[0.5, 0.99]], device=device))
    rollout_log_prob = torch.log(torch.tensor([[0.5, 0.50]], device=device))
    response_mask = torch.ones_like(old_log_prob)
    log_ratio = old_log_prob - rollout_log_prob

    modified_mask, metrics = compute_rollout_rejection_mask(
        log_ratio=log_ratio,
        response_mask=response_mask,
        rollout_rs="binary_kl",
        rollout_rs_threshold=1.0,
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
    )

    assert modified_mask[0, 0].item() == 1, "matched token should be kept"
    assert modified_mask[0, 1].item() == 0, "high-divergence token should be rejected"
    assert metrics["rollout_rs_binary_kl_masked_fraction"] == pytest.approx(0.5, abs=1e-6)


def test_binary_kl_keeps_everything_under_loose_threshold():
    """With a large phi every token survives and the mask is unchanged."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    old_log_prob = torch.randn(3, 7, device=device)
    rollout_log_prob = old_log_prob + torch.randn(3, 7, device=device) * 0.1
    response_mask = torch.ones_like(old_log_prob)

    modified_mask, _ = compute_rollout_rejection_mask(
        log_ratio=old_log_prob - rollout_log_prob,
        response_mask=response_mask,
        rollout_rs="binary_kl",
        rollout_rs_threshold=1e6,
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
    )
    assert torch.equal(modified_mask, response_mask)


def test_binary_kl_requires_logprobs():
    """binary_kl needs the raw log-probs; omitting them must raise a clear error."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_ratio = torch.randn(2, 4, device=device)
    response_mask = torch.ones_like(log_ratio)

    with pytest.raises(ValueError, match="binary_kl"):
        compute_rollout_rejection_mask(
            log_ratio=log_ratio,
            response_mask=response_mask,
            rollout_rs="binary_kl",
            rollout_rs_threshold=2.0,
        )


def test_binary_kl_through_unified_entrypoint():
    """End-to-end through compute_rollout_correction_and_rejection_mask."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    old_log_prob = torch.randn(4, 8, device=device)
    rollout_log_prob = old_log_prob + torch.randn(4, 8, device=device) * 0.15
    response_mask = torch.ones_like(old_log_prob)

    _, modified_mask, metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
        rollout_is=None,
        rollout_rs="binary_kl",
        rollout_rs_threshold=2.0,
    )

    assert modified_mask.shape == response_mask.shape
    assert "rollout_corr/rollout_rs_binary_kl_mean" in metrics
    assert "rollout_corr/rollout_rs_binary_kl_masked_fraction" in metrics


def test_decoupled_token_kpop_preset():
    """The convenience preset wires binary_kl with phi and disables IS weights."""
    cfg = RolloutCorrectionConfig.decoupled_token_kpop(phi=2.5)
    assert cfg.rollout_rs == "binary_kl"
    assert cfg.rollout_rs_threshold == 2.5
    assert cfg.rollout_is is None


if __name__ == "__main__":
    print("=" * 60)
    print("Binary KL (KPop) Rejection Sampling Test Suite")
    print("=" * 60)

    try:
        test_binary_kl_self_divergence_is_zero()
        test_binary_kl_no_nan_at_probability_one_boundary()
        test_binary_kl_matches_reference_value()
        test_binary_kl_dtype_preserved()
        test_binary_kl_registered_as_token_level_option()
        test_binary_kl_rejects_high_divergence_tokens()
        test_binary_kl_keeps_everything_under_loose_threshold()
        test_binary_kl_requires_logprobs()
        test_binary_kl_through_unified_entrypoint()
        test_decoupled_token_kpop_preset()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)