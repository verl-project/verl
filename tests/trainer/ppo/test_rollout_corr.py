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
Quick Sanity Test for Rollout Correction

This is a standalone test script that can be run without pytest to quickly verify
the rollout correction implementation is working correctly. For comprehensive integration
tests, see: tests/trainer/ppo/test_rollout_corr_integration.py

Usage:
    python test_rollout_corr.py

This tests:
- Basic rollout correction functionality (IS weights + rejection sampling)
- Metrics completeness (IS metrics + rejection metrics + off-policy metrics)
- Edge cases
"""

import pytest
import torch

from verl.trainer.ppo.rollout_corr_helper import (
    SUPPORTED_ROLLOUT_RS_OPTIONS,
    compute_offpolicy_metrics,
    compute_rollout_correction_and_rejection_mask,
)


def test_basic_rollout_correction():
    """Test basic rollout correction functionality."""
    print("Testing basic rollout correction functionality...")

    # Create test data
    batch_size, seq_length = 4, 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create slightly different log probs (simulating BF16 vs FP32 mismatch)
    old_log_prob = torch.randn(batch_size, seq_length, device=device)
    rollout_log_prob = old_log_prob + torch.randn(batch_size, seq_length, device=device) * 0.1
    eos_mask = torch.ones(batch_size, seq_length, device=device)

    # Test token-level truncate mode
    print("\n1. Testing token-level truncate mode...")
    weights_proto, modified_response_mask, metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=eos_mask,
        rollout_is="token",  # Compute IS weights at token level
        rollout_is_threshold=2.0,
        rollout_rs=None,  # No rejection sampling (truncate mode)
    )

    weights = weights_proto.batch["rollout_is_weights"]
    print(f"   Weights shape: {weights.shape}")
    print(f"   Mean weight: {metrics['rollout_corr/rollout_is_mean']:.4f}")
    print(f"   Max weight: {metrics['rollout_corr/rollout_is_max']:.4f}")
    print(f"   Min weight: {metrics['rollout_corr/rollout_is_min']:.4f}")
    assert weights.shape == old_log_prob.shape
    assert weights.max() <= 2.0, "Weights should be capped at threshold"
    print("   ✓ Token-level truncate mode passed")

    # Test sequence-level mode
    print("\n2. Testing sequence-level mode...")
    weights_seq_proto, _, metrics_seq = compute_rollout_correction_and_rejection_mask(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=eos_mask,
        rollout_is="sequence",  # Compute IS weights at sequence level
        rollout_is_threshold=5.0,
        rollout_rs=None,  # No rejection sampling (truncate mode)
    )

    weights_seq = weights_seq_proto.batch["rollout_is_weights"]
    print(f"   Mean weight: {metrics_seq['rollout_corr/rollout_is_mean']:.4f}")
    print(f"   Effective sample size: {metrics_seq['rollout_corr/rollout_is_eff_sample_size']:.4f}")
    # Check that all tokens in a sequence have the same weight
    for i in range(batch_size):
        seq_weights = weights_seq[i, eos_mask[i].bool()]
        assert torch.allclose(seq_weights, seq_weights[0]), "All tokens in sequence should have same weight"
    print("   ✓ Sequence-level mode passed")

    # Test K1 sequence mean rejection sampling (mask mode)
    print("\n3. Testing K1 (sequence mean) rejection sampling...")
    weights_geo_proto, modified_mask_geo, metrics_geo = compute_rollout_correction_and_rejection_mask(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=eos_mask,
        rollout_is=None,  # No IS weights (pure mask mode)
        rollout_rs="seq_mean_k1",  # Rejection sampling with sequence-mean log ratio bounds
        rollout_rs_threshold="0.5_1.5",
    )

    print(f"   Masked fraction: {metrics_geo['rollout_corr/rollout_rs_masked_fraction']:.4f}")
    print("   ✓ K1 sequence mean rejection sampling passed")

    # Test disabled IS (rollout_is=None, rollout_rs=None)
    print("\n4. Testing disabled IS...")
    weights_disabled, modified_response_mask_disabled, metrics_disabled = compute_rollout_correction_and_rejection_mask(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=eos_mask,
        rollout_is=None,
        rollout_rs=None,
    )

    assert weights_disabled is None, "Should return None when IS is disabled"
    assert torch.equal(modified_response_mask_disabled, eos_mask), "Should return original mask unchanged"
    # Note: off-policy metrics are still computed even when IS/RS are disabled
    assert "rollout_corr/kl" in metrics_disabled, "Should still compute off-policy metrics"
    print("   ✓ Disabled IS passed")

    print("\n✓ All tests passed!")


@pytest.mark.parametrize(
    ("option", "threshold"),
    [
        ("token_k1", "0.5_1.5"),
        ("token_k2", 2.0),
        ("token_k3", 2.0),
        ("seq_sum_k1", "0.6_1.4"),
        ("seq_sum_k2", 2.5),
        ("seq_sum_k3", 2.5),
        ("seq_mean_k1", "0.5_1.5"),
        ("seq_mean_k2", 2.0),
        ("seq_mean_k3", 2.0),
        ("seq_max_k2", 2.0),
        ("seq_max_k3", 2.0),
    ],
)
def test_each_supported_rollout_rs_option(option: str, threshold):
    """Ensure every supported RS option produces metrics without error."""
    assert option in SUPPORTED_ROLLOUT_RS_OPTIONS

    batch_size, seq_length = 3, 7
    device = "cuda" if torch.cuda.is_available() else "cpu"

    old_log_prob = torch.randn(batch_size, seq_length, device=device)
    rollout_log_prob = old_log_prob + torch.randn(batch_size, seq_length, device=device) * 0.15
    response_mask = torch.ones(batch_size, seq_length, device=device)

    _, modified_mask, metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
        rollout_is=None,
        rollout_rs=option,
        rollout_rs_threshold=threshold,
    )

    expected_key = f"rollout_corr/rollout_rs_{option}_mean"
    assert expected_key in metrics, f"Missing metric for {option}"
    assert modified_mask.shape == response_mask.shape


def test_rollout_rs_multiple_options():
    """Verify multiple RS options with mixed threshold formats."""
    batch_size, seq_length = 2, 6
    device = "cuda" if torch.cuda.is_available() else "cpu"

    old_log_prob = torch.randn(batch_size, seq_length, device=device)
    rollout_log_prob = old_log_prob + torch.randn(batch_size, seq_length, device=device) * 0.2
    response_mask = torch.ones(batch_size, seq_length, device=device)

    rollout_rs = "token_k1,seq_max_k3"
    rollout_rs_threshold = "0.4_1.8,3.0"

    _, _, metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
        rollout_is=None,
        rollout_rs=rollout_rs,
        rollout_rs_threshold=rollout_rs_threshold,
    )

    for option in rollout_rs.split(","):
        key = f"rollout_corr/rollout_rs_{option}_mean"
        assert key in metrics, f"Metrics missing for chained option {option}"


def test_metrics_completeness():
    """Test that all expected metrics are returned."""
    print("\nTesting metrics completeness...")

    batch_size, seq_length = 3, 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    old_log_prob = torch.randn(batch_size, seq_length, device=device)
    rollout_log_prob = old_log_prob + torch.randn(batch_size, seq_length, device=device) * 0.2
    eos_mask = torch.ones(batch_size, seq_length, device=device)

    _, _, metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=eos_mask,
        rollout_is="token",
        rollout_is_threshold=2.5,
        rollout_rs=None,
    )

    # Expected IS metrics
    expected_is_metrics = [
        "rollout_corr/rollout_is_mean",
        "rollout_corr/rollout_is_max",
        "rollout_corr/rollout_is_min",
        "rollout_corr/rollout_is_std",
        "rollout_corr/rollout_is_eff_sample_size",
        "rollout_corr/rollout_is_ratio_fraction_high",
        "rollout_corr/rollout_is_ratio_fraction_low",
    ]

    # Expected off-policy diagnostic metrics (also included now)
    expected_offpolicy_metrics = [
        "rollout_corr/training_ppl",
        "rollout_corr/training_log_ppl",
        "rollout_corr/kl",
        "rollout_corr/k3_kl",
        "rollout_corr/rollout_ppl",
        "rollout_corr/rollout_log_ppl",
        "rollout_corr/log_ppl_diff",
        "rollout_corr/log_ppl_abs_diff",
        "rollout_corr/log_ppl_diff_max",
        "rollout_corr/log_ppl_diff_min",
        "rollout_corr/ppl_ratio",
        "rollout_corr/chi2_token",
        "rollout_corr/chi2_seq",
    ]

    expected_metrics = expected_is_metrics + expected_offpolicy_metrics

    missing_metrics = [m for m in expected_metrics if m not in metrics]
    if missing_metrics:
        print(f"   ✗ Missing metrics: {missing_metrics}")
        return False

    print(f"   ✓ All {len(expected_metrics)} expected metrics present")
    print(f"   Total metrics returned: {len(metrics)}")
    return True


def test_offpolicy_metrics():
    """Test off-policy metrics computation."""
    print("\nTesting off-policy metrics computation...")

    batch_size, seq_length = 4, 12
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test data with some mismatch
    old_log_prob = torch.randn(batch_size, seq_length, device=device) - 2.0  # training policy
    rollout_log_prob = torch.randn(batch_size, seq_length, device=device) - 1.5  # rollout policy (more confident)
    response_mask = torch.ones(batch_size, seq_length, device=device)

    # Test with rollout log probs
    metrics = compute_offpolicy_metrics(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
    )

    expected_metrics = [
        "training_ppl",
        "training_log_ppl",
        "kl",
        "k3_kl",
        "rollout_ppl",
        "rollout_log_ppl",
        "log_ppl_diff",
        "log_ppl_abs_diff",
        "log_ppl_diff_max",
        "log_ppl_diff_min",
        "ppl_ratio",
        "chi2_token",
        "chi2_seq",
    ]

    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"

    print(f"   Training PPL: {metrics['training_ppl']:.4f}")
    print(f"   Rollout PPL: {metrics['rollout_ppl']:.4f}")
    print(f"   KL divergence: {metrics['kl']:.6f}")
    print(f"   K3 KL: {metrics['k3_kl']:.6f}")
    print(f"   PPL ratio: {metrics['ppl_ratio']:.4f}")
    print(f"   ✓ All {len(expected_metrics)} off-policy metrics present")

    # Test without rollout log probs
    metrics_no_rollout = compute_offpolicy_metrics(
        old_log_prob=old_log_prob,
        rollout_log_prob=None,
        response_mask=response_mask,
    )

    assert "training_ppl" in metrics_no_rollout
    assert "rollout_ppl" not in metrics_no_rollout
    print("   ✓ Off-policy metrics work without rollout log probs")


def test_mask_mode():
    """Test mask mode applies rejection via response_mask, keeps true IS weights."""
    print("\nTesting mask mode behavior...")

    batch_size = 2
    seq_length = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sequence 0: ratio ≈ 0.37 (below 0.5, should be rejected)
    # Sequence 1: ratio ≈ 1.65 (in [0.5, 2.0], should be accepted)
    old_log_prob = torch.tensor([[-2.0] * seq_length, [-2.0] * seq_length], device=device)
    rollout_log_prob = torch.tensor(
        [
            [-1.0] * seq_length,  # exp(-2.0 - (-1.0)) = exp(-1.0) ≈ 0.37
            [-2.5] * seq_length,  # exp(-2.0 - (-2.5)) = exp(0.5) ≈ 1.65
        ],
        device=device,
    )
    response_mask = torch.ones(batch_size, seq_length, device=device)

    weights_proto, modified_response_mask, metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
        rollout_is="token",  # Compute IS weights
        rollout_is_threshold=2.0,
        rollout_rs="token_k1",  # Also apply rejection sampling (mask mode)
        rollout_rs_threshold="0.5_2.0",
    )

    weights = weights_proto.batch["rollout_is_weights"]

    # KEY FIX: Weights should be safety-bounded ratios (NOT zeroed)
    assert torch.all(weights[0, :] > 0), "Weights should remain as safety-bounded ratios (not zeroed)"
    assert torch.allclose(weights[0, 0], torch.tensor(0.368, device=device), atol=0.01), (
        "First seq ratio should be ≈0.37"
    )
    assert torch.allclose(weights[1, 0], torch.tensor(1.649, device=device), atol=0.01), (
        "Second seq ratio should be ≈1.65"
    )

    # Rejection should be applied via response_mask
    assert torch.all(modified_response_mask[0, :] == 0), "First sequence should be rejected via mask"
    assert torch.all(modified_response_mask[1, :] == 1), "Second sequence should be accepted"

    # Verify rejection sampling metrics exist
    assert "rollout_corr/rollout_rs_masked_fraction" in metrics, "Should have rollout_rs_masked_fraction metric"
    assert abs(metrics["rollout_corr/rollout_rs_masked_fraction"] - 0.5) < 0.01, "Should reject 50% of tokens"

    print(f"   First seq IS weight: {weights[0, 0]:.4f} (expected ≈0.37)")
    print(f"   Second seq IS weight: {weights[1, 0]:.4f} (expected ≈1.65)")
    print(f"   First seq mask: {modified_response_mask[0, 0]:.0f} (expected 0 - rejected)")
    print(f"   Second seq mask: {modified_response_mask[1, 0]:.0f} (expected 1 - accepted)")
    print(f"   Masked fraction: {metrics['rollout_corr/rollout_rs_masked_fraction']:.2f}")
    print("   ✓ Mask mode correctly separates IS weights from rejection")


if __name__ == "__main__":
    print("=" * 60)
    print("Rollout Correction Test Suite")
    print("=" * 60)

    try:
        test_basic_rollout_correction()
        test_metrics_completeness()
        test_offpolicy_metrics()
        test_mask_mode()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
