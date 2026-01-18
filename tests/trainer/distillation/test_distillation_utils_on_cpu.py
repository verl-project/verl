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

import unittest

import pytest
import torch

from verl.trainer.distillation.utils import (
    Stage,
    get_topk_keys,
    topk_logprobs_from_logits,
)


class TestStageEnum(unittest.TestCase):
    """Test the Stage enum."""

    def test_stage_values(self):
        """Test that Stage enum has expected values."""
        self.assertEqual(Stage.OLD_LOG_PROB.value, "old_log_prob")
        self.assertEqual(Stage.REF_LOG_PROB.value, "ref_log_prob")
        self.assertEqual(Stage.ACTOR_UPDATE.value, "actor_update")


class TestGetTopkKeys(unittest.TestCase):
    """Test the get_topk_keys function."""

    def test_with_stage_enum(self):
        """Test get_topk_keys with Stage enum."""
        logprobs_key, indices_key = get_topk_keys(Stage.OLD_LOG_PROB)
        self.assertEqual(logprobs_key, "old_log_prob_topk_log_probs")
        self.assertEqual(indices_key, "old_log_prob_topk_indices")

    def test_with_string(self):
        """Test get_topk_keys with string stage."""
        logprobs_key, indices_key = get_topk_keys("ref_log_prob")
        self.assertEqual(logprobs_key, "ref_log_prob_topk_log_probs")
        self.assertEqual(indices_key, "ref_log_prob_topk_indices")

    def test_all_stages(self):
        """Test get_topk_keys for all Stage enum values."""
        for stage in Stage:
            logprobs_key, indices_key = get_topk_keys(stage)
            self.assertTrue(logprobs_key.endswith("_topk_log_probs"))
            self.assertTrue(indices_key.endswith("_topk_indices"))
            self.assertTrue(logprobs_key.startswith(stage.value))
            self.assertTrue(indices_key.startswith(stage.value))


class TestTopkLogprobsFromLogits(unittest.TestCase):
    """Test the topk_logprobs_from_logits function."""

    def test_compute_topk_only(self):
        """Test computing top-k from logits without gathering."""
        torch.manual_seed(42)
        batch_size = 4
        seq_len = 16
        vocab_size = 100
        k = 10

        logits = torch.randn(batch_size * seq_len, vocab_size)

        topk_logprobs, topk_indices = topk_logprobs_from_logits(
            logits=logits,
            k=k,
            compute_topk=True,
            gather_topk=False,
            topk_indices=None,
        )

        self.assertEqual(topk_logprobs.shape, (batch_size * seq_len, k))
        self.assertEqual(topk_indices.shape, (batch_size * seq_len, k))

        # Verify that indices are within vocab range
        self.assertTrue((topk_indices >= 0).all())
        self.assertTrue((topk_indices < vocab_size).all())

        # Verify that logprobs are properly computed (should be log_softmax values)
        expected_logprobs = torch.log_softmax(logits, dim=-1)
        for i in range(batch_size * seq_len):
            for j in range(k):
                idx = topk_indices[i, j]
                torch.testing.assert_close(
                    topk_logprobs[i, j], expected_logprobs[i, idx], atol=1e-5, rtol=1e-5
                )

    def test_gather_topk_only(self):
        """Test gathering log probs at provided indices."""
        torch.manual_seed(42)
        batch_size = 4
        seq_len = 16
        vocab_size = 100
        k = 10

        logits = torch.randn(batch_size * seq_len, vocab_size)
        # Pre-computed indices to gather
        topk_indices_input = torch.randint(0, vocab_size, (batch_size * seq_len, k))

        topk_logprobs, topk_indices = topk_logprobs_from_logits(
            logits=logits,
            k=k,
            compute_topk=False,
            gather_topk=True,
            topk_indices=topk_indices_input,
        )

        self.assertEqual(topk_logprobs.shape, (batch_size * seq_len, k))
        self.assertEqual(topk_indices.shape, (batch_size * seq_len, k))

        # Indices should be the same as input
        torch.testing.assert_close(topk_indices, topk_indices_input)

        # Verify gathered logprobs
        expected_logprobs = torch.log_softmax(logits, dim=-1)
        gathered = torch.gather(expected_logprobs, dim=-1, index=topk_indices_input)
        torch.testing.assert_close(topk_logprobs, gathered, atol=1e-5, rtol=1e-5)

    def test_compute_and_gather_topk(self):
        """Test both computing and gathering top-k (for JSD with both student and teacher)."""
        torch.manual_seed(42)
        batch_size = 4
        seq_len = 16
        vocab_size = 100
        k = 10

        logits = torch.randn(batch_size * seq_len, vocab_size)
        # Pre-computed indices from another model (e.g., student top-k)
        other_topk_indices = torch.randint(0, vocab_size, (batch_size * seq_len, k))

        topk_logprobs, topk_indices = topk_logprobs_from_logits(
            logits=logits,
            k=k,
            compute_topk=True,
            gather_topk=True,
            topk_indices=other_topk_indices,
        )

        # Should have 2*k logprobs (k from gather + k from compute)
        self.assertEqual(topk_logprobs.shape, (batch_size * seq_len, 2 * k))
        self.assertEqual(topk_indices.shape, (batch_size * seq_len, 2 * k))

    def test_gather_without_indices_raises_error(self):
        """Test that gather_topk=True without indices raises error."""
        logits = torch.randn(4, 100)

        with self.assertRaises(ValueError) as cm:
            topk_logprobs_from_logits(
                logits=logits,
                k=10,
                compute_topk=False,
                gather_topk=True,
                topk_indices=None,
            )
        self.assertIn("topk_indices to be provided", str(cm.exception))

    def test_indices_provided_without_gather_raises_error(self):
        """Test that providing indices without gather_topk=True raises error."""
        logits = torch.randn(4, 100)
        indices = torch.randint(0, 100, (4, 10))

        with self.assertRaises(ValueError) as cm:
            topk_logprobs_from_logits(
                logits=logits,
                k=10,
                compute_topk=True,
                gather_topk=False,
                topk_indices=indices,
            )
        self.assertIn("None when gather_topk is False", str(cm.exception))

    def test_wrong_indices_shape_raises_error(self):
        """Test that indices with wrong shape raise error."""
        logits = torch.randn(4, 100)
        k = 10
        wrong_k = 5  # Neither k nor 2*k
        indices = torch.randint(0, 100, (4, wrong_k))

        with self.assertRaises(ValueError) as cm:
            topk_logprobs_from_logits(
                logits=logits,
                k=k,
                compute_topk=False,
                gather_topk=True,
                topk_indices=indices,
            )
        self.assertIn("shape", str(cm.exception).lower())

    def test_deduplication_with_overlapping_topk(self):
        """Test that overlapping indices between computed and gathered top-k are deduplicated."""
        torch.manual_seed(42)
        vocab_size = 100
        k = 10

        # Create logits where top-k will have specific indices
        logits = torch.zeros(1, vocab_size)
        top_indices = list(range(k))
        for i, idx in enumerate(top_indices):
            logits[0, idx] = 10 - i  # Make these the top-k

        # Provide same indices for gathering (complete overlap)
        topk_indices_input = torch.tensor([top_indices])

        topk_logprobs, topk_indices = topk_logprobs_from_logits(
            logits=logits,
            k=k,
            compute_topk=True,
            gather_topk=True,
            topk_indices=topk_indices_input,
        )

        # Should have 2*k entries but duplicates should have -inf logprobs
        self.assertEqual(topk_logprobs.shape, (1, 2 * k))

        # Count non-inf logprobs - should be exactly k due to complete overlap
        non_inf_count = (topk_logprobs > float("-inf")).sum().item()
        self.assertEqual(non_inf_count, k)

    def test_deduplication_with_no_overlap(self):
        """Test that non-overlapping indices are not deduplicated."""
        torch.manual_seed(42)
        vocab_size = 100
        k = 10

        # Create logits where top-k will have specific indices
        logits = torch.zeros(1, vocab_size)
        for i in range(k):
            logits[0, i] = 10 - i  # Indices 0-9 are top-k

        # Provide completely different indices for gathering
        topk_indices_input = torch.tensor([[i for i in range(k, 2 * k)]])  # Indices 10-19

        topk_logprobs, topk_indices = topk_logprobs_from_logits(
            logits=logits,
            k=k,
            compute_topk=True,
            gather_topk=True,
            topk_indices=topk_indices_input,
        )

        # Should have 2*k entries with no duplicates
        self.assertEqual(topk_logprobs.shape, (1, 2 * k))

        # All logprobs should be finite (no -inf from deduplication)
        non_inf_count = (topk_logprobs > float("-inf")).sum().item()
        self.assertEqual(non_inf_count, 2 * k)

    def test_topk_logprobs_are_sorted(self):
        """Test that computed top-k logprobs are in descending order."""
        torch.manual_seed(42)
        logits = torch.randn(8, 100)
        k = 10

        topk_logprobs, _ = topk_logprobs_from_logits(
            logits=logits,
            k=k,
            compute_topk=True,
            gather_topk=False,
            topk_indices=None,
        )

        # Check that logprobs are in descending order
        for i in range(topk_logprobs.shape[0]):
            diffs = topk_logprobs[i, :-1] - topk_logprobs[i, 1:]
            self.assertTrue((diffs >= -1e-6).all(), f"Row {i} not sorted: {topk_logprobs[i]}")


@pytest.mark.parametrize(
    "batch_size,seq_len,vocab_size,k",
    [
        (1, 1, 10, 5),
        (4, 16, 100, 10),
        (8, 32, 1000, 50),
        (16, 64, 10000, 128),
    ],
)
def test_topk_logprobs_shapes(batch_size: int, seq_len: int, vocab_size: int, k: int):
    """Test topk_logprobs_from_logits with various input shapes."""
    torch.manual_seed(42)
    total_tokens = batch_size * seq_len
    logits = torch.randn(total_tokens, vocab_size)

    topk_logprobs, topk_indices = topk_logprobs_from_logits(
        logits=logits,
        k=k,
        compute_topk=True,
        gather_topk=False,
        topk_indices=None,
    )

    assert topk_logprobs.shape == (total_tokens, k)
    assert topk_indices.shape == (total_tokens, k)
    assert topk_indices.dtype == torch.int64
    print(f"[topk_logprobs] shape=({batch_size}*{seq_len}, {vocab_size}) k={k} - OK")


@pytest.mark.parametrize("k", [1, 5, 10, 50])
def test_topk_logprobs_correct_values(k: int):
    """Test that topk_logprobs returns correct top-k log probabilities."""
    torch.manual_seed(42)
    vocab_size = 100
    batch_size = 4

    logits = torch.randn(batch_size, vocab_size)

    topk_logprobs, topk_indices = topk_logprobs_from_logits(
        logits=logits,
        k=k,
        compute_topk=True,
        gather_topk=False,
        topk_indices=None,
    )

    # Compute expected values
    expected_logprobs = torch.log_softmax(logits, dim=-1)
    expected_topk_logprobs, expected_topk_indices = torch.topk(expected_logprobs, k=k, dim=-1)

    torch.testing.assert_close(topk_logprobs, expected_topk_logprobs, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(topk_indices, expected_topk_indices)


if __name__ == "__main__":
    unittest.main()
