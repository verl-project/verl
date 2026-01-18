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

import torch

from verl.trainer.distillation.utils import topk_logprobs_from_logits


class TestTopkLogprobsFromLogits:
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
            topk_indices=None,
        )

        assert topk_logprobs.shape == (batch_size * seq_len, k)
        assert topk_indices.shape == (batch_size * seq_len, k)

        # Verify that indices are within vocab range
        assert (topk_indices >= 0).all()
        assert (topk_indices < vocab_size).all()

        # Verify that logprobs are properly computed (should be log_softmax values)
        expected_logprobs = torch.log_softmax(logits, dim=-1)
        for i in range(batch_size * seq_len):
            for j in range(k):
                idx = topk_indices[i, j]
                torch.testing.assert_close(topk_logprobs[i, j], expected_logprobs[i, idx], atol=1e-5, rtol=1e-5)

    def test_gather_only(self):
        """Test gathering log probs at provided indices (compute_topk=False)."""
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
            topk_indices=topk_indices_input,
        )

        assert topk_logprobs.shape == (batch_size * seq_len, k)
        assert topk_indices.shape == (batch_size * seq_len, k)

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
            topk_indices=other_topk_indices,
        )

        # Should have 2*k logprobs (k from gather + k from compute)
        assert topk_logprobs.shape == (batch_size * seq_len, 2 * k)
        assert topk_indices.shape == (batch_size * seq_len, 2 * k)

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
            topk_indices=topk_indices_input,
        )

        # Should have 2*k entries but duplicates should have -inf logprobs
        assert topk_logprobs.shape == (1, 2 * k)

        # Count non-inf logprobs - should be exactly k due to complete overlap
        non_inf_count = (topk_logprobs > float("-inf")).sum().item()
        assert non_inf_count == k

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
            topk_indices=topk_indices_input,
        )

        # Should have 2*k entries with no duplicates
        assert topk_logprobs.shape == (1, 2 * k)

        # All logprobs should be finite (no -inf from deduplication)
        non_inf_count = (topk_logprobs > float("-inf")).sum().item()
        assert non_inf_count == 2 * k
