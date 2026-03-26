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
Test that demonstrates the cross-sequence contamination bug in sft_loss()
when using torch.roll on flattened nested tensor values.

Bug: In NO_PADDING mode, sft_loss flattens the loss_mask nested tensor with
.values() and applies torch.roll(shifts=-1) on the 1D buffer. This circular
shift causes the last element of each sequence to receive the mask value from
the *next* sequence (and the last sequence wraps around to the first).

The correct behavior is to left-shift the loss_mask independently per sequence
so that each sequence's boundary is respected.
"""

import torch
import pytest


def _build_nested_tensor(tensors):
    """Build a jagged nested tensor from a list of 1D tensors."""
    return torch.nested.nested_tensor(tensors, layout=torch.jagged)


def sft_loss_mask_buggy(log_prob_nested, loss_mask_nested):
    """Current (buggy) implementation: global roll on flattened values."""
    log_prob_flatten = log_prob_nested.values()
    loss_mask_flatten = loss_mask_nested.values()
    loss_mask_shifted = torch.roll(loss_mask_flatten, shifts=-1, dims=0)
    return loss_mask_shifted


def sft_loss_mask_fixed(log_prob_nested, loss_mask_nested):
    """Reference fixed implementation: per-sequence shift using a loop."""
    loss_mask_flatten = loss_mask_nested.values()
    offsets = loss_mask_nested.offsets()
    shifted = torch.empty_like(loss_mask_flatten)
    for i in range(len(offsets) - 1):
        start = offsets[i]
        end = offsets[i + 1]
        # Left-shift by 1 within this sequence; last position gets 0
        shifted[start : end - 1] = loss_mask_flatten[start + 1 : end]
        shifted[end - 1] = 0
    return shifted


def sft_loss_mask_fixed_vectorized(log_prob_nested, loss_mask_nested):
    """Vectorized fix: global roll + zero out sequence-boundary positions.

    This matches the actual fix applied in losses.py."""
    loss_mask_flatten = loss_mask_nested.values().clone()
    offsets = loss_mask_nested.offsets()
    loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)
    loss_mask_flatten[offsets[1:] - 1] = 0
    return loss_mask_flatten


class TestSftLossRollBug:
    """Demonstrate that torch.roll on flattened nested tensor causes
    cross-sequence mask contamination."""

    def test_identical_when_single_sequence(self):
        """With a single sequence, there's no cross-sequence contamination."""
        loss_mask = _build_nested_tensor([torch.tensor([0.0, 0.0, 1.0, 1.0])])
        log_prob = _build_nested_tensor([torch.tensor([-1.0, -2.0, -0.5, -0.3])])

        buggy = sft_loss_mask_buggy(log_prob, loss_mask)
        fixed = sft_loss_mask_fixed(log_prob, loss_mask)

        # Single sequence: roll wraps within itself, same as per-seq shift
        # (except the last element, which wraps vs gets zeroed)
        # buggy last = loss_mask[0] = 0.0, fixed last = 0.0
        # In this case they happen to match because loss_mask[0] == 0
        assert torch.allclose(buggy, fixed)

    def test_cross_sequence_contamination(self):
        """With multiple sequences where boundary mask values differ,
        torch.roll causes incorrect mask values at sequence boundaries.

        This is THE bug demonstration."""
        # Seq A: prompt=[0,1,1,1] — first token masked, rest are loss tokens
        # Seq B: prompt=[1,0,1]   — first token is loss, second masked, third is loss
        seq_a_mask = torch.tensor([0.0, 1.0, 1.0, 1.0])
        seq_b_mask = torch.tensor([1.0, 0.0, 1.0])

        loss_mask = _build_nested_tensor([seq_a_mask, seq_b_mask])
        # log_prob values don't matter for mask computation, just need same shape
        log_prob = _build_nested_tensor([torch.randn(4), torch.randn(3)])

        buggy = sft_loss_mask_buggy(log_prob, loss_mask)
        fixed = sft_loss_mask_fixed(log_prob, loss_mask)

        # Flattened: [0, 1, 1, 1, 1, 0, 1]
        #
        # Buggy (global roll -1):
        #   [1, 1, 1, 1, 0, 1, 0]
        #          pos3=1 (from seq_b[0]=1) ← WRONG
        #                      pos6=0 (from seq_a[0]=0) ← WRONG
        #
        # Fixed (per-sequence shift, last gets 0):
        #   [1, 1, 1, 0, 0, 1, 0]
        #          pos3=0 (end of seq_a, zeroed) ← CORRECT
        #                      pos6=0 (end of seq_b, zeroed) ← CORRECT

        expected_buggy = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
        expected_fixed = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])

        assert torch.allclose(buggy, expected_buggy), f"Buggy mismatch: {buggy} vs {expected_buggy}"
        assert torch.allclose(fixed, expected_fixed), f"Fixed mismatch: {fixed} vs {expected_fixed}"

        # The key difference: position 3 (last token of seq A)
        # Buggy gives 1.0 (from seq B's first element), but correct is 0.0
        assert buggy[3] != fixed[3], "Position 3 should differ between buggy and fixed"
        assert buggy[3] == 1.0, "Buggy position 3 gets seq_b[0]=1.0 (contaminated)"
        assert fixed[3] == 0.0, "Fixed position 3 should be 0.0 (end-of-sequence)"

    def test_contamination_affects_loss(self):
        """Show that cross-sequence contamination changes the actual loss value."""
        # Construct sequences where the contamination changes which tokens are included
        # Seq A: loss_mask = [0, 0, 1, 1], log_prob = [-1, -2, -0.5, -3.0]
        # Seq B: loss_mask = [1, 1, 0],    log_prob = [-0.1, -0.2, -0.3]
        seq_a_mask = torch.tensor([0.0, 0.0, 1.0, 1.0])
        seq_b_mask = torch.tensor([1.0, 1.0, 0.0])
        seq_a_logp = torch.tensor([-1.0, -2.0, -0.5, -3.0])
        seq_b_logp = torch.tensor([-0.1, -0.2, -0.3])

        loss_mask = _build_nested_tensor([seq_a_mask, seq_b_mask])
        log_prob = _build_nested_tensor([seq_a_logp, seq_b_logp])

        buggy_mask = sft_loss_mask_buggy(log_prob, loss_mask)
        fixed_mask = sft_loss_mask_fixed(log_prob, loss_mask)

        log_prob_flat = log_prob.values()

        # Compute masked loss (simplified version of sft_loss)
        buggy_loss = -(log_prob_flat * buggy_mask).sum()
        fixed_loss = -(log_prob_flat * fixed_mask).sum()

        # Flattened: mask=[0,0,1,1, 1,1,0], logp=[-1,-2,-0.5,-3, -0.1,-0.2,-0.3]
        #
        # Buggy mask after roll: [0,1,1,1, 1,0,0]
        # Buggy loss = -((-2)*1 + (-0.5)*1 + (-3)*1 + (-0.1)*1) = 5.6
        #
        # Fixed mask after shift: [0,1,1,0, 1,0,0]
        # Fixed loss = -((-2)*1 + (-0.5)*1 + (-0.1)*1) = 2.6
        #
        # Difference: 3.0 (the log_prob at seq_a position 3, wrongly included)

        assert not torch.isclose(buggy_loss, fixed_loss), (
            f"Losses should differ! buggy={buggy_loss.item()}, fixed={fixed_loss.item()}"
        )
        assert abs(buggy_loss.item() - 5.6) < 1e-5, f"Buggy loss should be 5.6, got {buggy_loss.item()}"
        assert abs(fixed_loss.item() - 2.6) < 1e-5, f"Fixed loss should be 2.6, got {fixed_loss.item()}"

    def test_three_sequences(self):
        """Bug affects all sequence boundaries, not just the last one."""
        # Three sequences with different masks to show cascading contamination
        seq_a_mask = torch.tensor([0.0, 1.0])  # last=1
        seq_b_mask = torch.tensor([0.0, 0.0, 1.0])  # last=1
        seq_c_mask = torch.tensor([1.0, 0.0])  # last=0

        loss_mask = _build_nested_tensor([seq_a_mask, seq_b_mask, seq_c_mask])
        log_prob = _build_nested_tensor([torch.randn(2), torch.randn(3), torch.randn(2)])

        buggy = sft_loss_mask_buggy(log_prob, loss_mask)
        fixed = sft_loss_mask_fixed(log_prob, loss_mask)

        # Flattened: [0, 1, 0, 0, 1, 1, 0]
        # Buggy (global roll): [1, 0, 0, 1, 1, 0, 0]
        #   pos1 (last of A)=0 (from B[0]=0) vs fixed=0 (zeroed) — same by coincidence
        #   pos4 (last of B)=1 (from C[0]=1) vs fixed=0 (zeroed) — DIFFERENT!
        #   pos6 (last of C)=0 (from A[0]=0) vs fixed=0 (zeroed) — same by coincidence

        # Position 4 (last of seq B): buggy gets C[0]=1, fixed gets 0
        assert buggy[4] == 1.0, f"Buggy pos4 should be 1.0, got {buggy[4]}"
        assert fixed[4] == 0.0, f"Fixed pos4 should be 0.0, got {fixed[4]}"

    def test_per_sequence_shift_zeroes_last(self):
        """The per-sequence shift should always zero out the last position of each
        sequence, because there is no 'next token' prediction for the last position."""
        masks = [
            torch.tensor([1.0, 1.0, 1.0]),
            torch.tensor([1.0, 1.0]),
            torch.tensor([0.0, 1.0, 1.0, 1.0]),
        ]
        loss_mask = _build_nested_tensor(masks)
        log_prob = _build_nested_tensor([torch.randn(3), torch.randn(2), torch.randn(4)])
        offsets = loss_mask.offsets()

        fixed = sft_loss_mask_fixed(log_prob, loss_mask)

        # Last position of each sequence should be 0
        for i in range(len(offsets) - 1):
            end = offsets[i + 1] - 1
            assert fixed[end] == 0.0, f"Seq {i} last position should be 0, got {fixed[end]}"


    def test_vectorized_fix_matches_loop_fix(self):
        """The vectorized fix (global roll + zero boundaries) must produce
        the same result as the per-sequence loop fix."""
        masks = [
            torch.tensor([0.0, 1.0, 1.0, 1.0]),
            torch.tensor([1.0, 0.0, 1.0]),
            torch.tensor([1.0, 1.0]),
            torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0]),
        ]
        loss_mask = _build_nested_tensor(masks)
        log_prob = _build_nested_tensor([torch.randn(4), torch.randn(3), torch.randn(2), torch.randn(5)])

        loop_result = sft_loss_mask_fixed(log_prob, loss_mask)
        vec_result = sft_loss_mask_fixed_vectorized(log_prob, loss_mask)

        assert torch.allclose(loop_result, vec_result), (
            f"Vectorized fix must match loop fix:\n  loop: {loop_result}\n  vec:  {vec_result}"
        )

    def test_vectorized_fix_differs_from_buggy(self):
        """Verify the vectorized fix actually changes behavior vs buggy."""
        seq_a_mask = torch.tensor([0.0, 1.0, 1.0, 1.0])
        seq_b_mask = torch.tensor([1.0, 0.0, 1.0])

        loss_mask = _build_nested_tensor([seq_a_mask, seq_b_mask])
        log_prob = _build_nested_tensor([torch.randn(4), torch.randn(3)])

        buggy = sft_loss_mask_buggy(log_prob, loss_mask)
        fixed = sft_loss_mask_fixed_vectorized(log_prob, loss_mask)

        # Position 3 (last of seq A): buggy=1 (from seq_b[0]), fixed=0
        assert buggy[3] == 1.0
        assert fixed[3] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
