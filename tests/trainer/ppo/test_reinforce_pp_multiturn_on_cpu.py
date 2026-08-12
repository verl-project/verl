# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Tests for REINFORCE++ advantage computation with multi-turn observation spans.

Verifies that masked observation tokens (response_mask=0) are correctly
skipped during the reverse return scan, so that outcome rewards propagate
past observation spans to earlier assistant turns.
"""

from types import SimpleNamespace

import pytest
import torch

from verl.trainer.ppo.core_algos import compute_reinforce_plus_plus_outcome_advantage


def _config(gamma=1.0):
    return SimpleNamespace(gamma=gamma)


class TestReinforcePPMultiTurn:
    """Regression tests for #7278: observation spans must not block reward propagation."""

    def test_observation_span_does_not_block_returns(self):
        """Core regression test from the issue: inserting observation tokens
        between assistant tokens should not change valid-token returns."""
        config = _config(gamma=1.0)

        # Compact: 4 assistant tokens, reward=1.0 at the end
        compact_mask = torch.ones(1, 4)
        compact_rewards = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

        # Expanded: same 4 assistant tokens with 2 observation tokens (mask=0) in the middle
        expanded_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0]])
        expanded_rewards = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        _, compact_returns = compute_reinforce_plus_plus_outcome_advantage(compact_rewards, compact_mask, config=config)
        _, expanded_returns = compute_reinforce_plus_plus_outcome_advantage(
            expanded_rewards, expanded_mask, config=config
        )

        # Valid-token returns should match
        compact_valid = compact_returns[compact_mask.bool()]
        expanded_valid = expanded_returns[expanded_mask.bool()]
        torch.testing.assert_close(compact_valid, expanded_valid)

    def test_observation_positions_have_zero_returns(self):
        """Observation tokens (mask=0) should have returns=0."""
        config = _config(gamma=1.0)
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0]])
        rewards = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        _, returns = compute_reinforce_plus_plus_outcome_advantage(rewards, mask, config=config)

        # Observation positions (indices 2, 3) should be zero
        assert returns[0, 2].item() == 0.0
        assert returns[0, 3].item() == 0.0

    def test_trailing_padding_stays_zero(self):
        """Trailing padding (mask=0 at the end) should have returns=0 and
        should correctly reset running_return for EOS behavior."""
        config = _config(gamma=1.0)
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
        rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])

        _, returns = compute_reinforce_plus_plus_outcome_advantage(rewards, mask, config=config)

        # Valid tokens should all see the reward
        assert returns[0, 0].item() == 1.0
        assert returns[0, 1].item() == 1.0
        assert returns[0, 2].item() == 1.0
        # Padding should be zero
        assert returns[0, 3].item() == 0.0
        assert returns[0, 4].item() == 0.0

    def test_gamma_discount_across_observation_span(self):
        """Gamma discount should only apply across valid tokens, not
        observation tokens — observations are skipped, not discounted."""
        config = _config(gamma=0.5)
        # 2 valid tokens, then 2 observations, then 1 valid token with reward
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0]])
        rewards = torch.tensor([[0.0, 0.0, 0.0, 0.0, 2.0]])

        _, returns = compute_reinforce_plus_plus_outcome_advantage(rewards, mask, config=config)

        # Position 4: reward=2.0 -> return=2.0
        assert returns[0, 4].item() == 2.0
        # Observations (2,3): skipped, return=0
        assert returns[0, 2].item() == 0.0
        assert returns[0, 3].item() == 0.0
        # Position 1: gamma * running_return from position 4 = 0.5 * 2.0 = 1.0
        assert returns[0, 1].item() == pytest.approx(1.0)
        # Position 0: gamma * 1.0 = 0.5
        assert returns[0, 0].item() == pytest.approx(0.5)

    def test_batch_dimension(self):
        """Verify correct behavior across a batch of sequences."""
        config = _config(gamma=1.0)
        mask = torch.tensor(
            [
                [1.0, 0.0, 1.0, 1.0],  # observation at position 1
                [1.0, 1.0, 1.0, 0.0],  # padding at position 3
            ]
        )
        rewards = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )

        _, returns = compute_reinforce_plus_plus_outcome_advantage(rewards, mask, config=config)

        # Sequence 0: positions 0,2,3 are valid; observation at 1 is skipped
        assert returns[0, 0].item() == 1.0  # reward propagated past observation
        assert returns[0, 1].item() == 0.0  # observation -> zero
        assert returns[0, 2].item() == 1.0
        assert returns[0, 3].item() == 1.0

        # Sequence 1: positions 0,1,2 are valid; padding at 3
        assert returns[1, 0].item() == 1.0
        assert returns[1, 1].item() == 1.0
        assert returns[1, 2].item() == 1.0
        assert returns[1, 3].item() == 0.0

    def test_no_observation_tokens_unchanged(self):
        """Without observation tokens, behavior should be identical to before."""
        config = _config(gamma=1.0)
        mask = torch.ones(1, 5)
        rewards = torch.tensor([[0.0, 0.0, 0.0, 0.0, 3.0]])

        _, returns = compute_reinforce_plus_plus_outcome_advantage(rewards, mask, config=config)

        # All positions should see the cumulative return
        for t in range(5):
            assert returns[0, t].item() == 3.0
