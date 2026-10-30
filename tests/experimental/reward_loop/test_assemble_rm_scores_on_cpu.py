# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import pytest
import torch

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    compute_reinforce_plus_plus_outcome_advantage,
    compute_remax_outcome_advantage,
)


def _batch(response_attention_mask, response_mask=None):
    """Build a real padded reward batch, including a left-padded prompt."""
    attention = torch.tensor(response_attention_mask, dtype=torch.long)
    batch_size, response_length = attention.shape
    prompts = torch.tensor([0, 10, 11]).expand(batch_size, -1).clone()
    tensors = {
        "prompts": prompts,
        "responses": torch.arange(response_length).expand(batch_size, -1).clone(),
        "attention_mask": torch.cat((prompts.ne(0).long(), attention), dim=1),
    }
    if response_mask is not None:
        tensors["response_mask"] = torch.tensor(response_mask, dtype=torch.long)
    return DataProto.from_dict(tensors=tensors)


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.long, torch.float32])
def test_assemble_scores_on_last_generated_token(mask_dtype):
    """Masked template suffixes must not swallow colocated RM outcomes (#8367)."""
    data = _batch(
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]],
        [[1, 1, 0, 1, 1, 0], [1, 0, 1, 0, 0, 0]],
    )
    data.batch["response_mask"] = data.batch["response_mask"].to(mask_dtype)
    before = data.batch.clone()

    scores = RewardManagerBase.assemble_rm_scores(data, [2.0, -1.0])

    torch.testing.assert_close(scores, torch.tensor([[0, 0, 0, 0, 2, 0], [0, 0, -1, 0, 0, 0]]).float())
    for key in before.keys():
        torch.testing.assert_close(data.batch[key], before[key])


@pytest.mark.parametrize("with_mask", [False, True])
def test_assemble_scores_preserves_single_turn_placement(with_mask):
    """Single-turn batches and callers without response_mask retain their scores."""
    attention = [[1, 1, 1, 1], [1, 1, 0, 0]]
    data = _batch(attention, attention if with_mask else None)

    scores = RewardManagerBase.assemble_rm_scores(data, [1.25, -0.5])

    torch.testing.assert_close(scores, torch.tensor([[0, 0, 0, 1.25], [0, -0.5, 0, 0]]))


def test_assemble_scores_without_generated_tokens_uses_legacy_fallback():
    """An all-zero generation mask preserves the last valid-token fallback."""
    data = _batch([[1, 1, 1, 0]], [[0, 0, 0, 0]])

    scores = RewardManagerBase.assemble_rm_scores(data, [3.0])

    torch.testing.assert_close(scores, torch.tensor([[0.0, 0.0, 3.0, 0.0]]))


def test_assemble_scores_ignores_generated_marks_on_padding():
    """Padding cannot become a reward destination even if its response mask is set."""
    data = _batch([[1, 1, 1, 0, 0]], [[1, 1, 0, 1, 1]])

    scores = RewardManagerBase.assemble_rm_scores(data, [2.0])

    torch.testing.assert_close(scores, torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0]]))


def test_assemble_scores_handles_empty_batch():
    """A zero-row batch keeps its response width and floating score dtype."""
    data = DataProto.from_dict(
        tensors={
            "prompts": torch.empty((0, 2), dtype=torch.long),
            "responses": torch.empty((0, 4), dtype=torch.long),
            "attention_mask": torch.empty((0, 6), dtype=torch.long),
            "response_mask": torch.empty((0, 4), dtype=torch.long),
        }
    )

    scores = RewardManagerBase.assemble_rm_scores(data, [])

    assert scores.shape == (0, 4)
    assert scores.dtype == torch.float32


@pytest.mark.parametrize("estimator", ["gae", "reinforce_plus_plus", "remax"])
def test_colocated_outcome_reaches_returns_and_policy_gradient(estimator):
    """Real estimators must retain the RM reward across masked observation spans."""
    data = _batch(
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]],
        [[1, 1, 0, 1, 1, 0], [1, 0, 1, 0, 0, 0]],
    )
    scores = RewardManagerBase.assemble_rm_scores(data, [2.0, -1.0])
    mask = data.batch["response_mask"].float()
    if estimator == "gae":
        _, returns = compute_gae_advantage_return(scores, torch.zeros_like(scores), mask, gamma=1.0, lam=1.0)
    elif estimator == "reinforce_plus_plus":
        _, returns = compute_reinforce_plus_plus_outcome_advantage(scores, mask, config=AlgoConfig(gamma=1.0))
    else:
        _, returns = compute_remax_outcome_advantage(scores, torch.zeros(2), mask)

    expected = torch.tensor([2.0, -1.0]).unsqueeze(1).expand_as(mask)
    torch.testing.assert_close(scores.sum(dim=1), torch.tensor([2.0, -1.0]))
    torch.testing.assert_close(returns[mask.bool()], expected[mask.bool()])
    log_probs = torch.zeros_like(scores, requires_grad=True)
    loss = -(log_probs * returns.detach() * mask).sum()
    loss.backward()
    torch.testing.assert_close(log_probs.grad, -expected * mask)
