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

from verl.utils.trajectory import (
    apply_loss_weight_to_advantages,
    resolve_agent_loop_loss_weight,
    validate_loss_weights,
)


def test_validate_loss_weights_preserves_explicit_multipliers():
    weights = torch.tensor([0.25, 0.5, 1.25])

    prepared = validate_loss_weights(weights)

    torch.testing.assert_close(prepared, weights)


def test_validate_loss_weights_zeroes_invalid_samples():
    weights = torch.tensor([0.25, 0.5, 1.25])
    valid_mask = torch.tensor([True, True, False])

    prepared = validate_loss_weights(weights, valid_mask=valid_mask)

    torch.testing.assert_close(prepared, torch.tensor([0.25, 0.5, 0.0]))


def test_zeroed_padding_rows_survive_the_apply_step():
    """The padding rows zeroed above must not trip the positivity check downstream."""
    weights = validate_loss_weights(torch.tensor([0.5, 1.0]), valid_mask=torch.tensor([True, False]))

    weighted = apply_loss_weight_to_advantages(torch.ones(2, 2), weights)

    torch.testing.assert_close(weighted, torch.tensor([[0.5, 0.5], [0.0, 0.0]]))


def test_validate_loss_weights_rejects_invalid_values():
    with pytest.raises(ValueError, match="positive"):
        validate_loss_weights(torch.tensor([1.0, 0.0]))

    with pytest.raises(ValueError, match="finite"):
        validate_loss_weights(torch.tensor([1.0, float("nan")]))


def test_resolve_agent_loop_loss_weight_defaults_and_reads_extra_fields():
    class Output:
        loss_weight = None
        extra_fields = {"loss_weight": 0.25}

    assert resolve_agent_loop_loss_weight(Output()) == pytest.approx(0.25)

    class DefaultOutput:
        loss_weight = None
        extra_fields = {}

    # A missing weight is always neutral 1.0: the adapter must not guess 1/N, whose
    # correctness depends on actor.loss_agg_mode.
    assert resolve_agent_loop_loss_weight(DefaultOutput()) == pytest.approx(1.0)


def test_apply_loss_weight_to_advantages_supports_vector_and_column_shapes():
    advantages = torch.ones(2, 3)

    vector_result = apply_loss_weight_to_advantages(advantages, torch.tensor([0.25, 0.5]))
    column_result = apply_loss_weight_to_advantages(advantages, torch.tensor([[0.25], [0.5]]))

    torch.testing.assert_close(vector_result, torch.tensor([[0.25, 0.25, 0.25], [0.5, 0.5, 0.5]]))
    torch.testing.assert_close(column_result, vector_result)


def test_apply_loss_weight_to_advantages_rejects_invalid_shape_and_values():
    advantages = torch.ones(2, 3)

    with pytest.raises(ValueError, match="shape"):
        apply_loss_weight_to_advantages(advantages, torch.ones(2, 2))

    # Strict positivity is enforced at the entry point (validate_loss_weights); the apply
    # step only rejects negative/non-finite values so it stays compatible with the zeroed
    # padding rows that validate_loss_weights itself produces.
    with pytest.raises(ValueError, match="non-negative"):
        apply_loss_weight_to_advantages(advantages, torch.tensor([1.0, -0.5]))

    with pytest.raises(ValueError, match="finite"):
        apply_loss_weight_to_advantages(advantages, torch.tensor([1.0, float("nan")]))


@pytest.mark.parametrize(
    ("loss_agg_mode", "invariant_weight"),
    [
        # These normalize by token count, which splitting preserves -> neutral 1.0.
        ("token-mean", 1.0),
        ("token-sum", 1.0),
        ("seq-mean-token-sum", 1.0),
        # This normalizes by row count, which splitting inflates 1 -> N -> needs 1/N.
        ("seq-mean-token-mean", 0.5),
    ],
)
def test_split_trajectory_is_invariant_only_under_the_mode_specific_weight(loss_agg_mode, invariant_weight):
    """Pin which weight makes a split trajectory equivalent to an unsplit one.

    A blanket ``1 / N`` default is wrong for the default ``token-mean`` mode: it
    would shrink the trajectory's contribution by ``N``. This is why the adapter
    stays neutral and leaves the choice to the agent loop.
    """
    from verl.trainer.ppo.core_algos import agg_loss

    norm = {"batch_num_tokens": 32, "global_batch_size": 8}
    # One trajectory of 6 tokens with non-uniform per-token loss...
    unsplit = agg_loss(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]), torch.ones(1, 6), loss_agg_mode, **norm)
    # ...versus the same tokens split into two 3-token segments.
    segments = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    split = agg_loss(segments * invariant_weight, torch.ones(2, 3), loss_agg_mode, **norm)

    torch.testing.assert_close(split, unsplit)

    # And the other candidate weight is measurably wrong for this mode.
    wrong_weight = 0.5 if invariant_weight == 1.0 else 1.0
    wrong = agg_loss(segments * wrong_weight, torch.ones(2, 3), loss_agg_mode, **norm)
    assert not torch.isclose(wrong, unsplit), f"{wrong_weight} should not be invariant for {loss_agg_mode}"


def test_apply_loss_weight_to_advantages_rejects_advantages_without_token_dim():
    # A 1-D [batch_size] advantage broadcasts against the [batch_size, 1] weight into a
    # bogus [batch_size, batch_size] tensor, so it must fail loudly instead.
    with pytest.raises(ValueError, match=r"\[batch_size, response_length\]"):
        apply_loss_weight_to_advantages(torch.tensor([1.0, 2.0]), torch.tensor([0.5, 0.5]))
