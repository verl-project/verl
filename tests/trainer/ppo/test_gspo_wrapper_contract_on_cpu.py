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

"""GSPO's stop-gradient contract through the real production PPO loss wrapper.

The oracle uses scalar arithmetic and an analytic derivative, not agg_loss or
a copy of the production tensor expression. DP averaging is simulated here.
"""

import math

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType
from verl.workers.config.actor import ActorConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss

_LENGTHS = [1, 3, 2, 4]
_PROMPT_LENGTHS = [2, 1, 3, 2]
_PARTITIONS = [
    [[[0, 1, 2, 3]]],
    [[[0], [1, 2], [3]]],
    [[[3], [0]], [[1, 2]]],
]


def _fixture(dtype):
    ratios = torch.tensor([[1.05, 1, 1, 1], [1.3, 1.4, 1.5, 1], [0.6, 0.7, 1, 1], [1.15, 0.95, 1.1, 20]], dtype=dtype)
    delta = ratios.log().requires_grad_()
    old = torch.full_like(delta, -3.0)
    mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    advantages = torch.tensor([1.0, 2.0, -2.0, -1.0], dtype=dtype)[:, None].expand_as(delta)
    # Unequal token weights distinguish the intended stop-gradient derivative from
    # differentiating the sequence average, even with sequence-constant advantages.
    weights = torch.tensor([[1, 1, 1, 1], [0.5, 1, 2, 1], [1, 2, 1, 1], [0.4, 1.5, 0.8, 1]], dtype=dtype)
    return delta, old, mask, advantages, weights


def _pack(values, rows):
    # Model outputs predict the next token: response output begins at prompt_len - 1.
    return torch.cat(
        [
            torch.cat(
                (
                    values.new_full((_PROMPT_LENGTHS[row] - 1,), 71),
                    values[row, : _LENGTHS[row]],
                    values.new_full((1,), 83),
                )
            )
            for row in rows
        ]
    )


def _batch(rows, old, mask, advantages, weights, layout, dp_size):
    fields = {"old_log_probs": old, "response_mask": mask, "advantages": advantages, "rollout_is_weights": weights}
    fields["ref_log_prob"] = old + 0.05
    if layout == "nested":
        values = {
            key: torch.nested.as_nested_tensor([value[row, : _LENGTHS[row]] for row in rows], layout=torch.jagged)
            for key, value in fields.items()
        }
        values["prompts"] = torch.nested.as_nested_tensor(
            [torch.ones(_PROMPT_LENGTHS[row], dtype=torch.long) for row in rows], layout=torch.jagged
        )
        values["responses"] = torch.nested.as_nested_tensor(
            [torch.ones(_LENGTHS[row], dtype=torch.long) for row in rows], layout=torch.jagged
        )
    else:
        values = {key: value[rows] for key, value in fields.items()}
        values["prompts"] = torch.zeros(len(rows), 3, dtype=torch.long)
        values["responses"] = torch.zeros(len(rows), 4, dtype=torch.long)
        attention = torch.zeros(len(rows), 7, dtype=torch.long)
        for index, row in enumerate(rows):
            attention[index, 3 - _PROMPT_LENGTHS[row] : 3 + _LENGTHS[row]] = 1
        values["attention_mask"] = attention
    data = TensorDict(values, batch_size=[len(rows)])
    for key, value in {
        "dp_size": dp_size,
        "batch_num_tokens": int(mask.sum()),
        "global_batch_size": len(_LENGTHS),
    }.items():
        tu.assign_non_tensor_data(data, key, value)
    return data


def _oracle(delta, old, mask, advantages, weights, mode, regularize):
    """Return the scalar objective and its local stop-gradient derivative."""
    total = 0.0
    gradient = torch.zeros_like(delta)
    pg_total = 0.0
    for row in range(len(_LENGTHS)):
        valid = torch.where(mask[row])[0].tolist()
        ratio = math.exp(sum(float(delta[row, col].detach()) for col in valid) / len(valid))
        clipped = min(max(ratio, 0.8), 1.2)
        denominator = int(mask.sum()) if mode == "token-mean" else len(_LENGTHS) * len(valid)
        for col in valid:
            advantage = float(advantages[row, col])
            weight = float(weights[row, col])
            value = max(-advantage * ratio, -advantage * clipped) * weight / denominator
            pg_total += value
            # The sequence ratio is constant for backprop; each token's own log-prob
            # supplies the derivative. Do not differentiate the sequence mean again.
            unclipped = (advantage > 0 and ratio < 1.2) or (advantage < 0 and ratio > 0.8)
            derivative = -advantage * ratio * weight if unclipped else 0.0
            if regularize:
                log_prob = float((old[row, col] + delta[row, col]).detach())
                value -= 0.03 * (0.2 * log_prob + 1.0) / denominator
                value += 0.04 * (log_prob - float(old[row, col]) - 0.05) / denominator
                derivative += -0.03 * 0.2 + 0.04
            total += value
            gradient[row, col] = derivative / denominator
    return total, gradient, pg_total


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["token-mean", "seq-mean-token-mean"])
@pytest.mark.parametrize("layout", ["padded", "nested"])
@pytest.mark.parametrize("regularize", [False, True])
@pytest.mark.parametrize("ranks", _PARTITIONS, ids=["full", "uneven-microbatches", "simulated-dp-reordered"])
def test_gspo_wrapper_matches_objective_and_analytic_gradient(dtype, mode, layout, regularize, ranks):
    delta, old, mask, advantages, weights = _fixture(dtype)
    expected_loss, expected_grad, expected_pg = _oracle(delta, old, mask, advantages, weights, mode, regularize)
    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(loss_mode="gspo"),
        clip_ratio=0.2,
        loss_agg_mode=mode,
        entropy_coeff=0.03 if regularize else 0.0,
        use_kl_loss=regularize,
        kl_loss_coef=0.04,
        kl_loss_type="kl",
    )
    loss = delta.new_zeros(())
    logged_pg = 0.0
    for microbatches in ranks:
        for rows in microbatches:
            current = old + delta
            output = {"log_probs": _pack(current, rows)}
            if regularize:
                output["entropy"] = _pack(0.2 * current + 1.0, rows)
            data = _batch(rows, old, mask, advantages, weights, layout, len(ranks))
            part, metrics = ppo_loss(config, output, data)
            loss = loss + part / len(ranks)
            assert metrics["actor/pg_loss"].aggregation == AggregationType.SUM
            logged_pg += metrics["actor/pg_loss"].aggregate() / len(ranks)

    loss.backward()
    tolerance = {"rtol": 1e-5, "atol": 1e-6} if dtype == torch.float32 else {"rtol": 1e-10, "atol": 1e-10}
    torch.testing.assert_close(loss.detach(), loss.new_tensor(expected_loss), **tolerance)
    torch.testing.assert_close(delta.grad, expected_grad, **tolerance)
    assert logged_pg == pytest.approx(expected_pg, rel=tolerance["rtol"], abs=tolerance["atol"])
    assert torch.equal(delta.grad[~mask], torch.zeros_like(delta.grad[~mask]))
    # The fixture exercises both clipped branches, not just a no-clipping happy path.
    if not regularize:
        assert torch.count_nonzero(delta.grad[1:3]) == 0
        assert torch.count_nonzero(delta.grad[0]) > 0
        assert torch.count_nonzero(delta.grad[3]) > 0
