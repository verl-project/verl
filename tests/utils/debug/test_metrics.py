# Copyright 2025 Individual Contributor: TomQunChaoA
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

import math
import unittest

import numpy as np
import pytest
import torch

from verl.protocol import DataProto
from verl.utils.debug.metrics import calculate_debug_metrics, calculate_train_infer_discrepancy, partial_rollout_mask


class TestMetrics(unittest.TestCase):
    def test_calculate_debug_metrics(self):
        data = DataProto.from_dict(
            {
                "rollout_log_probs": torch.tensor(
                    [
                        [-1.5085, -0.1200, -0.6650, -0.4823, -0.1426, -1.5557, -2.8532, -0.3919, -0.4294, -0.4700],
                        [-0.0585, -0.0573, -0.4681, -0.5187, -0.7451, -1.2737, -0.0682, -0.4284, -0.5754, -0.0611],
                    ]
                ),
                "old_log_probs": torch.tensor(
                    [
                        [-1.8636, -0.7863, -0.2136, -0.4376, -2.0257, -0.2579, -1.1547, -0.5203, -0.3802, -0.9872],
                        [-0.3507, -0.5426, -0.2725, -0.4637, -0.3577, -0.3733, -1.7560, -1.9542, -0.4229, -1.3098],
                    ]
                ),
                "loss_mask": torch.tensor([[1, 0, 0, 0, 1, 1, 0, 1, 1, 0], [1, 0, 1, 0, 1, 1, 1, 0, 1, 1]]),
                "responses": torch.zeros((2, 10)),
            }
        )
        metrics = calculate_debug_metrics(data)
        print(metrics)
        assert metrics["training/rollout_probs_diff_valid"] == 1
        assert metrics["training/rollout_logprobs_mismatch_count"] == 20
        assert metrics["training/rollout_logprobs_diff_max"] > 0

    def test_one_ulp_logprob_difference_is_hidden_by_probs_diff(self):
        rollout = torch.tensor([[-0.10000000149, -2.0, -0.5]], dtype=torch.float32)
        actor = torch.tensor([[-0.10000000894, -2.0, -0.75]], dtype=torch.float32)
        assert torch.exp(rollout[0, 0]) == torch.exp(actor[0, 0])
        data = DataProto.from_dict(
            {
                "rollout_log_probs": rollout,
                "old_log_probs": actor,
                "response_mask": torch.tensor([[1, 1, 0]]),
                "responses": torch.zeros((1, 3)),
            }
        )
        metrics = calculate_debug_metrics(data)
        assert metrics["training/rollout_probs_diff_max"] == 0
        assert metrics["training/rollout_logprobs_mismatch_count"] == 1
        assert metrics["training/rollout_logprobs_diff_max"] == (rollout[0, 0] - actor[0, 0]).abs().item()
        assert metrics["training/rollout_logprobs_diff_mean"] == metrics["training/rollout_logprobs_diff_max"] / 2

    def test_mismatch_count_follows_the_casted_diff_across_dtypes(self):
        rollout = torch.tensor([[-0.5, -2.0, -0.25]], dtype=torch.bfloat16)
        actor = torch.tensor([[-0.5, -2.0, -0.75]], dtype=torch.float32)
        data = DataProto.from_dict(
            {
                "rollout_log_probs": rollout,
                "old_log_probs": actor,
                "response_mask": torch.tensor([[1, 1, 1]]),
                "responses": torch.zeros((1, 3)),
            }
        )
        metrics = calculate_debug_metrics(data)
        assert metrics["training/rollout_logprobs_mismatch_count"] == 1
        assert metrics["training/rollout_logprobs_diff_max"] == 0.5

    def test_all_masked_returns_invalid(self):
        data = DataProto.from_dict(
            {
                "rollout_log_probs": torch.tensor([[-1.0, -2.0]]),
                "old_log_probs": torch.tensor([[-1.5, -2.0]]),
                "response_mask": torch.tensor([[0, 0]]),
                "responses": torch.zeros((1, 2)),
            }
        )
        metrics = calculate_debug_metrics(data)
        assert metrics["training/rollout_probs_diff_valid"] == 0
        assert metrics["training/rollout_logprobs_mismatch_count"] == 0

    def test_train_infer_discrepancy_matches_the_r3_definitions(self):
        # One token is 4x more likely under the training engine, the other matches exactly.
        log_ratio = math.log(4.0)
        rollout = torch.tensor([[-1.0, -2.0]])
        actor = rollout + torch.tensor([[log_ratio, 0.0]])

        metrics = calculate_train_infer_discrepancy(actor, rollout, torch.tensor([[1, 1]]))

        # k3 estimator (Eq. 3): r - 1 - log r, averaged over tokens.
        assert metrics["training/train_infer_kl_k3"] == pytest.approx((4.0 - 1 - log_ratio) / 2, rel=1e-5)
        # F(tau=2) (Eq. 4): only the 4x token is extreme.
        assert metrics["training/train_infer_extreme_token_frac"] == pytest.approx(0.5)
        assert metrics["training/train_infer_tokens_sum"] == 2

    def test_train_infer_discrepancy_splits_partial_trajectories(self):
        # Splitting is the point: a trajectory resumed across a weight sync is the suspect, and
        # pooling it with the majority of single-version trajectories hides its discrepancy.
        rollout = torch.zeros(2, 2)
        actor = torch.tensor([[math.log(4.0), math.log(4.0)], [0.0, 0.0]])
        mask = torch.ones(2, 2, dtype=torch.long)

        metrics = calculate_train_infer_discrepancy(actor, rollout, mask, torch.tensor([True, False]))

        assert metrics["training/train_infer_extreme_token_frac_partial"] == pytest.approx(1.0)
        assert metrics["training/train_infer_extreme_token_frac_nonpartial"] == pytest.approx(0.0)
        assert metrics["training/train_infer_kl_k3_nonpartial"] == pytest.approx(0.0)
        assert metrics["training/train_infer_tokens_sum_partial"] == 2

    def test_train_infer_discrepancy_omits_empty_groups(self):
        # The fully_async aggregator averages a step's value into the whole sync cycle, so a
        # nan from a step without partial trajectories would discard the steps that had them.
        metrics = calculate_train_infer_discrepancy(
            torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.long), torch.tensor([False])
        )

        assert "training/train_infer_kl_k3_partial" not in metrics
        assert "training/train_infer_tokens_sum_partial" not in metrics
        assert metrics["training/train_infer_tokens_sum_nonpartial"] == 2

    def test_train_infer_discrepancy_sums_allow_exact_reaggregation(self):
        # Averaging per-step means would weight a step with few tokens like a step with many.
        rollout = torch.zeros(1, 2)
        # Two tokens at a 4x ratio in one step, a single matching token in the next.
        steps = [
            calculate_train_infer_discrepancy(
                torch.full((1, 2), math.log(4.0)), rollout, torch.ones(1, 2, dtype=torch.long)
            ),
            calculate_train_infer_discrepancy(torch.zeros(1, 2), rollout, torch.tensor([[1, 0]])),
        ]

        k3 = 4.0 - 1 - math.log(4.0)
        pooled = sum(m["training/train_infer_kl_k3_sum"] for m in steps) / sum(
            m["training/train_infer_scored_tokens_sum"] for m in steps
        )
        assert pooled == pytest.approx(2 * k3 / 3, rel=1e-5)
        # F(tau) (Eq. 4) must stay recoverable the same way.
        assert sum(m["training/train_infer_extreme_tokens_sum"] for m in steps) == 2
        assert sum(m["training/train_infer_tokens_sum"] for m in steps) == 3
        # The unweighted mean of the per-step means cannot see the 2:1 token imbalance.
        assert sum(m["training/train_infer_kl_k3"] for m in steps) / 2 == pytest.approx(k3 / 2, rel=1e-5)

    def test_train_infer_discrepancy_survives_impossible_rollout_tokens(self):
        # A rollout engine may report -inf for a token, which makes k3 inf - inf. One such
        # token must not turn the whole average into nan.
        rollout = torch.tensor([[-float("inf"), -2.0]])
        actor = torch.tensor([[-1.0, -2.0]])

        metrics = calculate_train_infer_discrepancy(actor, rollout, torch.tensor([[1, 1]]))

        assert metrics["training/train_infer_kl_k3"] == pytest.approx(0.0)
        assert metrics["training/train_infer_tokens_sum"] == 2
        assert metrics["training/train_infer_nonfinite_token_frac"] == pytest.approx(0.5)
        # The cause matters: an impossible token is a discrepancy, a nan input is a bug.
        assert metrics["training/train_infer_impossible_under_rollout_token_frac"] == pytest.approx(0.5)
        assert metrics["training/train_infer_impossible_under_actor_token_frac"] == pytest.approx(0.0)
        assert metrics["training/train_infer_nan_input_token_frac"] == pytest.approx(0.0)

    def test_partial_rollout_mask_needs_both_weight_versions(self):
        def tags(starts, ends=None):
            if ends is None:
                return [{"min_global_steps": start} for start in starts]
            return [
                {"min_global_steps": start, "max_global_steps": end}
                for start, end in zip(starts, ends, strict=True)
            ]

        assert partial_rollout_mask(tags([3, 4], [3, 6])).tolist() == [False, True]
        # Backends that do not report versions must not be reported as single-version.
        assert partial_rollout_mask(tags([3, 4])) is None
        assert partial_rollout_mask(tags([None, 4], [3, 6])) is None
        assert partial_rollout_mask([]) is None


if __name__ == "__main__":
    unittest.main()
