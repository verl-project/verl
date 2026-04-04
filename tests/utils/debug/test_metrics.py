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

import unittest

import torch

from verl.protocol import DataProto
from verl.utils.debug.metrics import _find_contiguous_segments, calculate_debug_metrics


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

    def test_find_contiguous_segments(self):
        # Single segment
        mask = torch.tensor([1, 1, 1, 0, 0])
        assert _find_contiguous_segments(mask) == [(0, 3)]

        # Multiple segments (multi-turn)
        mask = torch.tensor([1, 1, 0, 0, 1, 1, 1, 0, 1])
        assert _find_contiguous_segments(mask) == [(0, 2), (4, 7), (8, 9)]

        # All zeros
        mask = torch.tensor([0, 0, 0])
        assert _find_contiguous_segments(mask) == []

        # All ones
        mask = torch.tensor([1, 1, 1])
        assert _find_contiguous_segments(mask) == [(0, 3)]

    def test_per_round_metrics_single_turn(self):
        """Single contiguous response should produce 1 round."""
        data = DataProto.from_dict(
            {
                "rollout_log_probs": torch.tensor([[-1.0, -2.0, -3.0, -4.0]]),
                "old_log_probs": torch.tensor([[-1.1, -2.1, -3.1, -4.1]]),
                "response_mask": torch.tensor([[1, 1, 1, 1]]),
                "responses": torch.zeros((1, 4)),
            }
        )
        metrics = calculate_debug_metrics(data)
        assert metrics["per_round/total_rounds"] == 1
        assert "per_round/round_0_abs_diff_mean" in metrics
        self.assertAlmostEqual(metrics["per_round/round_0_abs_diff_mean"], 0.1, places=5)

    def test_per_round_metrics_multi_turn(self):
        """Multi-turn: two rounds separated by env tokens."""
        # Round 0: positions 0-1, identical logprobs -> diff=0
        # Round 1: positions 4-5, different logprobs -> diff=1.0
        data = DataProto.from_dict(
            {
                "rollout_log_probs": torch.tensor([[-1.0, -2.0, -9.0, -9.0, -3.0, -4.0]]),
                "old_log_probs": torch.tensor([[-1.0, -2.0, -9.0, -9.0, -4.0, -5.0]]),
                "response_mask": torch.tensor([[1, 1, 0, 0, 1, 1]]),
                "responses": torch.zeros((1, 6)),
            }
        )
        metrics = calculate_debug_metrics(data)
        assert metrics["per_round/total_rounds"] == 2
        # Round 0: identical logprobs
        self.assertAlmostEqual(metrics["per_round/round_0_abs_diff_mean"], 0.0, places=5)
        # Round 1: diff of 1.0 each
        self.assertAlmostEqual(metrics["per_round/round_1_abs_diff_mean"], 1.0, places=5)
        # Max diff should be round 1
        assert metrics["per_round/max_round_diff"] == 1
        self.assertAlmostEqual(metrics["per_round/max_diff_value"], 1.0, places=5)

    def test_per_round_metrics_batch(self):
        """Batch with different number of rounds per sample."""
        # Sample 0: 1 round (positions 0-2)
        # Sample 1: 2 rounds (positions 0-1, positions 3-4)
        data = DataProto.from_dict(
            {
                "rollout_log_probs": torch.tensor(
                    [
                        [-1.0, -2.0, -3.0, -9.0, -9.0],
                        [-1.0, -2.0, -9.0, -3.0, -4.0],
                    ]
                ),
                "old_log_probs": torch.tensor(
                    [
                        [-1.0, -2.0, -3.0, -9.0, -9.0],
                        [-1.0, -2.0, -9.0, -3.5, -4.5],
                    ]
                ),
                "response_mask": torch.tensor(
                    [
                        [1, 1, 1, 0, 0],
                        [1, 1, 0, 1, 1],
                    ]
                ),
                "responses": torch.zeros((2, 5)),
            }
        )
        metrics = calculate_debug_metrics(data)
        # Max rounds across batch is 2
        assert metrics["per_round/total_rounds"] == 2
        assert "per_round/round_0_abs_diff_mean" in metrics
        assert "per_round/round_1_abs_diff_mean" in metrics
        assert metrics["per_round/round_0_token_count"] == 5  # 3 from sample 0 + 2 from sample 1
        assert metrics["per_round/round_1_token_count"] == 2  # only from sample 1


if __name__ == "__main__":
    unittest.main()
