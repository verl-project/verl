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

import torch

from verl.protocol import DataProto
from verl.utils.debug.metrics import calculate_debug_metrics

DEBUG_METRIC_KEYS = {
    "training/rollout_probs_diff_valid",
    "training/rollout_probs_diff_max",
    "training/rollout_probs_diff_mean",
    "training/rollout_probs_diff_std",
    "training/rollout_actor_probs_pearson_corr",
}


def _make_data(rollout_log_probs, old_log_probs, mask):
    rollout_log_probs = torch.tensor(rollout_log_probs, dtype=torch.float32)
    old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
    mask = torch.tensor(mask)
    return DataProto.from_dict(
        {
            "rollout_log_probs": rollout_log_probs,
            "old_log_probs": old_log_probs,
            "response_mask": mask,
            "responses": torch.zeros_like(rollout_log_probs),
        }
    )


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
        # all five debug metrics must be emitted, including the Pearson correlation
        assert DEBUG_METRIC_KEYS.issubset(metrics.keys())

    def test_pearson_corr_perfectly_correlated(self):
        # identical rollout/actor log-probs => probs are identical => correlation == 1.0
        log_probs = [[-0.1, -0.5, -1.0, -2.0]]
        data = _make_data(log_probs, log_probs, [[1, 1, 1, 1]])
        metrics = calculate_debug_metrics(data)
        assert metrics["training/rollout_probs_diff_valid"] == 1
        self.assertAlmostEqual(metrics["training/rollout_actor_probs_pearson_corr"], 1.0, places=4)

    def test_pearson_corr_anti_correlated(self):
        # probs [0.1,0.2,0.3,0.4] vs [0.4,0.3,0.2,0.1] are perfectly anti-correlated => -1.0
        rollout = [[math.log(p) for p in (0.1, 0.2, 0.3, 0.4)]]
        actor = [[math.log(p) for p in (0.4, 0.3, 0.2, 0.1)]]
        data = _make_data(rollout, actor, [[1, 1, 1, 1]])
        metrics = calculate_debug_metrics(data)
        self.assertAlmostEqual(metrics["training/rollout_actor_probs_pearson_corr"], -1.0, places=4)

    def test_empty_mask_returns_nan(self):
        # no valid tokens => valid flag 0 and nan statistics (guards against div-by-zero)
        log_probs = [[-0.1, -0.5, -1.0, -2.0]]
        data = _make_data(log_probs, log_probs, [[0, 0, 0, 0]])
        metrics = calculate_debug_metrics(data)
        assert metrics["training/rollout_probs_diff_valid"] == 0
        assert math.isnan(metrics["training/rollout_actor_probs_pearson_corr"])


if __name__ == "__main__":
    unittest.main()
