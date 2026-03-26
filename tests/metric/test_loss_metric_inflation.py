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

"""
Test that loss metrics are correctly averaged across mini-batches and ppo_epochs.

Regression test for a bug introduced by PR #4711 where actor/pg_loss,
actor/kl_loss, and critic/vf_loss were accumulated as float scalars via +=
across ALL mini-batches and epochs. Since reduce_metrics calls np.mean on
each value, floats pass through as-is (np.mean(scalar) == scalar), so the
final metric was a sum instead of a mean — inflated by
(num_mini_batches * ppo_epochs).

The fix accumulates loss within each mini-batch's micro-batches, then appends
the mini-batch total to a list via append_to_dict. This way reduce_metrics
correctly averages across mini-batches and epochs.
"""

import unittest

import numpy as np

from verl.utils.metric.utils import reduce_metrics
from verl.utils.py_functional import append_to_dict


class TestLossMetricInflation(unittest.TestCase):
    """Test the metrics aggregation pattern used in dp_actor and dp_critic."""

    def _simulate_old_metrics(self, loss_values, scale_factors, num_mini_batches, ppo_epochs):
        """
        Simulate the OLD (buggy) pattern from PR #4711.

        loss_values: list of per-micro-batch loss values
        scale_factors: list of scale factors (1/grad_accum for each micro-batch)
        Each mini-batch has len(loss_values)//num_mini_batches micro-batches.
        """
        metrics = {"actor/pg_loss": 0.0}  # float, not list
        micro_batches_per_mini = len(loss_values) // num_mini_batches

        idx = 0
        for _epoch in range(ppo_epochs):
            for _mb in range(num_mini_batches):
                for _ub in range(micro_batches_per_mini):
                    metrics["actor/pg_loss"] += loss_values[idx % len(loss_values)] * scale_factors[idx % len(scale_factors)]
                    idx += 1

        # reduce_metrics: np.mean on a float returns the float as-is
        return reduce_metrics(metrics)["actor/pg_loss"]

    def _simulate_new_metrics(self, loss_values, scale_factors, num_mini_batches, ppo_epochs):
        """
        Simulate the NEW (fixed) pattern.

        Loss is accumulated per mini-batch, then appended to list.
        reduce_metrics averages the list across mini-batches and epochs.
        """
        metrics = {}
        micro_batches_per_mini = len(loss_values) // num_mini_batches

        idx = 0
        for _epoch in range(ppo_epochs):
            for _mb in range(num_mini_batches):
                mini_batch_loss = 0.0
                for _ub in range(micro_batches_per_mini):
                    mini_batch_loss += loss_values[idx % len(loss_values)] * scale_factors[idx % len(scale_factors)]
                    idx += 1
                # append mini-batch total to list
                append_to_dict(metrics, {"actor/pg_loss": mini_batch_loss})

        return reduce_metrics(metrics)["actor/pg_loss"]

    def test_single_mini_batch_single_epoch(self):
        """When num_mini_batches=1, ppo_epochs=1: old and new should agree."""
        # 4 micro-batches, grad_accum=4, scale_factor=0.25 each
        losses = [1.0, 2.0, 3.0, 4.0]
        scales = [0.25, 0.25, 0.25, 0.25]

        old = self._simulate_old_metrics(losses, scales, num_mini_batches=1, ppo_epochs=1)
        new = self._simulate_new_metrics(losses, scales, num_mini_batches=1, ppo_epochs=1)

        # sum(loss_i * 0.25) = (1+2+3+4)*0.25 = 2.5
        self.assertAlmostEqual(old, 2.5)
        self.assertAlmostEqual(new, 2.5)

    def test_two_mini_batches_old_inflated(self):
        """With 2 mini-batches: old code returns sum, new returns mean."""
        # 2 mini-batches, each with 2 micro-batches, grad_accum=2, scale=0.5
        losses = [1.0, 2.0, 3.0, 4.0]  # mb1: [1,2], mb2: [3,4]
        scales = [0.5, 0.5, 0.5, 0.5]

        old = self._simulate_old_metrics(losses, scales, num_mini_batches=2, ppo_epochs=1)
        new = self._simulate_new_metrics(losses, scales, num_mini_batches=2, ppo_epochs=1)

        # mini-batch 1: 1*0.5 + 2*0.5 = 1.5
        # mini-batch 2: 3*0.5 + 4*0.5 = 3.5
        # Old: sum = 5.0 (inflated by 2x)
        # New: mean([1.5, 3.5]) = 2.5
        self.assertAlmostEqual(old, 5.0)
        self.assertAlmostEqual(new, 2.5)

    def test_two_epochs_old_inflated(self):
        """With 2 ppo_epochs: old code returns sum, new returns mean."""
        losses = [1.0, 2.0]
        scales = [0.5, 0.5]

        old = self._simulate_old_metrics(losses, scales, num_mini_batches=1, ppo_epochs=2)
        new = self._simulate_new_metrics(losses, scales, num_mini_batches=1, ppo_epochs=2)

        # Each epoch: 1*0.5 + 2*0.5 = 1.5
        # Old: 1.5 + 1.5 = 3.0 (inflated by 2x)
        # New: mean([1.5, 1.5]) = 1.5
        self.assertAlmostEqual(old, 3.0)
        self.assertAlmostEqual(new, 1.5)

    def test_two_mini_batches_two_epochs(self):
        """Combined: 2 mini-batches * 2 epochs = 4x inflation in old code."""
        losses = [1.0, 1.0, 1.0, 1.0]
        scales = [0.5, 0.5, 0.5, 0.5]

        old = self._simulate_old_metrics(losses, scales, num_mini_batches=2, ppo_epochs=2)
        new = self._simulate_new_metrics(losses, scales, num_mini_batches=2, ppo_epochs=2)

        # Each mini-batch: 1*0.5 + 1*0.5 = 1.0
        # 4 mini-batch iterations total (2 epochs * 2 mini-batches)
        # Old: sum of all = 4.0 (inflated by 4x)
        # New: mean([1.0, 1.0, 1.0, 1.0]) = 1.0
        self.assertAlmostEqual(old, 4.0)
        self.assertAlmostEqual(new, 1.0)

    def test_inflation_factor(self):
        """Verify inflation factor = num_mini_batches * ppo_epochs."""
        losses = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        scales = [1/3, 1/3, 1/3, 1/3, 1/3, 1/3]  # 3 micro-batches per mini-batch

        num_mini_batches = 2
        ppo_epochs = 3
        inflation_factor = num_mini_batches * ppo_epochs  # 6

        old = self._simulate_old_metrics(losses, scales, num_mini_batches, ppo_epochs)
        new = self._simulate_new_metrics(losses, scales, num_mini_batches, ppo_epochs)

        self.assertAlmostEqual(old / new, inflation_factor)

    def test_metrics_dict_types(self):
        """After fix, all metric values should be lists (not mixed float/list)."""
        metrics = {}

        # Simulate fixed pattern: loss appended at mini-batch boundary
        for _mb in range(3):
            mini_batch_loss = 1.5
            micro_metrics = {"actor/entropy": 0.3, "actor/clipfrac": 0.1}
            append_to_dict(metrics, micro_metrics)
            append_to_dict(metrics, micro_metrics)  # 2 micro-batches

            mini_batch_metrics = {
                "actor/pg_loss": mini_batch_loss,
                "actor/kl_loss": 0.0,
                "actor/grad_norm": 0.5,
            }
            append_to_dict(metrics, mini_batch_metrics)

        # All values should be lists
        for key, val in metrics.items():
            self.assertIsInstance(val, list, f"metrics['{key}'] should be a list, got {type(val)}")

        # reduce_metrics should work correctly on all-list dict
        reduced = reduce_metrics(metrics)
        self.assertAlmostEqual(reduced["actor/pg_loss"], 1.5)


if __name__ == "__main__":
    unittest.main()
