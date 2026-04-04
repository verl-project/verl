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
Tests for the metric utilities in verl.trainer.ppo.metric_utils.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from verl.trainer.ppo.metric_utils import (
    bootstrap_metric,
    calc_maj_val,
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.metric.utils import (
    AggregationType,
    Metric,
)


class TestReduceMetrics(unittest.TestCase):
    """Tests for the reduce_metrics function."""

    def test_reduce_metrics_basic(self):
        """Test that reduce_metrics correctly computes means."""
        metrics = {
            "loss": [1.0, 2.0, 3.0],
            "accuracy": [0.0, 0.5, 1.0],
        }
        result = reduce_metrics(metrics)

        self.assertEqual(result["loss"], 2.0)
        self.assertEqual(result["accuracy"], 0.5)

    def test_reduce_metrics_empty(self):
        """Test that reduce_metrics handles empty lists."""
        metrics = {
            "empty": [],
        }
        result = reduce_metrics(metrics)

        self.assertTrue(np.isnan(result["empty"]))

    def test_reduce_metrics_single_value(self):
        """Test that reduce_metrics works with single values."""
        metrics = {
            "single": [5.0],
        }
        result = reduce_metrics(metrics)

        self.assertEqual(result["single"], 5.0)


class TestMetric(unittest.TestCase):
    """Tests for the Metric class."""

    # ================= init tests =================
    def test_init_with_string_aggregation(self):
        """Test Metric initialization with string aggregation type."""
        metric = Metric(aggregation="mean")
        self.assertEqual(metric.aggregation, AggregationType.MEAN)
        self.assertEqual(metric.count, 0)

    def test_init_with_enum_aggregation(self):
        """Test Metric initialization with AggregationType enum."""
        metric = Metric(aggregation=AggregationType.SUM)
        self.assertEqual(metric.aggregation, AggregationType.SUM)
        self.assertEqual(metric.count, 0)

    def test_init_with_value(self):
        """Test Metric initialization with an initial value."""
        x = 5.0
        metric = Metric(aggregation="mean", value=x)
        self.assertEqual(metric.count, 1)
        self.assertEqual(metric.aggregate(), x)

    def test_init_with_invalid_aggregation(self):
        """Test Metric initialization with invalid aggregation type."""
        with self.assertRaises(ValueError):
            Metric(aggregation="invalid")

    # ================= accumulate tests =================
    def test_accumulate_float(self):
        """Test accumulating float values."""
        metric = Metric(aggregation="mean")
        x1, x2 = 1.0, 2.0
        metric.accumulate(x1)
        metric.accumulate(x2)
        self.assertEqual(metric.count, 2)
        self.assertEqual(metric.aggregate(), np.mean([x1, x2]))

    def test_accumulate_int(self):
        """Test accumulating int values."""
        metric = Metric(aggregation="mean")
        x1, x2 = 1, 2
        metric.accumulate(x1)
        metric.accumulate(x2)
        self.assertEqual(metric.count, 2)
        self.assertEqual(metric.aggregate(), np.mean([x1, x2]))

    def test_accumulate_scalar_tensor(self):
        """Test accumulating scalar tensor values."""
        metric = Metric(aggregation="mean")
        x1, x2 = torch.tensor(3.0), torch.tensor(4.0)
        metric.accumulate(x1)
        metric.accumulate(x2)
        self.assertEqual(metric.count, 2)
        self.assertEqual(metric.aggregate(), np.mean([x1, x2]))

    def test_accumulate_tensor_flattens(self):
        """Test accumulating non-scalar tensors flattens their values."""
        metric = Metric(aggregation="sum")
        tensor = torch.tensor([1.0, 2.0])
        metric.accumulate(tensor)
        self.assertEqual(metric.count, 2)
        self.assertEqual(metric.aggregate(), tensor.sum().item())

    def test_accumulate_scalar_ndarray(self):
        """Test accumulating scalar numpy array values."""
        metric = Metric(aggregation="mean")
        x1, x2 = np.array(3.0), np.array(5.0)
        metric.accumulate(x1)
        metric.accumulate(x2)
        self.assertEqual(metric.count, 2)
        self.assertEqual(metric.aggregate(), np.mean([x1, x2]))

    def test_accumulate_ndarray_flattens(self):
        """Test accumulating numpy arrays flattens their values."""
        metric = Metric(aggregation="sum")
        array = np.array([[1.0, 2.0], [3.0, 4.0]])
        metric.accumulate(array)
        self.assertEqual(metric.count, array.size)
        self.assertEqual(metric.aggregate(), array.sum())

    def test_accumulate_list(self):
        """Test accumulating a list of values."""
        metric = Metric(aggregation="mean")
        ls = [1.0, 2.0, 3.0]
        metric.accumulate(ls)
        self.assertEqual(metric.count, len(ls))
        self.assertEqual(metric.aggregate(), np.mean(ls))

    def test_accumulate_metric(self):
        """Test accumulating another Metric merges aggregate state."""
        ls = [1.0, 2.0, 3.0]
        metric1 = Metric(aggregation="mean", value=ls[0])
        metric1.accumulate(ls[1:-1])

        metric2 = Metric(aggregation="mean", value=ls[-1])
        metric2.accumulate(metric1)

        self.assertEqual(metric2.count, len(ls))
        self.assertEqual(metric2.aggregate(), np.mean(ls))

    def test_accumulate_aggregation_mismatch_raises(self):
        """Test that accumulating mismatched aggregation raises ValueError."""
        metric1 = Metric(aggregation="mean")
        metric2 = Metric(aggregation="sum")

        with self.assertRaises(ValueError):
            metric1.accumulate(metric2)

    def test_accumulate_empty_list_is_noop(self):
        """Test accumulating an empty list leaves metric empty."""
        metric = Metric(aggregation="sum")
        metric.accumulate([])
        self.assertEqual(metric.count, 0)
        self.assertTrue(np.isnan(metric.aggregate()))

    def test_accumulate_tuple(self):
        """Test accumulating a tuple of scalar values."""
        metric = Metric(aggregation="sum")
        tup = (1.0, 2.0, 3.0)
        metric.accumulate(tup)
        self.assertEqual(metric.count, len(tup))
        self.assertEqual(metric.aggregate(), sum(list(tup)))

    def test_accumulate_nested_list_raises(self):
        """Test nested lists are rejected."""
        metric = Metric(aggregation="mean")
        with self.assertRaises(ValueError):
            metric.accumulate([[1.0], [2.0]])

    def test_accumulate_list_of_tensors_raises(self):
        """Test Python sequences of tensors are rejected."""
        metric = Metric(aggregation="mean")
        with self.assertRaises(ValueError):
            metric.accumulate([torch.tensor(1.0), torch.tensor(2.0)])

    # ================= aggregate tests =================
    def test_aggregate_mean(self):
        """Test aggregation with mean."""
        metric = Metric(aggregation="mean")
        ls = [1.0, 2.0, 3.0, 4.0]
        metric.accumulate(ls)
        self.assertEqual(metric.aggregate(), np.mean(ls))

    def test_aggregate_sum(self):
        """Test aggregation with sum."""
        metric = Metric(aggregation="sum")
        ls = [1.0, 2.0, 3.0, 4.0]
        metric.accumulate(ls)
        self.assertEqual(metric.aggregate(), sum(ls))

    def test_aggregate_min(self):
        """Test aggregation with min."""
        metric = Metric(aggregation="min")
        ls = [3.0, 1.0, 4.0, 2.0]
        metric.accumulate(ls)
        self.assertEqual(metric.aggregate(), min(ls))

    def test_aggregate_max(self):
        """Test aggregation with max."""
        metric = Metric(aggregation="max")
        ls = [3.0, 1.0, 4.0, 2.0]
        metric.accumulate(ls)
        self.assertEqual(metric.aggregate(), max(ls))

    def test_aggregate_empty_returns_nan(self):
        """Test aggregating an empty metric returns NaN."""
        self.assertTrue(np.isnan(Metric(aggregation="mean").aggregate()))

    # ================= union tests =================
    def test_union_sum_mean(self):
        """Test union with SUM and MEAN aggregations."""
        ls1, ls2 = [1.0, 2.0], [3.0, 4.0]
        metric1 = Metric(aggregation="sum", value=ls1)
        metric2 = Metric(aggregation="sum", value=ls2)
        self.assertEqual(Metric.union(metric1, metric2).aggregate(), sum(ls1 + ls2))

        metric4 = Metric(aggregation="mean", value=ls1)
        metric5 = Metric(aggregation="mean", value=ls2)
        self.assertEqual(Metric.union(metric4, metric5).aggregate(), np.mean(ls1 + ls2))

    def test_union_min_max(self):
        """Test union with MIN and MAX aggregations."""
        metric1 = Metric(aggregation="max")
        ls1, ls2 = [1.0, 2.0], [3.0, 4.0]
        metric1.accumulate(ls1)

        metric2 = Metric(aggregation="max")
        metric2.accumulate(ls2)

        self.assertEqual(Metric.union(metric1, metric2).aggregate(), max(ls1 + ls2))

        metric4 = Metric(aggregation="min")
        metric4.accumulate(ls1)

        metric5 = Metric(aggregation="min")
        metric5.accumulate(ls2)

        self.assertEqual(Metric.union(metric4, metric5).aggregate(), min(ls1 + ls2))

    def test_union_empty_raises(self):
        """Test union raises on empty input."""
        with self.assertRaises(ValueError):
            Metric.union()

    def test_union_aggregation_mismatch_raises(self):
        """Test union raises on mismatched aggregations."""
        metric1 = Metric(aggregation="sum")
        metric2 = Metric(aggregation="mean")
        with self.assertRaises(ValueError):
            Metric.union(metric1, metric2)

    def test_union_does_not_mutate_inputs(self):
        """Test union leaves input metrics unchanged."""
        ls1, ls2 = [1.0, 2.0], [3.0, 4.0]
        metric1 = Metric(aggregation="sum", value=ls1)
        metric2 = Metric(aggregation="sum", value=ls2)
        merged = Metric.union(metric1, metric2)

        self.assertEqual(metric1.aggregate(), sum(ls1))
        self.assertEqual(metric2.aggregate(), sum(ls2))
        self.assertEqual(merged.aggregate(), sum(ls1 + ls2))

    def test_union_weighted_mean_across_unequal_counts(self):
        """Test union computes weighted means across uneven local counts."""
        ls1, ls2, ls3, ls4 = [1.0] * 25, [2.0] * 24, [3.0] * 25, [4.0] * 26
        metric1 = Metric(aggregation="mean", value=ls1)
        metric2 = Metric(aggregation="mean", value=ls2)
        metric3 = Metric(aggregation="mean", value=ls3)
        metric4 = Metric(aggregation="mean", value=ls4)

        self.assertEqual(Metric.union(metric1, metric2, metric3, metric4).aggregate(), np.mean(ls1 + ls2 + ls3 + ls4))

    # ================= dp aggregate =================
    def test_aggregate_dp_sum_averages_across_ranks(self):
        """Test DP aggregation for SUM metrics averages rank contributions."""
        ls1, ls2 = [1.0, 2.0], [3.0, 4.0]
        metric1 = Metric(aggregation="sum", value=ls1)
        metric2 = Metric(aggregation="sum", value=ls2)
        self.assertEqual(Metric.aggregate_dp([metric1, metric2]), np.mean([sum(ls1), sum(ls2)]))

    def test_aggregate_dp_mean_is_weighted(self):
        """Test DP aggregation for MEAN metrics remains weighted by count."""
        ls1, ls2, ls3, ls4 = [1.0] * 25, [2.0] * 24, [3.0] * 25, [4.0] * 26
        metrics = [
            Metric(aggregation="mean", value=ls1),
            Metric(aggregation="mean", value=ls2),
            Metric(aggregation="mean", value=ls3),
            Metric(aggregation="mean", value=ls4),
        ]
        self.assertAlmostEqual(Metric.aggregate_dp(metrics), np.mean(ls1 + ls2 + ls3 + ls4))

    def test_aggregate_dp_min_max(self):
        """Test DP aggregation for MIN and MAX metrics uses global extrema."""
        ls1, ls2 = [3.0, 1.0], [4.0, 2.0]
        metric_min_1 = Metric(aggregation="min", value=ls1)
        metric_min_2 = Metric(aggregation="min", value=ls2)
        self.assertEqual(Metric.aggregate_dp([metric_min_1, metric_min_2]), min(ls1 + ls2))

        metric_max_1 = Metric(aggregation="max", value=ls1)
        metric_max_2 = Metric(aggregation="max", value=ls2)
        self.assertEqual(Metric.aggregate_dp([metric_max_1, metric_max_2]), max(ls1 + ls2))

    def test_aggregate_dp_empty_raises(self):
        """Test DP aggregation raises on empty input."""
        with self.assertRaises(ValueError):
            Metric.aggregate_dp([])

    # ================= reduce metrics tests =================
    def test_reduce_metrics_with_metric(self):
        """Test reduce_metrics correctly handles Metric objects."""
        ls1, ls2 = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
        metric = Metric(aggregation="mean")
        metric.accumulate(ls1)

        metrics = {
            "custom_metric": metric,
            "list_metric": ls2,
        }
        result = reduce_metrics(metrics)

        self.assertEqual(result["custom_metric"], np.mean(ls1))
        self.assertEqual(result["list_metric"], np.mean(ls2))


class TestComputeDataMetrics(unittest.TestCase):
    """Tests for the compute_data_metrics function."""

    def setUp(self):
        """Set up common test data."""
        # Create a mock DataProto object
        self.batch = MagicMock()
        self.batch.batch = {
            "token_level_scores": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "token_level_rewards": torch.tensor([[0.5, 1.0], [1.5, 2.0]]),
            "advantages": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            "returns": torch.tensor([[1.1, 1.2], [1.3, 1.4]]),
            "responses": torch.zeros((2, 2)),  # 2 samples, 2 tokens each
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1],  # 2 prompt tokens, 2 response tokens
                    [1, 1, 1, 1],
                ]
            ),
            "response_mask": torch.tensor(
                [
                    [1, 1],  # 2 response tokens
                    [1, 1],
                ]
            ),
            "values": torch.tensor([[0.9, 1.0], [1.1, 1.2]]),
        }

    def test_compute_data_metrics_with_critic(self):
        """Test compute_data_metrics with critic enabled."""
        metrics = compute_data_metrics(self.batch, use_critic=True)

        # Check that all expected metrics are present
        self.assertIn("critic/score/mean", metrics)
        self.assertIn("critic/rewards/mean", metrics)
        self.assertIn("critic/advantages/mean", metrics)
        self.assertIn("critic/returns/mean", metrics)
        self.assertIn("critic/values/mean", metrics)
        self.assertIn("critic/vf_explained_var", metrics)
        self.assertIn("response_length/mean", metrics)
        self.assertIn("prompt_length/mean", metrics)

        # Check some specific values
        self.assertAlmostEqual(metrics["critic/score/mean"], 5.0)  # Sum of token_level_scores
        self.assertAlmostEqual(metrics["critic/rewards/mean"], 2.5)  # Sum of token_level_rewards

    def test_compute_data_metrics_without_critic(self):
        """Test compute_data_metrics with critic disabled."""
        metrics = compute_data_metrics(self.batch, use_critic=False)

        # Check that critic-specific metrics are not present
        self.assertNotIn("critic/values/mean", metrics)
        self.assertNotIn("critic/vf_explained_var", metrics)

        # Check that other metrics are still present
        self.assertIn("critic/score/mean", metrics)
        self.assertIn("critic/rewards/mean", metrics)
        self.assertIn("response_length/mean", metrics)


class TestComputeTimingMetrics(unittest.TestCase):
    """Tests for the compute_timing_metrics function."""

    def setUp(self):
        """Set up common test data."""
        # Create a mock DataProto object
        self.batch = MagicMock()
        self.batch.batch = {
            "responses": torch.zeros((2, 3)),  # 2 samples, 3 response tokens each
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1],  # 3 prompt tokens, 3 response tokens
                    [1, 1, 1, 1, 1, 1],
                ]
            ),
        }

        # Mock the _compute_response_info function to return known values
        self.response_info = {
            "prompt_length": torch.tensor([3.0, 3.0]),
            "response_length": torch.tensor([3.0, 3.0]),
            "response_mask": torch.ones((2, 3)),
        }

    @patch("verl.trainer.ppo.metric_utils._compute_response_info")
    def test_compute_timing_metrics(self, mock_compute_response_info):
        """Test compute_timing_metrics with various timing data."""
        mock_compute_response_info.return_value = self.response_info

        timing_raw = {
            "gen": 0.5,  # 500ms
            "ref": 0.3,  # 300ms
            "values": 0.2,  # 200ms
        }

        metrics = compute_timing_metrics(self.batch, timing_raw)

        # Check raw timing metrics
        self.assertEqual(metrics["timing_s/gen"], 0.5)
        self.assertEqual(metrics["timing_s/ref"], 0.3)
        self.assertEqual(metrics["timing_s/values"], 0.2)

        # Check per-token timing metrics
        # gen uses only response tokens (6 tokens)
        self.assertAlmostEqual(metrics["timing_per_token_ms/gen"], 0.5 * 1000 / 6, places=5)

        # ref and values use all tokens (12 tokens)
        self.assertAlmostEqual(metrics["timing_per_token_ms/ref"], 0.3 * 1000 / 12, places=5)
        self.assertAlmostEqual(metrics["timing_per_token_ms/values"], 0.2 * 1000 / 12, places=5)


class TestComputeThroughputMetrics(unittest.TestCase):
    """Tests for the compute_throughout_metrics function."""

    def setUp(self):
        """Set up common test data."""
        # Create a mock DataProto object
        self.batch = MagicMock()
        self.batch.meta_info = {
            "global_token_num": [100, 200, 300],  # 600 tokens total
        }

    def test_compute_throughout_metrics(self):
        """Test compute_throughout_metrics with various timing data."""
        timing_raw = {
            "step": 2.0,  # 2 seconds per step
        }

        # Test with 1 GPU
        metrics = compute_throughout_metrics(self.batch, timing_raw, n_gpus=1)

        self.assertEqual(metrics["perf/total_num_tokens"], 600)
        self.assertEqual(metrics["perf/time_per_step"], 2.0)
        self.assertEqual(metrics["perf/throughput"], 600 / 2.0)  # 300 tokens/sec

        # Test with 2 GPUs
        metrics = compute_throughout_metrics(self.batch, timing_raw, n_gpus=2)

        self.assertEqual(metrics["perf/total_num_tokens"], 600)
        self.assertEqual(metrics["perf/time_per_step"], 2.0)
        self.assertEqual(metrics["perf/throughput"], 600 / (2.0 * 2))  # 150 tokens/sec/GPU


class TestBootstrapMetric(unittest.TestCase):
    """Tests for the bootstrap_metric function."""

    def test_bootstrap_metric_basic(self):
        """Test bootstrap_metric with simple data and functions."""
        data = [1, 2, 3, 4, 5]
        reduce_fns = [np.mean, np.max]

        # Use a fixed seed for reproducibility
        result = bootstrap_metric(data, subset_size=3, reduce_fns=reduce_fns, n_bootstrap=100, seed=42)

        # Check that we get two results (one for each reduce_fn)
        self.assertEqual(len(result), 2)

        # Each result should be a tuple of (mean, std)
        mean_result, max_result = result
        self.assertEqual(len(mean_result), 2)
        self.assertEqual(len(max_result), 2)

        # The mean of means should be close to the true mean (3.0)
        self.assertAlmostEqual(mean_result[0], 3.0, delta=0.3)

        # The mean of maxes should be close to the expected value for samples of size 3
        # For samples of size 3 from [1,2,3,4,5], the expected max is around 4.0-4.5
        self.assertGreater(max_result[0], 3.5)
        self.assertLess(max_result[0], 5.0)

    def test_bootstrap_metric_empty(self):
        """Test bootstrap_metric with empty data."""
        with self.assertRaises(ValueError):
            bootstrap_metric([], subset_size=1, reduce_fns=[np.mean])


class TestCalcMajVal(unittest.TestCase):
    """Tests for the calc_maj_val function."""

    def test_calc_maj_val_basic(self):
        """Test calc_maj_val with simple data."""
        data = [
            {"pred": "A", "val": 0.9},
            {"pred": "B", "val": 0.8},
            {"pred": "A", "val": 0.7},
        ]

        result = calc_maj_val(data, vote_key="pred", val_key="val")

        # "A" is the majority vote, so we should get the first "val" for "A"
        self.assertEqual(result, 0.9)

    def test_calc_maj_val_tie(self):
        """Test calc_maj_val with tied votes."""
        data = [
            {"pred": "A", "val": 0.9},
            {"pred": "B", "val": 0.8},
            {"pred": "B", "val": 0.7},
            {"pred": "A", "val": 0.6},
        ]

        # In case of a tie, the first key in sorted order wins
        # This depends on Python's dict implementation, but for this test
        # we just verify that one of the valid values is returned
        result = calc_maj_val(data, vote_key="pred", val_key="val")

        self.assertTrue(result in [0.9, 0.8])


class TestProcessValidationMetrics(unittest.TestCase):
    """Tests for the process_validation_metrics function."""

    def test_process_validation_metrics_basic(self):
        """Test process_validation_metrics with simple data."""
        data_sources = ["source1", "source1", "source2"]
        sample_inputs = ["prompt1", "prompt1", "prompt2"]
        infos_dict = {
            "score": [0.8, 0.9, 0.7],
        }

        result = process_validation_metrics(data_sources, sample_inputs, infos_dict, seed=42)

        # Check the structure of the result
        self.assertIn("source1", result)
        self.assertIn("source2", result)

        # Check that source1 has metrics for score
        self.assertIn("score", result["source1"])

        # Check that mean@2 is present for source1/score
        self.assertIn("mean@2", result["source1"]["score"])

        # Check the value of mean@2 for source1/score
        self.assertAlmostEqual(result["source1"]["score"]["mean@2"], 0.85)

    def test_process_validation_metrics_with_pred(self):
        """Test process_validation_metrics with prediction data."""
        data_sources = ["source1", "source1", "source1"]
        sample_inputs = ["prompt1", "prompt1", "prompt1"]
        infos_dict = {
            "score": [0.8, 0.9, 0.7],
            "pred": ["A", "B", "A"],
        }

        result = process_validation_metrics(data_sources, sample_inputs, infos_dict, seed=42)

        # Check that majority voting metrics are present
        self.assertIn("maj@2/mean", result["source1"]["score"])

        # For bootstrap with n=2, the majority vote could be either A or B
        # depending on the random sampling, so we don't check the exact value


if __name__ == "__main__":
    unittest.main()
