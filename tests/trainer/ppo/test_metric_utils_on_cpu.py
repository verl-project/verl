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
    build_metric_batch_fields,
    calc_maj_val,
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    metric_length_caps,
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

    def test_init_with_string_aggregation(self):
        """Test Metric initialization with string aggregation type."""
        metric = Metric(aggregation="mean")
        self.assertEqual(metric.aggregation, AggregationType.MEAN)
        self.assertEqual(metric.values, [])

    def test_init_with_enum_aggregation(self):
        """Test Metric initialization with AggregationType enum."""
        metric = Metric(aggregation=AggregationType.SUM)
        self.assertEqual(metric.aggregation, AggregationType.SUM)
        self.assertEqual(metric.values, [])

    def test_init_with_value(self):
        """Test Metric initialization with an initial value."""
        metric = Metric(aggregation="mean", value=5.0)
        self.assertEqual(metric.values, [5.0])

    def test_init_with_invalid_aggregation(self):
        """Test Metric initialization with invalid aggregation type."""
        with self.assertRaises(ValueError):
            Metric(aggregation="invalid")

    def test_append_float(self):
        """Test appending float values."""
        metric = Metric(aggregation="mean")
        metric.append(1.0)
        metric.append(2.0)
        self.assertEqual(metric.values, [1.0, 2.0])

    def test_append_int(self):
        """Test appending int values."""
        metric = Metric(aggregation="mean")
        metric.append(1)
        metric.append(2)
        self.assertEqual(metric.values, [1, 2])

    def test_append_tensor(self):
        """Test appending scalar tensor values."""
        metric = Metric(aggregation="mean")
        metric.append(torch.tensor(3.0))
        metric.append(torch.tensor(4.0))
        self.assertEqual(metric.values, [3.0, 4.0])

    def test_append_non_scalar_tensor_raises(self):
        """Test that appending non-scalar tensor raises ValueError."""
        metric = Metric(aggregation="mean")
        with self.assertRaises(ValueError):
            metric.append(torch.tensor([1.0, 2.0]))

    def test_append_metric(self):
        """Test appending another Metric extends values."""
        metric1 = Metric(aggregation="mean", value=1.0)
        metric1.append(2.0)

        metric2 = Metric(aggregation="mean", value=3.0)
        metric2.append(metric1)

        self.assertEqual(metric2.values, [3.0, 1.0, 2.0])

    def test_extend_with_list(self):
        """Test extending with a list of values."""
        metric = Metric(aggregation="mean")
        metric.extend([1.0, 2.0, 3.0])
        self.assertEqual(metric.values, [1.0, 2.0, 3.0])

    def test_extend_with_metric(self):
        """Test extending with another Metric."""
        metric1 = Metric(aggregation="mean")
        metric1.extend([1.0, 2.0])

        metric2 = Metric(aggregation="mean")
        metric2.extend([3.0, 4.0])
        metric2.extend(metric1)

        self.assertEqual(metric2.values, [3.0, 4.0, 1.0, 2.0])

    def test_extend_aggregation_mismatch_raises(self):
        """Test that extending with mismatched aggregation raises ValueError."""
        metric1 = Metric(aggregation="mean")
        metric2 = Metric(aggregation="sum")

        with self.assertRaises(ValueError):
            metric1.extend(metric2)

    def test_aggregate_mean(self):
        """Test aggregation with mean."""
        metric = Metric(aggregation="mean")
        metric.extend([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(metric.aggregate(), 2.5)

    def test_aggregate_sum(self):
        """Test aggregation with sum."""
        metric = Metric(aggregation="sum")
        metric.extend([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(metric.aggregate(), 10.0)

    def test_aggregate_min(self):
        """Test aggregation with min."""
        metric = Metric(aggregation="min")
        metric.extend([3.0, 1.0, 4.0, 2.0])
        self.assertEqual(metric.aggregate(), 1.0)

    def test_aggregate_max(self):
        """Test aggregation with max."""
        metric = Metric(aggregation="max")
        metric.extend([3.0, 1.0, 4.0, 2.0])
        self.assertEqual(metric.aggregate(), 4.0)

    def test_aggregate_dp_sum_mean(self):
        """Test aggregate_dp with SUM and MEAN aggregations."""
        # Test with SUM: mean over DP ranks, then sum
        metric1 = Metric(aggregation="sum")
        metric1.extend([1.0, 2.0])

        metric2 = Metric(aggregation="sum")
        metric2.extend([3.0, 4.0])

        result = Metric.aggregate_dp([metric1, metric2])

        # value_arrays = [[1.0, 2.0], [3.0, 4.0]]
        # mean over axis 0 = [2.0, 3.0]
        # sum = 5.0
        self.assertEqual(result, 5.0)

        # Test with MEAN: mean over DP ranks, then mean
        metric4 = Metric(aggregation="mean")
        metric4.extend([1.0, 2.0])

        metric5 = Metric(aggregation="mean")
        metric5.extend([3.0, 4.0])

        result = Metric.aggregate_dp([metric4, metric5])

        # value_arrays = [[1.0, 2.0], [3.0, 4.0]]
        # mean over axis 0 = [2.0, 3.0]
        # mean = 2.5
        self.assertEqual(result, 2.5)

    def test_aggregate_dp_min_max(self):
        """Test aggregate_dp with MIN and MAX aggregations."""
        # Test with MAX: flatten, then max
        metric1 = Metric(aggregation="max")
        metric1.extend([1.0, 2.0])

        metric2 = Metric(aggregation="max")
        metric2.extend([3.0, 4.0])

        result = Metric.aggregate_dp([metric1, metric2])

        # value_arrays = [[1.0, 2.0], [3.0, 4.0]]
        # flatten = [1.0, 2.0, 3.0, 4.0]
        # max = 4.0
        self.assertEqual(result, 4.0)

        # Test with MIN: flatten, then min
        metric4 = Metric(aggregation="min")
        metric4.extend([1.0, 2.0])

        metric5 = Metric(aggregation="min")
        metric5.extend([3.0, 4.0])

        result = Metric.aggregate_dp([metric4, metric5])

        # value_arrays = [[1.0, 2.0], [3.0, 4.0]]
        # flatten = [1.0, 2.0, 3.0, 4.0]
        # min = 1.0
        self.assertEqual(result, 1.0)

    def test_aggregate_dp_mismatched_lengths(self):
        """Test aggregate_dp raises error with mismatched value lengths."""
        metric1 = Metric(aggregation="sum")
        metric1.extend([1.0, 2.0])

        metric2 = Metric(aggregation="sum")
        metric2.extend([3.0, 4.0, 5.0])  # Different length

        with self.assertRaises(ValueError):
            Metric.aggregate_dp([metric1, metric2])

    def test_from_dict(self):
        """Test from_dict creates Metrics from dictionary."""
        data = {"loss": 1.0, "accuracy": 0.9}
        metrics = Metric.from_dict(data, aggregation="mean")

        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertEqual(metrics["loss"].values, [1.0])
        self.assertEqual(metrics["accuracy"].values, [0.9])
        self.assertEqual(metrics["loss"].aggregation, AggregationType.MEAN)

    def test_init_list(self):
        """Test init_list creates new empty Metric with same aggregation."""
        metric = Metric(aggregation="max")
        metric.extend([1.0, 2.0])

        new_metric = metric.init_list()

        self.assertEqual(new_metric.aggregation, AggregationType.MAX)
        self.assertEqual(new_metric.values, [])

    def test_reduce_metrics_with_metric(self):
        """Test reduce_metrics correctly handles Metric objects."""
        metric = Metric(aggregation="mean")
        metric.extend([1.0, 2.0, 3.0])

        metrics = {
            "custom_metric": metric,
            "list_metric": [4.0, 5.0, 6.0],
        }
        result = reduce_metrics(metrics)

        self.assertEqual(result["custom_metric"], 2.0)
        self.assertEqual(result["list_metric"], 5.0)


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
            "prompts": torch.zeros((2, 2)),  # 2 samples, 2 tokens for each prompt
            "responses": torch.zeros((2, 2)),  # 2 samples, 2 tokens for each response
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

    @patch("verl.trainer.ppo.metric_utils._compute_response_info")
    def test_compute_timing_metrics_zero_tokens(self, mock_compute_response_info):
        """Regression test: zero tokens should return 0.0, not crash or report misleading values."""
        zero_response_info = {
            "prompt_length": torch.tensor([0.0, 0.0]),
            "response_length": torch.tensor([0.0, 0.0]),
            "response_mask": torch.zeros((2, 3)),
        }
        mock_compute_response_info.return_value = zero_response_info

        timing_raw = {
            "gen": 0.5,
            "ref": 0.3,
            "values": 0.2,
        }

        metrics = compute_timing_metrics(self.batch, timing_raw)

        # All per-token metrics should be 0.0 when there are no tokens
        self.assertEqual(metrics["timing_per_token_ms/gen"], 0.0)
        self.assertEqual(metrics["timing_per_token_ms/ref"], 0.0)
        self.assertEqual(metrics["timing_per_token_ms/values"], 0.0)

        # Raw timing should still be reported
        self.assertEqual(metrics["timing_s/gen"], 0.5)


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

    def test_process_validation_metrics_counts_missing_sessions_as_incorrect(self):
        data_sources = ["source1", "source1"]
        sample_uids = ["prompt1", "prompt1"]
        infos_dict = {
            "reward": [0.4, 0.6],
            "acc": [1.0, 1.0],
        }

        result = process_validation_metrics(
            data_sources,
            sample_uids,
            infos_dict,
            expected_acc_counts={
                ("source1", "prompt1"): 4,
                ("source1", "prompt2"): 4,
            },
        )

        self.assertAlmostEqual(result["source1"]["acc"]["mean@4"], 0.25)
        self.assertAlmostEqual(result["source1"]["acc_success_only"]["mean@2"], 1.0)
        self.assertAlmostEqual(result["source1"]["reward"]["mean@2"], 0.5)

    def test_process_validation_metrics_does_not_add_acc_to_reward_only_source(self):
        result = process_validation_metrics(
            data_sources=["acc_source", "reward_source"],
            sample_uids=["acc_prompt", "reward_prompt"],
            infos_dict={"reward": [1.0, 0.5], "acc": [1.0, None]},
            expected_acc_counts={
                ("acc_source", "acc_prompt"): 2,
                ("reward_source", "reward_prompt"): 2,
            },
        )

        self.assertAlmostEqual(result["acc_source"]["acc"]["mean@2"], 0.5)
        self.assertAlmostEqual(result["acc_source"]["acc_success_only"]["mean@1"], 1.0)
        self.assertNotIn("acc", result["reward_source"])
        self.assertNotIn("acc_success_only", result["reward_source"])


class TestComputeDataMetricsLengthCaps(unittest.TestCase):
    """Regression tests: truncation rates must use the configured length caps.

    V1 stores jagged sequences and pads them to the longest sample in the batch,
    so ``prompts``/``responses`` tensor widths are not the configured caps.
    """

    def _make_v1_batch(self, meta_info=None):
        batch = MagicMock()
        batch.meta_info = dict(meta_info or {})
        batch.non_tensor_batch = {}
        batch.batch = {
            "token_level_scores": torch.zeros((4, 512)),
            "token_level_rewards": torch.zeros((4, 512)),
            "advantages": torch.zeros((4, 512)),
            "returns": torch.zeros((4, 512)),
            "values": torch.zeros((4, 512)),
            "prompts": torch.zeros((4, 70)),
            "responses": torch.zeros((4, 512)),
            "response_mask": torch.ones((4, 512)),
            # V1 carries the true per-sample lengths as separate columns.
            "prompt_length": torch.tensor([50.0, 60.0, 70.0, 70.0]),
            "response_length": torch.tensor([100.0, 200.0, 512.0, 512.0]),
        }
        return batch

    def test_meta_info_caps_report_no_truncation(self):
        # Batch max width is 512/70 but the configured caps are 4096/1024 and
        # nothing was truncated, so both clip ratios must be 0.0 (they were 0.5
        # when the width was used as the cap).
        batch = self._make_v1_batch(meta_info={"max_prompt_length": 1024, "max_response_length": 4096})

        metrics = compute_data_metrics(batch, use_critic=True)

        self.assertAlmostEqual(metrics["response_length/clip_ratio"], 0.0)
        self.assertAlmostEqual(metrics["prompt_length/clip_ratio"], 0.0)
        self.assertAlmostEqual(metrics["response_length_non_aborted/clip_ratio"], 0.0)

    def test_meta_info_caps_detect_real_truncation(self):
        # Two of the four responses hit the configured 512-token cap.
        batch = self._make_v1_batch(meta_info={"max_prompt_length": 70, "max_response_length": 512})

        metrics = compute_data_metrics(batch, use_critic=True)

        self.assertAlmostEqual(metrics["response_length/clip_ratio"], 0.5)
        self.assertAlmostEqual(metrics["prompt_length/clip_ratio"], 0.5)

    def test_explicit_caps_take_precedence_over_meta_info(self):
        batch = self._make_v1_batch(meta_info={"max_prompt_length": 1, "max_response_length": 1})

        metrics = compute_data_metrics(
            batch,
            use_critic=True,
            max_prompt_length=1024,
            max_response_length=4096,
        )

        self.assertAlmostEqual(metrics["response_length/clip_ratio"], 0.0)
        self.assertAlmostEqual(metrics["prompt_length/clip_ratio"], 0.0)

    def test_width_fallback_preserves_legacy_behaviour(self):
        # No caps supplied: keep the historical width-based comparison used by
        # V0-style callers and existing external users.
        batch = self._make_v1_batch()

        metrics = compute_data_metrics(batch, use_critic=True)

        self.assertAlmostEqual(metrics["response_length/clip_ratio"], 0.5)
        self.assertAlmostEqual(metrics["prompt_length/clip_ratio"], 0.5)

    def test_invalid_caps_fall_back_to_width(self):
        # 0 / bool / non-integer caps must not crash torch.eq; they are ignored.
        batch = self._make_v1_batch(meta_info={"max_prompt_length": 0, "max_response_length": True})

        metrics = compute_data_metrics(batch, use_critic=True)

        self.assertAlmostEqual(metrics["response_length/clip_ratio"], 0.5)
        self.assertAlmostEqual(metrics["prompt_length/clip_ratio"], 0.5)


class TestComputeVarianceProxyMetrics(unittest.TestCase):
    """The V1 metrics projection must keep the variance-proxy inputs together."""

    def _make_batch(self, with_variance_inputs: bool = True):
        tensors = {
            "response_mask": torch.ones((1, 1)),
            "advantages": torch.tensor([[2.0]]),
        }
        if with_variance_inputs:
            # Uniform policy over two tokens: pi(y) = 0.5, sum_pi_squared = 0.5,
            # so W = 1 - 2*0.5 + 0.5 = 0.5 and proxy2 = A^2 * W = 2.0.
            tensors["old_log_probs"] = torch.tensor([[0.5]]).log()
            tensors["sum_pi_squared"] = torch.tensor([[0.5]])
        batch = MagicMock()
        batch.batch = tensors
        return batch

    def test_emits_variance_proxy_metrics(self):
        metrics = compute_variance_proxy_metrics(self._make_batch())

        self.assertAlmostEqual(metrics["variance_proxy/proxy2_total_power"], 2.0)
        self.assertAlmostEqual(metrics["variance_proxy/expected_a_squared"], 4.0)

    def test_returns_empty_when_inputs_are_not_projected(self):
        # This is the V1 regression: without old_log_probs/sum_pi_squared the
        # function silently returns {} instead of emitting the proxy metrics.
        self.assertEqual(compute_variance_proxy_metrics(self._make_batch(with_variance_inputs=False)), {})


class TestBuildMetricBatchFields(unittest.TestCase):
    """The V1 metrics projection must fetch every field the metric functions read."""

    def test_always_requests_data_metric_inputs(self):
        fields = build_metric_batch_fields(use_critic=False, calculate_sum_pi_squared=False)

        self.assertEqual(
            fields,
            [
                "prompts",
                "responses",
                "response_mask",
                "advantages",
                "returns",
                "rm_scores",
                "token_level_rewards",
                "num_turns",
            ],
        )

    def test_adds_values_for_critic(self):
        fields = build_metric_batch_fields(use_critic=True, calculate_sum_pi_squared=False)

        self.assertIn("values", fields)
        self.assertNotIn("old_log_probs", fields)
        self.assertNotIn("sum_pi_squared", fields)

    def test_requests_variance_proxy_inputs(self):
        # compute_variance_proxy_metrics needs both fields, so the projection must
        # fetch them whenever the actor emits sum_pi_squared.
        fields = build_metric_batch_fields(use_critic=True, calculate_sum_pi_squared=True)

        self.assertIn("old_log_probs", fields)
        self.assertIn("sum_pi_squared", fields)


class TestMetricLengthCaps(unittest.TestCase):
    """The V1 trainer passes the configured caps through ``batch.meta_info``."""

    def test_uses_rollout_lengths(self):
        from omegaconf import OmegaConf

        config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"prompt_length": 1024, "response_length": 4096}}})

        self.assertEqual(metric_length_caps(config), {"max_prompt_length": 1024, "max_response_length": 4096})

    def test_ignores_missing_and_invalid_values(self):
        from omegaconf import OmegaConf

        for prompt_length, response_length, expected in (
            (None, 4096, {"max_response_length": 4096}),
            (0, -1, {}),
            (True, "512", {}),
        ):
            with self.subTest(prompt_length=prompt_length, response_length=response_length):
                config = OmegaConf.create(
                    {
                        "actor_rollout_ref": {
                            "rollout": {"prompt_length": prompt_length, "response_length": response_length}
                        }
                    }
                )
                self.assertEqual(metric_length_caps(config), expected)


if __name__ == "__main__":
    unittest.main()
