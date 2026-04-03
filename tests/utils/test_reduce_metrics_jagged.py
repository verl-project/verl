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
Tests for reduce_metrics handling of jagged (nested) metric lists.

When legacy workers call prepare_dynamic_batch without dp_group, different
DP ranks can produce different numbers of micro-batches.  The resulting
metric dicts then contain jagged lists (e.g. [[1.0, 2.0], [3.0]]) instead
of flat lists.  Before the fix, calling np.mean / np.max / np.min on such
a structure would raise:
    ValueError: setting an array element with a sequence

These tests verify the defensive _flatten_metric_values helper and the
updated reduce_metrics function handle this case correctly.
"""

import math

import pytest

from verl.utils.metric.utils import Metric, _flatten_metric_values, reduce_metrics

# ---------------------------------------------------------------------------
# _flatten_metric_values
# ---------------------------------------------------------------------------


class TestFlattenMetricValues:
    """Tests for the _flatten_metric_values helper."""

    def test_flat_list_unchanged(self):
        """A flat list of scalars should be returned as-is."""
        values = [1.0, 2.0, 3.0]
        assert _flatten_metric_values(values) == [1.0, 2.0, 3.0]

    def test_jagged_list(self):
        """A jagged nested list should be flattened to scalars."""
        values = [[1.0, 2.0], [3.0]]
        assert _flatten_metric_values(values) == [1.0, 2.0, 3.0]

    def test_uniform_nested_list(self):
        """A uniformly nested list should also be flattened."""
        values = [[1.0, 2.0], [3.0, 4.0]]
        assert _flatten_metric_values(values) == [1.0, 2.0, 3.0, 4.0]

    def test_deeply_nested(self):
        """Deeply nested structures should be recursively flattened."""
        values = [[[1.0], [2.0, 3.0]], [4.0]]
        assert _flatten_metric_values(values) == [1.0, 2.0, 3.0, 4.0]

    def test_empty_list(self):
        """An empty list should return an empty list."""
        assert _flatten_metric_values([]) == []

    def test_mixed_scalars_and_lists(self):
        """A mix of scalars and sub-lists should be flattened correctly."""
        values = [1.0, [2.0, 3.0], 4.0]
        assert _flatten_metric_values(values) == [1.0, 2.0, 3.0, 4.0]

    def test_tuples_flattened(self):
        """Tuples should also be recursively flattened."""
        values = [(1.0, 2.0), (3.0,)]
        assert _flatten_metric_values(values) == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# reduce_metrics with jagged lists
# ---------------------------------------------------------------------------


class TestReduceMetricsJagged:
    """Tests that reduce_metrics handles jagged nested lists without crashing."""

    def test_mean_jagged(self):
        """Mean over a jagged list should flatten first, then compute."""
        metrics = {"loss": [[1.0, 2.0], [3.0]]}
        result = reduce_metrics(metrics)
        assert result["loss"] == pytest.approx(2.0)

    def test_max_jagged(self):
        """Max over a jagged list (key contains 'max')."""
        metrics = {"max_reward": [[5.0, 8.0], [6.0]]}
        result = reduce_metrics(metrics)
        assert result["max_reward"] == pytest.approx(8.0)

    def test_min_jagged(self):
        """Min over a jagged list (key contains 'min')."""
        metrics = {"min_error": [[0.3, 0.1], [0.2]]}
        result = reduce_metrics(metrics)
        assert result["min_error"] == pytest.approx(0.1)

    def test_flat_list_still_works(self):
        """Flat lists should continue to work as before (regression check)."""
        metrics = {
            "loss": [1.0, 2.0, 3.0],
            "max_reward": [5.0, 8.0, 6.0],
            "min_error": [0.1, 0.05, 0.2],
        }
        result = reduce_metrics(metrics)
        assert result["loss"] == pytest.approx(2.0)
        assert result["max_reward"] == pytest.approx(8.0)
        assert result["min_error"] == pytest.approx(0.05)

    def test_single_value(self):
        """Single-element list should still work."""
        metrics = {"loss": [42.0]}
        result = reduce_metrics(metrics)
        assert result["loss"] == pytest.approx(42.0)

    def test_empty_list_returns_nan(self):
        """Empty list should produce NaN (existing behavior)."""
        metrics = {"loss": []}
        result = reduce_metrics(metrics)
        assert math.isnan(result["loss"])

    def test_metric_object_not_affected(self):
        """Metric objects should still use their own aggregate method."""
        metric = Metric(aggregation="mean")
        metric.extend([1.0, 2.0, 3.0])
        metrics = {"custom": metric, "loss": [[1.0], [2.0, 3.0]]}
        result = reduce_metrics(metrics)
        assert result["custom"] == pytest.approx(2.0)
        assert result["loss"] == pytest.approx(2.0)

    def test_deeply_nested_jagged(self):
        """Deeply nested jagged lists should be handled gracefully."""
        metrics = {"loss": [[[1.0, 2.0], [3.0]], [4.0, 5.0]]}
        result = reduce_metrics(metrics)
        assert result["loss"] == pytest.approx(3.0)
