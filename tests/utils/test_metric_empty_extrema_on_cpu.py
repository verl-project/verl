# Copyright 2026 Individual Contributor: Zupeng Wang
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

import math

import pytest

from verl.utils.metric import AggregationType, Metric


@pytest.mark.parametrize("aggregation", [AggregationType.MIN, AggregationType.MAX])
def test_empty_extrema(aggregation):
    metric = Metric(aggregation)
    assert math.isnan(Metric.aggregate_dp([metric, Metric(aggregation)]))


@pytest.mark.parametrize("aggregation, expected", [(AggregationType.MIN, 2.0), (AggregationType.MAX, 7.0)])
@pytest.mark.parametrize("observations", [[[], [2.0, 7.0]], [[2.0], [], [7.0]], [[2.0, 3.0], [7.0]]])
def test_dp_extrema_ignore_missing_observations(aggregation, expected, observations):
    metrics = []
    for values in observations:
        metric = Metric(aggregation)
        metric.extend(values)
        metrics.append(metric)
    assert Metric.aggregate_dp(metrics) == expected


@pytest.mark.parametrize("aggregation", [AggregationType.SUM, AggregationType.MEAN])
def test_dp_sum_and_mean_keep_equal_microbatch_count_requirement(aggregation):
    with pytest.raises(ValueError, match="same number of values"):
        Metric.aggregate_dp([Metric(aggregation), Metric(aggregation, 1.0)])
