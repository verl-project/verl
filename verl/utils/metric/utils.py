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
"""Metrics utils."""

import logging
import os
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def reduce_metrics(metrics: dict[str, Union["Metric", list[Any]]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains "max", np.max is used
    - If the key contains "min", np.min is used
    - Otherwise, np.mean is used

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    for key, val in metrics.items():
        if isinstance(val, Metric):
            metrics[key] = val.aggregate()
        elif "max" in key:
            metrics[key] = np.max(val)
        elif "min" in key:
            metrics[key] = np.min(val)
        else:
            metrics[key] = np.mean(val)
    return metrics


class AggregationType(Enum):
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"


Numeric = int | float | torch.Tensor | np.ndarray


class Metric:
    """
    A metric aggregator.

    This class stores metric states for specified aggregation type.
    New values are added via `accumulate`, and multiple metrics can be merged via
    `union` or by accumulating another Metric instance.

    Args:
        aggregation: The aggregation method to use. Can be a string ("mean", "sum", "min", "max")
            or an AggregationType enum value.
        value: Optional initial value(s) to add. Can be a single numeric value or a list of values.

    Example:
        >>> metric = Metric(aggregation="mean", value=1.0)
        >>> metric.accumulate(2.0)
        >>> metric.accumulate(3.0)
        >>> metric.aggregate()
        2.0
    """

    def __init__(self, aggregation: str | AggregationType, value: Optional[Numeric | list[Numeric]] = None) -> None:
        if isinstance(aggregation, str):
            self.aggregation = AggregationType(aggregation)
        else:
            self.aggregation = aggregation
        if not isinstance(self.aggregation, AggregationType):
            raise ValueError(f"Unsupported aggregation type: {aggregation}")
        self.count = 0
        self.total = 0.0
        self.value = float("inf") if self.aggregation == AggregationType.MIN else float("-inf")
        if value is not None:
            self.accumulate(value)

    @staticmethod
    def _flatten_values(value: Numeric | list[Numeric] | tuple[Numeric, ...]) -> list[float | int]:
        if isinstance(value, int | float):
            return [value]
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().flatten().tolist()
        if isinstance(value, np.ndarray):
            return value.flatten().tolist()
        if isinstance(value, list | tuple):
            if not all(isinstance(item, int | float) for item in value):
                raise ValueError("Lists and tuples passed to Metric.accumulate must contain only int or float values")
            return list(value)
        raise ValueError(f"Unsupported value type: {type(value)}")

    def accumulate(self, value: Union[Numeric, "Metric", list[Numeric], tuple[Numeric, ...]]) -> None:
        if isinstance(value, Metric):
            if value.aggregation != self.aggregation:
                raise ValueError(f"Aggregation type mismatch: {self.aggregation} != {value.aggregation}")
            if value.count == 0:
                return
            match self.aggregation:
                case AggregationType.SUM | AggregationType.MEAN:
                    self.total += value.total
                case AggregationType.MIN:
                    self.value = min(self.value, value.value)
                case AggregationType.MAX:
                    self.value = max(self.value, value.value)
            self.count += value.count
            return

        values = self._flatten_values(value)
        if not values:
            return
        match self.aggregation:
            case AggregationType.SUM | AggregationType.MEAN:
                self.total += sum(values)
            case AggregationType.MIN:
                self.value = min(self.value, min(values))
            case AggregationType.MAX:
                self.value = max(self.value, max(values))
        self.count += len(values)

    def aggregate(self) -> float:
        if self.count == 0:
            logging.warning("Aggregating metric with no values. Returning NaN.")
            return float("nan")
        match self.aggregation:
            case AggregationType.MEAN:
                return self.total / self.count
            case AggregationType.SUM:
                return self.total
            case AggregationType.MIN | AggregationType.MAX:
                return self.value

    @classmethod
    def union(cls, *metrics: "Metric") -> "Metric":
        if not metrics:
            raise ValueError("Cannot union an empty set of metrics.")
        aggregation = metrics[0].aggregation
        merged = cls(aggregation=aggregation)
        for metric in metrics:
            if metric.aggregation != aggregation:
                raise ValueError(f"Aggregation type mismatch: {aggregation} != {metric.aggregation}")
            merged.accumulate(metric)
        return merged

    @classmethod
    def aggregate_dp(cls, metrics: list["Metric"]) -> float:
        """Aggregate a list of Metric instances across data parallel ranks.
        For SUM metrics, this averages the contributions from each rank.
        """
        if not metrics:
            raise ValueError("Cannot aggregate an empty set of metrics.")
        merged = cls.union(*metrics)
        result = merged.aggregate()
        if merged.aggregation == AggregationType.SUM:
            result /= len(metrics)
        return result
