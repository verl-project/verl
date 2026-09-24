from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class MetricAggregationType(Enum):
    """
    Enumeration of supported metric aggregation types.

    Attributes:
        MEAN: Calculate the arithmetic mean of values
        SUM: Calculate the sum of all values
        MAX: Find the maximum value
        MIN: Find the minimum value
        STD: Calculate the standard deviation
        MEDIAN: Calculate the median value
    """

    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    STD = "std"
    MEDIAN = "median"


@dataclass
class MetricConfig:
    """
    Configuration for a metric including aggregation type and formatting.

    Attributes:
        name: The name of the metric
        aggregation: How to aggregate multiple values for this metric
        format_str: Format string for displaying the metric value
    """

    name: str
    aggregation: MetricAggregationType = MetricAggregationType.MEAN
    format_str: str = "{:.3f}"


class Metrics:
    def __init__(self):
        self.data: Dict[str, List] = defaultdict(list)
        self.configs: Dict[str, MetricConfig] = {}

    def add(self, name: str, value: Any, config: Optional[MetricConfig] = None):
        """
        Add a single value to a metric.

        Automatically converts tensors and arrays to scalar values. If this is the first
        time adding this metric, a configuration will be auto-inferred from the name
        unless explicitly provided.

        Args:
            name: The metric name (supports hierarchical naming like 'actor/loss')
            value: The value to add (int, float, torch.Tensor, np.ndarray)
            config: Optional configuration for this metric. If None and this is a new
                   metric, configuration will be auto-inferred.

        Raises:
            ValueError: If tensor/array value is not scalar

        Example:
            >>> metrics = Metrics()
            >>> metrics.add("actor/loss", 0.5)
            >>> metrics.add("actor/loss", torch.tensor(0.4))  # Converted to scalar
            >>>
            >>> # With custom config
            >>> config = MetricConfig("custom_metric", MetricAggregationType.MAX)
            >>> metrics.add("custom_metric", 1.0, config)
        """
        if isinstance(value, (torch.Tensor, np.ndarray)):
            if value.ndim != 0:
                raise ValueError(f"Metric value for '{name}' must be a scalar (0-dim), but got shape {value.shape}")
            value = value.item()

        if name not in self.configs:
            # a new metric is added
            if config is None:
                config = self._auto_infer_config(name)
            self.configs[name] = config
        else:
            # a metric is already added, and user wants to override the config
            if config is not None:
                self.configs[name] = config

        self.data[name].append(value)

    def _auto_infer_config(self, name: str) -> MetricConfig:
        """
        Automatically infer metric configuration from the metric name.

        Uses keyword matching to determine appropriate aggregation type and formatting:
        - Names containing 'sum' -> SUM aggregation
        - Names containing 'max', 'peak' -> MAX aggregation
        - Names containing 'min', 'lowest' -> MIN aggregation
        - Names containing 'lr' -> 6 decimal places formatting
        - Default -> MEAN aggregation, 3 decimal places

        Args:
            name: The metric name to analyze

        Returns:
            MetricConfig: Auto-configured metric configuration
        """
        name_lower = name.lower()

        if any(word in name_lower for word in ("sum",)):
            agg = MetricAggregationType.SUM
        elif any(word in name_lower for word in ("max", "peak")):
            agg = MetricAggregationType.MAX
        elif any(word in name_lower for word in ("min", "lowest")):
            agg = MetricAggregationType.MIN
        else:
            # default to mean
            agg = MetricAggregationType.MEAN

        if any(word in name_lower for word in ("lr", "learning_rate")):
            fmt = "{:.6f}"
        else:
            # default to 3 decimal places
            fmt = "{:.3f}"

        return MetricConfig(name, agg, fmt)

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any], configs: Optional[Dict[str, MetricConfig]] = None) -> "Metrics":
        """
        Create a Metrics object from a dictionary of values.

        Args:
            data_dict: Dictionary where keys are metric names and values can be:
                      - Single values (int, float, torch.Tensor, np.ndarray)
                      - Lists of values
            configs: Optional dictionary of MetricConfig objects for each metric name.
                    If not provided, configs will be auto-inferred from metric names.

        Returns:
            Metrics: New Metrics object initialized with the provided data

        Example:
            >>> # Simple usage with single values
            >>> metrics = Metrics.from_dict({"loss": 0.5, "accuracy": 0.95})

            >>> # With lists of values
            >>> metrics = Metrics.from_dict({
            ...     "loss": [0.5, 0.4, 0.3],
            ...     "accuracy": [0.9, 0.95, 0.92]
            ... })

            >>> # With custom configurations
            >>> configs = {
            ...     "loss": MetricConfig("loss", MetricAggregationType.MEAN, "{:.4f}")
            ... }
            >>> metrics = Metrics.from_dict({"loss": 0.5}, configs)
        """
        metrics = cls()

        copied_data_dict = deepcopy(data_dict)

        for name, value in copied_data_dict.items():
            # Get config for this metric if provided
            config = configs.get(name) if configs else None

            if isinstance(value, (list, tuple)):
                # If value is a list/tuple, add each element
                for v in value:
                    metrics.add(name, v, config=config)
                    config = None  # Only apply config on first add
            else:
                # Single value
                metrics.add(name, value, config=config)

        return metrics

    def merge(self, other: "Metrics", strict_config: bool = False) -> "Metrics":
        """
        Merge data from another Metrics object by extending existing data lists.

        This method extends the current metric data with values from another Metrics object.
        Existing data is preserved and new data is appended.

        Args:
            other: Another Metrics object to merge from
            strict_config: If True, raise an error if config for a metric differs
                          between self and other. If False, self's config is preserved.

        Returns:
            self: Returns self for method chaining

        Raises:
            TypeError: If other is not a Metrics object or dict
            ValueError: If strict_config=True and configs differ for the same metric
        """
        if isinstance(other, dict):
            other = Metrics.from_dict(other)
        if not isinstance(other, Metrics):
            raise TypeError(f"Expected Metrics or dict, got {type(other)}")

        for name, values in other.data.items():
            if name not in self.data:
                self.data[name] = deepcopy(values)
                self.configs[name] = deepcopy(other.configs[name])
            else:
                if strict_config and self.configs[name] != other.configs[name]:
                    raise ValueError(f"Config for metric {name} is different between self and other")
                self.data[name].extend(deepcopy(values))
        return self

    def update(self, other: "Metrics", strict_config: bool = False) -> "Metrics":
        """
        Update this Metrics object by replacing data with another Metrics object.

        This method replaces the current metric data with values from another Metrics object.
        Existing data is overwritten.

        Args:
            other: Another Metrics object or dict to update from
            strict_config: If True, raise an error if config for a metric differs
                          between self and other. If False, self's config is preserved.

        Returns:
            self: Returns self for method chaining

        Raises:
            TypeError: If other is not a Metrics object or dict
            ValueError: If strict_config=True and configs differ for the same metric
        """
        if isinstance(other, dict):
            other = Metrics.from_dict(other)
        if not isinstance(other, Metrics):
            raise TypeError(f"Expected Metrics or dict, got {type(other)}")

        for name, values in other.data.items():
            if name not in self.data:
                self.data[name] = deepcopy(values)
                self.configs[name] = deepcopy(other.configs[name])
            else:
                if strict_config and self.configs[name] != other.configs[name]:
                    raise ValueError(f"Config for metric {name} is different between self and other")
                self.data[name] = deepcopy(values)
        return self

    def _aggregate_value(self, values: List[Any], agg: MetricAggregationType) -> Any:
        """
        Apply aggregation function to a list of values.

        Args:
            values: List of numeric values to aggregate
            agg: The aggregation type to apply

        Returns:
            Any: The aggregated result (type depends on aggregation function)

        Raises:
            ValueError: If aggregation type is not supported

        Example:
            >>> metrics = Metrics()
            >>> values = [1, 2, 3, 4, 5]
            >>> result = metrics._aggregate_value(values, MetricAggregationType.MEAN)
            >>> print(result)  # 3.0
        """
        if agg == MetricAggregationType.MEAN:
            return np.mean(values)
        if agg == MetricAggregationType.SUM:
            return np.sum(values)
        if agg == MetricAggregationType.MAX:
            return np.max(values)
        if agg == MetricAggregationType.MIN:
            return np.min(values)
        if agg == MetricAggregationType.STD:
            return np.std(values)
        if agg == MetricAggregationType.MEDIAN:
            return np.median(values)

        raise ValueError(f"Invalid aggregation type: {agg}")

    def get_aggregated_value(self, name: str, formatted_str: bool = False):
        """
        Get the aggregated value for a specific metric.

        Args:
            name: The metric name
            formatted: If True, return a formatted string using the metric's format_str.
                      If False, return the raw numeric value.

        Returns:
            Union[float, str]: The aggregated value, formatted or raw

        Raises:
            KeyError: If the metric name is not found
        """
        if name not in self.data:
            raise KeyError(f"Metric {name} not found")

        values = self.data[name]
        config = self.configs[name]
        agg = config.aggregation

        val = self._aggregate_value(values, agg)

        if formatted_str:
            return config.format_str.format(val)
        else:
            return val

    def get_raw_value(self, name: str):
        """
        Get the raw list of values for a specific metric.

        Args:
            name: The metric name

        Returns:
            List: The raw list of values for this metric

        Raises:
            KeyError: If the metric name is not found
        """
        if name not in self.data:
            raise KeyError(f"Metric {name} not found")
        return self.data[name]

    def __setitem__(self, name: str, value: Any):
        self.add(name, value)

    def __getitem__(self, name: str):
        return self.get_raw_value(name)

    def __contains__(self, name: str) -> bool:
        return name in self.data

    def __len__(self) -> int:
        return len(self.data)
