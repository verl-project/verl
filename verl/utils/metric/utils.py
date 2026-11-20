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
Metrics utils.
"""

from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import torch


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


NumericType = int, float, torch.Tensor, np.ndarray
Numeric = int | float | torch.Tensor | np.ndarray


def scalars_to_host(tensors: Sequence[torch.Tensor]) -> list[Any]:
    """Reads single-element tensors back to the host as Python numbers, with one transfer per (device, dtype).

    Equivalent to ``[t.item() for t in tensors]``, but each ``.item()`` on an accelerator is a blocking
    device-to-host round trip, while this stacks the tensors of each (device, dtype) group and copies them with a
    single ``.cpu()``. Tensor subclasses (e.g. ``DTensor``) are read one by one with ``.item()``.
    """
    host: list[Any] = [None] * len(tensors)
    groups: dict[tuple[torch.device, torch.dtype], list[int]] = {}
    for i, t in enumerate(tensors):
        if t.numel() != 1:
            raise ValueError(f"Only single-element tensors can be converted to a Python number, got {t.shape}")
        if type(t) is torch.Tensor:
            groups.setdefault((t.device, t.dtype), []).append(i)
        else:
            host[i] = t.detach().item()
    with torch.no_grad():
        for idx in groups.values():
            if len(idx) == 1:
                host[idx[0]] = tensors[idx[0]].detach().item()
                continue
            group = [tensors[i].detach().reshape(()) for i in idx]
            # Pad to a power of two (repeating an existing tensor) so backends that compile one program per shape
            # (XLA / TPU) see a logarithmic number of distinct stack sizes as the micro-batch count varies.
            group += group[:1] * ((1 << (len(group) - 1).bit_length()) - len(group))
            stacked = torch.stack(group).cpu().tolist()
            for i, v in zip(idx, stacked, strict=False):
                host[i] = v
    return host


def materialize_scalars(scalars: Optional[list[Any]] = None, metrics: Iterable["Metric"] = ()) -> None:
    """In place, replaces every tensor in ``scalars`` and in the values of ``metrics`` by a Python number.

    All of them are read back together (see :func:`scalars_to_host`), so a training step that kept its per-micro-batch
    losses and metrics on the device pays one device-to-host transfer instead of one blocking ``.item()`` per value.
    """
    slots: list[tuple[list[Any], int]] = []
    if scalars is not None:
        slots.extend((scalars, i) for i, v in enumerate(scalars) if isinstance(v, torch.Tensor))
    for m in metrics:
        slots.extend((m._values, i) for i, v in enumerate(m._values) if isinstance(v, torch.Tensor))
    if not slots:
        return
    for (lst, i), v in zip(slots, scalars_to_host([lst[i] for lst, i in slots]), strict=True):
        lst[i] = v


class Metric:
    """
    A metric aggregator for collecting and aggregating numeric values.

    This class accumulates numeric values (int, float, or scalar tensors) and computes
    an aggregate statistic based on the specified aggregation type (MEAN, SUM, MIN, or MAX).

    Scalar tensors are not read back when they are appended: on an accelerator ``.item()`` would block the host until
    the device has computed them (e.g. in the middle of a micro-batch, before its backward is dispatched). They are
    kept as detached copies and converted to Python numbers on first read of ``values``, in ``aggregate`` /
    ``aggregate_dp``, or when pickled, in one batched transfer (``Metric.materialize_all`` batches across metrics).
    Tensor subclasses such as ``DTensor`` are still converted immediately.

    Args:
        aggregation: The aggregation method to use. Can be a string ("mean", "sum", "min", "max")
            or an AggregationType enum value.
        value: Optional initial value(s) to add. Can be a single numeric value or a list of values.

    Example:
        >>> metric = Metric(aggregation="mean", value=1.0)
        >>> metric.append(2.0)
        >>> metric.append(3.0)
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
        self._values: list[Numeric] = []
        if value is not None:
            self.append(value)

    @property
    def values(self) -> list[Numeric]:
        """The appended values as Python numbers (device tensors are read back on first access)."""
        self.materialize()
        return self._values

    @values.setter
    def values(self, values: list[Numeric]) -> None:
        self._values = list(values)

    def append(self, value: Union[Numeric, "Metric"]) -> None:
        if isinstance(value, Metric):
            self.extend(value)
            return
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Only scalar tensors can be converted to float")
            value = value.detach()
            if type(value) is torch.Tensor:
                # Keep it on the device. Clone, so a later in-place update of the source tensor (e.g. ``loss -= ...``)
                # does not change the recorded value.
                value = value.reshape(()).clone()
            else:
                value = value.item()
        if not isinstance(value, NumericType):
            raise ValueError(f"Unsupported value type: {type(value)}")
        self._values.append(value)

    def extend(self, values: Union["Metric", list[Numeric]]) -> None:
        if isinstance(values, Metric):
            if values.aggregation != self.aggregation:
                raise ValueError(f"Aggregation type mismatch: {self.aggregation} != {values.aggregation}")
            self._values.extend(values._values)  # already validated; keeps device tensors lazy
            return
        for value in values:
            self.append(value)

    def materialize(self) -> None:
        """Converts any device tensors among the values to Python numbers (one batched transfer)."""
        materialize_scalars(metrics=(self,))

    @staticmethod
    def materialize_all(metrics: Iterable["Metric"]) -> None:
        """Like :meth:`materialize`, for many metrics at once with a single batched transfer."""
        materialize_scalars(metrics=metrics)

    def __getstate__(self) -> dict[str, Any]:
        # Never ship device tensors across processes (Ray / all_gather_object): send Python numbers.
        self.materialize()
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        if "values" in state:  # pickled by a version that stored a plain ``values`` attribute
            state["_values"] = state.pop("values")
        self.__dict__.update(state)

    def aggregate(self) -> float:
        return self._aggregate(self.values, self.aggregation)

    @classmethod
    def _aggregate(cls, values: list[Numeric], aggregation: AggregationType) -> float:
        match aggregation:
            case AggregationType.MEAN:
                return np.mean(values)
            case AggregationType.SUM:
                return np.sum(values)
            case AggregationType.MIN:
                return np.min(values)
            case AggregationType.MAX:
                return np.max(values)

    @classmethod
    def aggregate_dp(cls, metric_lists: list["Metric"]) -> float:
        if not metric_lists:
            raise ValueError("Cannot aggregate an empty list of metrics.")
        cls.materialize_all(metric_lists)
        value_lists = [ml.values for ml in metric_lists]
        if not all(len(ls) == len(value_lists[0]) for ls in value_lists):
            raise ValueError(
                f"All Metric instances must have the same number of values "
                f"for dp aggregation: {[len(ls) for ls in value_lists]}"
            )
        value_arrays = np.array(value_lists)  # [num_dp, num_grad_accumulation]
        aggregation = metric_lists[0].aggregation
        match aggregation:
            case AggregationType.SUM | AggregationType.MEAN:
                return cls._aggregate(
                    values=np.mean(value_arrays, axis=0), aggregation=aggregation
                )  # mean over dp ranks
            case AggregationType.MIN | AggregationType.MAX:
                return cls._aggregate(values=value_arrays.flatten(), aggregation=aggregation)  # min/max over all values

    @classmethod
    def from_dict(cls, data: dict[str, Numeric], aggregation: str | AggregationType) -> dict[str, "Metric"]:
        return {key: cls(value=value, aggregation=aggregation) for key, value in data.items()}

    def init_list(self) -> "Metric":
        return Metric(aggregation=self.aggregation)
