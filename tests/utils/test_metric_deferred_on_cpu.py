# Copyright (c) Meta Platforms, Inc. and affiliates.
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
"""Unit tests of verl.utils.metric.deferred (deferred host scalars)."""

import math

import pytest
import torch

from verl.utils.metric import utils as metric_utils
from verl.utils.metric.deferred import (
    DeferredScalar,
    active_scope,
    check,
    defer_host_scalars,
    host_copy,
    host_scalar,
)


def test_resolved_values_are_the_eager_values():
    """Every placeholder resolves to exactly what ``item()`` returns (type, bits, repr)."""
    values = [
        torch.tensor(0.1, dtype=torch.float32),
        torch.tensor(1 / 3, dtype=torch.float64),
        torch.tensor(7, dtype=torch.int64),
        torch.tensor(True),
        torch.tensor(2.5, dtype=torch.bfloat16),
        torch.tensor(-0.0),
        torch.tensor(float("inf")),
        torch.tensor(float("nan")),
        torch.tensor([3.25]),
    ]
    eager = [v.item() for v in values]
    with defer_host_scalars() as scope:
        deferred = [host_scalar(v) for v in values]
        assert all(isinstance(d, DeferredScalar) for d in deferred)
        metric = metric_utils.Metric("mean", torch.tensor(0.7))
        tree = {"values": list(deferred), "metric": metric}
        filled = scope.resolve(tree)
    assert filled == len(values) + 1
    assert [type(x) for x in tree["values"]] == [type(x) for x in eager]
    assert [repr(x) for x in tree["values"]] == [repr(x) for x in eager]
    assert math.copysign(1.0, tree["values"][5]) == -1.0
    assert metric.values == [torch.tensor(0.7).item()]
    # One stacked read per dtype: float32 (6 incl. the metric), float64, int64, bool, bfloat16.
    assert scope.reads == 5


def test_host_functions_run_on_the_host_values():
    with defer_host_scalars() as scope:
        ratio = host_scalar(torch.tensor(4.0), fn=lambda x: 1.0 / x)
        gated = host_scalar(torch.tensor(True), torch.tensor(2.0), fn=lambda ok, x: x if ok else 0.0)
        tree = [ratio, gated]
        scope.resolve(tree)
    assert tree == [0.25, 2.0]
    assert host_scalar(torch.tensor(4.0), torch.tensor(False), fn=lambda a, b: a if b else 0.0) == 0.0


def test_outside_a_scope_the_helpers_are_eager():
    assert active_scope() is None
    assert host_scalar(torch.tensor(3.0)) == 3.0
    copy = host_copy(torch.arange(3))
    assert copy.device.type == "cpu" and copy.tolist() == [0, 1, 2]
    with pytest.raises(AssertionError, match="eager"):
        check(torch.tensor(False), AssertionError, "eager")
    check(torch.tensor(True), AssertionError, "never raised")


def test_checks_raise_at_resolve_with_the_same_type_and_message():
    with defer_host_scalars() as scope:
        check(torch.tensor(False), ValueError, "boom")
        value = host_scalar(torch.tensor(1.0))
        with pytest.raises(ValueError, match="^boom$"):
            scope.resolve([value])


def test_host_copy_is_filled_by_resolve():
    src = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    with defer_host_scalars() as scope:
        dest = host_copy(src)
        assert dest.shape == src.shape and torch.isnan(dest).all()
        scope.resolve()
    assert torch.equal(dest, src)


def test_placeholders_refuse_numeric_use_before_resolve():
    with defer_host_scalars():
        value = host_scalar(torch.tensor(2.0))
        with pytest.raises(RuntimeError, match="before HostScalarScope.resolve"):
            float(value)
        with pytest.raises((RuntimeError, TypeError)):
            metric_utils.Metric("mean", value).aggregate()


def test_in_place_modification_before_resolve_is_refused():
    t = torch.tensor(1.0)
    with defer_host_scalars() as scope:
        value = host_scalar(t)
        t.add_(1)
        with pytest.raises(RuntimeError, match="modified in place"):
            scope.resolve([value])


def test_scopes_do_not_nest_and_resolve_once():
    with defer_host_scalars() as scope:
        with pytest.raises(RuntimeError, match="not nestable"):
            with defer_host_scalars():
                pass
        scope.resolve()
        with pytest.raises(RuntimeError, match="already resolved"):
            scope.resolve()
    assert active_scope() is None


def test_materialize_is_called_per_tensor_in_order():
    seen = []
    with defer_host_scalars(materialize=lambda t: seen.append(float(t))) as scope:
        total = host_scalar(torch.tensor(1.0), torch.tensor(2.0), fn=lambda a, b: a + b)
        check(torch.tensor(3.0) > 0, ValueError, "positive")
        tree = [total]
        scope.resolve(tree)
    assert seen == [1.0, 2.0, 1.0]
    assert tree == [3.0]


def test_only_one_element_tensors_are_accepted():
    with defer_host_scalars():
        with pytest.raises(ValueError, match="one-element"):
            host_scalar(torch.zeros(2))
        with pytest.raises(TypeError):
            host_scalar(1.0)


def test_tuples_cannot_hold_placeholders():
    with defer_host_scalars() as scope:
        value = host_scalar(torch.tensor(1.0))
        with pytest.raises(TypeError, match="tuple"):
            scope.resolve((value,))
