# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Pin how the outer-chunked step turns per-pass losses into the reported number.

The passes are disjoint contributions to one mean over the same global mask, so the report
is their SUM. This was measured the hard way: averaging divided the K=2 loss by the pass
count (4.18 -> 1.39 at 512K) and looked like a broken kernel rather than a reporting bug.

The module is loaded by path so the suite needs neither torch nor megatron-core (the
package __init__ imports both).
"""

import importlib.util
import pathlib

import pytest

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "verl" / "workers" / "engine" / "megatron" / "chunk_losses.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("dsv41_chunk_losses", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cl = _load()

# 512K measured: only the graded pass owns the 12 supervised tokens; the prefill and the
# replay of pass 0 contribute exactly 0.
MEASURED_PASSES = [0.0, 4.1766, 0.0]


def test_passes_are_summed_not_averaged():
    assert cl.aggregate_pass_losses(MEASURED_PASSES, fallback=-1.0) == pytest.approx(4.1766)


def test_averaging_would_be_wrong_by_the_pass_count():
    averaged = sum(MEASURED_PASSES) / len(MEASURED_PASSES)
    assert cl.aggregate_pass_losses(MEASURED_PASSES, fallback=-1.0) == pytest.approx(averaged * 3)


def test_nan_entries_are_dropped():
    assert cl.aggregate_pass_losses([float("nan"), 2.0, 0.5], fallback=-1.0) == pytest.approx(2.5)


def test_per_token_scaled_entries_are_dropped():
    """The scan pass reports mean * routed_num_tokens / dp_size, which is not a mean."""
    scaled = 4.18 * 2.0e5
    assert cl.aggregate_pass_losses([scaled, 4.18], fallback=-1.0) == pytest.approx(4.18)


def test_fallback_when_nothing_is_reportable():
    assert cl.aggregate_pass_losses([float("nan"), 1.0e6], fallback=7.5) == 7.5
    assert cl.aggregate_pass_losses([], fallback=3.0) == 3.0
