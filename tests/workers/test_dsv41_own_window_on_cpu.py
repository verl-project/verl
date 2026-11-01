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
"""Pin the row-window arithmetic of the outer-chunked step.

`chunk_window.own_window_rows` decides which rows of `log_probs` a pass is allowed to
contribute. Getting it wrong is silent: the tensor keeps its shape and the run keeps
moving, while the pass that carries the gradient contributes an empty window and the
loss collapses toward zero. These tests pin the three modes and, in particular, that the
default ("keep") still reproduces the shipped expression exactly.

The module under test is loaded by path so that the suite does not need torch or
megatron-core, which the `verl.workers.engine.megatron` package `__init__` imports.
"""

import importlib.util
import pathlib

import pytest

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "verl" / "workers" / "engine" / "megatron" / "chunk_window.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("dsv41_chunk_window", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cw = _load_module()

# The 512K SFT step: real whole-sequence rows, physical CP, K=2.
REAL_NUMEL = 483_495
REAL_CP = 32
REAL_K = 2


def shipped_window(numel: int, cp_size: int, index: int) -> tuple[int, int]:
    """The expression this switch replaced, kept here as the reference."""
    win = numel // cp_size
    start = index * cp_size * win
    return start, min(start + cp_size * win, numel)


@pytest.mark.parametrize("numel", [1, 7, 8, 245_760, 483_488, 483_495, 491_520])
@pytest.mark.parametrize("cp_size", [1, 2, 4, 32])
@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_keep_matches_the_shipped_expression(numel, cp_size, index):
    # Compared as row ranges: the helper clamps an out-of-range start to `numel`, which
    # yields the same (empty) slice the shipped expression produced.
    assert range(*cw.own_window_rows(numel, cp_size, REAL_K, index, cw.KEEP)) == range(
        *shipped_window(numel, cp_size, index)
    )


def test_keep_gives_every_later_pass_an_empty_window_on_real_shape():
    # This is the shipped behaviour, pinned so a reader can see the defect it documents:
    # width == cp * (numel // cp) == the whole tensor, so pass 1 starts at the end.
    start0, end0 = cw.own_window_rows(REAL_NUMEL, REAL_CP, REAL_K, 0, cw.KEEP)
    start1, end1 = cw.own_window_rows(REAL_NUMEL, REAL_CP, REAL_K, 1, cw.KEEP)
    assert (start0, end0) == (0, 483_488)
    assert end1 - start1 == 7


def test_chunk_mode_splits_the_timeline_evenly_and_covers_the_tail_window():
    windows = [cw.own_window_rows(REAL_NUMEL, REAL_CP, REAL_K, index, cw.CHUNK) for index in range(REAL_K)]
    # `numel % chunk_count` rows go to the last pass, so the widths differ by at most the
    # remainder; what must hold is that the windows tile the timeline with no gap.
    assert windows == [(0, 241_747), (241_747, REAL_NUMEL)]
    assert sum(end - start for start, end in windows) == REAL_NUMEL
    # The supervised tail (last 12 rows) must live inside exactly one pass's window.
    tail = range(REAL_NUMEL - 12, REAL_NUMEL)
    owning = [i for i, (start, end) in enumerate(windows) if start <= tail[0] and tail[-1] < end]
    assert owning == [REAL_K - 1], owning


def test_chunk_mode_windows_do_not_overlap():
    prev_end = 0
    for index in range(REAL_K):
        start, end = cw.own_window_rows(REAL_NUMEL, REAL_CP, REAL_K, index, cw.CHUNK)
        assert start == prev_end
        prev_end = end


def test_off_keeps_the_whole_tensor_for_every_pass():
    for index in range(REAL_K):
        assert cw.own_window_rows(REAL_NUMEL, REAL_CP, REAL_K, index, cw.OFF) == (0, REAL_NUMEL)


def test_window_never_exceeds_the_tensor():
    for mode in (cw.KEEP, cw.CHUNK):
        for index in range(8):
            start, end = cw.own_window_rows(REAL_NUMEL, REAL_CP, REAL_K, index, mode)
            assert 0 <= start <= end <= REAL_NUMEL


def test_unknown_mode_fails_loudly():
    with pytest.raises(ValueError, match="unknown window mode"):
        cw.own_window_rows(REAL_NUMEL, REAL_CP, REAL_K, 0, "offf")


def test_invalid_sizes_are_rejected():
    with pytest.raises(ValueError):
        cw.own_window_rows(REAL_NUMEL, 0, REAL_K, 0, cw.KEEP)
    with pytest.raises(ValueError):
        cw.own_window_rows(REAL_NUMEL, REAL_CP, 0, 0, cw.CHUNK)


def test_default_mode_is_the_chunk_window_not_keep():
    """The default must be the ownership-correct window, never the retired ``keep`` formula.

    ``keep`` hands pass 0 the whole tensor, so a K>1 run silently zeroes the rows later
    passes computed. That corrupts the loss *and* the gradient, because the window is
    applied to ``log_probs`` before the loss is taken. Pinning the default here is what
    keeps a future refactor from reintroducing it as the shipped behaviour.
    """
    for index in range(REAL_K):
        assert cw.own_window_rows(REAL_NUMEL, REAL_CP, REAL_K, index) == cw.own_window_rows(
            REAL_NUMEL, REAL_CP, REAL_K, index, cw.CHUNK
        )
        assert cw.own_window_rows(REAL_NUMEL, REAL_CP, REAL_K, index) != cw.own_window_rows(
            REAL_NUMEL, REAL_CP, REAL_K, index, cw.KEEP
        )


def test_chunk_window_is_the_identity_at_one_chunk():
    """K=1 must be untouched by the window: the default has to be a no-op there."""
    assert cw.own_window_rows(REAL_NUMEL, REAL_CP, 1, 0) == (0, REAL_NUMEL)


def test_chunk_windows_stay_disjoint_when_the_length_does_not_divide():
    """An indivisible length must still tile [0, numel) with no gap and no overlap."""
    numel, cp_size, k = 483495, 32, 4
    spans = [cw.own_window_rows(numel, cp_size, k, i) for i in range(k)]
    assert spans[0][0] == 0
    assert spans[-1][1] == numel
    for i in range(len(spans) - 1):
        assert spans[i][1] == spans[i + 1][0], (
            f"gap or overlap between pass {i} end {spans[i][1]} and pass {i + 1} start {spans[i + 1][0]}"
        )
