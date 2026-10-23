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
"""On-policy self-distillation: the teacher's rows must land on the student's grid.

OPSD reads the teacher under a privileged context (the verified solution), so the
teacher sequence is ``[teacher_prefix][response]`` while the student sequence is
``[prompt][response]``. Both score the same response, but a response token the
student predicts at position ``i`` sits at teacher position
``i + delta, delta = len(teacher_prefix) - len(prompt)``.

``realign_teacher_rows`` is the function that performs that shift; the megatron
distillation loss asserts the teacher tensors against ``student_logits.shape[:2]``,
so a wrong length here is a hard failure, and a wrong shift silently trains on the
distribution of the wrong token.

The cases below pin: identity when the prefixes match, the shift in both
directions, the fill contract for rows the teacher never saw, and the shape
mismatch guard.
"""

import pytest

torch = pytest.importorskip("torch")

from verl.experimental.teacher_loop.teacher_manager import realign_teacher_rows  # noqa: E402


def _rows(values):
    """(seq, k) teacher rows from a list of [(id, logp), ...] pairs."""
    ids = torch.tensor([[i for i, _ in row] for row in values], dtype=torch.int64)
    lps = torch.tensor([[lp for _, lp in row] for row in values], dtype=torch.float32)
    return ids, lps


def _assert_col(actual, expected):
    """Compare one column, treating -inf as a value and floats with fp32 tolerance."""
    assert len(actual) == len(expected), (actual, expected)
    for got, want in zip(actual, expected, strict=True):
        if want == float("-inf") or want == float("inf"):
            assert got == want, (got, want)
        else:
            assert got == pytest.approx(want, abs=1e-6), (got, want)


def test_identity_when_prefixes_match():
    """Same teacher and student prompt: the rows must pass through untouched."""
    ids, lps = _rows([[(5, -0.1), (6, -2.0)], [(7, -0.3), (8, -1.5)], [(9, -0.2), (10, -3.0)]])
    out_ids, out_lps = realign_teacher_rows(ids, lps, student_len=3, delta=0)
    assert torch.equal(out_ids, ids)
    assert torch.equal(out_lps, lps)


def test_shifts_when_teacher_has_longer_prefix():
    """Teacher prefix longer by 2: student row i reads teacher row i + 2."""
    ids, lps = _rows([[(1, -0.1), (2, -0.2)]] * 5)
    # make every row distinguishable
    for r in range(5):
        ids[r, 0] = 100 + r
        lps[r, 0] = -0.1 * r
    out_ids, out_lps = realign_teacher_rows(ids, lps, student_len=3, delta=2)
    assert out_ids[:, 0].tolist() == [102, 103, 104]
    _assert_col(out_lps[:, 0].tolist(), [-0.2, -0.3, -0.4])


def test_shifts_when_teacher_prefix_is_shorter():
    """delta < 0 (teacher prefix shorter): leading student rows fall before the teacher."""
    ids, lps = _rows([[(100, -0.5)], [(101, -0.6)]])
    out_ids, out_lps = realign_teacher_rows(ids, lps, student_len=4, delta=-1)
    # student row 0 maps to teacher row -1 -> fill; rows 1,2 map to 0,1; row 3 -> fill
    assert out_ids[:, 0].tolist() == [0, 100, 101, 0]
    _assert_col(out_lps[:, 0].tolist(), [float("-inf"), -0.5, -0.6, float("-inf")])


def test_rows_beyond_the_teacher_are_filled_not_wrapped():
    """A student longer than the teacher must not wrap around to row 0."""
    ids, lps = _rows([[(7, -0.1)], [(8, -0.2)]])
    out_ids, out_lps = realign_teacher_rows(ids, lps, student_len=4, delta=0)
    assert out_ids[:, 0].tolist() == [7, 8, 0, 0]
    _assert_col(out_lps[:, 0].tolist(), [-0.1, -0.2, float("-inf"), float("-inf")])


def test_fill_rows_contribute_no_teacher_mass():
    """The fill sentinel must be exp(-inf) == 0, so it cannot pull the student."""
    ids, lps = _rows([[(7, -0.1)]])
    # delta=0: student row 0 reads the teacher's only row; rows 1 and 2 are fill.
    _, out_lps = realign_teacher_rows(ids, lps, student_len=3, delta=0)
    mass = out_lps.exp().sum(dim=-1)
    assert mass[0].item() == pytest.approx(0.904837, rel=1e-4)
    assert mass[1].item() == pytest.approx(0.0, abs=1e-12)
    assert mass[2].item() == pytest.approx(0.0, abs=1e-12)


def test_rejects_shape_mismatch():
    ids = torch.zeros((3, 2), dtype=torch.int64)
    lps = torch.zeros((3, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="same shape"):
        realign_teacher_rows(ids, lps, student_len=3, delta=0)


def test_rejects_non_2d():
    ids = torch.zeros(3, dtype=torch.int64)
    lps = torch.zeros(3, dtype=torch.float32)
    with pytest.raises(ValueError, match="2-D"):
        realign_teacher_rows(ids, lps, student_len=3, delta=0)
