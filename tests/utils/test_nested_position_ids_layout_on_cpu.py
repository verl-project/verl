# Copyright 2026 The verl authors
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
"""Red-green regression tests for the 3D (mRoPE) position_ids nested-layout fix.

Eight distilled cases; every one fails on unfixed master and passes with the fix:

- the equal-length `as_nested_tensor` quirk (uniform inputs yield a ragged@1 layout)
  combined with the blind `_ragged_idx = 2` workaround in `maybe_fix_3d_position_ids`,
  which detonates as `split_with_sizes` at the next `unbind()` (cases 1, 3, 6) or
  silently returns transposed chunks on the square corner (case 2);
- `index_select_tensor_dict` failing with a cryptic low-level error instead of an
  actionable one (case 4);
- `concat_nested_tensors` silently rebuilding mixed-layout inputs with the first
  tensor's tag (case 5);
- `normalize_3d_position_ids` itself: component-count-agnostic repair (case 7) and
  zero-cost idempotent passthrough for canonical input (case 8).

All fixtures are CPU-only integer tensors; every expectation is a bitwise comparison.
"""

import pytest
import torch
from tensordict import TensorDict

import verl.protocol as vp
from verl.utils import tensordict_utils as tu


# ---------------------------------------------------------------- fixtures


def _dense_samples(lengths, num_components):
    """Deterministic per-sample (num_components, L_i) tensors with distinct content."""
    out, base = [], 0
    for length in lengths:
        out.append(torch.arange(base, base + num_components * length, dtype=torch.int64).reshape(num_components, length))
        base += num_components * length
    return out


def _canonical(samples, ragged_idx=2):
    """Build the canonical layout via the repo's own builder."""
    return tu.nested_tensor_from_tensor_list(samples, ragged_idx=ragged_idx)


def _quirky_uniform(length, batch, num_components=4):
    """Reproduce the as_nested_tensor uniform-input quirk (ragged@1). Returns (nt, samples)."""
    samples = [
        torch.arange(i * 1000, i * 1000 + num_components * length, dtype=torch.int64).reshape(num_components, length)
        for i in range(batch)
    ]
    return torch.nested.as_nested_tensor(samples, layout=torch.jagged), samples


def _wrap(nt, batch):
    return TensorDict({"position_ids": nt}, batch_size=[batch])


def _select(td, idx):
    return tu.index_select_tensor_dict(td, idx)


# ------------------------------------------------ E2E: quirk + blind tag chain


def test_1_equal_length_quirk_e2e_index_select():
    """Real-scale equal-length batch (128 x (4, 34135)): maybe_fix + index_select must work.

    Unfixed master: the quirk produces ragged@1, the blind retag makes it self-inconsistent,
    and index_select crashes with `split_with_sizes expects split_sizes to sum exactly to
    34135 ... but got split_sizes=[4, 4, 4, ...]`.
    """
    num_components, length, batch = 4, 34135, 128
    samples = [
        torch.arange(i * 1000, i * 1000 + num_components * length, dtype=torch.int64).reshape(num_components, length)
        for i in range(batch)
    ]
    nt = torch.nested.as_nested_tensor(samples, layout=torch.jagged)
    td = _wrap(nt, batch)

    tu.maybe_fix_3d_position_ids(td)
    out = _select(td, [0, 1, 2])["position_ids"]
    rebuilt = list(out.unbind())
    assert [tuple(r.shape) for r in rebuilt] == [(num_components, length)] * 3
    assert all(torch.equal(r, s) for r, s in zip(rebuilt, samples[:3]))


def test_2_square_corner_equal_length_e2e():
    """Square corner (values=(32,32): B=8, C=4, L=32): must be rebuilt, not mis-read.

    Unfixed master: the blind retag makes offsets[-1]==values.shape[1] hold, so unbind
    silently returns transposed (L, C) chunks with no error — wrong data, no crash.
    """
    nt, samples = _quirky_uniform(32, 8)  # values=(32, 32), square
    td = _wrap(nt, 8)

    tu.maybe_fix_3d_position_ids(td)
    out = _select(td, [3, 5])["position_ids"]
    rebuilt = list(out.unbind())
    assert [tuple(r.shape) for r in rebuilt] == [(4, 32), (4, 32)]
    assert all(torch.equal(r, s) for r, s in zip(rebuilt, [samples[3], samples[5]]))


def test_3_blind_tagged_legacy_layout_e2e():
    """A coordinate-packed tensor already blind-retagged to 2 (the old workaround's output)
    must be rebuilt into a consistent canonical layout."""
    nt, samples = _quirky_uniform(48, 8)  # values=(32, 48), non-square
    nt._ragged_idx = 2  # what the old workaround leaves behind
    td = _wrap(nt, 8)

    tu.maybe_fix_3d_position_ids(td)
    out = _select(td, [0, 7])["position_ids"]
    rebuilt = list(out.unbind())
    assert [tuple(r.shape) for r in rebuilt] == [(4, 48), (4, 48)]
    assert all(torch.equal(r, s) for r, s in zip(rebuilt, [samples[0], samples[7]]))


# ------------------------------------------------------- guards / diagnostics


def test_4_index_select_inconsistent_layout_readable_error():
    """index_select on an inconsistent layout must fail fast with an actionable message
    (key name, ragged_idx, sums), not the raw split_with_sizes error."""
    nt, _ = _quirky_uniform(48, 8)
    nt._ragged_idx = 2  # sum(lengths)=32 != values.shape[1]=48
    td = _wrap(nt, 8)

    with pytest.raises(RuntimeError, match=r"key='position_ids'.*sum\(lengths\)=32"):
        _select(td, [0, 1])


def test_5_concat_mixed_ragged_idx_rejected():
    """Mixed-layout concat must fail fast; silently rebuilding with the first tensor's
    tag produces a wrong tensor."""
    nt1, _ = _quirky_uniform(48, 4)  # ragged_idx=1
    nt2 = _canonical(_dense_samples([10, 24], num_components=4))  # ragged_idx=2
    with pytest.raises(AssertionError, match=r"inconsistent ragged_idx across inputs: \[1, 2\]"):
        tu.concat_nested_tensors([nt1, nt2])


# ------------------------------------------------------- serialization chain


def test_6_serialize_deserialize_roundtrip_e2e():
    """An equal-length canonical batch must survive serialize -> deserialize -> maybe_fix
    -> index_select. Unfixed master: deserialize re-rolls as_nested_tensor, the quirk
    yields ragged@1 again, and the blind retag detonates at index_select."""
    samples = _dense_samples([16, 16, 16, 16], num_components=4)  # equal lengths
    td = TensorDict(
        {"position_ids": _canonical(samples), "dense": torch.arange(12).reshape(4, 3)}, batch_size=[4]
    )
    arr = vp.serialize_tensordict(td)
    td2 = vp.deserialize_tensordict(arr)

    tu.maybe_fix_3d_position_ids(td2)
    out = _select(td2, [1, 2])["position_ids"]
    rebuilt = list(out.unbind())
    assert [tuple(r.shape) for r in rebuilt] == [(4, 16), (4, 16)]
    assert all(torch.equal(r, s) for r, s in zip(rebuilt, [samples[1], samples[2]]))
    assert torch.equal(td2["dense"], td["dense"])


# ------------------------------------------- normalize_3d_position_ids itself


def test_7_branch_a_sequence_packed_c6_rebuild():
    """Sequence-packed stale metadata (serialization reset) with C=6: the repair must be
    component-count agnostic (no `C in (3, 4)` hardcode) and rebuild per-sample content
    bitwise."""
    samples = _dense_samples([9, 15, 11], num_components=6)
    nt = _canonical(samples)
    nt._ragged_idx = 1  # metadata reset by serialization
    data = _wrap(nt, 3)

    tu.normalize_3d_position_ids(data)
    out = data["position_ids"]
    assert out._ragged_idx == 2
    rebuilt = list(out.unbind())
    assert [tuple(r.shape) for r in rebuilt] == [(6, 9), (6, 15), (6, 11)]
    assert all(torch.equal(r, s) for r, s in zip(rebuilt, samples))


def test_8_canonical_passthrough_idempotent():
    """Canonical input (variable-length and single-sample square) passes through untouched
    at zero cost — the fix must not disturb healthy layouts."""
    samples = _dense_samples([10, 24, 7], num_components=4)
    nt = _canonical(samples)
    data = _wrap(nt, 3)
    tu.normalize_3d_position_ids(data)
    assert data["position_ids"] is nt  # same object, zero-cost passthrough
    assert all(torch.equal(r, s) for r, s in zip(data["position_ids"].unbind(), samples))

    # B=1 degenerate square canonical (values=(4, 4)): both interpretations coincide;
    # must stay untouched rather than be "rebuilt"
    single = torch.arange(16, dtype=torch.int64).reshape(4, 4)
    nt1 = _canonical([single])
    data1 = _wrap(nt1, 1)
    tu.normalize_3d_position_ids(data1)
    assert data1["position_ids"] is nt1
    assert torch.equal(list(data1["position_ids"].unbind())[0], single)
