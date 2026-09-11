# Copyright 2025 Meituan Ltd. and/or its affiliates
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

"""CPU tests for the flex (flex_attention) prefix-tree path.

Covers the flex-specific surface that magi tests do not:
- ``_build_flex_key`` mask semantics against a trie-derived oracle
- ``_flex_aux`` closure retention on the returned BlockMask
- ``_prepare_attn_inputs`` flex contract (full layout, flex_attention_key kwarg)
- ``_finalize_prefix_tree_batch`` attention_type dispatch (flex <-> magi)

The oracle: a token at flat position q (in leaf node L) may attend exactly
``union(ancestor node ranges) + causal-within-L``. Nodes NOT on the root->leaf
path of L's owning sample must be invisible (cross-leaf blocked), and no
future token (kv > q within L) may be visible.

Requires ``tests/utils/prefix_tree/conftest.py`` stubs (magi_attention,
megatron, apex, transformer_engine) - forward.py hard-imports magi_attention.
"""

from __future__ import annotations

import pytest
import torch
from _helpers import make_pt_batch

from verl.utils.prefix_tree.dynamic import build_tree_dynamic
from verl.utils.prefix_tree.forward import (
    _build_flex_key,
    _finalize_prefix_tree_batch,
    _prepare_attn_inputs,
)
from verl.utils.prefix_tree.utils import build_layout_from_tree_node


def _layout(samples):
    subtrie = build_tree_dynamic(samples)
    assert subtrie is not None, "expected a shared-prefix trie"
    return build_layout_from_tree_node(samples, subtrie), subtrie


def _visible(block_mask, q: int, total: int) -> set[int]:
    """Evaluate mask_mod over all kv positions for a given q (CPU oracle probe)."""
    b = torch.tensor(0)
    h = torch.tensor(0)
    q_t = torch.tensor(q)
    return {kv for kv in range(total) if bool(block_mask.mask_mod(b, h, q_t, torch.tensor(kv)))}


def _oracle(subtrie, params, leaf_pos: int) -> dict[int, set[int]]:
    """Reference visibility: q -> visible kv set, from trie structure.

    For the leaf at position ``leaf_pos`` in ``params.leaf_ranges``, each token
    sees: every ancestor node's flat range (full), plus its own leaf causally.
    """
    nodes_by_idx = {n.node_idx: n for n in subtrie.nodes}

    # Rebuild flat ranges by replicating layout Pass 1 (contiguous BFS order).
    flat_start: dict[int, int] = {}
    pos = 0
    for node in subtrie.bfs():
        flat_start[node.node_idx] = pos
        pos += len(node.input_ids)

    start, end = params.leaf_ranges[leaf_pos]
    nid = subtrie.leaf_node_ids[leaf_pos]
    cur = nodes_by_idx[nid]
    path_ranges: list[tuple[int, int]] = []
    while cur is not None:
        if len(cur.input_ids) > 0 and cur.node_idx in flat_start:
            fs = flat_start[cur.node_idx]
            path_ranges.append((fs, fs + len(cur.input_ids)))
        cur = getattr(cur, "ancestor", None)

    visible: dict[int, set[int]] = {}
    for q in range(start, end):
        vis: set[int] = set()
        for rs, re_ in path_ranges:
            if (rs, re_) == (start, end):
                vis |= {x for x in range(rs, re_) if x <= q}  # own leaf: causal
            else:
                vis |= set(range(rs, re_))  # ancestor: full
        visible[q] = vis
    return visible


def _assert_mask_matches_oracle(samples, case: str):
    params, subtrie = _layout(samples)
    block_mask = _build_flex_key(params, torch.device("cpu"), subtrie=subtrie)
    total = params.total_seqlen_q
    for leaf_pos in range(len(params.leaf_ranges)):
        expected = _oracle(subtrie, params, leaf_pos)
        for q, exp in expected.items():
            got = _visible(block_mask, q, total)
            assert got == exp, f"{case}: leaf {leaf_pos} q={q}: flex visible {sorted(got)} != oracle {sorted(exp)}"


# ---------------------------------------------------------------------------
# Mask semantics vs oracle (bug lockers)
# ---------------------------------------------------------------------------

_T = torch.tensor

_MASK_CASES = [
    ("shared-prefix-only", [_T([1, 2, 3, 4, 5, 6]), _T([1, 2, 3, 4, 7, 8])]),
    ("intermediate-branch", [_T([1, 2, 3, 4, 5, 6]), _T([1, 2, 3, 4, 7, 8]), _T([1, 2, 9, 9, 9, 9])]),
    ("mixed-depths", [_T([1, 2, 3, 4, 5]), _T([1, 2, 3, 6, 7]), _T([1, 2, 8]), _T([1, 2, 8, 9])]),
    ("strict-prefix", [_T([1, 2, 3]), _T([1, 2, 3, 4, 5])]),
    ("duplicates", [_T([1, 2, 3, 4]), _T([1, 2, 3, 4]), _T([1, 2, 5, 6])]),
]


@pytest.mark.parametrize(("case", "samples"), _MASK_CASES)
def test_flex_mask_matches_oracle(case, samples):
    """_build_flex_key's mask must equal the trie oracle on every topology.
    Bug lockers per case:
    - shared-prefix-only: depth-2 trie (shared prompt, divergent responses) -
      passes today; regression guard.
    - intermediate-branch: middle branch node (shared response prefix beyond
      the prompt) must be visible to its descendants - leaf-only leaf_id made
      these tokens invisible, so samples could not attend their own shared tokens.
    - mixed-depths: unequal branch depths; middle nodes and a mid-tree branch
      must all be visible.
    - strict-prefix: one sample's sequence is a strict prefix of another's: the
      trie root IS that sample's leaf; its tokens must attend only causally
      (``in_prefix_k & (q_leaf >= 0)`` let q=0 see the whole root).
    - duplicates: identical sequences share one leaf - mask must stay correct
      per leaf."""
    _assert_mask_matches_oracle(samples, case)


# ---------------------------------------------------------------------------
# BlockMask artifact contract
# ---------------------------------------------------------------------------


def test_flex_block_mask_keeps_leaf_id_closure_alive():
    """block_mask._flex_aux pins the closure tensors so they survive GC; must exist
    and span ALL flat positions (sentinel -1 outside node spans, node id inside)."""
    params, subtrie = _layout(
        [
            torch.tensor([1, 2, 3, 4, 5, 6]),
            torch.tensor([1, 2, 3, 4, 7, 8]),
        ]
    )
    block_mask = _build_flex_key(params, torch.device("cpu"), subtrie=subtrie)
    flex_aux = getattr(block_mask, "_flex_aux", None)
    assert flex_aux is not None, "block_mask._flex_aux missing (closure may be GC'd)"
    pos_node = flex_aux[0]
    assert pos_node.shape[0] == params.total_seqlen_q, (
        f"pos_node spans {pos_node.shape[0]} positions != total {params.total_seqlen_q}"
    )
    # Positions inside a node span carry that node's id; every real flat
    # position belongs to exactly one trie node, so all must be >= 0.
    in_node = pos_node >= 0
    assert in_node.all(), "every flat position should map to a trie node"


# ---------------------------------------------------------------------------
# _prepare_attn_inputs flex contract
# ---------------------------------------------------------------------------


def test_prepare_attn_inputs_flex_returns_full_layout():
    """Flex path: no dispatch - full tree-packed tokens, batch dim added, flex key kwarg."""
    params, subtrie = _layout(
        [
            torch.tensor([10, 20, 30, 41, 42]),
            torch.tensor([10, 20, 30, 51]),
        ]
    )
    block_mask = _build_flex_key(params, torch.device("cpu"), subtrie=subtrie)
    pb = make_pt_batch(params, subtrie, flex_key=block_mask)

    input_ids, position_ids, attn_kwargs = _prepare_attn_inputs(pb, "flex")

    assert input_ids.shape == (1, params.total_seqlen_q)
    assert position_ids.shape == (1, params.total_seqlen_q)
    assert torch.equal(input_ids.squeeze(0), pb.tree_packed_input_ids)
    assert torch.equal(position_ids.squeeze(0), pb.tree_packed_position_ids)
    assert attn_kwargs == {"flex_attention_key": block_mask}
    assert "magi_attention_key" not in attn_kwargs


def test_prepare_attn_inputs_rejects_flex_without_key():
    """Flex branch with flex_key=None must fail loudly (silent None would hit the
    FA3 fallback and produce full-causal attention over the packed layout)."""
    params, subtrie = _layout(
        [
            torch.tensor([10, 20, 30, 41, 42]),
            torch.tensor([10, 20, 30, 51]),
        ]
    )
    pb = make_pt_batch(params, subtrie)
    with pytest.raises(RuntimeError, match="flex_key"):
        _prepare_attn_inputs(pb, "flex")


# ---------------------------------------------------------------------------
# _finalize_prefix_tree_batch attention_type dispatch
# ---------------------------------------------------------------------------


def test_finalize_dispatches_attention_type(monkeypatch):
    """attention_type='flex' -> flex_key set, magi_key None; 'magi' -> the inverse.
    Magi key building is stubbed (needs real mpu + magi_attention)."""
    samples = [
        torch.tensor([10, 20, 30, 41, 42]),
        torch.tensor([10, 20, 30, 51]),
    ]

    params, subtrie = _layout(samples)
    pb_flex = _finalize_prefix_tree_batch(params, model=None, num_samples=2, attention_type="flex", subtrie=subtrie)
    assert pb_flex.flex_key is not None
    assert pb_flex.magi_key is None

    import verl.utils.prefix_tree.forward as ptf

    monkeypatch.setattr(ptf, "_build_magi_key", lambda model, params: object())
    params2, subtrie2 = _layout(samples)
    pb_magi = _finalize_prefix_tree_batch(params2, model=None, num_samples=2, attention_type="magi")
    assert pb_magi.magi_key is not None
    assert pb_magi.flex_key is None


def test_finalize_rejects_unknown_attention_type():
    params, _ = _layout(
        [
            torch.tensor([10, 20, 30, 41, 42]),
            torch.tensor([10, 20, 30, 51]),
        ]
    )
    with pytest.raises(ValueError):
        _finalize_prefix_tree_batch(params, model=None, num_samples=2, attention_type="bogus")
