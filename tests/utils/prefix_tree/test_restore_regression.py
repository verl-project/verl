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


"""Worker-side prefix-tree restore contracts.

These tests walk the EXACT worker-side flow used in production:

    build_global_trie (deepest-node leaf_idx)
    → mbs_groups_from_leaf_idx / create_and_attach_subtrie_views
      (leaf_to_sample = LOCAL positions within the micro-batch)
    → build_layout_from_tree_node
    → restore_flat_to_nested

Each test locks a regression fixed in this suite:

- owner resolution is order-independent (node-id keyed + descendant
  propagation), so mbs whose global sample ids are >= mb size restore exactly
  instead of raising "max leaf range end must equal total sequence length"
  (previously ``owner_of`` cross-matched GLOBAL sequence_ids against LOCAL
  leaf_to_sample keys).
- duplicate leaves (identical sequences sharing one leaf) restore exactly
  even when non-adjacent in the micro-batch (leaf_ranges is emitted per
  leaf_node_ids position, aligned with subtrie.leaf_to_sample).
- strict-prefix samples (one response is an exact prefix of another) group
  and restore instead of crashing in mbs_groups_from_leaf_idx.
"""

from __future__ import annotations

import torch

from verl.utils.prefix_tree.dynamic import build_tree_dynamic, mbs_groups_from_leaf_idx
from verl.utils.prefix_tree.magi import PackRestorationParam, PrefixTreeMagiBatch, restore_flat_to_nested
from verl.utils.prefix_tree.trainer import build_global_trie
from verl.utils.prefix_tree.tree import PrefixSubTrie
from verl.utils.prefix_tree.utils import build_layout_from_tree_node


def _grpo_samples(n_prompts, rollout_n, prefix_len, resp_len, seed=0, duplicate_pair=None):
    """prompt shared per group; responses random. duplicate_pair=(p, r0, r1) makes two identical responses."""
    g = torch.Generator().manual_seed(seed)
    samples = []
    for p in range(n_prompts):
        prefix = torch.randint(0, 100000, (prefix_len,), generator=g)
        resps = [torch.randint(0, 100000, (resp_len,), generator=g) for _ in range(rollout_n)]
        if duplicate_pair is not None and duplicate_pair[0] == p:
            resps[duplicate_pair[1]] = resps[duplicate_pair[2]].clone()
        for resp in resps:
            samples.append(torch.cat([prefix, resp]))
    return samples


def _build_pb(samples, subtrie, params):
    """Mirror _finalize_prefix_tree_batch's PrefixTreeMagiBatch wrapping."""
    return PrefixTreeMagiBatch(
        tree_packed_input_ids=params.tree_packed_tokens,
        tree_packed_position_ids=params.tree_packed_position_ids,
        tree_packed_labels=params.tree_packed_labels,
        magi_key=None,
        flex_key=None,
        restoration=PackRestorationParam(
            segment_ranges=params.leaf_ranges,
            prefix_range=params.prefix_range,
            ancestor_segment_ranges=getattr(params, "_leaf_ancestor_ranges", None),
            boundary_registry=getattr(params, "boundary_registry", None),
        ),
        subtrie=subtrie,
        real_tokens=params.tree_packed_tokens.shape[0],
    )


def _worker_restore(samples, trie, leaf_idx, order):
    """Build the worker-style subtrie (LOCAL leaf_to_sample) and restore."""
    subtrie = PrefixSubTrie(
        source=trie,
        leaf_node_ids=[int(leaf_idx[i]) for i in order],
        leaf_to_sample=list(range(len(order))),
        batch_size=len(order),
    )
    samples_mb = [samples[i] for i in order]
    params = build_layout_from_tree_node(samples_mb, subtrie)
    pb = _build_pb(samples_mb, subtrie, params)
    restored = restore_flat_to_nested(pb.tree_packed_input_ids, pb)
    lengths = restored.offsets().diff().tolist()
    assert lengths == [len(s) for s in samples_mb], (
        f"restored lengths {lengths} != expected {[len(s) for s in samples_mb]}"
    )
    vals = restored.values()
    pos = 0
    for i, s in enumerate(samples_mb):
        assert torch.equal(vals[pos : pos + len(s)], s), f"sample {i} token mismatch"
        pos += len(s)


def test_owner_resolution_order_independent_late_global_ids():
    """mb containing only samples with global ids >= mb_size restores exactly."""
    samples = _grpo_samples(4, 8, prefix_len=300, resp_len=200, seed=42)
    trie, leaf_idx, _ = build_global_trie(samples)
    _worker_restore(samples, trie, leaf_idx, list(range(16, 32)))


def test_owner_resolution_order_independent_shuffled_ids():
    """mb order != global id order must still restore exactly (was: wrong content)."""
    samples = _grpo_samples(4, 8, prefix_len=300, resp_len=200, seed=42)
    trie, leaf_idx, _ = build_global_trie(samples)
    order = list(range(32))
    order = order[8:] + order[:8]
    _worker_restore(samples, trie, leaf_idx, order)


def test_identity_order_with_adjacent_duplicates():
    samples = _grpo_samples(3, 4, prefix_len=100, resp_len=50, seed=42, duplicate_pair=(1, 2, 3))
    trie, leaf_idx, _ = build_global_trie(samples)
    assert int(leaf_idx[6]) == int(leaf_idx[7])
    _worker_restore(samples, trie, leaf_idx, list(range(len(samples))))


def test_non_adjacent_duplicate_leaves_restore_exactly():
    """identical samples at non-adjacent mb positions (was: content scramble)."""
    g = torch.Generator().manual_seed(11)
    prefix = torch.randint(0, 100000, (200,), generator=g)
    resp = [torch.randint(0, 100000, (50,), generator=g) for _ in range(4)]
    samples = [
        torch.cat([prefix, resp[0]]),
        torch.cat([prefix, resp[1]]),
        torch.cat([prefix, resp[2]]),
        torch.cat([prefix, resp[3]]),
        torch.cat([prefix, resp[1].clone()]),
    ]
    trie, leaf_idx, _ = build_global_trie(samples)
    assert int(leaf_idx[1]) == int(leaf_idx[4])
    _worker_restore(samples, trie, leaf_idx, list(range(5)))


def test_strict_prefix_sample_restores_exactly():
    """one response is an exact prefix of another (was: non-leaf ValueError)."""
    g = torch.Generator().manual_seed(3)
    prefix = torch.randint(0, 100000, (100,), generator=g)
    short = torch.randint(0, 100000, (20,), generator=g)
    long_resp = torch.cat([short, torch.randint(0, 100000, (30,), generator=g)])
    samples = [
        torch.cat([prefix, short]),
        torch.cat([prefix, long_resp]),
        torch.cat([prefix, torch.randint(0, 100000, (40,), generator=g)]),
    ]
    trie, leaf_idx, _ = build_global_trie(samples)
    assert trie is not None
    groups = mbs_groups_from_leaf_idx(leaf_idx, trie, max_token_len=10**6)
    assert sorted(i for mb in groups for i in mb) == list(range(len(samples)))
    for idx in groups:
        _worker_restore(samples, trie, leaf_idx, idx)


def test_worker_restore_round_trip_full_batch():
    """full worker flow: group by budget then restore every micro-batch exactly."""
    samples = _grpo_samples(6, 8, prefix_len=300, resp_len=200, seed=42, duplicate_pair=(2, 1, 5))
    trie, leaf_idx, _ = build_global_trie(samples)
    for budget in (10**9, 5000):
        groups = mbs_groups_from_leaf_idx(leaf_idx, trie, max_token_len=budget)
        assert sorted(i for mb in groups for i in mb) == list(range(len(samples)))
        for idx in groups:
            _worker_restore(samples, trie, leaf_idx, idx)


def test_unfused_expand_first_no_boundary_patch(monkeypatch):
    """Unfused post-processing expands per-sample BEFORE the logits processor:
    every sample's boundary position gets ITS OWN label's log-prob (no flat
    boundary patch)."""
    import verl.utils.prefix_tree.forward as pt_forward

    # model=None makes the real clear_rope_pids crash via unwrap_model's
    # isinstance against stub classnames; the rope context is unused here.
    monkeypatch.setattr(pt_forward, "clear_rope_pids", lambda model: None)

    tensors = [torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 5]), torch.tensor([1, 2, 3, 6])]
    subtrie = build_tree_dynamic(tensors)
    assert subtrie is not None
    params = build_layout_from_tree_node(tensors, subtrie)
    pb = _build_pb(tensors, subtrie, params)
    pb.per_sample_labels = [torch.cat([s[1:], torch.zeros(1, dtype=torch.long)]) for s in tensors]

    flat_len = pb.tree_packed_input_ids.shape[0]
    logits = torch.zeros(flat_len, 8)
    for i, lbl in enumerate(pb.tree_packed_labels.tolist()):
        logits[i, lbl] = 100.0
    # Boundary (flat pos 2, shared token 3): give every sample's OWN next token a
    # high, distinct score so per-sample values are distinguishable.
    for k, nxt in enumerate((4, 5, 6)):
        logits[2, nxt] = 100.0 + 10.0 * k

    def processor(logits_, label, temperature=1.0, **kw):
        lp = torch.log_softmax(logits_.squeeze(1), dim=-1)
        log_probs = lp.gather(1, label.long())
        probs = torch.softmax(logits_.squeeze(1), dim=-1)
        entropy = -(probs * lp).sum(-1)
        return {"log_probs": log_probs.squeeze(-1), "entropy": entropy}

    ctx = pt_forward.TreeForwardCtx(pb, None, None, "flex", model=None)
    out = pt_forward.tree_post_processing(ctx, logits.unsqueeze(0), processor, {"temperature": 1.0}, post_process=True)

    lengths = out["log_probs"].offsets().diff().tolist()
    assert lengths == [4, 4, 4]
    vals = out["log_probs"].values()
    pos = 0
    for j, s in enumerate(tensors):
        rolled = torch.cat([s[1:], torch.zeros(1, dtype=torch.long)])
        p_start, p_end = pb.restoration.prefix_range
        s_start, s_end = pb.restoration.segment_ranges[j]
        rows = list(range(p_start, p_end)) + list(range(s_start, s_end))
        lp = torch.log_softmax(logits[rows], dim=-1)
        expected = lp.gather(1, rolled.unsqueeze(1)).squeeze(-1)
        assert torch.allclose(vals[pos : pos + 4], expected), f"sample {j} boundary log-prob mismatch"
        pos += 4
