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


"""CPU tests for verl/utils/prefix_tree/magi.py: flat layout build,"""

from __future__ import annotations

import torch

from verl.utils.prefix_tree import magi as magi_mod
from verl.utils.prefix_tree.dynamic import build_tree_dynamic
from verl.utils.prefix_tree.magi import PrefixTreeMagiBatch, restore_flat_to_nested
from verl.utils.prefix_tree.utils import build_layout_from_tree_node


def _build_params(tokens):
    result = build_tree_dynamic(tokens)
    assert result is not None, "Expected shared prefix trie"
    return build_layout_from_tree_node(tokens, result), result


def _build_pt_batch(tokens):
    from verl.utils.prefix_tree.magi import PackRestorationParam

    params, subtrie = _build_params(tokens)
    return PrefixTreeMagiBatch(
        tree_packed_input_ids=params.tree_packed_tokens,
        tree_packed_position_ids=params.tree_packed_position_ids,
        tree_packed_labels=params.tree_packed_labels,
        magi_key=None,
        flex_key=None,
        restoration=PackRestorationParam(
            segment_ranges=params.leaf_ranges,
            prefix_range=params.prefix_range,
        ),
        subtrie=subtrie,
    )


def _build_pt_batch(tokens):
    from verl.utils.prefix_tree.magi import PackRestorationParam

    params, subtrie = _build_params(tokens)
    return PrefixTreeMagiBatch(
        tree_packed_input_ids=params.tree_packed_tokens,
        tree_packed_position_ids=params.tree_packed_position_ids,
        tree_packed_labels=params.tree_packed_labels,
        magi_key=None,
        flex_key=None,
        restoration=PackRestorationParam(
            segment_ranges=params.leaf_ranges,
            prefix_range=params.prefix_range,
        ),
        subtrie=subtrie,
    )


def test_basic_shared_prefix_flat_layout_and_flex_rects():
    tokens = [
        torch.tensor([10, 20, 30, 41, 42]),
        torch.tensor([10, 20, 30, 51]),
        torch.tensor([10, 20, 30, 61, 62, 63]),
    ]
    params, _ = _build_params(tokens)
    assert list(params.tree_packed_tokens[:3].tolist()) == [10, 20, 30]
    assert params.prefix_range[0] == 0 and params.prefix_range[1] >= 1
    assert len(params.leaf_ranges) == 3  # one per sample
    assert params.total_seqlen_q >= max(t.numel() for t in tokens)
    short = [torch.tensor([10, 20, 30, 41, 42]), torch.tensor([10, 20, 30, 51])]
    sp, _ = _build_params(short)
    rects = set(zip(sp.q_ranges, sp.k_ranges, sp.mask_types, strict=False))
    assert any(m == "causal" for _, _, m in rects)
    assert any(m == "full" for _, _, m in rects)


def test_restore_token_ids_round_trip():
    tokens = [
        torch.tensor([10, 20, 30, 41, 42]),
        torch.tensor([10, 20, 30, 51]),
        torch.tensor([10, 20, 30, 61, 62, 63]),
    ]
    pt_batch = _build_pt_batch(tokens)
    restored = restore_flat_to_nested(pt_batch.tree_packed_input_ids, pt_batch)
    offsets, vals = restored.offsets(), restored.values()
    lengths = offsets.diff().tolist()
    assert lengths == [5, 4, 6]
    pos = 0
    for i, orig in enumerate(tokens):
        assert torch.equal(vals[pos : pos + int(lengths[i])], orig), f"sample {i} mismatch"
        pos += int(lengths[i])


def test_build_prefix_tree_micro_batch_unpacks_nested(monkeypatch):
    """Integration: NestedTensor input -> flat layout via build_prefix_tree_micro_batch."""
    import types

    pytest = __import__("pytest")
    pytest.importorskip("codetiming")
    import verl.utils.prefix_tree.forward as ptf
    import verl.utils.prefix_tree.magi as ptm

    monkeypatch.setattr(ptf, "_build_magi_key", lambda model, params: object())
    cfg = types.SimpleNamespace(num_attention_heads=8, num_query_groups=8, kv_channels=128, fp8=None)
    model = types.SimpleNamespace(config=cfg, pre_process=True, post_process=True)
    tensors = [torch.tensor(t) for t in [[10, 20, 30, 41, 42], [10, 20, 30, 51], [10, 20, 30, 61, 62, 63]]]
    input_ids = torch.nested.nested_tensor(tensors, layout=torch.jagged)
    subtrie = build_tree_dynamic(tensors)
    assert subtrie is not None
    result = ptm.build_prefix_tree_micro_batch(model, input_ids, subtrie=subtrie)
    assert result is not None and len(result.restoration.segment_ranges) == 3
    assert list(result.tree_packed_input_ids[:3].tolist()) == [10, 20, 30]


class _Rotary(torch.nn.Module):
    """Stand-in for megatron RotaryEmbedding: just needs a ``_pids`` slot."""


class _GPTModel(torch.nn.Module):
    """Stand-in for megatron GPTModel: exposes ``rotary_pos_emb``."""

    def __init__(self) -> None:
        super().__init__()
        self.rotary_pos_emb = _Rotary()


class _WrappedEngine(torch.nn.Module):
    """Stand-in for the DDP/FSDP-wrapped engine passed to prepare_prefix_tree.

    Critically, it does NOT expose ``rotary_pos_emb`` directly (mirrors the real
    wrappers whose ``__getattr__`` does not delegate submodule attributes), so a
    bare ``getattr(model, "rotary_pos_emb")`` returns None — the bug condition.
    The inner GPTModel is reachable only via ``unwrap_model``.
    """

    def __init__(self, gpt: _GPTModel) -> None:
        super().__init__()
        self.module = gpt


def test_set_rope_pids_sets_pids_on_inner_rotary_through_wrapper(monkeypatch):
    """set_rope_pids must reach the GPTModel's rotary even when the top-level"""
    gpt = _GPTModel()
    wrapped = _WrappedEngine(gpt)
    assert not hasattr(wrapped, "rotary_pos_emb")
    assert hasattr(gpt, "rotary_pos_emb")
    assert getattr(gpt.rotary_pos_emb, "_pids", None) is None

    monkeypatch.setattr(magi_mod, "unwrap_model", lambda m: gpt if m is wrapped else m)

    pids = torch.tensor([0, 3, 1, 7], dtype=torch.long)
    magi_mod.set_rope_pids(wrapped, pids)

    assert gpt.rotary_pos_emb._pids is not None, (
        "set_rope_pids no-op'd: _pids was not set on the inner rotary "
        "(wrapped model hides rotary_pos_emb -> fallback RoPE bug)"
    )
    assert torch.equal(gpt.rotary_pos_emb._pids, pids.reshape(-1))


def test_clear_rope_pids_clears_inner_rotary(monkeypatch):
    """clear_rope_pids must clear _pids on the inner GPTModel's rotary."""
    gpt = _GPTModel()
    wrapped = _WrappedEngine(gpt)
    gpt.rotary_pos_emb._pids = torch.tensor([1, 2, 3], dtype=torch.long)

    monkeypatch.setattr(magi_mod, "unwrap_model", lambda m: gpt if m is wrapped else m)
    magi_mod.clear_rope_pids(wrapped)

    assert gpt.rotary_pos_emb._pids is None


def test_set_rope_pids_noop_when_position_ids_none(monkeypatch):
    """None position_ids must not set _pids (no false activation of the patch)."""
    gpt = _GPTModel()
    wrapped = _WrappedEngine(gpt)
    monkeypatch.setattr(magi_mod, "unwrap_model", lambda m: gpt if m is wrapped else m)

    magi_mod.set_rope_pids(wrapped, None)
    assert getattr(gpt.rotary_pos_emb, "_pids", None) is None
