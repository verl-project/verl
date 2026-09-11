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


"""CPU tests for verl/utils/prefix_tree/magi.py, including strict-prefix junction
coverage for the fused LCE boundary registry.

Junction rule: where a sample terminates (its whole sequence is a strict token
prefix of another sample's), the junction must be registered like a fork: the
continuing sample reads the junction position, so it needs its own next-token
label. Pre-fix, only >=2-children nodes were registered, so the continuing
sample's boundary log-prob came from the terminating owner's rolled 0-pad
label — silently wrong. Triggers on multi-turn/agentic data, not on ordinary
shared-prompt forks or GRPO n-siblings.
"""

from __future__ import annotations

import types

import pytest
import torch
from _helpers import build_layout

from verl.utils.prefix_tree import magi as magi_mod
from verl.utils.prefix_tree.dynamic import build_tree_dynamic
from verl.utils.prefix_tree.magi import restore_flat_to_nested


def test_build_prefix_tree_micro_batch_unpacks_nested(monkeypatch):
    """Integration: NestedTensor input -> flat layout via build_prefix_tree_micro_batch."""
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


# ---- strict-prefix junction registration (boundary registry) ----


def test_strict_prefix_junction_registered():
    """A=[1,2,3] is a strict prefix of B=[1,2,3,4]: the junction (last shared
    token, flat pos 2) must be registered with B's next token 4."""
    _, params = build_layout([torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4])])
    assert params.boundary_registry == [(2, [(1, 4)])], params.boundary_registry


def test_fork_registry():
    """Ordinary forks must not gain entries from the strict-prefix junction fix:
    a pure 3-way fork keeps ONE boundary with all three leaves; when one branch
    also terminates mid-way, the fork boundary carries the continuing leaves and
    the terminating branch's junction carries only the continuing leaf."""
    _, params = build_layout([torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 5]), torch.tensor([1, 2, 3, 6])])
    assert params.boundary_registry == [(2, [(0, 4), (1, 5), (2, 6)])], params.boundary_registry
    _, params = build_layout([torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 5])])
    assert params.boundary_registry == [(2, [(1, 4), (2, 5)])], params.boundary_registry


def test_chain_junctions_registered():
    """A=[1,2] ⊂ B=[1,2,3] ⊂ C=[1,2,3,4]: junction 1 is read by BOTH B and C
    (B terminates at the internal node [3], not a childless leaf); junction 2
    is read only by C."""
    tensors = [torch.tensor([1, 2]), torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4])]
    _, params = build_layout(tensors)
    assert params.boundary_registry == [(1, [(1, 3), (2, 3)]), (2, [(2, 4)])], params.boundary_registry


def test_strict_prefix_junction_restore():
    """End-to-end restore: only the continuing sample B has a boundary entry
    (A terminates — its label at the junction is the masked 0-pad, so A keeps
    the flat value). B's tensor must carry B's own log-prob at the junction."""
    tensors = [torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4])]
    pb, params = build_layout(tensors)
    (boundary_pos, [(sample_idx, _next_token)]) = params.boundary_registry[0]
    assert (boundary_pos, sample_idx) == (2, 1)

    b_val = torch.tensor(-7.0)
    pb._boundary_logps = {sample_idx: [(boundary_pos, b_val)]}

    flat = torch.arange(pb.tree_packed_input_ids.shape[0], dtype=torch.float32)
    restored = restore_flat_to_nested(flat, pb, apply_boundary_patch=True)
    lengths = restored.offsets().diff().tolist()
    assert lengths == [len(s) for s in tensors]
    vals = restored.values()
    # A (sample 0): no entry → keeps flat value at the junction.
    assert vals[2] == flat[boundary_pos], "terminating sample's junction row was modified"
    assert vals[0] == flat[0] and vals[1] == flat[1], "terminating sample's prefix rows corrupted"
    # B (sample 1): junction patched to its own log-prob.
    assert vals[3 + 2] == b_val, "continuing sample's junction not patched"
    assert vals[3 + 3] == flat[3], "continuing sample's extension row corrupted"
