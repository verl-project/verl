# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Unit tests for ``verl.utils.veomni.router_replay.VeOmniRouterReplay``.

Pure-CPU coverage for the controller state machine. The patched
``SparseMoeBlock`` integration (the actual end-to-end forward through a
real router) lives in VeOmni's invariant test suite; the engine-driver
glue (snapshot side-channel + nested rebuild) lives in
``tests/workers/test_router_replay_engine_helpers_on_cpu.py``.

What's covered
--------------
* RECORD lifecycle (single mb, multi-mb).
* Recompute under whole-model AND per-layer activation checkpointing.
  Per-layer checkpointing fires backward recompute for each layer
  *independently in reverse order* — the failure mode that breaks any
  monotonic-cursor design.
* REPLAY first step (R3 case): set_microbatch_targets must work
  before any RECORD has populated the id table.
* REPLAY strict missing-target error path.
* Snapshot clone semantics.
* id-mapping stability across micro-batches.
* ``install`` / ``uninstall`` against a stubbed VeOmni hook surface.
"""

import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from verl.utils.veomni.router_replay import RouterReplayAction, VeOmniRouterReplay

# ----------------------------------------------------------------- fixtures


@pytest.fixture
def ctrl():
    """Fresh controller per test. Does NOT call ``install()`` — that
    would need a stubbed ``veomni.utils.moe_router_replay`` module.
    The controller's ``on_router_forward`` / ``begin_*`` / ``clear``
    methods do not depend on ``install()`` having been called, so most
    tests don't need it. The ``install`` / ``uninstall`` paths get
    their own dedicated tests below."""
    return VeOmniRouterReplay()


# Each FakeRouter is a distinct nn.Module so ``id(router)`` is unique
# and stable for the test's lifetime — exactly what the production code
# relies on for FSDP2-wrapped MoE routers.
class _FakeRouter(nn.Module):
    pass


@pytest.fixture
def routers():
    """Three distinct router instances, mimicking three MoE layers."""
    return [_FakeRouter() for _ in range(3)]


# Toy shapes — small enough that any failure prints readable tensors.
_NNZ = 16
_TOPK = 2


def _scores():
    return torch.randn(_NNZ, 8)


def _idx():
    """Return ``[_NNZ, _TOPK]`` int indices with DISTINCT entries per
    row. Distinct-per-row matters for tests that assert REPLAY returns
    the target verbatim — the duplicate-detection fallback in
    ``on_router_forward`` would otherwise treat duplicate-top-k rows
    as corrupted and return native instead. Real router top-k output
    always picks distinct experts, so this matches production
    semantics."""
    # Random distinct top-k per row via sampling without replacement.
    return torch.stack(
        [torch.randperm(8)[:_TOPK] for _ in range(_NNZ)],
        dim=0,
    ).to(torch.int64)


def _fire_all(ctrl, routers):
    """Fire each router once in order and return the controller's outputs."""
    return [ctrl.on_router_forward(r, _scores(), _idx()) for r in routers]


# ===========================================================
# RECORD lifecycle
# ===========================================================


def test_record_multiple_microbatches(ctrl, routers):
    """advance_record_microbatch bumps the slot fill count without
    touching the id mapping. The id table must persist across mbs in
    the same step (recompute detection in on_router_forward depends on
    ``len(slot) == _mb_index + 1``, which only holds with stable ids)."""
    ctrl.begin_record()
    n_mb = 3
    for mb in range(n_mb):
        if mb > 0:
            ctrl.advance_record_microbatch()
        _fire_all(ctrl, routers)
    assert all(len(slot) == n_mb for slot in ctrl._recorded)
    assert ctrl.action is RouterReplayAction.RECORD


# ===========================================================
# Activation-checkpointing recompute (the bug class id-keying solves)
# ===========================================================


def test_record_recompute_per_layer_reverse(ctrl, routers):
    """Per-layer checkpointing: backward replays each layer independently
    in REVERSE order. This is the realistic VeOmni MoE training case
    that breaks any monotonic-cursor design. Subsumes whole-model
    recompute (which is just sequential forward order)."""
    ctrl.begin_record()
    _fire_all(ctrl, routers)  # forward layer 0..L-1
    for r in reversed(routers):  # backward recompute, reverse order
        ctrl.on_router_forward(r, _scores(), _idx())
    assert all(len(slot) == 1 for slot in ctrl._recorded), (
        f"per-layer reverse recompute leaked: {[len(s) for s in ctrl._recorded]}"
    )


# ===========================================================
# REPLAY
# ===========================================================


def test_replay_first_step_without_prior_discovery(ctrl, routers):
    """R3 first step: REPLAY runs before any RECORD has populated the id
    table. set_microbatch_targets just stashes the list; lookup happens
    lazily during forward."""
    ctrl.begin_replay()
    targets = [_idx() for _ in routers]
    ctrl.set_microbatch_targets(targets)
    returned = _fire_all(ctrl, routers)
    for i, ret in enumerate(returned):
        assert torch.equal(ret, targets[i]), f"REPLAY layer {i} returned wrong target"


def test_replay_per_layer_reverse_recompute(ctrl, routers):
    """REPLAY recompute under per-layer checkpointing: same id ->
    same target, regardless of fire order."""
    ctrl.begin_replay()
    targets = [_idx() for _ in routers]
    ctrl.set_microbatch_targets(targets)
    _fire_all(ctrl, routers)  # forward populates id mapping
    # backward recompute in reverse — each layer must hit its OWN target
    for r, want_pos in zip(reversed(routers), reversed(range(len(routers))), strict=True):
        ret = ctrl.on_router_forward(r, _scores(), _idx())
        assert torch.equal(ret, targets[want_pos]), f"REPLAY recompute layer {want_pos} returned wrong target"


def test_replay_strict_missing_target_raises(ctrl, routers):
    """Layer position with no target must raise — no silent fallback."""
    ctrl.begin_replay()
    ctrl.set_microbatch_targets([_idx()])  # only 1 target for 3 layers
    ctrl.on_router_forward(routers[0], _scores(), _idx())  # pos 0 OK
    with pytest.raises(RuntimeError, match="pos=1.*no target"):
        ctrl.on_router_forward(routers[1], _scores(), _idx())


def test_replay_with_mask_substitutes_only_masked_tokens(ctrl, routers):
    """R3 prompt-token regression: when ``replay_mask`` is provided,
    only tokens with mask=True get the recorded target; mask=False
    tokens fall through to native indices.

    This is what prevents R3 from sending all prompt tokens to expert 0
    (the rollout backend writes zeros for prompt tokens because it
    never captured prefill-time routing). Without the mask, those
    zeros would be substituted, corrupting the EP all-to-all and
    surfacing as ``RuntimeError: Split sizes doesn't match total dim
    0 size`` mid-forward.
    """
    ctrl.begin_replay()
    # 16 tokens; first 10 are "prompt" (mask=False), last 6 are "response" (mask=True).
    mask = torch.tensor([False] * 10 + [True] * 6)
    targets = [torch.zeros(_NNZ, _TOPK, dtype=torch.int64) for _ in routers]
    # Fill response-portion of each target with DISTINCT-per-slot nonzero
    # values so they (a) differ from the native sentinel, (b) don't
    # contain duplicate top-k slots that would trigger the duplicate
    # fallback.
    for layer_pos, t in enumerate(targets):
        for k in range(_TOPK):
            t[10:, k] = layer_pos * 10 + k + 1  # e.g., layer 0 row: [1, 2], layer 1 row: [11, 12]
    ctrl.set_microbatch_targets(targets, replay_mask=mask)

    for layer_pos, r in enumerate(routers):
        # Native values must also be distinct per slot so they don't
        # themselves trigger the duplicate fallback.
        native = torch.stack(
            [torch.full((_NNZ,), 90 + k, dtype=torch.int64) for k in range(_TOPK)],
            dim=-1,
        )  # row pattern: [90, 91]
        out = ctrl.on_router_forward(r, _scores(), native)

        # First 10 rows (prompt, mask=False): native values preserved.
        assert torch.equal(out[:10], native[:10]), (
            f"layer {layer_pos}: prompt rows must keep native indices, got {out[:10]}"
        )
        # Last 6 rows (response, mask=True): substituted with the target.
        assert torch.equal(out[10:], targets[layer_pos][10:]), (
            f"layer {layer_pos}: response rows must use replay target"
        )


def test_replay_duplicate_topk_falls_back_to_native(ctrl, routers):
    """R3 robustness regression: rows where the substituted top-k
    contains duplicates (e.g. a placeholder ``[0, 0, 0, ...]`` that
    slipped through the mask, or any rollout-side corruption) must
    fall through to native routing.

    Without this fallback, VeOmni's MoE expert dispatch silently
    dedupes the duplicate top-k slots inside ``permute()``, while
    ``input_splits`` keeps counting all of them — the EP all-to-all
    then crashes with ``Split sizes doesn't match total dim 0 size``
    several layers deep.
    """
    ctrl.begin_replay()
    # All-True mask — exercise the fallback purely on duplicate detection.
    mask = torch.ones(_NNZ, dtype=torch.bool)

    # Build a target with two corruption patterns:
    #   row 0: all-zeros (extreme duplicate — every slot is expert 0)
    #   row 1: partial duplicate (slots 0 and 1 both expert 5)
    # The other rows are clean (distinct top-k).
    targets = []
    for _ in routers:
        t = torch.stack(
            [torch.arange(_NNZ, dtype=torch.int64) + k for k in range(_TOPK)],
            dim=-1,
        )  # row i: [i, i+1] — distinct
        t[0] = 0  # all-zero row
        t[1, 0] = 5
        t[1, 1] = 5  # duplicate row
        targets.append(t)
    ctrl.set_microbatch_targets(targets, replay_mask=mask)

    for r, t in zip(routers, targets, strict=True):
        # Native: distinct per row, distinct per slot.
        native = torch.stack(
            [torch.arange(_NNZ, dtype=torch.int64) + 100 + k for k in range(_TOPK)],
            dim=-1,
        )
        out = ctrl.on_router_forward(r, _scores(), native)

        # Corrupted rows fall back to native.
        assert torch.equal(out[0], native[0]), "all-zero row must fall back to native"
        assert torch.equal(out[1], native[1]), "partial-duplicate row must fall back to native"
        # Clean rows substitute normally.
        for i in range(2, _NNZ):
            assert torch.equal(out[i], t[i]), f"clean row {i} must use replay target"


def test_replay_mask_shape_mismatch_raises(ctrl, routers):
    """Defensive: a mask sliced with a different SP rule than the
    targets would silently corrupt routing. The controller refuses
    rather than broadcast."""
    ctrl.begin_replay()
    targets = [_idx() for _ in routers]
    bad_mask = torch.ones(_NNZ + 4, dtype=torch.bool)  # too long
    ctrl.set_microbatch_targets(targets, replay_mask=bad_mask)
    with pytest.raises(RuntimeError, match="replay_mask has .* rows but top_indices"):
        ctrl.on_router_forward(routers[0], _scores(), _idx())


def test_replay_set_targets_outside_replay_mode_raises(ctrl):
    """set_microbatch_targets is REPLAY-only."""
    ctrl.begin_record()
    with pytest.raises(RuntimeError, match="requires REPLAY"):
        ctrl.set_microbatch_targets([_idx()])


# ===========================================================
# State management
# ===========================================================


def test_advance_record_microbatch_outside_record_raises(ctrl):
    """advance_record_microbatch is RECORD-only."""
    ctrl.begin_replay()
    with pytest.raises(RuntimeError, match="requires RECORD"):
        ctrl.advance_record_microbatch()


def test_clear_resets_state(ctrl, routers):
    """clear() is the always-safe reset; state must be empty after."""
    ctrl.begin_record()
    _fire_all(ctrl, routers)
    assert ctrl.action is RouterReplayAction.RECORD
    assert ctrl._recorded
    ctrl.clear()
    assert ctrl.action is RouterReplayAction.DISABLED
    assert ctrl._recorded == []
    assert ctrl._targets == []
    assert ctrl._id_to_pos == {}


# ===========================================================
# Snapshot clone semantics
# ===========================================================


def test_record_snapshot_independent_of_source_tensor(ctrl, routers):
    """The captured tensor must NOT alias the source — otherwise
    autograd-graph mutations would corrupt recorded indices."""
    ctrl.begin_record()
    src = _idx()
    ctrl.on_router_forward(routers[0], _scores(), src)
    src.fill_(99)
    captured = ctrl._recorded[0][0]
    assert (captured != 99).all(), "snapshot must be independent of the source tensor (.detach().clone())"


# ===========================================================
# Disabled state
# ===========================================================


def test_disabled_passes_through_indices(ctrl, routers):
    """When no begin_*() has been called (DISABLED), on_router_forward
    is a no-op pass-through."""
    src = _idx()
    out = ctrl.on_router_forward(routers[0], _scores(), src)
    assert torch.equal(out, src)
    assert ctrl._recorded == []
    assert ctrl._targets == []


# ===========================================================
# install / uninstall against a stubbed VeOmni hook surface
# ===========================================================


def _make_veomni_stub():
    """Return a (stub_module, captured_state) tuple where ``stub_module``
    is the fake ``veomni.utils.moe_router_replay`` and
    ``captured_state['active']`` records the last value passed to
    ``set_active_replay``."""
    stub = MagicMock()
    captured = {"active": None}

    def _set(x):
        captured["active"] = x

    stub.set_active_replay.side_effect = _set
    stub.get_active_replay.side_effect = lambda: captured["active"]
    stub.validate_model_for_replay.return_value = None
    return stub, captured


def test_install_uninstall_roundtrip(ctrl):
    """install() registers the controller in VeOmni's global slot and
    runs the model validator; uninstall() clears the slot back to None."""
    stub, captured = _make_veomni_stub()
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "veomni.utils.moe_router_replay", stub)
        ctrl.install(nn.Linear(1, 1))
        assert captured["active"] is ctrl
        stub.validate_model_for_replay.assert_called_once()
        ctrl.uninstall()
        assert captured["active"] is None


def test_install_without_veomni_raises(ctrl):
    """If the VeOmni hook surface is missing, install() must raise a
    typed RuntimeError pointing the user at the dependency, not a raw
    ImportError."""
    with pytest.MonkeyPatch.context() as mp:
        # Setting sys.modules[name] = None makes Python's import machinery
        # raise ImportError on `from veomni.utils.moe_router_replay import ...`
        # without needing to manipulate sys.path (which doesn't help when
        # veomni is installed in site-packages).
        mp.setitem(sys.modules, "veomni.utils.moe_router_replay", None)
        with pytest.raises(RuntimeError, match="VeOmni build"):
            ctrl.install(nn.Linear(1, 1))
