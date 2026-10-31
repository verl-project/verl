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
real router) lives in VeOmni's invariant test suite; the engine-side
pad / Ulysses / nested-rebuild glue lives in
``tests/workers/test_router_replay_engine_helpers_on_cpu.py``.

What's covered
--------------
* RECORD lifecycle for a single micro-batch, and the reset between
  micro-batches.
* Recompute under per-layer activation checkpointing, which fires each
  layer *independently in reverse order* — the failure mode that breaks
  any monotonic-cursor design.
* ``take_recorded`` stacking order and shape.
* REPLAY first step (R3 case): targets must work before any RECORD has
  populated the id table.
* REPLAY strict missing-target and shape-mismatch error paths.
* Replay validity is driven by ``routed_experts_mask`` (explicit) or by
  an all-zero row (fallback) -- never by inspecting top-k values.
* ``begin_microbatch`` mode validation.
* Snapshot clone semantics.
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
    row. Distinct-per-row keeps these fixtures obviously valid; the
    controller no longer inspects top-k values, so duplicates would be
    replayed verbatim too (see ``test_replay_all_zero_rows_fall_back_without_mask``)."""
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


def test_take_recorded_is_layer_major(ctrl, routers):
    """``take_recorded`` stacks layers on dim=1, so the result matches the
    trainer-side ``routed_experts`` layout ``[nnz, L, topk]``."""
    ctrl.begin_record()
    ctrl.begin_microbatch()
    fired = _fire_all(ctrl, routers)

    recorded = ctrl.take_recorded()
    assert recorded.shape == (_NNZ, len(routers), _TOPK)
    for pos, native in enumerate(fired):
        assert torch.equal(recorded[:, pos, :], native), f"layer {pos} landed on the wrong slice"


def test_record_begin_microbatch_drops_previous_microbatch(ctrl, routers):
    """The controller holds exactly one micro-batch. Re-arming must clear
    the previous one, otherwise ``take_recorded`` would stack stale layers
    and the engine's per-mb gather would see the wrong nnz."""
    ctrl.begin_record()
    for _ in range(3):
        ctrl.begin_microbatch()
        _fire_all(ctrl, routers)
        assert len(ctrl._recorded) == len(routers)
    assert ctrl.action is RouterReplayAction.RECORD


def test_take_recorded_before_any_router_fired_raises(ctrl):
    """An empty buffer means the model's SparseMoeBlock never called the
    hook — fail with a message that points at the wiring."""
    ctrl.begin_record()
    ctrl.begin_microbatch()
    with pytest.raises(RuntimeError, match="no router fired"):
        ctrl.take_recorded()


def test_take_recorded_outside_record_mode_raises(ctrl):
    ctrl.begin_replay()
    with pytest.raises(RuntimeError, match="requires RECORD"):
        ctrl.take_recorded()


# ===========================================================
# Activation-checkpointing recompute (the bug class id-keying solves)
# ===========================================================


def test_record_recompute_per_layer_reverse(ctrl, routers):
    """Per-layer checkpointing: backward replays each layer independently
    in REVERSE order. This is the realistic VeOmni MoE training case
    that breaks any monotonic-cursor design. Subsumes whole-model
    recompute (which is just sequential forward order)."""
    ctrl.begin_record()
    ctrl.begin_microbatch()
    fired = _fire_all(ctrl, routers)  # forward layer 0..L-1
    for r in reversed(routers):  # backward recompute, reverse order
        ctrl.on_router_forward(r, _scores(), _idx())

    assert len(ctrl._recorded) == len(routers), f"per-layer reverse recompute leaked extra slots: {len(ctrl._recorded)}"
    # The recompute fired with *different* indices; the original snapshot wins.
    for pos, native in enumerate(fired):
        assert torch.equal(ctrl._recorded[pos], native), f"layer {pos} was overwritten by recompute"


# ===========================================================
# REPLAY
# ===========================================================


def test_replay_first_step_without_prior_discovery(ctrl, routers):
    """R3 first step: REPLAY runs before any RECORD has populated the id
    table. begin_microbatch just stashes the list; lookup happens
    lazily during forward."""
    ctrl.begin_replay()
    targets = [_idx() for _ in routers]
    ctrl.begin_microbatch(targets=targets)
    returned = _fire_all(ctrl, routers)
    for i, ret in enumerate(returned):
        assert torch.equal(ret, targets[i]), f"REPLAY layer {i} returned wrong target"


def test_replay_per_layer_reverse_recompute(ctrl, routers):
    """REPLAY recompute under per-layer checkpointing: same id ->
    same target, regardless of fire order."""
    ctrl.begin_replay()
    targets = [_idx() for _ in routers]
    ctrl.begin_microbatch(targets=targets)
    _fire_all(ctrl, routers)  # forward populates id mapping
    # backward recompute in reverse — each layer must hit its OWN target
    for r, want_pos in zip(reversed(routers), reversed(range(len(routers))), strict=True):
        ret = ctrl.on_router_forward(r, _scores(), _idx())
        assert torch.equal(ret, targets[want_pos]), f"REPLAY recompute layer {want_pos} returned wrong target"


def test_replay_strict_missing_target_pos_raises(ctrl, routers):
    """Layer position with no target must raise — no silent fallback."""
    ctrl.begin_replay()
    ctrl.begin_microbatch(targets=[_idx()])  # only 1 target for 3 layers
    ctrl.on_router_forward(routers[0], _scores(), _idx())  # pos 0 OK
    with pytest.raises(RuntimeError, match="pos=1.*no target"):
        ctrl.on_router_forward(routers[1], _scores(), _idx())


def test_replay_all_zero_rows_fall_back_without_mask(ctrl, routers):
    """Without an explicit mask, only an entirely-zero row is treated as
    padding. A row whose top-k happens to repeat one expert is a perfectly
    valid replay target and must be replayed verbatim -- the previous
    duplicate heuristic dropped those rows back to native routing."""
    ctrl.begin_replay()

    targets = []
    for _ in routers:
        t = torch.stack(
            [torch.arange(_NNZ, dtype=torch.int64) + k for k in range(_TOPK)],
            dim=-1,
        )  # row i: [i, i+1] -- distinct
        t[0] = 0  # all-zero row == padding
        t[1, 0] = 5
        t[1, 1] = 5  # valid, but repeats one expert
        targets.append(t)
    ctrl.begin_microbatch(targets=targets)

    for r, t in zip(routers, targets, strict=True):
        native = torch.stack(
            [torch.arange(_NNZ, dtype=torch.int64) + 100 + k for k in range(_TOPK)],
            dim=-1,
        )
        out = ctrl.on_router_forward(r, _scores(), native)

        assert torch.equal(out[0], native[0]), "all-zero row must fall back to native"
        assert torch.equal(out[1], t[1]), "repeated-expert row is still a valid target"
        for i in range(2, _NNZ):
            assert torch.equal(out[i], t[i]), f"clean row {i} must use replay target"


def test_replay_respects_explicit_mask(ctrl, routers):
    """An explicit mask wins over the value-based fallback: a masked-in row is
    replayed even when it is all-zero, a masked-out row stays native even when
    it carries plausible-looking indices."""
    ctrl.begin_replay()
    targets = [torch.arange(_NNZ * _TOPK, dtype=torch.int64).view(_NNZ, _TOPK) + 7 for _ in routers]
    targets = [t.clone() for t in targets]
    targets[0][0] = 0  # looks like padding, but the mask says it is real
    mask = torch.ones(_NNZ, dtype=torch.bool)
    mask[5] = False
    ctrl.begin_microbatch(targets=targets, mask=mask)

    native = torch.stack(
        [torch.arange(_NNZ, dtype=torch.int64) + 100 + k for k in range(_TOPK)],
        dim=-1,
    )
    out = ctrl.on_router_forward(routers[0], _scores(), native)

    assert torch.equal(out[0], targets[0][0]), "masked-in all-zero row must be replayed"
    assert torch.equal(out[5], native[5]), "masked-out row must stay native"
    for i in (1, 2, 3, 4, 6, 7):
        assert torch.equal(out[i], targets[0][i])


def test_replay_topk_one_supported(ctrl, routers):
    """topk=1 no longer breaks replay: padding is identified by the mask (or
    by an all-zero row), never by a duplicate that a single slot cannot form."""
    ctrl.begin_replay()
    targets = [torch.full((_NNZ, 1), 3, dtype=torch.int64) for _ in routers]
    targets[0] = targets[0].clone()
    targets[0][0] = 0  # padding row
    ctrl.begin_microbatch(targets=targets)

    native = torch.full((_NNZ, 1), 7, dtype=torch.int64)
    out = ctrl.on_router_forward(routers[0], _scores(), native)

    assert torch.equal(out[0], native[0]), "topk=1 padding row must fall back to native"
    assert torch.equal(out[1], targets[0][1]), "topk=1 valid row must be replayed"


def test_replay_mask_row_count_mismatch_raises(ctrl, routers):
    """A mask packed with a different pad/SP rule than the targets would shift
    every row's validity; refuse instead of broadcasting silently."""
    ctrl.begin_replay()
    with pytest.raises(RuntimeError, match="mask has"):
        ctrl.begin_microbatch(
            targets=[torch.zeros(_NNZ, _TOPK, dtype=torch.int64) for _ in routers],
            mask=torch.zeros(_NNZ + 3, dtype=torch.bool),
        )


def test_replay_target_row_count_mismatch_raises(ctrl, routers):
    """Defensive: a target sliced with a different SP rule than input_ids
    would silently misalign routing. The controller refuses rather than
    letting the downstream ``torch.where`` broadcast."""
    ctrl.begin_replay()
    ctrl.begin_microbatch(targets=[torch.zeros(_NNZ + 4, _TOPK, dtype=torch.int64) for _ in routers])
    with pytest.raises(RuntimeError, match="target has shape"):
        ctrl.on_router_forward(routers[0], _scores(), _idx())


# ===========================================================
# begin_microbatch mode validation
# ===========================================================


def test_begin_microbatch_replay_without_targets_raises(ctrl):
    ctrl.begin_replay()
    with pytest.raises(RuntimeError, match="REPLAY requires per-layer targets"):
        ctrl.begin_microbatch()


def test_begin_microbatch_record_with_targets_raises(ctrl):
    ctrl.begin_record()
    with pytest.raises(RuntimeError, match="RECORD does not take targets"):
        ctrl.begin_microbatch(targets=[_idx()])


def test_begin_microbatch_while_disabled_raises(ctrl):
    with pytest.raises(RuntimeError, match="requires RECORD or REPLAY"):
        ctrl.begin_microbatch()


# ===========================================================
# State management
# ===========================================================


def test_clear_resets_state(ctrl, routers):
    """clear() is the always-safe reset; state must be empty after."""
    ctrl.begin_record()
    ctrl.begin_microbatch()
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
    ctrl.begin_microbatch()
    src = _idx()
    ctrl.on_router_forward(routers[0], _scores(), src)
    src.fill_(99)
    captured = ctrl._recorded[0]
    assert (captured != 99).all(), "snapshot must be independent of the source tensor (.detach().clone())"


# ===========================================================
# Disabled state
# ===========================================================


def test_disabled_passes_through_indices(ctrl, routers):
    """When no begin_*() has been called (DISABLED), on_router_forward
    is a no-op pass-through and does not even assign layer positions."""
    src = _idx()
    out = ctrl.on_router_forward(routers[0], _scores(), src)
    assert torch.equal(out, src)
    assert ctrl._recorded == []
    assert ctrl._targets == []
    assert ctrl._id_to_pos == {}


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
