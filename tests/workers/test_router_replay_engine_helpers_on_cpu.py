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

"""Engine-driver glue tests for VeOmni router replay.

Complements ``tests/utils/veomni/test_router_replay_on_cpu.py`` (which
covers the controller state machine in isolation). This file tests the
engine-side glue helpers in ``verl/workers/engine/veomni/transformer_impl.py``:

* ``VeOmniEngineWithLMHead._maybe_push_router_replay_state`` — the
  per-micro-batch snapshot side-channel (pad_size + cu_seqlens) pushed
  back from ``prepare_model_inputs`` to ``forward_backward_batch``,
  plus the jagged-NestedTensor input assertion and the strict
  missing-routed_experts error path in REPLAY mode.

The real production helpers are imported here (not mirrored) — the
``veomni.*`` package surfaces that ``transformer_impl`` needs at module
load are stubbed with ``MagicMock`` via ``sys.modules`` patching, the
same pattern the rest of the verl test suite uses for optional deps
(see ``test_rollout_trace_on_cpu.py`` for the ``weave`` precedent).

Out of scope (covered by manual GPU smoke + the e2e shell scripts):
real ``VeOmniEngineWithLMHead`` instantiation, FSDP wrapping,
multi-rank SP all-gather, end-to-end forward through the patched
``SparseMoeBlock``.
"""

import sys
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

# ----------------------------------------------------------------- veomni stub
#
# ``transformer_impl.py`` imports several ``veomni.*`` submodules at
# module top level (``OpsImplementationConfig``, ``parallel_state``,
# ``build_foundation_model`` etc.). MagicMock auto-creates any
# attribute on access, so a single stub per submodule is enough to let
# the import succeed in environments without VeOmni installed (which is
# the standard verl CPU-unit-tests image — VeOmni is only present in
# the e2e_*.yml workflows). On environments where VeOmni IS installed,
# ``setdefault`` is a no-op and the real package is used.

for _mod in (
    "veomni",
    "veomni.arguments",
    "veomni.distributed",
    "veomni.distributed.offloading",
    "veomni.distributed.torch_parallelize",
    "veomni.models",
    "veomni.models.auto",
    "veomni.optim",
    "veomni.utils",
    "veomni.utils.moe_router_replay",
    "veomni.utils.seqlen_pos_transform_utils",
):
    sys.modules.setdefault(_mod, MagicMock())


from verl.utils.veomni.router_replay import RouterReplayAction, VeOmniRouterReplay  # noqa: E402
from verl.workers.engine.veomni.transformer_impl import VeOmniEngineWithLMHead  # noqa: E402

# ----------------------------------------------------------------- helpers


def _make_jagged_input_ids(seq_lens: list[int]) -> torch.Tensor:
    """Build a jagged NestedTensor mimicking what ``left_right_2_no_padding``
    produces for ``input_ids``."""
    pieces = [torch.randint(0, 100, (s,), dtype=torch.int64) for s in seq_lens]
    return torch.nested.as_nested_tensor(pieces, layout=torch.jagged)


def _make_jagged_routed_experts(seq_lens: list[int], L: int, topk: int) -> torch.Tensor:
    """Build a jagged NestedTensor mimicking the trainer-side
    ``routed_experts`` shape: ``[bs, jagged_seq, L, topk]``."""
    pieces = [torch.randint(0, 8, (s, L, topk), dtype=torch.int64) for s in seq_lens]
    return torch.nested.as_nested_tensor(pieces, layout=torch.jagged)


def _make_engine_with_controller(
    controller: VeOmniRouterReplay,
    mode: str = "R2",
) -> VeOmniEngineWithLMHead:
    """Build a bare ``VeOmniEngineWithLMHead`` instance without invoking
    its ``__init__`` (which requires a torch.distributed process group,
    parallel_state init, etc.). Only the controller and the mode string
    (used to gate R3-specific replay-mask construction) are needed for
    the helper methods exercised here."""
    engine = VeOmniEngineWithLMHead.__new__(VeOmniEngineWithLMHead)
    engine._router_replay = controller
    engine._router_replay_mode = mode
    return engine


@pytest.fixture
def controller():
    return VeOmniRouterReplay()


# ===========================================================
# _maybe_push_router_replay_state
# ===========================================================


class TestMaybePushRouterReplayState:
    """The per-mb side-channel + REPLAY input prep glue. Exercised on a
    bare-instance ``VeOmniEngineWithLMHead`` (no real init needed —
    the method only reads ``self._router_replay``)."""

    def test_pad_and_cu_seqlens_snapshot_population(self, controller):
        """The two snapshot sinks should receive one entry per call,
        in the order ``prepare_model_inputs`` ran."""
        controller.begin_record()
        engine = _make_engine_with_controller(controller)

        pad_sink: list[int] = []
        cu_sink: list[torch.Tensor] = []

        for mb in range(3):
            mb_lens = [3 + mb, 5 + mb]
            input_ids = _make_jagged_input_ids(mb_lens)
            td = TensorDict({"input_ids": input_ids}, batch_size=[len(mb_lens)])
            td.set_non_tensor("_router_replay_pad_size_out", pad_sink)
            td.set_non_tensor("_router_replay_cu_seqlens_out", cu_sink)
            output_args = {"pad_size": mb}  # pretend ulysses pad varies
            engine._maybe_push_router_replay_state(td, output_args)

        assert pad_sink == [0, 1, 2]
        assert len(cu_sink) == 3
        # Each cu_seqlens entry should reflect input_ids offsets at the
        # time of capture (cloned, not aliased to the now-freed mb).
        assert cu_sink[0].tolist() == [0, 3, 8]
        assert cu_sink[1].tolist() == [0, 4, 10]
        assert cu_sink[2].tolist() == [0, 5, 12]

    def test_side_channel_list_can_be_stashed_on_micro_batch(self):
        """Regression: ``forward_backward_batch`` stashes the snapshot
        list via ``tu.assign_non_tensor_data(...)`` onto a micro_batch
        that has a non-empty batch_size. The original code used
        ``tu.assign_non_tensor`` (auto-dispatch), which detects the
        value as a list and routes to ``assign_non_tensor_stack``,
        producing a ``NonTensorStack`` with ``batch_size=[len(list)]``.
        For an empty list (``[]``), the stack has ``batch_size=[0]``,
        which mismatches the micro_batch's ``batch_size=[1]`` and
        triggers ``RuntimeError: ... Modifying the batch size of a
        lazy representation of a tensordict is not permitted`` because
        ``NonTensorStack`` is itself a lazy TD.

        The fix: use ``assign_non_tensor_data`` (singular), which wraps
        the entire list as one ``NonTensorData(val)`` regardless of type
        — the list IS the value, not a per-sample sequence to stack.
        """
        from verl.utils import tensordict_utils as tu

        # Plain TensorDict with batch_size=[1] — same shape that
        # prepare_micro_batches yields for a 1-sample micro-batch.
        td = TensorDict({}, batch_size=[1])

        # Sanity: the OLD (broken) auto-dispatch path raises on empty list.
        with pytest.raises(RuntimeError, match="lazy representation"):
            tu.assign_non_tensor(td, _broken_sink=[])

        # The fixed singular API tolerates any value, including an empty
        # list, by wrapping it as a single NonTensorData.
        sink: list[int] = []
        tu.assign_non_tensor_data(td, "_sink", sink)

        # Mutate via reference and read back — the side-channel contract
        # the engine driver relies on.
        sink.append(7)
        sink.append(11)
        recovered = tu.get_non_tensor_data(td, "_sink", default=None)
        assert recovered == [7, 11], f"side-channel list lost mutations: {recovered}"

    def test_non_jagged_input_ids_raises(self, controller):
        """The engine init guard rejects use_remove_padding=False, but
        the per-mb assertion catches future bypass paths — a dense
        tensor would otherwise fail mid-step on .offsets() with an
        opaque AttributeError."""
        controller.begin_record()
        engine = _make_engine_with_controller(controller)
        td = TensorDict(
            {"input_ids": torch.randint(0, 100, (2, 8), dtype=torch.int64)},
            batch_size=[2],
        )
        with pytest.raises(RuntimeError, match="must be a jagged NestedTensor"):
            engine._maybe_push_router_replay_state(td, {"pad_size": 0})

    def test_replay_missing_routed_experts_raises(self, controller):
        """Strict mode: REPLAY without routed_experts on the micro_batch
        is a plumbing bug (compute_log_prob → update_actor lost the
        field), not a soft fallback."""
        controller.begin_replay()
        engine = _make_engine_with_controller(controller)
        td = TensorDict({"input_ids": _make_jagged_input_ids([3, 5])}, batch_size=[2])
        with pytest.raises(RuntimeError, match="missing 'routed_experts'"):
            engine._maybe_push_router_replay_state(td, {"pad_size": 0})

    def test_replay_with_routed_experts_feeds_targets(self, controller):
        """Successful R2 REPLAY path: routed_experts.values() is sliced
        per-layer and forwarded to set_microbatch_targets. The
        controller's ``_targets`` should then contain L per-layer
        tensors. R2 must NOT build a replay_mask — RECORD captured
        full-sequence routing, so REPLAY substitutes uniformly."""
        L, topk = 3, 2
        controller.begin_replay()
        engine = _make_engine_with_controller(controller, mode="R2")

        seq_lens = [4, 6]
        routed = _make_jagged_routed_experts(seq_lens, L, topk)
        td = TensorDict(
            {
                "input_ids": _make_jagged_input_ids(seq_lens),
                "routed_experts": routed,
                # R2 must ignore response_mask even when present —
                # gating prompt tokens out would re-introduce routing
                # divergence on prompt tokens that propagates through
                # attention KV into response logits/grad.
                "response_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.int64),
            },
            batch_size=[2],
        )
        engine._maybe_push_router_replay_state(td, {"pad_size": 0})
        assert len(controller._targets) == L
        for t in controller._targets:
            assert t.shape == (sum(seq_lens), topk)
        assert controller._replay_mask is None, "R2 must NOT build a replay_mask"

    def test_r3_with_response_mask_builds_per_token_mask(self, controller):
        """Regression: R3 must build a per-rmpad-token mask from
        ``response_mask`` (strided ``(bs, max_response_len)``) via
        per-sample length arithmetic — calling ``response_mask.values()``
        directly raises ``RuntimeError: values expected sparse tensor
        layout but got Strided``.

        Layout the test mimics: 2 samples with total lens [5, 7], where
        the last 2 / 3 tokens of each are response. The expected flat
        mask is ``[0,0,0,1,1, 0,0,0,0,1,1,1]``.
        """
        L, topk = 2, 1
        controller.begin_replay()
        engine = _make_engine_with_controller(controller, mode="R3")

        # Sample 0: 5 tokens (3 prompt + 2 response).
        # Sample 1: 7 tokens (4 prompt + 3 response).
        seq_lens = [5, 7]
        max_response_len = 4  # padded; sample 0 has 2 response tokens, sample 1 has 3
        response_mask = torch.zeros(2, max_response_len, dtype=torch.int64)
        response_mask[0, :2] = 1  # sample 0: 2 response tokens
        response_mask[1, :3] = 1  # sample 1: 3 response tokens

        routed = _make_jagged_routed_experts(seq_lens, L, topk)
        td = TensorDict(
            {
                "input_ids": _make_jagged_input_ids(seq_lens),
                "routed_experts": routed,
                "response_mask": response_mask,
            },
            batch_size=[2],
        )
        engine._maybe_push_router_replay_state(td, {"pad_size": 0})

        assert controller._replay_mask is not None, "R3 must build a replay_mask"
        expected = torch.tensor(
            [
                False,
                False,
                False,
                True,
                True,  # sample 0: 3 prompt + 2 response
                False,
                False,
                False,
                False,
                True,
                True,
                True,  # sample 1: 4 prompt + 3 response
            ],
            dtype=torch.bool,
        )
        assert torch.equal(controller._replay_mask, expected), (
            f"replay_mask mismatch: got {controller._replay_mask.tolist()}, expected {expected.tolist()}"
        )

    def test_r3_missing_response_mask_raises(self, controller):
        """R3 needs response_mask to know which tokens to substitute.
        Missing it is a plumbing bug, not a soft-fallback case."""
        L, topk = 2, 1
        controller.begin_replay()
        engine = _make_engine_with_controller(controller, mode="R3")
        seq_lens = [3, 4]
        td = TensorDict(
            {
                "input_ids": _make_jagged_input_ids(seq_lens),
                "routed_experts": _make_jagged_routed_experts(seq_lens, L, topk),
            },
            batch_size=[2],
        )
        with pytest.raises(RuntimeError, match="R3.*missing 'response_mask'"):
            engine._maybe_push_router_replay_state(td, {"pad_size": 0})

    def test_r3_response_longer_than_total_raises(self, controller):
        """Defensive: if response_mask describes more tokens than the
        actor's input, prompt_lens goes negative — fail-fast with a
        typed error instead of letting repeat_interleave produce a
        malformed mask that surfaces later as the EP all-to-all crash."""
        L, topk = 2, 1
        controller.begin_replay()
        engine = _make_engine_with_controller(controller, mode="R3")
        # Sample 0: 3 tokens but response_mask claims 5 → prompt_lens = -2.
        seq_lens = [3, 4]
        response_mask = torch.zeros(2, 6, dtype=torch.int64)
        response_mask[0, :5] = 1  # impossible: 5 response > 3 total
        response_mask[1, :2] = 1
        td = TensorDict(
            {
                "input_ids": _make_jagged_input_ids(seq_lens),
                "routed_experts": _make_jagged_routed_experts(seq_lens, L, topk),
                "response_mask": response_mask,
            },
            batch_size=[2],
        )
        with pytest.raises(RuntimeError, match="response_mask sum exceeds total token"):
            engine._maybe_push_router_replay_state(td, {"pad_size": 0})

    def test_record_does_not_consume_routed_experts(self, controller):
        """RECORD path doesn't read routed_experts (it's the *output*,
        not an input). Even if the micro_batch happens to carry one,
        it must not be touched."""
        L, topk = 2, 1
        controller.begin_record()
        engine = _make_engine_with_controller(controller)
        seq_lens = [3, 4]
        td = TensorDict(
            {
                "input_ids": _make_jagged_input_ids(seq_lens),
                "routed_experts": _make_jagged_routed_experts(seq_lens, L, topk),
            },
            batch_size=[2],
        )
        engine._maybe_push_router_replay_state(td, {"pad_size": 0})
        # No targets set — RECORD ignores the field entirely.
        assert controller._targets == []

    def test_no_controller_is_a_noop(self):
        """If router_replay is disabled on the engine, the helper is a
        pure no-op even when the micro_batch is malformed."""
        engine = _make_engine_with_controller(controller=None)
        # Deliberately broken micro_batch — would raise if the helper
        # progressed past the early ``return``.
        td = TensorDict(
            {"input_ids": torch.randint(0, 100, (2, 8), dtype=torch.int64)},
            batch_size=[2],
        )
        engine._maybe_push_router_replay_state(td, {"pad_size": 0})
        # No raise = pass.


def _silence_unused_action_import():
    """``RouterReplayAction`` is imported for symmetry with the engine
    code that consumes it; silence the lint warning in tests."""
    _ = RouterReplayAction
