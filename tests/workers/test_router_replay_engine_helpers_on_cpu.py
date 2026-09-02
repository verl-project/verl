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
two engine hooks in ``verl/workers/engine/veomni/transformer_impl.py``
that own everything shape-related:

* ``VeOmniEngineWithLMHead._prepare_router_replay_inputs`` — arms the
  controller, and in REPLAY mode reshapes ``routed_experts`` with the
  same pad rule ``super().prepare_model_inputs`` applied to
  ``input_ids``.
* ``VeOmniEngineWithLMHead.prepare_model_outputs`` — unpads the recorded
  indices and re-wraps them per sample, like ``log_probs``.

The real production helpers are imported here (not mirrored) — the
``veomni.*`` package surfaces that ``transformer_impl`` needs at module
load are stubbed with ``MagicMock`` via ``sys.modules`` patching, the
same pattern the rest of the verl test suite uses for optional deps
(see ``test_rollout_trace_on_cpu.py`` for the ``weave`` precedent).

Out of scope (covered by manual GPU smoke + the e2e shell scripts):
real ``VeOmniEngineWithLMHead`` instantiation, FSDP wrapping,
multi-rank SP slice/all-gather, end-to-end forward through the patched
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
    "veomni.models.checkpoint_tensor_loading",
    "veomni.models.checkpoint_tensor_loading.get_checkpoint_tensor_converter",
):
    sys.modules.setdefault(_mod, MagicMock())


from verl.utils.veomni.router_replay import VeOmniRouterReplay  # noqa: E402
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead  # noqa: E402
from verl.workers.engine.veomni.transformer_impl import VeOmniEngineWithLMHead  # noqa: E402

# ----------------------------------------------------------------- helpers

_L, _TOPK = 3, 2


def _make_jagged_input_ids(seq_lens: list[int]) -> torch.Tensor:
    """Build a jagged NestedTensor mimicking what ``left_right_2_no_padding``
    produces for ``input_ids``."""
    pieces = [torch.randint(0, 100, (s,), dtype=torch.int64) for s in seq_lens]
    return torch.nested.as_nested_tensor(pieces, layout=torch.jagged)


def _make_jagged_routed_experts(seq_lens: list[int], L: int = _L, topk: int = _TOPK) -> torch.Tensor:
    """Build a jagged NestedTensor mimicking the trainer-side
    ``routed_experts`` shape: ``[bs, jagged_seq, L, topk]``."""
    pieces = [torch.randint(0, 8, (s, L, topk), dtype=torch.int64) for s in seq_lens]
    return torch.nested.as_nested_tensor(pieces, layout=torch.jagged)


def _make_engine(controller: VeOmniRouterReplay | None, mode: str = "R2") -> VeOmniEngineWithLMHead:
    """Build a bare ``VeOmniEngineWithLMHead`` instance without invoking
    its ``__init__`` (which requires a torch.distributed process group,
    parallel_state init, etc.). Only the attributes the two helpers read
    are populated."""
    engine = VeOmniEngineWithLMHead.__new__(VeOmniEngineWithLMHead)
    engine._router_replay = controller
    engine._router_replay_mode = mode
    engine.use_ulysses_sp = False
    return engine


@pytest.fixture
def controller():
    return VeOmniRouterReplay()


# ===========================================================
# _prepare_router_replay_inputs
# ===========================================================


class TestPrepareRouterReplayInputs:
    def test_record_only_arms_the_controller(self, controller):
        """RECORD doesn't read ``routed_experts`` (it's the *output*, not an
        input). Even when the micro_batch carries one, it must be ignored —
        and the previous micro-batch's buffer must be dropped."""
        controller.begin_record()
        controller._recorded = [torch.zeros(4, _TOPK, dtype=torch.int64)]  # stale mb
        engine = _make_engine(controller)

        seq_lens = [3, 4]
        td = TensorDict(
            {
                "input_ids": _make_jagged_input_ids(seq_lens),
                "routed_experts": _make_jagged_routed_experts(seq_lens),
            },
            batch_size=[2],
        )
        engine._prepare_router_replay_inputs(td, {"pad_size": 0})

        assert controller._recorded == []
        assert controller._targets == []

    def test_replay_missing_routed_experts_raises(self, controller):
        """Strict mode: REPLAY without routed_experts is a plumbing bug
        (compute_log_prob → update_actor lost the field), not a soft
        fallback."""
        controller.begin_replay()
        engine = _make_engine(controller)
        td = TensorDict({"input_ids": _make_jagged_input_ids([3, 5])}, batch_size=[2])
        with pytest.raises(RuntimeError, match="missing 'routed_experts'"):
            engine._prepare_router_replay_inputs(td, {"pad_size": 0})

    def test_replay_unbinds_targets_per_layer(self, controller):
        """``routed_experts`` covers prompt + response for every token, so with
        no padding the targets go through verbatim: one ``[total_nnz, topk]``
        tensor per layer."""
        controller.begin_replay()
        engine = _make_engine(controller)

        seq_lens = [4, 6]
        routed = _make_jagged_routed_experts(seq_lens)
        td = TensorDict(
            {
                "input_ids": _make_jagged_input_ids(seq_lens),
                "routed_experts": routed,
            },
            batch_size=[2],
        )
        engine._prepare_router_replay_inputs(td, {"pad_size": 0})

        assert len(controller._targets) == _L
        flat = routed.values()
        for pos, t in enumerate(controller._targets):
            assert t.shape == (sum(seq_lens), _TOPK)
            assert torch.equal(t, flat[:, pos, :])

    def test_replay_pad_extends_targets_with_zero_rows(self, controller):
        """With ``pad_to_length`` (or the Ulysses alignment) the packed sequence
        the routers see is longer than the recorded routing, so the targets have
        to grow by ``pad_size``. The tail is zero-filled, which the controller's
        duplicate-top-k check reads as "route this natively"."""
        controller.begin_replay()
        engine = _make_engine(controller)

        seq_lens = [4, 6]
        pad_size = 5
        td = TensorDict(
            {
                "input_ids": _make_jagged_input_ids(seq_lens),
                "routed_experts": _make_jagged_routed_experts(seq_lens),
            },
            batch_size=[2],
        )
        engine._prepare_router_replay_inputs(td, {"pad_size": pad_size})

        real_nnz = sum(seq_lens)
        assert len(controller._targets) == _L
        for t in controller._targets:
            assert t.shape == (real_nnz + pad_size, _TOPK)
            assert (t[real_nnz:] == 0).all(), "pad tail must be zero so the duplicate fallback fires"

    def test_replay_is_uniform_across_r2_and_r3(self, controller):
        """R3 records at the rollout backend, which reports routing for every
        token it forwarded — prompt included. So R3 needs no response-only
        gate: it takes the same path as R2."""
        seq_lens = [5, 7]
        td_fields = {
            "input_ids": _make_jagged_input_ids(seq_lens),
            "routed_experts": _make_jagged_routed_experts(seq_lens),
            # Present but irrelevant: gating prompt tokens out would let them
            # fall back to native routing and diverge from the rollout.
            "response_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.int64),
        }

        targets_per_mode = {}
        for mode in ("R2", "R3"):
            ctrl = VeOmniRouterReplay()
            ctrl.begin_replay()
            engine = _make_engine(ctrl, mode=mode)
            engine._prepare_router_replay_inputs(TensorDict(dict(td_fields), batch_size=[2]), {"pad_size": 0})
            targets_per_mode[mode] = ctrl._targets
        for r2, r3 in zip(targets_per_mode["R2"], targets_per_mode["R3"], strict=True):
            assert torch.equal(r2, r3)

    def test_no_controller_is_a_noop(self):
        """If router_replay is disabled on the engine, the helper is a
        pure no-op even when the micro_batch is malformed."""
        engine = _make_engine(controller=None)
        td = TensorDict(
            {"input_ids": torch.randint(0, 100, (2, 8), dtype=torch.int64)},
            batch_size=[2],
        )
        engine._prepare_router_replay_inputs(td, {"pad_size": 0})

    def test_disabled_controller_is_a_noop(self, controller):
        """Same for a controller that exists but is between steps."""
        engine = _make_engine(controller)
        td = TensorDict(
            {"input_ids": torch.randint(0, 100, (2, 8), dtype=torch.int64)},
            batch_size=[2],
        )
        engine._prepare_router_replay_inputs(td, {"pad_size": 0})
        assert controller._targets == []


# ===========================================================
# prepare_model_outputs
# ===========================================================


class TestPrepareModelOutputs:
    """``super().prepare_model_outputs`` needs a real HF model output, so it is
    stubbed out here — these tests only cover the routed-experts addendum."""

    @pytest.fixture(autouse=True)
    def _stub_super(self, monkeypatch):
        monkeypatch.setattr(
            FSDPEngineWithLMHead,
            "prepare_model_outputs",
            lambda self, output, output_args, micro_batch, logits_processor_func: {"log_probs": None},
        )

    def test_record_output_is_unpadded_and_wrapped_per_sample(self, controller):
        """The recorded indices arrive as ``[nnz + pad, L, topk]``; the pad
        suffix is dropped and the rest is re-wrapped on ``input_ids`` offsets
        so ``postprocess_batch_func`` can restore batch order."""
        seq_lens = [4, 6]
        pad_size = 3
        controller.begin_record()
        controller.begin_microbatch()

        routers = [torch.nn.Linear(1, 1) for _ in range(_L)]
        fired = []
        for r in routers:
            idx = torch.randint(0, 8, (sum(seq_lens) + pad_size, _TOPK), dtype=torch.int64)
            controller.on_router_forward(r, torch.randn(idx.size(0), 8), idx)
            fired.append(idx)

        engine = _make_engine(controller)
        input_ids = _make_jagged_input_ids(seq_lens)
        td = TensorDict({"input_ids": input_ids}, batch_size=[2])
        model_output = engine.prepare_model_outputs(
            output=None,
            output_args={"pad_size": pad_size},
            micro_batch=td,
            logits_processor_func=None,
        )

        routed = model_output["routed_experts"]
        assert routed.is_nested
        per_sample = routed.unbind()
        assert [t.shape for t in per_sample] == [(s, _L, _TOPK) for s in seq_lens]
        # Values line up with what the routers reported, pad suffix removed. The trip
        # through the uint8 view is a reinterpretation, so it must not perturb a
        # single id, and the result must come back as int16.
        assert routed.dtype == torch.int16
        expected = torch.stack(fired, dim=1)[: sum(seq_lens)]
        assert torch.equal(routed.values(), expected)

    def test_record_output_preserves_ids_above_uint8_range(self, controller):
        """The whole point of int16 storage: expert ids past 255 must survive the
        record -> gather -> unpad path. uint8 would wrap them silently."""
        seq_lens = [3]
        controller.begin_record()
        controller.begin_microbatch()
        idx = torch.tensor([[0, 255], [256, 511], [1023, 300]], dtype=torch.int64)
        controller.on_router_forward(torch.nn.Linear(1, 1), torch.randn(3, 1024), idx)

        engine = _make_engine(controller)
        td = TensorDict({"input_ids": _make_jagged_input_ids(seq_lens)}, batch_size=[1])
        model_output = engine.prepare_model_outputs(
            output=None, output_args={"pad_size": 0}, micro_batch=td, logits_processor_func=None
        )
        assert torch.equal(model_output["routed_experts"].values(), idx.unsqueeze(1))

    def test_replay_output_carries_no_routed_experts(self, controller):
        """REPLAY consumes routing, it doesn't produce it."""
        controller.begin_replay()
        engine = _make_engine(controller)
        td = TensorDict({"input_ids": _make_jagged_input_ids([3, 4])}, batch_size=[2])
        model_output = engine.prepare_model_outputs(
            output=None, output_args={"pad_size": 0}, micro_batch=td, logits_processor_func=None
        )
        assert "routed_experts" not in model_output

    def _replay_forward(self, controller, num_routers: int, seq_lens: list[int], L: int = _L):
        """Arm REPLAY with ``L`` layer slots, then fire ``num_routers`` routers."""
        controller.begin_replay()
        nnz = sum(seq_lens)
        controller.begin_microbatch(targets=[torch.randint(0, 8, (nnz, _TOPK)) for _ in range(L)])
        # Positions are keyed on ``id(module)``, so the routers must be kept
        # alive for the whole forward -- CPython recycles the address of a
        # temporary, which would collapse them onto one position.
        routers = [torch.nn.Linear(1, 1) for _ in range(num_routers)]
        for r in routers:
            controller.on_router_forward(
                r,
                torch.randn(nnz, 8),
                torch.randint(0, 8, (nnz, _TOPK), dtype=torch.int64),
            )
        engine = _make_engine(controller)
        return engine.prepare_model_outputs(
            output=None,
            output_args={"pad_size": 0},
            micro_batch=TensorDict({"input_ids": _make_jagged_input_ids(seq_lens)}, batch_size=[len(seq_lens)]),
            logits_processor_func=None,
        )

    def test_replay_rejects_fewer_routers_than_layer_slots(self, controller):
        """The DeepSeek-V4 failure mode: a family whose layers use more than one
        router class, with only some classes hooked. Targets are matched by fire
        order while the rollout indexes by absolute layer, so the unhooked layers'
        slots are still consumed and every later layer replays the wrong experts.
        ``on_router_forward`` cannot see this (it only trips when a router finds
        no slot at all), and nothing downstream fails — the run just trains on
        mis-routed experts. So it has to be caught once the forward is over."""
        with pytest.raises(RuntimeError, match=r"2 routers fired but routed_experts carries 3"):
            self._replay_forward(controller, num_routers=_L - 1, seq_lens=[4, 6])

    def test_replay_accepts_one_router_per_layer_slot(self, controller):
        """The contract holding is the common case and must stay silent."""
        model_output = self._replay_forward(controller, num_routers=_L, seq_lens=[4, 6])
        assert "routed_experts" not in model_output

    def test_no_controller_output_is_untouched(self):
        engine = _make_engine(controller=None)
        td = TensorDict({"input_ids": _make_jagged_input_ids([3, 4])}, batch_size=[2])
        model_output = engine.prepare_model_outputs(
            output=None, output_args={"pad_size": 0}, micro_batch=td, logits_processor_func=None
        )
        assert "routed_experts" not in model_output
