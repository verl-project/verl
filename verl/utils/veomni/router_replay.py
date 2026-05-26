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

"""VeOmni-flavored MoE router replay.

Self-contained record/replay controller for verl's VeOmni engine. Tapped
into VeOmni's MoE ``SparseMoeBlock.forward`` via
``veomni.utils.moe_router_replay.set_active_replay`` (a module-level
singleton slot); the patched forward calls
``maybe_replay_indices(module, routing_scores, top_indices)`` on each MoE
layer, which delegates to :meth:`VeOmniRouterReplay.on_router_forward`.

Lifecycle (per ``forward_backward_batch``)::

    # Outside every micro-batch loop:
    replay.begin_record()      # R2 compute_log_prob
        or replay.begin_replay()  # R2 actor update, or R3 everywhere

    # Inside the micro-batch loop, before each forward:
    replay.set_microbatch_targets(per_layer_targets)  # REPLAY only

    # After every micro-batch finishes:
    # (nothing — state carries across recompute-in-backward too)

    # After the whole step:
    routed_experts = replay.collect_recorded(...)     # RECORD only
    replay.clear()

Layer indexing
--------------
RR uses **id-keyed positional** indexing. The first time each MoE router
fires, we assign the next position (``len(_id_to_pos)``) to ``id(module)``;
every subsequent call for that module reuses the same position. This is
the key correctness property under **activation checkpointing**: backward
recompute fires the same router modules again (in any order — per-layer
checkpoint segments differentiate from L-1 down to 0), and the id-keyed
lookup gives the correct position regardless of the order. A
monotonically-incremented cursor would walk past ``len(_targets)`` (REPLAY
crash) or grow phantom slots in ``_recorded`` (RECORD corruption).

Layer position L is learned implicitly from input shapes:
    * RECORD: ``len(_recorded)`` after the first full forward establishes L.
    * REPLAY: ``set_microbatch_targets`` stashes a list of length L — no
      prior id-mapping discovery needed (this is what unblocks R3 step 1,
      where REPLAY runs *before* any RECORD has populated the id table).

Parallelism scope
-----------------
VeOmni uses FSDP2 + optional Ulysses SP, no pipeline parallelism. The
RECORD all-gather and REPLAY pad+slice both go through
``verl.utils.ulysses`` so the SP layout matches what
``super().prepare_model_inputs`` applies to ``input_ids``.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from verl.utils.ulysses import all_gather_tensor, slice_input_tensor

if TYPE_CHECKING:
    import torch.nn as nn


__all__ = ["RouterReplayAction", "VeOmniRouterReplay"]


class RouterReplayAction(Enum):
    DISABLED = "disabled"
    RECORD = "record"
    REPLAY = "replay"


class VeOmniRouterReplay:
    """Router replay controller for VeOmni (FSDP2 + optional Ulysses SP).

    Single self-contained class: state machine, layer discovery, RECORD /
    REPLAY forward dispatch, cross-rank aggregation, and the VeOmni-side
    install/uninstall hookup all live here. No abstract base — see the
    module docstring for why VeOmni and Megatron/mindspeed intentionally
    keep separate RR implementations.
    """

    # ------------------------------------------------------------ lifecycle

    def __init__(self, sp_group: dist.ProcessGroup | None = None) -> None:
        self._sp_group = sp_group
        self._action: RouterReplayAction = RouterReplayAction.DISABLED
        # id(router_module) -> position. Populated lazily on first sight
        # of each router; stable across the lifetime of the controller
        # (FSDP2 / LoRA wrappers don't mutate module identity per forward).
        # Re-discovered between RECORD and REPLAY phases (cleared by
        # ``begin_record`` / ``begin_replay``); this is just an optimization
        # bookkeeping reset — the same model produces the same id table.
        self._id_to_pos: dict[int, int] = {}
        # RECORD: per-layer-position list of [local_nnz, topk] tensors, one
        # per micro-batch. Inner list length == num_micro_batches at collect
        # time; outer list grows as new routers fire on the *first*
        # micro-batch. Recompute-in-backward (which fires the same routers
        # again, in any order under per-layer checkpointing) is detected via
        # ``len(_recorded[pos]) == _mb_index + 1`` and skipped.
        self._recorded: list[list[torch.Tensor]] = []
        # REPLAY: positional list of target tensors for the *current*
        # micro-batch. Re-built per micro-batch by ``set_microbatch_targets``.
        self._targets: list[torch.Tensor] = []
        # REPLAY only: optional per-token mask of shape ``[local_nnz]``,
        # bool, True where the recorded targets are valid (substitute) and
        # False where they are placeholders (fall through to native
        # routing). Populated by ``set_microbatch_targets``. R3 needs this
        # because the rollout backend only captures response-token routing
        # and writes zeros for prompt tokens; substituting all tokens with
        # those zeros sends every prompt token's topk slots to expert 0,
        # which corrupts the EP all-to-all token distribution.
        self._replay_mask: torch.Tensor | None = None
        # RECORD only: which micro-batch is currently being recorded. Bumped
        # by ``advance_record_microbatch`` between micro-batches; used both
        # to detect recompute (slot already filled for this mb) and to
        # validate the per-layer slot list is dense (no skipped mbs).
        self._mb_index: int = 0
        # Env-gated shape sanity check.
        self._debug: bool = os.environ.get("VERL_ROUTER_REPLAY_DEBUG") == "1"
        self._installed: bool = False

    @property
    def action(self) -> RouterReplayAction:
        return self._action

    @property
    def num_layers(self) -> int:
        """Number of MoE layers discovered so far. RECORD: established by the
        first full forward (``len(_recorded)``). REPLAY: known directly from
        ``set_microbatch_targets`` input length (``len(_targets)``)."""
        return max(len(self._recorded), len(self._targets))

    # -------------------------------------------------------- install/uninstall

    def install(self, model: nn.Module) -> None:
        """Register this controller with VeOmni's global ``set_active_replay`` slot.

        After returning, every MoE router forward in ``model`` should reach
        :meth:`on_router_forward` via ``maybe_replay_indices``.
        """
        try:
            from veomni.utils.moe_router_replay import set_active_replay, validate_model_for_replay  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "router_replay.mode != 'disabled' requires a VeOmni build that "
                "exposes `veomni.utils.moe_router_replay.set_active_replay`. "
                "Either upgrade VeOmni or set router_replay.mode='disabled'."
            ) from e
        # Fail fast if this model family has not been wired for replay
        # (would otherwise surface as a cryptic mid-forward error from the
        # controller's collect/set_microbatch_targets path).
        validate_model_for_replay(model)
        set_active_replay(self)
        self._installed = True

    def uninstall(self) -> None:
        """Reverse :meth:`install`. Idempotent."""
        if not self._installed:
            return
        try:
            from veomni.utils.moe_router_replay import set_active_replay  # type: ignore

            set_active_replay(None)
        except ImportError:
            pass
        self._installed = False

    # ----------------------------------------------------- router-side entry

    def on_router_forward(
        self,
        module: nn.Module,
        routing_scores: torch.Tensor,
        top_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Called from each MoE router forward via VeOmni's hook surface.

        Indices-only: records ``top_indices`` in RECORD mode or returns
        substituted target indices in REPLAY mode. All model-specific
        post-topk weight math (gather, renorm, scaling, dtype cast) lives
        in the per-family patched ``SparseMoeBlock.forward``, not here —
        that keeps the controller model-agnostic across MoE families
        (softmax/sigmoid gating, group topk, expert bias, scaling factors,
        etc.). ``routing_scores`` is accepted for optional debug inspection
        but NOT used to derive weights.

        Position assignment is keyed on ``id(module)`` so backward recompute
        under activation checkpointing (which fires routers again, possibly
        in non-sequential order under per-layer checkpoint segments) lands
        on the same position as the original forward.
        """
        mid = id(module)
        if mid in self._id_to_pos:
            pos = self._id_to_pos[mid]
            new_layer = False
        else:
            pos = len(self._id_to_pos)
            self._id_to_pos[mid] = pos
            new_layer = True

        if self._debug:
            # Cheap shape sanity check — cross-family safe (no weight math).
            # Env-gated; off by default.
            if routing_scores.dim() != 2 or top_indices.dim() != 2:
                raise AssertionError(
                    f"router_replay: expected 2D tensors, got routing_scores "
                    f"{tuple(routing_scores.shape)} and top_indices "
                    f"{tuple(top_indices.shape)}."
                )
            if routing_scores.shape[0] != top_indices.shape[0]:
                raise AssertionError(
                    f"router_replay: routing_scores / top_indices row count "
                    f"mismatch: {routing_scores.shape[0]} vs {top_indices.shape[0]}."
                )

        if self._action is RouterReplayAction.RECORD:
            if new_layer:
                # First time we've seen this router — grow the slot list.
                # Must be on the first micro-batch; later mbs would see the
                # router already mapped.
                self._recorded.append([])
            slot = self._recorded[pos]
            # Recompute detection: this layer has already been captured for
            # the current mb. Skip the append (re-recording would duplicate
            # an already-deterministic value). Snapshot was via
            # ``.detach().clone()`` so the originally captured tensor is
            # independent of the autograd graph that produced it.
            if len(slot) == self._mb_index + 1:
                return top_indices
            if len(slot) != self._mb_index:
                # Should never happen — would indicate skipped or extra
                # micro-batches earlier in the step.
                raise RuntimeError(
                    f"router_replay RECORD invariant violated at layer pos={pos}: "
                    f"slot has {len(slot)} entries, expected {self._mb_index} "
                    f"before appending mb {self._mb_index}. Possible cause: "
                    "router fired in some micro-batches but not others."
                )
            slot.append(top_indices.detach().clone())
            return top_indices

        if self._action is RouterReplayAction.REPLAY:
            # Strict: every layer position must have a target. There is no
            # silent fallback — a missing target indicates a real plumbing
            # bug (routed_experts not propagated, layer count mismatch
            # between RECORD and REPLAY models, or the engine forgot to call
            # set_microbatch_targets before this forward).
            if pos >= len(self._targets):
                raise RuntimeError(
                    f"router_replay REPLAY: layer pos={pos} has no target "
                    f"(only {len(self._targets)} targets set for this "
                    "micro-batch). Likely cause: model has more MoE layers "
                    "than the recorded routed_experts tensor describes, or "
                    "set_microbatch_targets was not called before forward."
                )
            target = self._targets[pos]
            if self._replay_mask is None:
                substituted = target
            else:
                # Per-token gated substitution: where the mask is True the
                # recorded target is valid (substitute); where False the
                # target is a placeholder and we must fall through to native
                # routing. R3 needs this to skip prompt tokens that the
                # rollout backend wrote zeros for.
                mask = self._replay_mask
                if mask.shape[0] != top_indices.shape[0]:
                    raise RuntimeError(
                        f"router_replay REPLAY: replay_mask has {mask.shape[0]} rows "
                        f"but top_indices has {top_indices.shape[0]}. The mask must "
                        "be sliced with the same Ulysses SP rule as the targets."
                    )
                substituted = torch.where(mask.unsqueeze(-1), target, top_indices)

            # Defensive duplicate-detection.
            #
            # VeOmni's MoE expert dispatch (``permute()`` in
            # ``veomni/distributed/moe/moe_utils.py``) builds the permuted
            # tensor via ``routing_map.bool().masked_select(...)``, which
            # collapses duplicate top-k slots within one token to a single
            # entry. ``input_splits`` keeps counting every slot, so the two
            # diverge whenever ANY token has duplicate top-k experts and the
            # EP all-to-all crashes with
            # ``RuntimeError: Split sizes doesn't match total dim 0 size``.
            #
            # Recorded targets can contain such duplicate rows when the
            # rollout backend writes zero placeholders (e.g. left/right pad,
            # or capture regions that don't match the response_mask). The
            # mask filters most of these, but the exact contract differs
            # across backends — fall back to native routing for any row whose
            # substituted top-k contains a duplicate. Native indices are
            # always distinct top-k choices, so this is correct regardless
            # of what the rollout produces.
            sorted_sub, _ = substituted.sort(dim=-1)
            has_duplicate = (sorted_sub[:, 1:] == sorted_sub[:, :-1]).any(dim=-1)
            return torch.where(has_duplicate.unsqueeze(-1), top_indices, substituted)

        return top_indices

    # --------------------------------------------------- engine-side drivers

    def begin_record(self) -> None:
        """Enter RECORD mode. Must be called before the micro-batch loop."""
        self._action = RouterReplayAction.RECORD
        self._recorded = []
        self._targets = []
        self._replay_mask = None
        self._id_to_pos = {}
        self._mb_index = 0

    def begin_replay(self) -> None:
        """Enter REPLAY mode. Must be called before the micro-batch loop."""
        self._action = RouterReplayAction.REPLAY
        self._targets = []
        self._replay_mask = None
        self._id_to_pos = {}
        # _mb_index is unused in REPLAY (recompute is detected via id-keyed
        # lookup hitting an already-mapped module, not via mb counters).
        self._mb_index = 0

    def advance_record_microbatch(self) -> None:
        """Mark the start of a new RECORD micro-batch.

        Call this in the engine driver immediately after the previous
        micro-batch's forward+backward returns and before the next one
        starts. Bumps ``_mb_index`` so :meth:`on_router_forward` knows which
        slot to fill on the next router fire. Recompute-in-backward (which
        fires within the *same* micro-batch's backward) does NOT call this —
        it's detected via id-keyed lookup hitting a slot whose length already
        equals ``_mb_index + 1``.
        """
        if self._action is not RouterReplayAction.RECORD:
            raise RuntimeError(f"advance_record_microbatch requires RECORD action, got {self._action}")
        self._mb_index += 1

    def set_microbatch_targets(
        self,
        per_layer_targets: list[torch.Tensor],
        replay_mask: torch.Tensor | None = None,
    ) -> None:
        """Load per-layer target indices for the upcoming micro-batch forward.

        ``per_layer_targets[i]`` is ``[local_nnz, topk]`` int64 on device,
        ordered by layer position (matches the order in which routers fire
        during forward — established when each router gets its position
        assigned in :meth:`on_router_forward`). Strict: REPLAY mode must
        already be active. ``L`` is taken from the input list length — no
        prior id-mapping discovery is required, which is what unblocks R3
        step 1 where REPLAY runs before any RECORD has populated the table.

        ``replay_mask`` (optional, ``[local_nnz]`` bool): per-token gate.
        Where True, substitute with the recorded target. Where False, fall
        through to the native router output. Required for R3, where the
        rollout backend captures only response-token routing and writes
        zeros for prompt tokens — without the mask, those zeros would be
        substituted, sending every prompt token's topk slots to expert 0
        and corrupting the EP all-to-all token distribution.

        The mask must be in the rmpad ``[local_nnz]`` layout (same SP
        slice rule as ``per_layer_targets``). The engine driver builds it
        from ``response_mask`` (strided ``(bs, max_response_len)``) plus
        ``input_ids.offsets()`` via per-sample length arithmetic, then
        runs it through :meth:`slice_microbatch_replay_mask`. Callers
        should not pass the strided ``response_mask`` / ``loss_mask``
        directly — they have neither the right shape nor a valid
        ``.values()`` for a strided layout.

        R2 callers should pass ``None`` (uniform substitution): R2
        RECORD captures the actor's full-sequence routing (prompt +
        response), so applying a response-only gate would leak prompt-
        token routing divergence into the bit-equal forward guarantee.
        """
        if self._action is not RouterReplayAction.REPLAY:
            raise RuntimeError(f"set_microbatch_targets requires REPLAY action, got {self._action}")
        self._targets = list(per_layer_targets)
        self._replay_mask = replay_mask.bool() if replay_mask is not None else None

    def clear(self) -> None:
        """Reset the state machine between steps."""
        self._action = RouterReplayAction.DISABLED
        self._recorded = []
        self._targets = []
        self._replay_mask = None
        self._id_to_pos = {}
        self._mb_index = 0

    # ---------------------------------------------------- cross-rank gather

    def _all_gather_recorded(self, local: torch.Tensor) -> torch.Tensor:
        """All-gather a ``[local_nnz, L, topk]`` tensor along Ulysses SP.

        Returns ``[nnz_padded, L, topk]`` where
        ``nnz_padded = local_nnz * sp_size``. The caller trims the ulysses
        pad suffix.
        """
        if self._sp_group is None or dist.get_world_size(self._sp_group) == 1:
            return local
        return all_gather_tensor(local.contiguous(), group=self._sp_group)

    def collect_recorded(
        self,
        pad_size_per_mb: list[int],
        num_micro_batches: int,
    ) -> list[torch.Tensor]:
        """Aggregate recorded indices across ranks and unpack per micro-batch.

        For each micro-batch:
          * Stack per-layer-position [local_nnz, topk] into [local_nnz, L, topk]
          * All-gather along Ulysses SP to restore full nnz+pad
          * Trim the ulysses pad suffix

        Returns a list of length ``num_micro_batches``, each element a
        ``[total_nnz_mb, L, topk]`` int tensor. The caller (engine) is
        responsible for (a) reordering micro-batches back to batch order
        using the indices returned by ``prepare_micro_batches``, and (b)
        converting to the nested/jagged layout expected by the trainer.
        """
        if self._action is not RouterReplayAction.RECORD:
            raise RuntimeError(f"collect_recorded requires RECORD action, got {self._action}")
        if not self._recorded:
            raise RuntimeError("collect_recorded called before any router fired.")
        if len(pad_size_per_mb) != num_micro_batches:
            raise ValueError(f"pad_size_per_mb length {len(pad_size_per_mb)} != {num_micro_batches}")
        # Sanity: every recorded layer should have the same number of entries.
        for pos, slot in enumerate(self._recorded):
            if len(slot) != num_micro_batches:
                raise RuntimeError(
                    f"Router layer pos={pos} recorded {len(slot)} micro-batches, "
                    f"expected {num_micro_batches}. Possible causes: "
                    "router skipped on some micro-batches, or forward failed mid-step."
                )

        per_mb: list[torch.Tensor] = []
        for mb in range(num_micro_batches):
            per_layer = [slot[mb] for slot in self._recorded]
            local = torch.stack(per_layer, dim=1)  # [local_nnz, L, topk]
            gathered = self._all_gather_recorded(local)
            pad = pad_size_per_mb[mb]
            if pad > 0:
                gathered = gathered[:-pad]
            per_mb.append(gathered)
        return per_mb

    # ---------------------------------------------------- replay input prep

    def slice_microbatch_replay_targets(self, batch_routed_experts: torch.Tensor) -> list[torch.Tensor]:
        """Prepare per-layer replay targets from a micro-batch's routed_experts.

        Reuses :func:`verl.utils.ulysses.slice_input_tensor` so the pad+split
        rule matches the one ``super().prepare_model_inputs`` already applies
        to ``input_ids``. The SP rank/world_size are derived from
        ``self._sp_group`` internally — we never dig into ``parallel_state``
        here.

        Args:
            batch_routed_experts: ``[mb_nnz, L, topk]`` int tensor, flattened
                across the micro-batch (the ``.values()`` of the nested
                jagged ``routed_experts``).

        Returns:
            List of length ``L``, each ``[mb_nnz_local, topk]`` int64 tensor,
            ready to feed :meth:`set_microbatch_targets`.
        """
        if batch_routed_experts.dim() != 3:
            raise ValueError(f"routed_experts must be [mb_nnz, L, topk], got {batch_routed_experts.shape}")
        idx = batch_routed_experts.to(torch.int64)
        if self._sp_group is not None and dist.get_world_size(self._sp_group) > 1:
            idx = slice_input_tensor(idx, dim=0, padding=True, group=self._sp_group)
        return list(idx.unbind(dim=1))

    def slice_microbatch_replay_mask(self, batch_mask: torch.Tensor) -> torch.Tensor:
        """Prepare a per-token replay mask using the same pad+slice rule as
        :meth:`slice_microbatch_replay_targets`.

        Used to feed the optional ``replay_mask`` of
        :meth:`set_microbatch_targets`. The input is a flat
        ``[mb_nnz]`` bool/int tensor in the same rmpad layout as
        ``input_ids.values()`` — it is NOT the engine's
        ``response_mask`` directly (which is a strided
        ``(bs, max_response_len)`` tensor and has the wrong shape).
        The engine driver constructs the flat layout via per-sample
        prompt/response length arithmetic (see
        ``VeOmniEngineWithLMHead._maybe_push_router_replay_state``)
        before calling this helper.

        Padding values (added by ``slice_input_tensor`` to make the tensor
        SP-divisible) are filled with zero, matching the "no recorded data"
        semantics — pad rows shouldn't be substituted regardless.
        """
        if batch_mask.dim() != 1:
            raise ValueError(f"replay_mask must be 1-D [mb_nnz], got {batch_mask.shape}")
        m = batch_mask.to(torch.int64)
        if self._sp_group is not None and dist.get_world_size(self._sp_group) > 1:
            m = slice_input_tensor(m, dim=0, padding=True, group=self._sp_group)
        return m.bool()

    # --------------------------------------------------------- debug helpers

    def assert_layer_count(self, expected: int) -> None:
        """Assert the discovered layer count matches the model config."""
        if self.num_layers != expected:
            raise AssertionError(
                f"router_replay discovered {self.num_layers} MoE layers, "
                f"model config says {expected}. Layer discovery is broken."
            )
