# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Accumulate independently finalized group gradients without re-reducing old gradients."""

import math
from collections.abc import Mapping

import torch


class GroupGradientAccumulator:
    """Weighted sum of group means, with an exact-once barrier before an optimizer step.

    Megatron's schedule finalizes (and sometimes rescales) its gradient buffers on
    every call. Keep the running sum outside those buffers, zero the engine before
    each group, and restore the sum only after the entire synchronous batch arrives.
    Weights are valid response-token counts for token-mean, or valid sequence counts
    for sequence-mean. This uses one additional FP32 gradient buffer.

    The running sum may live on a different device than the gradients (typically
    host memory). On a full-parameter 4B/TP2 run the extra FP32 buffer is 8 GiB
    of GPU that must sit next to main_grad for the entire backward, which leaves
    no room for activations on a 48 GB card. Accumulating on CPU trades that
    memory for one D2H copy of each group's gradient plus one H2D at step time,
    while keeping the sum itself exactly FP32.
    """

    # Pinned staging buffer used to move one gradient chunk at a time when the
    # sum is host-resident. Pageable D2H runs at ~2 GB/s on this host; pinned
    # memory reaches ~10 GB/s, which is the difference between 20 s and 3 s per
    # four-group step. The buffer is bounded so host pinning stays cheap.
    STAGING_BYTES = 256 << 20

    def __init__(
        self,
        group_ids: list[str],
        sums: dict[str, torch.Tensor] | None = None,
        staging: torch.Tensor | None = None,
    ):
        if not group_ids or len(set(group_ids)) != len(group_ids):
            raise ValueError("Expected a nonempty list of unique group IDs")
        self.expected = frozenset(group_ids)
        self.seen: set[str] = set()
        # A caller may pass the same buffers back on every batch. Reusing one
        # persistent FP32 buffer avoids a fresh ~4 bytes/owned-parameter
        # allocation (7.5 GiB for 4B/TP2) in an already tight memory budget.
        self.sums: dict[str, torch.Tensor] = sums if sums is not None else {}
        for value in self.sums.values():
            value.zero_()
        self.weight = 0.0
        self.restored = False
        # A caller may also pass a persistent pinned staging buffer, so the
        # (expensive) pin/unpin does not repeat on every batch.
        self._staging: torch.Tensor | None = staging

    def check_group(self, group_id: str, weight: float) -> None:
        """Reject duplicate, foreign, or invalid groups before running backward."""
        if self.restored or group_id not in self.expected or group_id in self.seen:
            raise ValueError(f"Unexpected or duplicate group: {group_id}")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Group weight must be finite and positive, got {weight}")

    @torch.no_grad()
    def add(self, group_id: str, gradients: Mapping[str, torch.Tensor], weight: float) -> None:
        """Copy a group's already synchronized gradients into the running sum."""
        self.check_group(group_id, weight)
        if not gradients:
            raise ValueError("No Megatron gradient buffers were found")
        if self.sums and gradients.keys() != self.sums.keys():
            raise ValueError("Gradient buffer layout changed within a synchronous batch")
        for name, grad in gradients.items():
            if grad.dtype != torch.float32:
                raise ValueError("Group accumulation requires FP32 Megatron gradient buffers")
            if name in self.sums and self.sums[name].shape != grad.shape:
                raise ValueError(f"Gradient buffer changed within a synchronous batch: {name}")
        for name, grad in gradients.items():
            if name not in self.sums:
                self.sums[name] = torch.zeros_like(grad)
            target = self.sums[name]
            if grad.device == target.device:
                target.add_(grad, alpha=weight)
            else:
                # The sum may be host-resident (see class docstring); move this
                # group's gradient chunk by chunk through pinned memory.
                self._accumulate_host(target, grad, weight)
        self.weight += weight
        self.seen.add(group_id)

    def _staging_for(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a pinned 1-D staging buffer matching ``tensor``'s dtype."""
        if self._staging is None or self._staging.dtype != tensor.dtype:
            elements = max(1, self.STAGING_BYTES // tensor.element_size())
            self._staging = torch.empty(elements, dtype=tensor.dtype, pin_memory=True)
        return self._staging

    def _copy_chunked(self, destination: torch.Tensor, source: torch.Tensor, *, accumulate: bool, weight: float):
        """Move ``source`` into ``destination`` across devices through pinned memory."""
        staging = self._staging_for(source)
        source_flat = source.reshape(-1)
        destination_flat = destination.reshape(-1)
        step = staging.numel()
        for offset in range(0, destination_flat.numel(), step):
            count = min(step, destination_flat.numel() - offset)
            staging[:count].copy_(source_flat[offset : offset + count])
            if accumulate:
                destination_flat[offset : offset + count].add_(staging[:count], alpha=weight)
            else:
                destination_flat[offset : offset + count].copy_(staging[:count])

    def _accumulate_host(self, target: torch.Tensor, grad: torch.Tensor, weight: float) -> None:
        self._copy_chunked(target, grad, accumulate=True, weight=weight)

    @torch.no_grad()
    def restore(self, gradients: Mapping[str, torch.Tensor]) -> None:
        """Restore the global mean exactly once, before clipping and optimizer.step."""
        if self.restored or self.seen != self.expected:
            raise RuntimeError(f"Cannot step: received {len(self.seen)}/{len(self.expected)} groups")
        if gradients.keys() != self.sums.keys():
            raise ValueError("Gradient buffer layout changed before optimizer step")
        for name, grad in gradients.items():
            if grad.shape != self.sums[name].shape:
                raise ValueError(f"Gradient buffer changed before optimizer step: {name}")
        for name, grad in gradients.items():
            source = self.sums[name]
            if grad.device == source.device:
                grad.copy_(source)
            else:
                # One H2D per step, streamed through the pinned staging buffer.
                self._copy_chunked(grad, source, accumulate=False, weight=0.0)
            grad.div_(self.weight)
        self.restored = True
        # Keep self.sums: a caller may own these buffers and reuse them for the
        # next batch. The next accumulator zeroes them in __init__; clearing the
        # dict here would drop (and force reallocation of) a persistent buffer.


def megatron_gradient_buffers(modules) -> dict[str, torch.Tensor]:
    """Return native DDP buffers, including expert buffers and distributed-optimizer shards.

    Accumulating the physical grad_data storage preserves reduce-scatter results
    even when a distributed optimizer owns only slices of individual parameters.
    Megatron FSDP / precision-aware optimizer layouts are deliberately unsupported.
    """
    gradients = {}
    seen = set()
    for chunk_index, module in enumerate(modules):
        for kind in ("buffers", "expert_parallel_buffers"):
            for buffer_index, buffer in enumerate(getattr(module, kind, ())):
                grad = buffer.grad_data
                if id(grad) not in seen:
                    gradients[f"{chunk_index}/{kind}/{buffer_index}"] = grad
                    seen.add(id(grad))
    if not gradients:
        raise RuntimeError("group_delivery requires native Megatron DDP gradient buffers")
    return gradients
