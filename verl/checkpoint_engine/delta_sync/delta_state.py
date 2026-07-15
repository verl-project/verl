# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Pinned-CPU snapshot of broadcast weights for delta sync.

``DeltaState`` holds one pinned-host copy per HF tensor name that has been
broadcast. Each sync prefetches the relevant snapshot tiles to the GPU on a
side stream (overlapped with the previous chunk's compute), the diff/encode
runs on the default stream, and the updated values are copied back to host
on another side stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ParamDiff:
    """One per-parameter compute output.

    ``values`` is a view of the current GPU tensor (no copy); ``mask`` is a
    same-shape bool tensor marking the positions whose bytes differ from the
    snapshot.
    """

    name: str
    values: torch.Tensor
    mask: torch.Tensor


class DeltaState:
    """Pinned-CPU snapshot store plus side streams for D2H / H2D pipelining.

    Lifecycle:

    * The first sync seeds the snapshot from current model weights and skips
      the engine RPC -- the receiver is assumed to have loaded the same HF
      checkpoint at init.
    * Each subsequent sync: ``prefetch_snapshot`` for chunk N+1 is issued on
      the H2D stream before chunk N's compute; ``compute_diffs`` waits on the
      prefetch event; ``update_snapshot_async`` writes the just-sent values
      back to the pinned host buffer on the D2H stream.
    * ``flush_snapshot`` blocks until all enqueued D2H copies have landed.
    """

    def __init__(self) -> None:
        self.snapshot: dict[str, torch.Tensor] = {}
        self.d2h_stream: torch.cuda.Stream | None = None
        self.h2d_stream: torch.cuda.Stream | None = None
        self._snapshot_dirty = False
        # Pipelining via side streams and pinned host memory is only worthwhile
        # (and only available) when CUDA is. The CPU-only path keeps the same
        # data flow but degrades to plain blocking copies, which keeps unit
        # tests runnable on CPU CI.
        self._cuda = torch.cuda.is_available()

    @property
    def seeded(self) -> bool:
        return len(self.snapshot) > 0

    def seed(self, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        """Populate the snapshot from the current model state.

        Costs roughly one full D2H of every parameter; expected to be called
        exactly once, at the first ``update_weights`` invocation.
        """
        for name, tensor in named_tensors:
            self._allocate(name, tensor)
        for name, tensor in named_tensors:
            self.snapshot[name].copy_(tensor.detach(), non_blocking=False)
        self._snapshot_dirty = False

    def _allocate(self, name: str, tensor: torch.Tensor) -> None:
        if name in self.snapshot:
            return
        # pin_memory requires CUDA; fall back to a regular host buffer in CPU
        # CI / smoke tests.
        self.snapshot[name] = torch.empty_like(
            tensor, device=torch.device("cpu"), pin_memory=self._cuda
        )

    def prefetch_snapshot(
        self, named_tensors: list[tuple[str, torch.Tensor]]
    ) -> tuple[list[torch.Tensor], "torch.cuda.Event | None"]:
        """Start an async H2D copy of snapshot tiles matching ``named_tensors``.

        Returns the GPU-resident previous-step tensors and a CUDA event that
        ``compute_diffs`` waits on before reading them. On CPU (no CUDA) the
        copy is blocking and the event is ``None``.
        """
        if self._cuda:
            if self.h2d_stream is None:
                self.h2d_stream = torch.cuda.Stream()
            prev_gpu: list[torch.Tensor] = []
            with torch.cuda.stream(self.h2d_stream):
                for name, tensor in named_tensors:
                    if name not in self.snapshot:
                        raise KeyError(
                            f"missing snapshot for {name!r}; the first update_weights call "
                            "must seed the snapshot before any chunk is prefetched"
                        )
                    prev_gpu.append(
                        self.snapshot[name].to(
                            device=tensor.device, non_blocking=True
                        )
                    )
                event = self.h2d_stream.record_event()
            return prev_gpu, event
        # CPU fallback: snapshot already lives on host, no async transfer needed.
        prev_host: list[torch.Tensor] = []
        for name, tensor in named_tensors:
            if name not in self.snapshot:
                raise KeyError(
                    f"missing snapshot for {name!r}; the first update_weights call "
                    "must seed the snapshot before any chunk is prefetched"
                )
            prev_host.append(self.snapshot[name].to(device=tensor.device))
        return prev_host, None

    def compute_diffs(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        prefetched: tuple[list[torch.Tensor], "torch.cuda.Event | None"],
    ) -> list[ParamDiff]:
        """Wait for the prefetched H2D copy, then per-param bytewise diff."""
        from .encode import bytewise_diff_mask  # local import avoids cycle

        prev_gpu, event = prefetched
        if event is not None:
            event.wait()
        return [
            ParamDiff(name=name, values=current, mask=bytewise_diff_mask(current, prev))
            for (name, current), prev in zip(named_tensors, prev_gpu, strict=True)
        ]

    def update_snapshot_async(
        self, named_tensors: list[tuple[str, torch.Tensor]]
    ) -> None:
        """Enqueue a D2H copy of ``named_tensors`` into the pinned snapshot.

        Non-blocking on CUDA; the caller must invoke :meth:`flush_snapshot`
        before the next sync. On CPU the copy is blocking.
        """
        if not self._cuda:
            for name, tensor in named_tensors:
                self._allocate(name, tensor)
                self.snapshot[name].copy_(tensor.detach())
            self._snapshot_dirty = False
            return
        if self.d2h_stream is None:
            self.d2h_stream = torch.cuda.Stream()
        event = torch.cuda.current_stream().record_event()
        with torch.cuda.stream(self.d2h_stream):
            self.d2h_stream.wait_event(event)
            for name, tensor in named_tensors:
                self._allocate(name, tensor)
                self.snapshot[name].copy_(tensor.detach(), non_blocking=True)
        self._snapshot_dirty = True

    def flush_snapshot(self) -> None:
        """Block until every enqueued D2H snapshot copy has landed."""
        if not self._snapshot_dirty:
            return
        if self._cuda:
            if self.d2h_stream is not None:
                self.d2h_stream.synchronize()
            else:
                torch.cuda.synchronize()
        self._snapshot_dirty = False
