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
"""Adapter from verl's ``(name, tensor)`` generator API to delta flushes.

The verl rollout API speaks "generator of (HF name, tensor)". This module
wraps that generator with the delta machinery and yields one
:class:`DeltaFlush` per bucket boundary -- the checkpoint engine broadcasts
those flushes over NCCL directly.

The first call has to seed the snapshot (no flushes emitted, no engine RPCs);
subsequent calls produce one or more sparse flushes carrying only the
positions and values that actually changed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator, Iterable

import torch

from .encode import DeltaParam

logger = logging.getLogger(__name__)


@dataclass
class DeltaFlush:
    """One ready-to-dispatch flush.

    * ``positions_cpu`` is a uint8 positions blob. The base encode path
      produces it on the host; the sharded engines keep it on the GPU (the wire
      broadcasts from the GPU, so a host round-trip would be pure overhead).
    * ``values_gpu`` stays on the GPU until the checkpoint engine broadcasts it
      over NCCL.
    * ``params`` carries the per-parameter manifest the receiver needs to
      decode the blob (sent alongside the data over the zmq side-channel).
    """

    encoding: DeltaEncodingName
    params: list[DeltaParam]
    positions_cpu: torch.Tensor
    values_gpu: torch.Tensor
    checksum: int

    @property
    def nnz(self) -> int:
        return self.values_gpu.numel()

    @property
    def wire_bytes(self) -> int:
        return (
            self.positions_cpu.numel()
            + self.values_gpu.numel() * self.values_gpu.element_size()
        )
