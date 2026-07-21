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
"""Checkpoint-engine adapter for the ``remote_backend`` weight-sync path.

Registered under name ``"remote_backend"`` so
:class:`verl.checkpoint_engine.base.CheckpointEngineManager` can be
constructed against a plugin-driven backend that owns its own weight
sync (typically CUDA IPC between colocated processes). The trainer
short-circuits ``update_weights`` for this backend in
:meth:`CheckpointEngineManager.update_weights` and never invokes
``send_weights`` / ``receive_weights`` on this class -- the methods are
provided to satisfy the ABC only.

Contrast with the ``"naive"`` colocated backend: naive is used when
verl's own actor worker group owns the model and hands per-tensor
params directly to a colocated rollout server, while ``"remote_backend"``
delegates the whole weight-sync (and typically training/sampling too)
to the plugin's remote client.
"""

from __future__ import annotations

from collections.abc import Generator

import torch

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry


@CheckpointEngineRegistry.register("remote_backend")
class RemoteBackendCheckpointEngine(CheckpointEngine):
    """No-op stub; the plugin performs weight sync out-of-band."""

    def __init__(self, bucket_size: int = 0, is_master: bool = False, **_ignored) -> None:
        self.bucket_size = bucket_size
        self.is_master = is_master

    def prepare(self):
        return None

    def init_process_group(self, **kwargs):
        return None

    def finalize(self):
        return None

    @classmethod
    def build_topology(cls, *args, **kwargs):
        return {}, {}

    def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        raise NotImplementedError(
            "RemoteBackendCheckpointEngine.send_weights is never invoked: "
            "CheckpointEngineManager.update_weights short-circuits for backend='remote_backend'."
        )

    def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        raise NotImplementedError(
            "RemoteBackendCheckpointEngine.receive_weights is never invoked: "
            "CheckpointEngineManager.update_weights short-circuits for backend='remote_backend'."
        )
