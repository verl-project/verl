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

from __future__ import annotations

from typing import Any

from verl.workers.config import RolloutConfig


def _sglang_engine_kwargs(config: RolloutConfig) -> dict[str, Any]:
    engine_kwargs = getattr(config, "engine_kwargs", None) or {}
    if not isinstance(engine_kwargs, dict):
        engine_kwargs = dict(engine_kwargs)
    sglang_kwargs = engine_kwargs.get("sglang", {}) or {}
    if not isinstance(sglang_kwargs, dict):
        sglang_kwargs = dict(sglang_kwargs)
    return sglang_kwargs


def should_enable_p2p_weight_transfer_bootstrap(config: RolloutConfig) -> bool:
    """Return whether rollout should start EngineInfoBootstrapServer for P2P RDMA."""
    checkpoint_engine = getattr(config, "checkpoint_engine", None)
    backend = getattr(checkpoint_engine, "backend", None)
    return backend == "p2p"


def sanitize_sglang_engine_kwargs_for_p2p(config: RolloutConfig) -> dict[str, Any]:
    """Return a copy of SGLang engine kwargs with P2P bootstrap flags owned by Verl."""
    kwargs = dict(_sglang_engine_kwargs(config))
    # Verl auto-allocates bootstrap ports and enables P2P seeding when backend=p2p.
    kwargs.pop("engine_info_bootstrap_port", None)
    kwargs.pop("remote_instance_weight_loader_start_seed_via_transfer_engine", None)
    return kwargs
