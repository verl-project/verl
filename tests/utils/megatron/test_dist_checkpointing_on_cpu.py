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

"""Check the save API boundary without requiring Megatron's GPU dependencies."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _load_checkpointing(save):
    """Load the production wrapper against a minimal MCore module tree and supplied save API."""
    names = [
        "megatron",
        "megatron.core",
        "megatron.core.dist_checkpointing",
        "megatron.core.dist_checkpointing.strategies",
        "megatron.core.dist_checkpointing.strategies.fully_parallel",
        "megatron.core.dist_checkpointing.strategies.torch",
    ]
    modules = {name: ModuleType(name) for name in names}
    modules["megatron"].core = modules["megatron.core"]
    modules["megatron.core"].__version__ = "0.19.2"
    modules["megatron.core"].dist_checkpointing = modules["megatron.core.dist_checkpointing"]
    modules["megatron.core"].mpu = SimpleNamespace(get_data_parallel_group=MagicMock(return_value="dp_group"))
    modules["megatron.core.dist_checkpointing"].save = save
    for name in ("FullyParallelLoadStrategyWrapper", "FullyParallelSaveStrategyWrapper"):
        setattr(modules[names[-2]], name, MagicMock())
    for name in ("TorchDistLoadShardedStrategy", "TorchDistSaveShardedStrategy"):
        setattr(modules[names[-1]], name, MagicMock())

    path = Path(__file__).resolve().parents[3] / "verl/utils/megatron/dist_checkpointing.py"
    spec = importlib.util.spec_from_file_location("_checkpointing_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("async_save", [False, True])
def test_save_without_optional_nvrx_dependency(async_save):
    """Select native MCore for async requests while preserving the synchronous API default."""
    request = object()
    calls = []

    # Match MCore's default: async saves select NVRx unless explicitly overridden.
    def save(state, path, *, async_strategy="nvrx", **kwargs):
        """Model an MCore save API whose default async backend needs an optional dependency."""
        calls.append((state, path, async_strategy, kwargs))
        if kwargs["async_sharded_save"]:
            if async_strategy == "nvrx":
                raise ModuleNotFoundError("nvidia_resiliency_ext")
            return request
        return None

    checkpointing = _load_checkpointing(save)
    state = {"weights": object()}
    metadata = {"version": 2}
    result = checkpointing.save_dist_checkpointing(
        state, "checkpoint", async_save=async_save, content_metadata=metadata
    )

    assert result is (request if async_save else None)
    assert calls[0][:2] == (state, "checkpoint")
    assert calls[0][2] == ("mcore" if async_save else "nvrx")
    assert calls[0][3]["content_metadata"] is metadata
    assert calls[0][3]["validate_access_integrity"] is True


@pytest.mark.parametrize("async_save", [False, True])
def test_legacy_save_api_without_backend_selector(async_save):
    """Keep async and synchronous saves compatible with older MCore signatures."""
    request = object()

    def save(state, path, *, sharded_strategy, async_sharded_save, validate_access_integrity, content_metadata=None):
        """Accept only the legacy save keywords and return an async request when requested."""
        assert async_sharded_save is async_save
        return request if async_sharded_save else None

    checkpointing = _load_checkpointing(save)
    result = checkpointing.save_dist_checkpointing({}, "checkpoint", async_save=async_save)
    assert result is (request if async_save else None)
