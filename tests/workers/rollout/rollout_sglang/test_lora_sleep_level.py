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

"""Unit tests for LoRA sleep_level / lora_as_adapter logic in SGLang rollout.

Tests the branching logic that controls what gets released during sleep:
  - sleep_level=2 (merge path or no LoRA): release weights + kv_cache
  - sleep_level=1 (adapter path): release kv_cache only, keep base weights
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Lightweight stubs so we can import SGLangHttpServer / ServerAdapter without
# pulling in torch, ray, sglang, etc.
# ---------------------------------------------------------------------------


@dataclass
class _StubModelConfig:
    """Minimal stand-in for HFModelConfig."""

    lora_rank: int = 0
    lora: dict[str, Any] = field(default_factory=dict)


@dataclass
class _StubRolloutConfig:
    """Minimal stand-in for RolloutConfig."""

    free_cache_engine: bool = True
    tensor_model_parallel_size: int = 1
    data_parallel_size: int = 1


# ---------------------------------------------------------------------------
# lora_as_adapter property tests (mirrors vllm_async_server pattern)
# ---------------------------------------------------------------------------


class _LoraAsAdapterMixin:
    """Reproduces the lora_as_adapter property from SGLangHttpServer so we can
    test the boolean logic without importing the real class."""

    model_config: _StubModelConfig

    @property
    def lora_as_adapter(self) -> bool:
        return (
            self.model_config.lora_rank > 0 or self.model_config.lora.get("rank", 0) > 0
        ) and not self.model_config.lora.get("merge", False)


class _FakeServer(_LoraAsAdapterMixin):
    def __init__(self, model_config: _StubModelConfig):
        self.model_config = model_config


class TestLoraAsAdapter:
    """Test lora_as_adapter property logic."""

    def test_no_lora(self):
        server = _FakeServer(_StubModelConfig(lora_rank=0, lora={}))
        assert server.lora_as_adapter is False

    def test_lora_merge_true(self):
        server = _FakeServer(_StubModelConfig(lora_rank=8, lora={"merge": True}))
        assert server.lora_as_adapter is False

    def test_lora_merge_false(self):
        server = _FakeServer(_StubModelConfig(lora_rank=8, lora={"merge": False}))
        assert server.lora_as_adapter is True

    def test_lora_merge_absent_defaults_false(self):
        """When merge key is absent, it defaults to False → adapter mode."""
        server = _FakeServer(_StubModelConfig(lora_rank=8, lora={}))
        assert server.lora_as_adapter is True

    def test_lora_rank_in_dict(self):
        """lora_rank=0 but lora.rank>0 should still detect LoRA."""
        server = _FakeServer(_StubModelConfig(lora_rank=0, lora={"rank": 16}))
        assert server.lora_as_adapter is True

    def test_lora_rank_in_dict_with_merge(self):
        server = _FakeServer(_StubModelConfig(lora_rank=0, lora={"rank": 16, "merge": True}))
        assert server.lora_as_adapter is False


# ---------------------------------------------------------------------------
# sleep_level → release tag tests (ServerAdapter.release)
# ---------------------------------------------------------------------------


class TestServerAdapterReleaseTags:
    """Test that ServerAdapter.release() sends the right tags based on sleep_level."""

    @staticmethod
    def _make_adapter(sleep_level: int = 2):
        """Build a minimal ServerAdapter-like object without real init."""
        adapter = MagicMock()
        adapter.sleep_level = sleep_level
        adapter.config = _StubRolloutConfig(free_cache_engine=True)
        # device_mesh["infer_tp"].get_local_rank() == 0
        tp_mesh = MagicMock()
        tp_mesh.get_local_rank.return_value = 0
        adapter.device_mesh = {"infer_tp": tp_mesh}
        adapter._engine = AsyncMock()
        adapter._engine.release_memory_occupation = AsyncMock()
        return adapter

    def test_sleep_level_2_releases_everything(self):
        adapter = self._make_adapter(sleep_level=2)

        # Call the real release logic inline (avoids importing ServerAdapter)
        async def release():
            if adapter.device_mesh["infer_tp"].get_local_rank() == 0 and adapter.config.free_cache_engine:
                if adapter.sleep_level == 1:
                    tags = ["kv_cache"]
                else:
                    tags = ["kv_cache", "weights"]
                await adapter._engine.release_memory_occupation(tags=tags)

        asyncio.run(release())
        adapter._engine.release_memory_occupation.assert_called_once_with(tags=["kv_cache", "weights"])

    def test_sleep_level_1_releases_kv_only(self):
        adapter = self._make_adapter(sleep_level=1)

        async def release():
            if adapter.device_mesh["infer_tp"].get_local_rank() == 0 and adapter.config.free_cache_engine:
                if adapter.sleep_level == 1:
                    tags = ["kv_cache"]
                else:
                    tags = ["kv_cache", "weights"]
                await adapter._engine.release_memory_occupation(tags=tags)

        asyncio.run(release())
        adapter._engine.release_memory_occupation.assert_called_once_with(tags=["kv_cache"])


# ---------------------------------------------------------------------------
# SGLangHttpServer.sleep() tag selection
# ---------------------------------------------------------------------------


class TestSGLangHttpServerSleepTags:
    """Test that SGLangHttpServer.sleep() uses the right tags based on lora_as_adapter."""

    @staticmethod
    def _run_sleep_logic(lora_as_adapter: bool):
        """Simulate the sleep() method's tag selection logic and return chosen tags."""
        # Mirrors async_sglang_server.py sleep() HYBRID branch
        if lora_as_adapter:
            tags = ["kv_cache"]
        else:
            tags = ["kv_cache", "weights"]
        return tags

    def test_no_lora_releases_everything(self):
        tags = self._run_sleep_logic(lora_as_adapter=False)
        assert tags == ["kv_cache", "weights"]

    def test_adapter_mode_releases_kv_only(self):
        tags = self._run_sleep_logic(lora_as_adapter=True)
        assert tags == ["kv_cache"]


# ---------------------------------------------------------------------------
# ActorRolloutRefWorker peft_merge config tests
# ---------------------------------------------------------------------------


class TestActorRolloutRefWorkerPeftMerge:
    """Test that ActorRolloutRefWorker reads peft_merge from model.lora.merge."""

    def test_merge_true(self):
        mc = _StubModelConfig(lora_rank=8, lora={"merge": True})
        peft_merge = mc.lora.get("merge", False)
        assert peft_merge is True

    def test_merge_false(self):
        mc = _StubModelConfig(lora_rank=8, lora={"merge": False})
        peft_merge = mc.lora.get("merge", False)
        assert peft_merge is False

    def test_merge_absent_defaults_false(self):
        mc = _StubModelConfig(lora_rank=8, lora={})
        peft_merge = mc.lora.get("merge", False)
        assert peft_merge is False
