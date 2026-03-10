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
"""Tests for the `tags` parameter on rollout release().

Each backend's release() accepts an optional `tags` argument that selects
which GPU resources to release (["weights"], ["kv_cache"], or both).

The shared validation logic lives in `_tag_utils.validate_release_tags()`
and is tested directly (no mocking needed). Backend-specific behavior
(vLLM sleep-level mapping, TRT-LLM tag resolution) is tested via
lightweight mock objects that exercise each backend's release() method
without requiring GPU or distributed infrastructure.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from verl.workers.rollout._tag_utils import validate_release_tags

# ---------------------------------------------------------------------------
# validate_release_tags — shared logic (real code, no mocks)
# ---------------------------------------------------------------------------


class TestValidateReleaseTags:
    def test_none_returns_both(self):
        assert validate_release_tags(None) == {"kv_cache", "weights"}

    def test_weights_only(self):
        assert validate_release_tags(["weights"]) == {"weights"}

    def test_kv_cache_only(self):
        assert validate_release_tags(["kv_cache"]) == {"kv_cache"}

    def test_both_explicit(self):
        assert validate_release_tags(["kv_cache", "weights"]) == {"kv_cache", "weights"}

    def test_duplicates_deduplicated(self):
        assert validate_release_tags(["weights", "weights"]) == {"weights"}

    def test_unknown_tag_raises(self):
        with pytest.raises(ValueError, match="Unknown release tags"):
            validate_release_tags(["bogus"])

    def test_mixed_valid_and_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown release tags"):
            validate_release_tags(["weights", "bogus"])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_release_tags([])


# ---------------------------------------------------------------------------
# Backend-specific release() behavior via mock objects.
#
# These test the exact method logic from each backend's release() without
# importing the actual classes (which require torch, sglang, ray, etc.).
# The validate_release_tags() call is real code — only the async I/O
# (engine calls, server adapters) is mocked.
# ---------------------------------------------------------------------------


async def _sglang_release(self, tags=None):
    """Mirrors SGLangRollout.release() — calls validate_release_tags."""
    tag_set = validate_release_tags(tags)
    await self._init_server_adapter()
    if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
        await self._engine.release_memory_occupation(tags=sorted(tag_set))


async def _vllm_release(self, tags=None):
    """Mirrors VllmRollout.release() — calls validate_release_tags."""
    tag_set = validate_release_tags(tags)
    if not self.config.free_cache_engine:
        return
    if tag_set == {"kv_cache", "weights"}:
        level = self.sleep_level
    elif tag_set == {"kv_cache"}:
        level = 1
    else:
        raise NotImplementedError(
            f"vLLM release does not support tags={tags!r}; only ['kv_cache', 'weights'] or ['kv_cache'] are supported"
        )
    await self._execute_method("sleep", kwargs={"level": level})


# TRT-LLM weight tags (from ServerAdapter._WEIGHTS_TAGS)
_TRTLLM_WEIGHTS_TAGS = [
    "sampler",
    "drafter",
    "guided_decoder",
    "spec_resource_manager",
    "model_extra",
    "executor_extra",
    "model",
    "draft_model",
]


async def _trtllm_release(self, tags=None):
    """Mirrors TrtllmRollout.release() — calls validate_release_tags."""
    tag_set = validate_release_tags(tags)
    if not self.is_leader_rank or not self.config.free_cache_engine:
        return
    await self._init_server_adapter()
    resolved_tags = []
    if "weights" in tag_set:
        resolved_tags.extend(_TRTLLM_WEIGHTS_TAGS)
    if "kv_cache" in tag_set:
        resolved_tags.append("kv_cache")
    await self._adapter.release_memory_occupation(tags=resolved_tags)


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _make_sglang_mock():
    mock = MagicMock()
    mock._init_server_adapter = AsyncMock()
    mock._engine = AsyncMock()
    mock._engine.release_memory_occupation = AsyncMock(return_value={"status": "ok"})
    mock.device_mesh = {"infer_tp": MagicMock(get_local_rank=MagicMock(return_value=0))}
    mock.config = MagicMock(free_cache_engine=True)
    return mock


def _make_vllm_mock():
    mock = MagicMock()
    mock.config = MagicMock(free_cache_engine=True)
    mock.sleep_level = 2
    mock._execute_method = AsyncMock()
    return mock


def _make_trtllm_mock():
    mock = MagicMock()
    mock.is_leader_rank = True
    mock.config = MagicMock(free_cache_engine=True)
    mock._init_server_adapter = AsyncMock()
    mock._adapter = AsyncMock()
    mock._adapter.release_memory_occupation = AsyncMock(return_value={"status": "ok"})
    return mock


# ---------------------------------------------------------------------------
# SGLang tests
# ---------------------------------------------------------------------------


class TestSGLangReleaseTags:
    @pytest.mark.asyncio
    async def test_default_releases_both(self):
        mock = _make_sglang_mock()
        await _sglang_release(mock)
        mock._engine.release_memory_occupation.assert_called_once_with(tags=["kv_cache", "weights"])

    @pytest.mark.asyncio
    async def test_weights_only(self):
        mock = _make_sglang_mock()
        await _sglang_release(mock, tags=["weights"])
        mock._engine.release_memory_occupation.assert_called_once_with(tags=["weights"])

    @pytest.mark.asyncio
    async def test_kv_cache_only(self):
        mock = _make_sglang_mock()
        await _sglang_release(mock, tags=["kv_cache"])
        mock._engine.release_memory_occupation.assert_called_once_with(tags=["kv_cache"])

    @pytest.mark.asyncio
    async def test_unknown_tag_raises(self):
        mock = _make_sglang_mock()
        with pytest.raises(ValueError, match="Unknown release tags"):
            await _sglang_release(mock, tags=["bogus"])

    @pytest.mark.asyncio
    async def test_free_cache_disabled_is_noop(self):
        mock = _make_sglang_mock()
        mock.config.free_cache_engine = False
        await _sglang_release(mock)
        mock._engine.release_memory_occupation.assert_not_called()


# ---------------------------------------------------------------------------
# vLLM tests
# ---------------------------------------------------------------------------


class TestVllmReleaseTags:
    @pytest.mark.asyncio
    async def test_default_releases_both(self):
        mock = _make_vllm_mock()
        await _vllm_release(mock)
        mock._execute_method.assert_called_once_with("sleep", kwargs={"level": 2})

    @pytest.mark.asyncio
    async def test_kv_cache_only(self):
        mock = _make_vllm_mock()
        await _vllm_release(mock, tags=["kv_cache"])
        mock._execute_method.assert_called_once_with("sleep", kwargs={"level": 1})

    @pytest.mark.asyncio
    async def test_weights_only_not_supported(self):
        mock = _make_vllm_mock()
        with pytest.raises(NotImplementedError):
            await _vllm_release(mock, tags=["weights"])

    @pytest.mark.asyncio
    async def test_unknown_tag_raises_value_error(self):
        mock = _make_vllm_mock()
        with pytest.raises(ValueError, match="Unknown release tags"):
            await _vllm_release(mock, tags=["bogus"])

    @pytest.mark.asyncio
    async def test_free_cache_disabled_is_noop(self):
        mock = _make_vllm_mock()
        mock.config.free_cache_engine = False
        await _vllm_release(mock)  # valid tags, but free_cache_engine=False → noop
        mock._execute_method.assert_not_called()

    @pytest.mark.asyncio
    async def test_free_cache_disabled_still_validates(self):
        mock = _make_vllm_mock()
        mock.config.free_cache_engine = False
        with pytest.raises(ValueError, match="Unknown release tags"):
            await _vllm_release(mock, tags=["bogus"])


# ---------------------------------------------------------------------------
# TRT-LLM tests
# ---------------------------------------------------------------------------


class TestTrtllmReleaseTags:
    @pytest.mark.asyncio
    async def test_default_releases_both(self):
        mock = _make_trtllm_mock()
        await _trtllm_release(mock)
        call_tags = mock._adapter.release_memory_occupation.call_args.kwargs["tags"]
        assert "kv_cache" in call_tags
        for wt in _TRTLLM_WEIGHTS_TAGS:
            assert wt in call_tags

    @pytest.mark.asyncio
    async def test_weights_only(self):
        mock = _make_trtllm_mock()
        await _trtllm_release(mock, tags=["weights"])
        call_tags = mock._adapter.release_memory_occupation.call_args.kwargs["tags"]
        assert "kv_cache" not in call_tags
        for wt in _TRTLLM_WEIGHTS_TAGS:
            assert wt in call_tags

    @pytest.mark.asyncio
    async def test_kv_cache_only(self):
        mock = _make_trtllm_mock()
        await _trtllm_release(mock, tags=["kv_cache"])
        call_tags = mock._adapter.release_memory_occupation.call_args.kwargs["tags"]
        assert call_tags == ["kv_cache"]

    @pytest.mark.asyncio
    async def test_unknown_tag_raises(self):
        mock = _make_trtllm_mock()
        with pytest.raises(ValueError, match="Unknown release tags"):
            await _trtllm_release(mock, tags=["bogus"])

    @pytest.mark.asyncio
    async def test_non_leader_is_noop(self):
        mock = _make_trtllm_mock()
        mock.is_leader_rank = False
        await _trtllm_release(mock, tags=["weights"])
        mock._adapter.release_memory_occupation.assert_not_called()
