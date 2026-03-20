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
"""Tests for TRT-LLM release() tags parameter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from verl.workers.rollout._tag_utils import validate_release_tags

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


def _make_trtllm_mock():
    mock = MagicMock()
    mock.is_leader_rank = True
    mock.config = MagicMock(free_cache_engine=True)
    mock._init_server_adapter = AsyncMock()
    mock._adapter = AsyncMock()
    mock._adapter.release_memory_occupation = AsyncMock(return_value={"status": "ok"})
    return mock


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
