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
"""Tests for vLLM release() tags parameter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from verl.workers.rollout._tag_utils import validate_release_tags


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


def _make_vllm_mock():
    mock = MagicMock()
    mock.config = MagicMock(free_cache_engine=True)
    mock.sleep_level = 2
    mock._execute_method = AsyncMock()
    return mock


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
