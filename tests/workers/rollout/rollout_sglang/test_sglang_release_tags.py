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
"""Tests for SGLang release() tags parameter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from verl.workers.rollout._tag_utils import validate_release_tags


async def _sglang_release(self, tags=None):
    """Mirrors SGLangRollout.release() — calls validate_release_tags."""
    tag_set = validate_release_tags(tags)
    await self._init_server_adapter()
    if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
        await self._engine.release_memory_occupation(tags=sorted(tag_set))


def _make_sglang_mock():
    mock = MagicMock()
    mock._init_server_adapter = AsyncMock()
    mock._engine = AsyncMock()
    mock._engine.release_memory_occupation = AsyncMock(return_value={"status": "ok"})
    mock.device_mesh = {"infer_tp": MagicMock(get_local_rank=MagicMock(return_value=0))}
    mock.config = MagicMock(free_cache_engine=True)
    return mock


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
