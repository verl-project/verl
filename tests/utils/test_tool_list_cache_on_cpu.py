#!/usr/bin/env python3
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

from __future__ import annotations

from dataclasses import dataclass

import pytest

import verl.tools.utils.tool_registry as tool_registry
from verl.tools.utils.tool_registry import ToolListCache


@dataclass(frozen=True)
class _DummyTool:
    name: str


def test_tool_list_cache_returns_empty_for_none():
    ToolListCache._cache.clear()
    assert ToolListCache.get_tool_list(None) == []


def test_tool_list_cache_caches_and_deepcopies(monkeypatch: pytest.MonkeyPatch):
    ToolListCache._cache.clear()

    calls: list[tuple[str, float | None]] = []

    def _fake_init(tools_config_file: str, mcp_init_timeout_s: float | None = None):
        calls.append((tools_config_file, mcp_init_timeout_s))
        return [_DummyTool(name="t1")]

    monkeypatch.setattr(tool_registry, "initialize_tools_from_config", _fake_init)
    monkeypatch.setenv("VERL_TOOL_CACHE_INIT_TIMEOUT_S", "12")

    out1 = ToolListCache.get_tool_list("dummy.yaml")
    out2 = ToolListCache.get_tool_list("dummy.yaml")

    assert calls == [("dummy.yaml", 12.0)]
    assert out1 == out2
    assert out1 is not out2
    assert out1[0] is not out2[0]


def test_tool_list_cache_retries_before_failing(monkeypatch: pytest.MonkeyPatch):
    ToolListCache._cache.clear()

    attempts = {"n": 0}

    def _fake_init(tools_config_file: str, mcp_init_timeout_s: float | None = None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("boom")
        return [_DummyTool(name="t1")]

    sleeps: list[float] = []

    monkeypatch.setattr(tool_registry, "initialize_tools_from_config", _fake_init)
    monkeypatch.setattr(tool_registry.time, "sleep", lambda s: sleeps.append(float(s)))
    monkeypatch.setenv("VERL_TOOL_CACHE_INIT_RETRIES", "2")
    monkeypatch.setenv("VERL_TOOL_CACHE_INIT_TIMEOUT_S", "0.5")
    monkeypatch.setenv("VERL_TOOL_CACHE_INIT_BACKOFF_BASE_S", "0.01")
    monkeypatch.setenv("VERL_TOOL_CACHE_INIT_BACKOFF_MAX_S", "0.01")

    out = ToolListCache.get_tool_list("dummy.yaml")

    assert out == [_DummyTool(name="t1")]
    assert attempts["n"] == 2
    assert sleeps == [0.01]
