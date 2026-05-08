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
"""Coexistence test: yaml-defined native tools + ``@function_tool`` tools.

The reference yaml is ``recipe/search_agent/config/all_tool_config.yaml`` from
the open_verl repo; that file points at real classes that need a retrieval /
crawl service to work. To keep this test on CPU and self-contained, we
declare structurally-equivalent stub ``BaseTool`` subclasses in
``tests.tools._stub_search_tools`` and reference them from a tmp yaml using
the same tool names (``search`` / ``crawler``) and parameter keys.

What the test exercises is exactly what
:meth:`verl.experimental.agent_loop.tool_agent_loop.ToolAgentLoop.__init__`
does today: load native tools from yaml, load function tools from a Python
file, assert no name collisions, then merge into one ``{name: tool}`` map.
"""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

import pytest

from tests.tools._stub_search_tools import StubCrawlTool, StubSearchTool
from verl.tools.base_tool import BaseTool
from verl.tools.utils import function_tool as function_tool_mod
from verl.tools.utils.function_tool import (
    FUNCTION_TOOL_REGISTRY,
    FunctionTool,
    load_function_tools_from_path,
    normalize_function_tool_return,
)
from verl.tools.utils.tool_registry import initialize_tools_from_config


@pytest.fixture(autouse=True)
def _clean_registry():
    """Both the registry and the per-path cache are process-global; reset both."""
    FUNCTION_TOOL_REGISTRY.clear()
    function_tool_mod._LOADED_FUNCTION_TOOL_PATHS.clear()
    yield
    FUNCTION_TOOL_REGISTRY.clear()
    function_tool_mod._LOADED_FUNCTION_TOOL_PATHS.clear()


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

# Schemas mirror open_verl's all_tool_config.yaml verbatim (names, parameter
# shapes). Indentation matters for OmegaConf, so this is written as a literal
# string and dropped into tmp_path at runtime.
_NATIVE_TOOL_YAML = """\
tools:
  - class_name: "tests.tools._stub_search_tools.StubSearchTool"
    config:
      retrieval_service_url: http://stub/retrieve
      topk: 3
      type: native
    tool_schema:
      type: function
      function:
        name: search
        description: Stub web search.
        parameters:
          type: object
          properties:
            query_list:
              type: array
              description: A list of fully-formed semantic queries.
          required: ["query_list"]

  - class_name: "tests.tools._stub_search_tools.StubCrawlTool"
    config:
      crawl_service_url: http://stub/crawl
      type: native
    tool_schema:
      type: function
      function:
        name: crawler
        description: Stub crawl.
        parameters:
          type: object
          properties:
            url_list:
              type: array
              description: URLs to crawl.
          required: ["url_list"]
"""

_FUNCTION_TOOL_BODY = """\
from verl.tools.utils.function_tool import function_tool

@function_tool("get_weather")
def get_weather(city: str) -> dict:
    '''Get the current weather for a city.

    Args:
        city: the city to look up.
    '''
    return {"temperature_c": 17.3, "condition": "drizzle"}

@function_tool("calculator")
def calculator(expression: str) -> str:
    '''Evaluate an arithmetic expression.

    Args:
        expression: a python-style arithmetic expression.
    '''
    return str(eval(expression, {"__builtins__": {}}, {}))
"""


@pytest.fixture
def native_yaml_path(tmp_path: Path) -> str:
    p = tmp_path / "all_tool_config.yaml"
    p.write_text(_NATIVE_TOOL_YAML)
    return str(p)


@pytest.fixture
def function_tool_py_path(tmp_path: Path) -> str:
    p = tmp_path / "my_function_tools.py"
    p.write_text(textwrap.dedent(_FUNCTION_TOOL_BODY))
    return str(p)


def _merge_like_tool_agent_loop(
    native_yaml: str | None, function_path: str | None
) -> dict[str, BaseTool | FunctionTool]:
    """Replicates the loader-merge in ``ToolAgentLoop.__init__``.

    Keep this in sync with ``ToolAgentLoop.__init__``; in particular the
    name-collision assert is part of the public contract these tests pin.

    Note: in production, ``function_tools`` is pre-loaded by
    ``AgentLoopWorker`` and injected into ``ToolAgentLoop`` as a constructor
    kwarg — the path is loaded once per worker, not once per trajectory.
    Here we call ``load_function_tools_from_path`` directly to mirror that
    inject without needing a real worker.
    """
    tool_list: list = initialize_tools_from_config(native_yaml) if native_yaml else []
    function_tools = load_function_tools_from_path(function_path) if function_path else []
    if function_tools:
        existing_names = {tool.name for tool in tool_list}
        collisions = sorted(t.name for t in function_tools if t.name in existing_names)
        assert not collisions, (
            f"Function tool name(s) {collisions} collide with tools already declared in "
            f"'{native_yaml}'. Each tool name must be unique across `tool_config_path` "
            f"and `function_tool_path`; rename one of them."
        )
        tool_list.extend(function_tools)
    return {tool.name: tool for tool in tool_list}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_native_only_loader(native_yaml_path):
    """Sanity: the stub yaml alone yields exactly the two BaseTool instances."""
    tools = _merge_like_tool_agent_loop(native_yaml_path, None)

    assert sorted(tools) == ["crawler", "search"]
    assert isinstance(tools["search"], StubSearchTool)
    assert isinstance(tools["crawler"], StubCrawlTool)
    # No function tools should have been registered as a side effect.
    assert FUNCTION_TOOL_REGISTRY == {}


def test_function_only_loader(function_tool_py_path):
    """Sanity: the function-tool file alone yields FunctionTool instances."""
    tools = _merge_like_tool_agent_loop(None, function_tool_py_path)

    assert sorted(tools) == ["calculator", "get_weather"]
    assert isinstance(tools["get_weather"], FunctionTool)
    assert isinstance(tools["calculator"], FunctionTool)


def test_native_and_function_tools_coexist(native_yaml_path, function_tool_py_path):
    """All four tools land in one mapping with the right concrete types."""
    tools = _merge_like_tool_agent_loop(native_yaml_path, function_tool_py_path)

    assert sorted(tools) == ["calculator", "crawler", "get_weather", "search"]

    # Native tools come from initialize_tools_from_config → BaseTool subclasses.
    assert isinstance(tools["search"], StubSearchTool)
    assert isinstance(tools["crawler"], StubCrawlTool)
    # Function tools come from load_function_tools_from_path → FunctionTool.
    assert isinstance(tools["get_weather"], FunctionTool)
    assert isinstance(tools["calculator"], FunctionTool)
    # FunctionTool instances are *not* BaseTool subclasses; the dispatch in
    # ToolAgentLoop._call_tool relies on this isinstance check.
    assert not isinstance(tools["get_weather"], BaseTool)
    assert not isinstance(tools["calculator"], BaseTool)


def test_merged_schemas_are_well_formed(native_yaml_path, function_tool_py_path):
    """Each tool exposes a valid OpenAI function schema, so the chat template
    can serialize them together."""
    tools = _merge_like_tool_agent_loop(native_yaml_path, function_tool_py_path)

    schemas = {name: tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for name, tool in tools.items()}

    for name, sch in schemas.items():
        assert sch["type"] == "function", name
        assert sch["function"]["name"] == name, name
        assert sch["function"]["parameters"]["type"] == "object", name

    # Native tool schemas come from yaml (yaml-declared params).
    assert "query_list" in schemas["search"]["function"]["parameters"]["properties"]
    assert set(schemas["crawler"]["function"]["parameters"]["properties"]) == {"url_list"}

    # Function tool schemas were inferred from signature + docstring.
    weather_props = schemas["get_weather"]["function"]["parameters"]["properties"]
    assert weather_props["city"]["type"] == "string"
    assert weather_props["city"]["description"] == "the city to look up."


def test_dispatch_branches_match_tool_agent_loop(native_yaml_path, function_tool_py_path):
    """The two tool families are invoked through different code paths in
    ``ToolAgentLoop._call_tool``. Replicate both branches and check the
    end-to-end output is what callers expect.
    """
    tools = _merge_like_tool_agent_loop(native_yaml_path, function_tool_py_path)

    async def _drive():
        # --- function tool branch (FunctionTool.call → normalize) ---
        # ``get_weather`` returns a dict, so this also exercises the
        # dict→JSON branch of ``normalize_function_tool_return``.
        weather_raw = await tools["get_weather"].call({"city": "Tokyo"})
        weather_resp, weather_reward, weather_metrics = normalize_function_tool_return(weather_raw)
        assert "17.3" in weather_resp.text and "drizzle" in weather_resp.text
        assert weather_reward == 0.0
        assert weather_metrics == {}

        calc_raw = await tools["calculator"].call({"expression": "2 + 3 * 4"})
        calc_resp, _, _ = normalize_function_tool_return(calc_raw)
        assert calc_resp.text == "14"

        # --- BaseTool branch (create → execute → release) ---
        search_tool: StubSearchTool = tools["search"]
        instance_id, _ = await search_tool.create()
        s_resp, s_reward, s_metrics = await search_tool.execute(instance_id, {"query_list": ["foo", "bar"]})
        await search_tool.release(instance_id)
        assert "hits-for:foo" in s_resp.text and "hits-for:bar" in s_resp.text
        assert s_reward == 0.0
        assert s_metrics == {"num_queries": 2}
        assert search_tool.calls == [{"query_list": ["foo", "bar"]}]

        crawl_tool: StubCrawlTool = tools["crawler"]
        instance_id, _ = await crawl_tool.create()
        c_resp, _, c_metrics = await crawl_tool.execute(instance_id, {"url_list": ["http://a", "http://b"]})
        await crawl_tool.release(instance_id)
        assert c_resp.text == "crawled:http://a,http://b"
        assert c_metrics == {"num_urls": 2}

    asyncio.run(_drive())


def test_loader_merge_is_safe_to_call_per_trajectory(native_yaml_path, function_tool_py_path):
    """Regression: ``ToolAgentLoop`` is per-trajectory, so ``__init__`` runs
    the loader-merge once per sample. Simulate three trajectories worth of
    init back-to-back and check nothing crashes and the merged dict stays
    structurally identical.
    """
    runs = [_merge_like_tool_agent_loop(native_yaml_path, function_tool_py_path) for _ in range(3)]

    for tools in runs:
        assert sorted(tools) == ["calculator", "crawler", "get_weather", "search"]
        assert isinstance(tools["get_weather"], FunctionTool)
        assert isinstance(tools["calculator"], FunctionTool)
        assert isinstance(tools["search"], StubSearchTool)
        assert isinstance(tools["crawler"], StubCrawlTool)

    # Function tools must be the *same objects* across runs (cache hit).
    assert runs[0]["get_weather"] is runs[1]["get_weather"] is runs[2]["get_weather"]
    assert runs[0]["calculator"] is runs[1]["calculator"] is runs[2]["calculator"]
    # Native tools, by contrast, are re-instantiated per call (yaml loader has
    # no cache today). Pin this so anyone touching the loader sees the
    # asymmetry.
    assert runs[0]["search"] is not runs[1]["search"]


def test_function_tool_name_collision_with_native_tool_raises(native_yaml_path, tmp_path):
    """A function tool sharing a name with a native (yaml) tool must fail loudly.

    Pins the behavior of ``ToolAgentLoop.__init__``: silent override would let
    a stray ``@function_tool("search")`` shadow the production search tool
    without the user noticing, so we assert at the merge site instead.
    """
    colliding_path = tmp_path / "colliding.py"
    colliding_path.write_text(
        textwrap.dedent(
            """
            from verl.tools.utils.function_tool import function_tool

            @function_tool("search")
            def search(query_list: list) -> str:
                '''Stub function tool deliberately reusing the native name.

                Args:
                    query_list: queries.
                '''
                return "from-function-tool"
            """
        )
    )

    with pytest.raises(AssertionError, match=r"\['search'\].*collide"):
        _merge_like_tool_agent_loop(native_yaml_path, str(colliding_path))


def test_function_tool_name_collision_reports_all_offenders(native_yaml_path, tmp_path):
    """Multiple collisions are surfaced together, sorted, so users only need
    one round-trip to fix all of them."""
    colliding_path = tmp_path / "many_colliding.py"
    colliding_path.write_text(
        textwrap.dedent(
            """
            from verl.tools.utils.function_tool import function_tool

            @function_tool("crawler")
            def crawler(url_list: list) -> str:
                '''Collides.

                Args:
                    url_list: urls.
                '''
                return ""

            @function_tool("search")
            def search(query_list: list) -> str:
                '''Collides.

                Args:
                    query_list: queries.
                '''
                return ""

            @function_tool("ok_unique")
            def ok_unique(x: str) -> str:
                '''Fine.

                Args:
                    x: x.
                '''
                return x
            """
        )
    )

    with pytest.raises(AssertionError, match=r"\['crawler', 'search'\].*collide"):
        _merge_like_tool_agent_loop(native_yaml_path, str(colliding_path))
