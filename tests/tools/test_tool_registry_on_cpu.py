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

import os
import tempfile

import pytest

from verl.tools.base_tool import BaseTool
from verl.tools.registry import ToolRegistry, register_tool
from verl.tools.schemas import OpenAIFunctionToolSchema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_schema(name: str = "test_func") -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "string", "description": "input"},
                    },
                    "required": ["x"],
                },
            },
        }
    )


class _DummyTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)


def _can_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Tests: ToolRegistry core behaviour
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def setup_method(self):
        self._snapshot = dict(ToolRegistry._registry)

    def teardown_method(self):
        ToolRegistry._registry.clear()
        ToolRegistry._registry.update(self._snapshot)

    def test_register_and_get(self):
        @register_tool("my_tool")
        class MyTool(BaseTool):
            pass

        assert ToolRegistry.contains("my_tool")
        assert ToolRegistry.get("my_tool") is MyTool

    def test_list_tools_includes_registered(self):
        @register_tool("listed_tool")
        class ListedTool(BaseTool):
            pass

        assert "listed_tool" in ToolRegistry.list_tools()

    def test_get_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="not_registered"):
            ToolRegistry.get("not_registered")

    def test_contains_false_for_unknown(self):
        assert not ToolRegistry.contains("nonexistent_tool_xyz")

    def test_duplicate_registration_overwrites_with_warning(self):
        @register_tool("dup_tool")
        class First(BaseTool):
            pass

        @register_tool("dup_tool")
        class Second(BaseTool):
            pass

        assert ToolRegistry.get("dup_tool") is Second

    def test_clear(self):
        @register_tool("clearable")
        class Tmp(BaseTool):
            pass

        assert ToolRegistry.contains("clearable")
        ToolRegistry.clear()
        assert not ToolRegistry.contains("clearable")


# ---------------------------------------------------------------------------
# Tests: Built-in tools are registered
# ---------------------------------------------------------------------------


class TestBuiltinToolRegistration:
    """Verify that importing the built-in tool modules populates the registry."""

    def test_gsm8k_registered(self):
        import verl.tools.gsm8k_tool  # noqa: F401

        assert ToolRegistry.contains("gsm8k")

    def test_geo3k_registered(self):
        import verl.tools.geo3k_tool  # noqa: F401

        assert ToolRegistry.contains("geo3k")

    @pytest.mark.skipif(
        not _can_import("fastmcp"),
        reason="fastmcp not installed",
    )
    def test_mcp_base_registered(self):
        import verl.tools.mcp_base_tool  # noqa: F401

        assert ToolRegistry.contains("mcp_base")


# ---------------------------------------------------------------------------
# Tests: initialize_tools_from_config with tool_name
# ---------------------------------------------------------------------------


class TestInitializeToolsFromConfig:
    """Test that tool configs using ``tool_name`` work alongside ``class_name``."""

    def _write_config(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_class_name_still_works(self):
        """Backward-compat: ``class_name`` resolves via importlib as before."""
        from verl.tools.utils.tool_registry import initialize_tools_from_config

        config_path = self._write_config(
            """
tools:
  - class_name: "verl.tools.gsm8k_tool.Gsm8kTool"
    config:
      type: native
    tool_schema:
      type: "function"
      function:
        name: "calc_gsm8k_reward"
        description: "test"
        parameters:
          type: "object"
          properties:
            answer:
              type: "string"
              description: "answer"
          required: ["answer"]
"""
        )
        try:
            tools = initialize_tools_from_config(config_path)
            assert len(tools) == 1
            assert tools[0].name == "calc_gsm8k_reward"
        finally:
            os.unlink(config_path)

    def test_tool_name_registry_lookup(self):
        """New path: ``tool_name`` resolves via ToolRegistry."""
        from verl.tools.utils.tool_registry import initialize_tools_from_config

        config_path = self._write_config(
            """
tools:
  - tool_name: "gsm8k"
    config:
      type: native
    tool_schema:
      type: "function"
      function:
        name: "calc_gsm8k_reward"
        description: "test"
        parameters:
          type: "object"
          properties:
            answer:
              type: "string"
              description: "answer"
          required: ["answer"]
"""
        )
        try:
            tools = initialize_tools_from_config(config_path)
            assert len(tools) == 1
            assert tools[0].name == "calc_gsm8k_reward"
        finally:
            os.unlink(config_path)

    def test_missing_both_keys_raises(self):
        """Config with neither ``tool_name`` nor ``class_name`` should error."""
        from verl.tools.utils.tool_registry import initialize_tools_from_config

        config_path = self._write_config(
            """
tools:
  - config:
      type: native
    tool_schema:
      type: "function"
      function:
        name: "bad"
        description: "no class"
        parameters:
          type: "object"
          properties: {}
          required: []
"""
        )
        try:
            with pytest.raises(ValueError, match="tool_name.*class_name"):
                initialize_tools_from_config(config_path)
        finally:
            os.unlink(config_path)


# ---------------------------------------------------------------------------
# Tests: BaseTool uses logger instead of print
# ---------------------------------------------------------------------------


class TestBaseToolLogging:
    def test_init_does_not_print(self, capsys):
        """BaseTool.__init__ should not produce stdout output."""
        schema = _make_schema()
        _DummyTool(config={}, tool_schema=schema)
        captured = capsys.readouterr()
        assert captured.out == "", "BaseTool.__init__ should use logger, not print()"
