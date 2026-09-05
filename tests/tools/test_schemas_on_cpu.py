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
"""Unit tests for the OpenAI tool-schema carrier models."""

from __future__ import annotations

import pytest

from verl.tools import function_tool as function_tool_mod
from verl.tools.function_tool import FUNCTION_TOOL_REGISTRY, function_tool
from verl.tools.schemas import OpenAIFunctionToolSchema

# The dump kwargs the rollout path uses, e.g. ToolAgentLoop and rl_dataset when they
# hand tool schemas to a chat template.
ROLLOUT_DUMP_KWARGS = {"exclude_unset": True, "exclude_none": True}


@pytest.fixture(autouse=True)
def _clean_registry():
    FUNCTION_TOOL_REGISTRY.clear()
    function_tool_mod._LOADED_FUNCTION_TOOL_PATHS.clear()
    yield
    FUNCTION_TOOL_REGISTRY.clear()
    function_tool_mod._LOADED_FUNCTION_TOOL_PATHS.clear()


def _constrained_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read part of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "offset": {"type": "integer", "description": "Start line", "minimum": 0},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum lines",
                        "minimum": 1,
                        "maximum": 128,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }


def test_numeric_constraints_and_additional_properties_survive_a_round_trip():
    """Keywords the carriers do not name are still part of the caller's contract.

    Pydantic drops unknown keys by default, so ``minimum`` / ``maximum`` /
    ``additionalProperties`` used to be gone by the time ``model_dump`` reached the
    chat template -- the model saw a weaker contract than the caller wrote.
    """
    raw = _constrained_tool_schema()

    dumped = OpenAIFunctionToolSchema.model_validate(raw).model_dump(exclude_none=True)

    params = dumped["function"]["parameters"]
    assert params["additionalProperties"] is False
    assert params["properties"]["offset"]["minimum"] == 0
    assert params["properties"]["limit"]["minimum"] == 1
    assert params["properties"]["limit"]["maximum"] == 128


def test_rollout_dump_reproduces_the_supplied_schema_exactly():
    """The dump the rollout path actually performs must not lose or invent keys."""
    raw = _constrained_tool_schema()

    dumped = OpenAIFunctionToolSchema.model_validate(raw).model_dump(**ROLLOUT_DUMP_KWARGS)

    assert dumped == raw


def test_nested_array_and_object_keywords_survive():
    """A property's own sub-schema is untyped here, so it has to pass through whole."""
    raw = {
        "type": "function",
        "function": {
            "name": "batch_update",
            "description": "Apply edits",
            "parameters": {
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {"line": {"type": "integer", "minimum": 1}},
                            "required": ["line"],
                        },
                    },
                    "mode": {"type": "string", "pattern": "^(insert|replace)$"},
                },
                "required": ["edits"],
            },
        },
    }

    dumped = OpenAIFunctionToolSchema.model_validate(raw).model_dump(**ROLLOUT_DUMP_KWARGS)

    assert dumped == raw


def test_function_tool_keeps_an_explicit_schema_as_is():
    """``function_tool(schema=...)`` documents the supplied schema as used as-is."""
    raw = _constrained_tool_schema()

    @function_tool(schema=raw)
    def read_file(path: str, offset: int = 0, limit: int = 128) -> str:
        return ""

    entry = FUNCTION_TOOL_REGISTRY["read_file"]

    assert entry.tool_schema.model_dump(**ROLLOUT_DUMP_KWARGS) == raw


def test_declared_fields_are_still_validated():
    """Accepting extra keywords must not turn into accepting anything."""
    with pytest.raises(ValueError):
        OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "broken",
                    "description": "missing the required property type",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"description": "no type given"}},
                    },
                },
            }
        )
