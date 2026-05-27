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
"""Unit tests for the function-based tool API."""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path
from typing import Literal

import pytest

from verl.tools import function_tool as function_tool_mod
from verl.tools.function_tool import (
    FUNCTION_TOOL_REGISTRY,
    FunctionTool,
    function_tool,
    load_function_tools_from_path,
    normalize_function_tool_return,
)
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset both the registry and the per-path cache around every test."""
    FUNCTION_TOOL_REGISTRY.clear()
    function_tool_mod._LOADED_FUNCTION_TOOL_PATHS.clear()
    yield
    FUNCTION_TOOL_REGISTRY.clear()
    function_tool_mod._LOADED_FUNCTION_TOOL_PATHS.clear()


def _write_tool_file(tmp_path: Path, body: str) -> str:
    path = tmp_path / "my_tools.py"
    path.write_text(textwrap.dedent(body))
    return str(path)


# ---------------------------------------------------------------------------
# @function_tool decorator + schema inference
# ---------------------------------------------------------------------------


def test_decorator_registers_with_inferred_schema():
    @function_tool("greet")
    def greet(name: str, excited: bool = False) -> str:
        """Greet someone.

        Args:
            name: Person to greet.
            excited: Whether to add an exclamation mark.
        """
        return f"hi {name}{'!' if excited else ''}"

    assert "greet" in FUNCTION_TOOL_REGISTRY
    tool = FUNCTION_TOOL_REGISTRY["greet"]
    fn_schema = tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)["function"]

    assert fn_schema["name"] == "greet"
    assert fn_schema["description"].startswith("Greet someone.")
    assert fn_schema["parameters"]["properties"]["name"]["type"] == "string"
    assert fn_schema["parameters"]["properties"]["name"]["description"] == "Person to greet."
    assert fn_schema["parameters"]["properties"]["excited"]["type"] == "boolean"
    assert fn_schema["parameters"]["required"] == ["name"]


def test_int_float_union_emits_number_type():
    """Numeric unions must let the LLM produce decimals.

    transformers ``get_json_schema`` returns the JSON Schema-standard
    ``{"type": ["integer", "number"]}`` for ``int | float``; verl's loosened
    schema accepts list-typed ``type`` fields so the LLM is allowed to emit
    decimals rather than being told "must be integer".
    """

    @function_tool("num")
    def num(x: int | float) -> str:
        """Numeric.

        Args:
            x: a numeric value.
        """
        return str(x)

    params = FUNCTION_TOOL_REGISTRY["num"].tool_schema.model_dump(exclude_unset=True, exclude_none=True)["function"][
        "parameters"
    ]
    assert "number" in params["properties"]["x"]["type"]


def test_int_literal_emits_int_enum():
    """``Literal[1, 2, 3]`` -> JSON ``enum: [1, 2, 3]``.

    Pins the verl schema loosening that allows non-string ``enum`` values
    (otherwise pydantic rejects integer literals).
    """

    @function_tool("pick")
    def pick(x: Literal[1, 2, 3]) -> str:
        """Pick.

        Args:
            x: pick one.
        """
        return str(x)

    props = FUNCTION_TOOL_REGISTRY["pick"].tool_schema.model_dump(exclude_unset=True, exclude_none=True)["function"][
        "parameters"
    ]["properties"]
    assert props["x"]["enum"] == [1, 2, 3]


def test_decorator_default_name_uses_function_name():
    @function_tool()
    def my_special_tool(x: int) -> int:
        """Doc.

        Args:
            x: A number.
        """
        return x

    assert "my_special_tool" in FUNCTION_TOOL_REGISTRY


def test_bare_decorator_without_parentheses():
    """``@function_tool`` (no parens) registers under the function name."""

    @function_tool
    def bare_tool(x: int) -> int:
        """Doc.

        Args:
            x: A number.
        """
        return x

    assert "bare_tool" in FUNCTION_TOOL_REGISTRY
    fn_schema = FUNCTION_TOOL_REGISTRY["bare_tool"].tool_schema.model_dump(exclude_unset=True, exclude_none=True)[
        "function"
    ]
    assert fn_schema["name"] == "bare_tool"
    # Schema inference path is the same as the parenthesised form.
    assert fn_schema["parameters"]["properties"]["x"]["type"] == "integer"


def test_explicit_schema_dict_override_skips_inference():
    custom = {
        "type": "function",
        "function": {
            "name": "x",
            "description": "custom desc",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

    @function_tool("x", schema=custom)
    def x() -> str:
        """This docstring should be ignored.

        Args:
            ignored: ignored.
        """
        return ""

    schema = FUNCTION_TOOL_REGISTRY["x"].tool_schema.model_dump(exclude_unset=True, exclude_none=True)
    assert schema["function"]["description"] == "custom desc"
    assert schema["function"]["parameters"]["properties"] == {}


def test_explicit_schema_object_override():
    custom = OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "y",
                "description": "obj desc",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    )

    @function_tool("y", schema=custom)
    def y() -> str:
        return ""

    assert FUNCTION_TOOL_REGISTRY["y"].tool_schema is custom


def test_duplicate_name_raises():
    @function_tool("dup")
    def fn1(x: str) -> str:
        """Doc.

        Args:
            x: a string.
        """
        return x

    with pytest.raises(ValueError, match="already registered"):

        @function_tool("dup")
        def fn2(x: str) -> str:
            """Doc.

            Args:
                x: a string.
            """
            return x


def test_async_function_marked_is_async():
    @function_tool("aecho")
    async def aecho(text: str) -> str:
        """Echo.

        Args:
            text: text.
        """
        return text

    tool = FUNCTION_TOOL_REGISTRY["aecho"]
    assert tool.is_async is True
    assert asyncio.run(tool.call({"text": "ok"})) == "ok"


def test_sync_function_runs_in_thread():
    @function_tool("secho")
    def secho(text: str) -> str:
        """Echo.

        Args:
            text: text.
        """
        return text.upper()

    tool = FUNCTION_TOOL_REGISTRY["secho"]
    assert tool.is_async is False
    assert asyncio.run(tool.call({"text": "hi"})) == "HI"


def test_missing_docstring_raises_at_registration():
    """Schema inference is delegated to ``transformers.get_json_schema``,
    which raises ``DocstringParsingException`` when the function has no
    docstring at all. We verify the contract surfaces, not the exact
    exception type, to avoid coupling tests to transformers internals.
    """

    with pytest.raises(Exception, match=r"no docstring"):

        @function_tool("nodoc")
        def nodoc(x: str) -> str:
            return x


def test_missing_type_hint_raises_at_registration():
    """``transformers.get_json_schema`` raises when a parameter is unannotated."""

    with pytest.raises(Exception, match=r"missing a type hint"):

        @function_tool("untyped")
        def untyped(x) -> str:
            """Doc.

            Args:
                x: a thing.
            """
            return x


def test_missing_arg_description_raises_at_registration():
    """``transformers.get_json_schema`` requires every parameter to be
    described in the docstring's ``Args:`` block."""

    with pytest.raises(Exception, match=r"no description for the argument"):

        @function_tool("partial")
        def partial(x: int, y: int) -> int:
            """Add.

            Args:
                x: only x described.
            """
            return x + y


def test_var_args_raises_at_registration():
    """``*args`` / ``**kwargs`` can't be expressed as fixed JSON properties.

    We catch this before ``get_json_schema`` so the user gets a verl-specific
    pointer to the right fix (``param: list[T]``) instead of transformers'
    less actionable "missing type hint for args".
    """

    with pytest.raises(ValueError, match=r"variadic parameter"):

        @function_tool("varargs")
        def varargs(x: int, *args: int) -> int:
            """Sum.

            Args:
                x: x.
            """
            return x + sum(args)

    with pytest.raises(ValueError, match=r"variadic parameter"):

        @function_tool("varkw")
        def varkw(x: int, **kwargs: int) -> int:
            """Sum.

            Args:
                x: x.
            """
            return x + sum(kwargs.values())


# ---------------------------------------------------------------------------
# load_function_tools_from_path
# ---------------------------------------------------------------------------


def test_load_basic_returns_registered_tools(tmp_path):
    path = _write_tool_file(
        tmp_path,
        """
        from verl.tools.function_tool import function_tool

        @function_tool("greet")
        def greet(name: str) -> str:
            '''Greet someone.

            Args:
                name: who to greet.
            '''
            return f"hi {name}"
        """,
    )

    tools = load_function_tools_from_path(path)
    assert [t.name for t in tools] == ["greet"]
    assert FUNCTION_TOOL_REGISTRY["greet"] is tools[0]


def test_load_multiple_tools(tmp_path):
    path = _write_tool_file(
        tmp_path,
        """
        from verl.tools.function_tool import function_tool

        @function_tool("a")
        def a(x: str) -> str:
            '''A.

            Args:
                x: x.
            '''
            return x

        @function_tool("b")
        def b(x: str) -> str:
            '''B.

            Args:
                x: x.
            '''
            return x
        """,
    )
    tools = load_function_tools_from_path(path)
    assert sorted(t.name for t in tools) == ["a", "b"]


def test_missing_path_raises():
    with pytest.raises(FileNotFoundError, match="function_tool_path does not exist"):
        load_function_tools_from_path("/nonexistent/path/here.py")


def test_no_decorator_logs_warning(tmp_path, caplog):
    path = _write_tool_file(tmp_path, "x = 1\n")
    with caplog.at_level("WARNING"):
        tools = load_function_tools_from_path(path)
    assert tools == []
    assert any("no @function_tool decorators found" in rec.getMessage() for rec in caplog.records)


def test_load_is_idempotent_across_calls(tmp_path):
    """Loading the same path twice in one process must be a no-op.

    Production calls ``load_function_tools_from_path`` exactly once per
    worker process (from ``AgentLoopWorker.__init__``), so this is not a
    hot-path concern there. The contract still matters for tests, custom
    managers, or any code that re-enters the loader: without the
    :data:`_LOADED_FUNCTION_TOOL_PATHS` cache, the second call would
    re-exec the user file, the ``@function_tool`` decorator would run
    again with a *new* function object for the same name, and the
    decorator's dup-name guard would raise ``ValueError``.
    """
    path = _write_tool_file(
        tmp_path,
        """
        from verl.tools.function_tool import function_tool

        @function_tool("idem")
        def idem(x: str) -> str:
            '''Idem.

            Args:
                x: x.
            '''
            return x
        """,
    )

    first = load_function_tools_from_path(path)
    second = load_function_tools_from_path(path)

    assert [t.name for t in first] == ["idem"]
    assert second[0] is first[0]
    assert second[0].fn is first[0].fn


def test_load_returns_only_tools_added_by_this_file(tmp_path):
    """Pre-registering a tool from elsewhere must not leak into the loader's
    return value; the loader attributes only what its file added."""

    @function_tool("preexisting")
    def preexisting(x: str) -> str:
        """Pre.

        Args:
            x: x.
        """
        return x

    path = _write_tool_file(
        tmp_path,
        """
        from verl.tools.function_tool import function_tool

        @function_tool("only_mine")
        def only_mine(x: str) -> str:
            '''Mine.

            Args:
                x: x.
            '''
            return x
        """,
    )

    tools = load_function_tools_from_path(path)
    assert [t.name for t in tools] == ["only_mine"]
    assert "preexisting" in FUNCTION_TOOL_REGISTRY


def test_relative_path_resolved_against_cwd(tmp_path, monkeypatch):
    path_str = _write_tool_file(
        tmp_path,
        """
        from verl.tools.function_tool import function_tool

        @function_tool("rel")
        def rel(x: str) -> str:
            '''Rel.

            Args:
                x: x.
            '''
            return x
        """,
    )
    monkeypatch.chdir(tmp_path)
    tools = load_function_tools_from_path(Path(path_str).name)
    assert [t.name for t in tools] == ["rel"]


# ---------------------------------------------------------------------------
# normalize_function_tool_return
# ---------------------------------------------------------------------------


def test_normalize_str():
    resp, reward, metrics = normalize_function_tool_return("hello")
    assert resp == ToolResponse(text="hello")
    assert reward == 0.0
    assert metrics == {}


def test_normalize_tool_response_passthrough():
    src = ToolResponse(text="x")
    resp, reward, metrics = normalize_function_tool_return(src)
    assert resp is src
    assert reward == 0.0
    assert metrics == {}


def test_normalize_dict_serialized_as_json():
    resp, _, _ = normalize_function_tool_return({"a": 1, "b": "two"})
    assert "a" in resp.text and "two" in resp.text


def test_normalize_2_tuple_carries_reward():
    resp, reward, metrics = normalize_function_tool_return(("text", 1.5))
    assert resp.text == "text"
    assert reward == 1.5
    assert metrics == {}


def test_normalize_3_tuple_carries_metrics():
    resp, reward, metrics = normalize_function_tool_return(("text", 2.0, {"k": "v"}))
    assert resp.text == "text"
    assert reward == 2.0
    assert metrics == {"k": "v"}


def test_normalize_tuple_tolerates_none_reward_and_metrics():
    """Tools may legitimately omit reward/metrics by returning ``None``."""
    resp, reward, metrics = normalize_function_tool_return(("text", None))
    assert resp.text == "text"
    assert reward == 0.0
    assert metrics == {}

    resp, reward, metrics = normalize_function_tool_return(("text", None, None))
    assert resp.text == "text"
    assert reward == 0.0
    assert metrics == {}


def test_normalize_falsy_reward_is_preserved_not_coerced_to_default():
    """Regression: detect ``None`` via ``is None``, not truthiness.

    The earlier ``ret[1] or 0.0`` form swallowed every falsy reward value
    including ``False`` and integer ``0``, so a tool that legitimately
    reported "no progress this turn" via ``reward=0`` or ``reward=False``
    was indistinguishable from one that returned ``reward=None`` -- and
    more importantly, distinct from the ``or``-fallback path semantically.
    """
    # int 0 is the canonical "no signal" reward; must round-trip as 0.0,
    # and importantly come out of the ``int -> float`` branch, not the
    # ``or 0.0`` branch.
    _, reward, _ = normalize_function_tool_return(("t", 0))
    assert reward == 0.0
    assert isinstance(reward, float)

    # bool is a subclass of int so ``False`` is technically valid here.
    _, reward, _ = normalize_function_tool_return(("t", False))
    assert reward == 0.0


def test_normalize_tuple_of_invalid_length_raises():
    """0-length and >=4-length tuples almost always indicate a bug; we
    refuse rather than silently ``str(ret)``-ing the entire tuple, which
    would corrupt the ToolResponse shown to the LLM."""
    with pytest.raises(TypeError, match=r"length 1, 2, or 3"):
        normalize_function_tool_return(())

    with pytest.raises(TypeError, match=r"length 1, 2, or 3"):
        normalize_function_tool_return(("a", 1, {}, "extra"))


def test_normalize_arbitrary_object_falls_back_to_str():
    class Foo:
        def __str__(self) -> str:
            return "FOO"

    resp, reward, metrics = normalize_function_tool_return(Foo())
    assert resp.text == "FOO"
    assert reward == 0.0
    assert metrics == {}


# ---------------------------------------------------------------------------
# Rollout config field
# ---------------------------------------------------------------------------


def test_rollout_yaml_exposes_function_tool_path():
    """Smoke test the YAML default so ``ToolAgentLoop.__init__`` can read it."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("verl/trainer/config/rollout/rollout.yaml")
    assert "function_tool_path" in cfg.multi_turn
    assert cfg.multi_turn.function_tool_path is None


class _HydraProbe:
    def __init__(self, tools):
        self.tools = tools


def test_tool_list_wrap_survives_hydra_instantiate(tmp_path):
    """Without ``ToolListWrap``, ``hydra.utils.instantiate`` demotes each
    ``FunctionTool`` in a kwarg list to ``DictConfig`` and breaks
    ``isinstance(tool, FunctionTool)`` in ``ToolAgentLoop._call_tool``."""
    import hydra

    from verl.experimental.agent_loop.agent_loop import ToolListWrap

    path = _write_tool_file(
        tmp_path,
        """
        from verl.tools.function_tool import function_tool

        @function_tool
        def probe(text: str) -> str:
            '''Probe.

            Args:
                text: text.
            '''
            return text
        """,
    )
    tools = load_function_tools_from_path(path)
    assert all(isinstance(t, FunctionTool) for t in tools)

    # Without the wrap: hydra demotes each FunctionTool to DictConfig. If this
    # ever stops being true, ToolListWrap is obsolete and can be deleted.
    naked = hydra.utils.instantiate({"_target_": f"{__name__}._HydraProbe"}, tools=tools)
    assert not all(isinstance(t, FunctionTool) for t in naked.tools), (
        "hydra.utils.instantiate no longer demotes FunctionTool to DictConfig; ToolListWrap may be obsolete."
    )

    wrapped = hydra.utils.instantiate(
        {"_target_": f"{__name__}._HydraProbe"},
        tools=ToolListWrap(tools),
    )
    assert isinstance(wrapped.tools, ToolListWrap)
    assert all(isinstance(t, FunctionTool) for t in wrapped.tools.tools)
    assert callable(wrapped.tools.tools[0].fn)
