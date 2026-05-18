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
"""
Lightweight function-based tool registration.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional

from transformers.utils import get_json_schema

from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# global registry
FUNCTION_TOOL_REGISTRY: dict[str, FunctionTool] = {}

_LOADED_FUNCTION_TOOL_PATHS: dict[str, list[FunctionTool]] = {}


@dataclass
class FunctionTool:
    """Carrier object stored in :data:`FUNCTION_TOOL_REGISTRY`.

    Exposes the minimal interface that the agent loop relies on:

    - ``name``: tool name (matches ``tool_schema.function.name``)
    - ``tool_schema``: ``OpenAIFunctionToolSchema`` for prompt assembly
    - ``fn``: the underlying callable
    """

    name: str
    fn: Callable[..., Any]
    tool_schema: OpenAIFunctionToolSchema
    is_async: bool = False

    async def call(self, parameters: dict[str, Any]) -> Any:
        """Invoke the underlying function with the LLM-supplied parameters."""
        if self.is_async:
            return await self.fn(**parameters)
        return await asyncio.to_thread(self.fn, **parameters)


def function_tool(
    name: Optional[str | Callable] = None,
    *,
    schema: Optional[OpenAIFunctionToolSchema | dict] = None,
):
    """Register a Python function as a verl tool.

    The OpenAI tool schema is inferred from the function via
    :func:`transformers.utils.get_json_schema`, so the function **must** carry:

    - a Google-style docstring summarising the tool;
    - a ``Args:`` block describing every parameter;
    - a type hint on every parameter.

    If any of those are missing, ``transformers`` raises
    ``DocstringParsingException`` / ``TypeHintParsingException`` at
    registration time.

    Supports both decorator forms::

        @function_tool                          # bare, name = fn.__name__
        def web_search(...): ...

        @function_tool("web_search")            # rename the tool
        def search(...): ...

    Args:
        name: Tool name exposed to the LLM. Defaults to the function name.
            When used as a bare ``@function_tool`` (no parentheses), this
            position receives the function being decorated.
        schema: Skip schema inference entirely and use the supplied
            ``OpenAIFunctionToolSchema`` (or a dict matching that shape) as-is.
            Use this only if your function's signature can't be expressed in
            JSON Schema -- the normal path is to fix the function.
    """

    def _make_decorator(tool_name_override: Optional[str]):
        def decorator(fn: Callable):
            tool_name = tool_name_override or fn.__name__

            if isinstance(schema, OpenAIFunctionToolSchema):
                built_schema = schema
            elif isinstance(schema, dict):
                built_schema = OpenAIFunctionToolSchema.model_validate(schema)
            else:
                built_schema = _build_schema_from_fn(fn, tool_name)

            entry = FunctionTool(
                name=tool_name,
                fn=fn,
                tool_schema=built_schema,
                is_async=inspect.iscoroutinefunction(fn),
            )

            existing = FUNCTION_TOOL_REGISTRY.get(tool_name)
            if existing is not None and existing.fn is not fn:
                raise ValueError(
                    f"Function tool '{tool_name}' is already registered to "
                    f"{existing.fn.__module__}.{existing.fn.__qualname__}; "
                    f"refusing to overwrite with {fn.__module__}.{fn.__qualname__}."
                )
            FUNCTION_TOOL_REGISTRY[tool_name] = entry
            logger.info("Registered function tool '%s' from %s.%s", tool_name, fn.__module__, fn.__qualname__)
            return fn

        return decorator

    if callable(name) and schema is None:
        fn = name
        return _make_decorator(None)(fn)

    return _make_decorator(name)


def get_function_tool(name: str) -> FunctionTool:
    """Look up a registered function tool by name. Raises ``KeyError`` if absent."""
    if name not in FUNCTION_TOOL_REGISTRY:
        raise KeyError(
            f"Function tool '{name}' not found in registry. Make sure its defining "
            f"file is referenced via the rollout `function_tool_path` config."
        )
    return FUNCTION_TOOL_REGISTRY[name]


def load_function_tools_from_path(path: str) -> list[FunctionTool]:
    """Execute a Python file at ``path`` and return its registered function tools."""
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"function_tool_path does not exist: {path}")

    if abs_path in _LOADED_FUNCTION_TOOL_PATHS:
        return _LOADED_FUNCTION_TOOL_PATHS[abs_path]

    before = set(FUNCTION_TOOL_REGISTRY)

    # Use a path-derived synthetic module name so the imported file can
    # ``from X import Y`` its siblings via ``sys.modules``.
    module_name = "_verl_function_tools_" + abs_path.replace(os.sep, "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for function_tool_path: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    new_names = sorted(set(FUNCTION_TOOL_REGISTRY) - before)
    if not new_names:
        logger.warning(
            "function_tool_path '%s' loaded but no @function_tool decorators found; "
            "did you forget to apply the decorator?",
            path,
        )
    else:
        logger.info("Loaded %d function tool(s) from %s: %s", len(new_names), path, new_names)

    tools = [FUNCTION_TOOL_REGISTRY[name] for name in new_names]
    _LOADED_FUNCTION_TOOL_PATHS[abs_path] = tools
    return tools


def _build_schema_from_fn(fn: Callable, tool_name: str) -> OpenAIFunctionToolSchema:
    """Infer the OpenAI tool schema for ``fn`` via transformers.

    The heavy lifting (signature inspection + Google-style docstring parsing
    + JSON-Schema type mapping) is delegated to
    :func:`transformers.utils.get_json_schema`.
    """
    sig = inspect.signature(fn)
    variadic = [
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if variadic:
        raise ValueError(
            f"@function_tool '{tool_name}' ({fn.__module__}.{fn.__qualname__}) "
            f"declares variadic parameter(s) {variadic}, which can't be "
            f"expressed in an OpenAI tool schema. Replace them with explicit "
            f"named parameters."
        )

    raw = get_json_schema(fn)
    raw["function"]["name"] = tool_name
    return OpenAIFunctionToolSchema.model_validate(raw)


def normalize_function_tool_return(ret: Any) -> tuple[ToolResponse, float, dict]:
    """Coerce a function's return value into the ``(ToolResponse, reward, metrics)`` triple.

    Accepted shapes:

    - ``ToolResponse``  -> as-is, reward 0.0, metrics {}
    - ``str``           -> ``ToolResponse(text=ret)``
    - ``dict``          -> ``ToolResponse(text=json.dumps(ret))``
    - ``(response,)`` / ``(response, reward)`` / ``(response, reward, metrics)``
      -- ``reward`` may be ``None`` (treated as ``0.0``); ``metrics`` may be
      ``None`` (treated as ``{}``).
    - anything else     -> ``ToolResponse(text=str(ret))``

    Tuples of length 0 or >= 4 raise ``TypeError`` rather than being silently
    stringified, since they almost always signal a tool authoring bug.
    ``None`` is detected via ``is None`` rather than truthiness, so a
    legitimate ``0`` / ``0.0`` / ``False`` reward is preserved.
    """
    if isinstance(ret, ToolResponse):
        return ret, 0.0, {}
    if isinstance(ret, str):
        return ToolResponse(text=ret), 0.0, {}
    if isinstance(ret, dict):
        return ToolResponse(text=json.dumps(ret, ensure_ascii=False)), 0.0, {}
    if isinstance(ret, tuple):
        if not 1 <= len(ret) <= 3:
            raise TypeError(
                f"@function_tool return tuple must have length 1, 2, or 3 "
                f"(got length {len(ret)}: {ret!r}). Use (response,), "
                f"(response, reward), or (response, reward, metrics)."
            )
        response = _coerce_response(ret[0])
        reward = 0.0 if len(ret) < 2 or ret[1] is None else float(ret[1])
        metrics = {} if len(ret) < 3 or ret[2] is None else dict(ret[2])
        return response, reward, metrics
    return ToolResponse(text=str(ret)), 0.0, {}


def _coerce_response(value: Any) -> ToolResponse:
    if isinstance(value, ToolResponse):
        return value
    if isinstance(value, str):
        return ToolResponse(text=value)
    if isinstance(value, dict):
        return ToolResponse(text=json.dumps(value, ensure_ascii=False))
    return ToolResponse(text=str(value))
