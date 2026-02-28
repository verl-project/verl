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

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verl.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ToolRegistry:
    """Global registry for tool classes.

    Mirrors the registration pattern used by ``_agent_loop_registry`` and
    ``ToolParser._registry`` elsewhere in the codebase.  Tools can self-register
    via the ``@register_tool`` decorator **or** be resolved at runtime from a
    fully-qualified class name (backward-compatible path).
    """

    _registry: dict[str, type[BaseTool]] = {}

    @classmethod
    def register(cls, tool_name: str):
        """Decorator that registers a ``BaseTool`` subclass under *tool_name*.

        Usage::

            @ToolRegistry.register("gsm8k")
            class Gsm8kTool(BaseTool):
                ...
        """

        def decorator(subclass: type[BaseTool]) -> type[BaseTool]:
            if tool_name in cls._registry:
                existing = cls._registry[tool_name]
                fqdn_existing = f"{existing.__module__}.{existing.__qualname__}"
                fqdn_new = f"{subclass.__module__}.{subclass.__qualname__}"
                logger.warning(
                    f"Tool '{tool_name}' is already registered as {fqdn_existing}. Overwriting with {fqdn_new}."
                )
            cls._registry[tool_name] = subclass
            return subclass

        return decorator

    @classmethod
    def get(cls, tool_name: str) -> type[BaseTool]:
        """Return the tool class registered under *tool_name*.

        Raises ``KeyError`` if *tool_name* has not been registered.
        """
        if tool_name not in cls._registry:
            raise KeyError(f"Tool '{tool_name}' is not registered. Available tools: {list(cls._registry.keys())}")
        return cls._registry[tool_name]

    @classmethod
    def list_tools(cls) -> list[str]:
        """Return the names of all registered tools."""
        return list(cls._registry.keys())

    @classmethod
    def contains(cls, tool_name: str) -> bool:
        """Check whether *tool_name* is registered."""
        return tool_name in cls._registry

    @classmethod
    def clear(cls) -> None:
        """Remove all entries. Intended for testing only."""
        cls._registry.clear()


# Convenience alias so users can write ``@register_tool("name")``
register_tool = ToolRegistry.register
