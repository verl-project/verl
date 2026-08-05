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

import importlib
import logging
import os
import sys
from enum import Enum
from typing import TYPE_CHECKING, Optional

from omegaconf import OmegaConf

from verl.tools.function_tool import FunctionTool, load_function_tools_from_path
from verl.tools.schemas import OpenAIFunctionToolSchema

if TYPE_CHECKING:
    from verl.tools.base_tool import BaseTool

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ToolType(Enum):
    # MCP tool is removed for now.
    NATIVE = "native"


def get_tool_class(cls_name):
    module_name, class_name = cls_name.rsplit(".", 1)
    if module_name not in sys.modules:
        spec = importlib.util.find_spec(module_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = sys.modules[module_name]

    tool_cls = getattr(module, class_name)
    return tool_cls


def initialize_tools_from_config(tools_config_file) -> list:
    """Instantiate ``BaseTool`` subclasses declared in a yaml config."""
    tools_config = OmegaConf.load(tools_config_file)
    tool_list = []

    for tool_config in tools_config.tools:
        cls_name = tool_config.class_name
        tool_type = ToolType(tool_config.config.type)
        tool_cls = get_tool_class(cls_name)

        match tool_type:
            case ToolType.NATIVE:
                if tool_config.get("tool_schema", None) is None:
                    tool_schema = None
                else:
                    tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
                    tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
                tool = tool_cls(
                    config=OmegaConf.to_container(tool_config.config, resolve=True),
                    tool_schema=tool_schema,
                )
                tool_list.append(tool)
            case _:
                raise NotImplementedError(f"Unsupported tool type: {tool_type}")

    return tool_list


def load_all_tools(
    tool_config_path: Optional[str],
    function_tool_path: Optional[str],
) -> list[BaseTool | FunctionTool]:
    """Load native + function tools, check for name collisions, return merged list."""
    native_tools: list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
    function_tools: list[FunctionTool] = load_function_tools_from_path(function_tool_path) if function_tool_path else []

    if function_tools and native_tools:
        existing = {t.name for t in native_tools}
        collisions = sorted(t.name for t in function_tools if t.name in existing)
        if collisions:
            raise ValueError(
                f"Function tool name(s) {collisions} collide with tools already declared in "
                f"'{tool_config_path}'. Each tool name must be unique across `tool_config_path` "
                f"and `function_tool_path`; rename one of them."
            )

    return native_tools + function_tools
