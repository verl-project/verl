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

import asyncio
import concurrent.futures
import copy
import importlib
import logging
import os
import sys
import threading
import time
from enum import Enum

from omegaconf import OmegaConf

from verl.tools.schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ToolType(Enum):
    NATIVE = "native"
    MCP = "mcp"


async def initialize_mcp_tool(tool_cls, tool_config) -> list:
    from verl.tools.utils.mcp_clients.McpClientManager import ClientManager

    tool_list = []
    mcp_servers_config_path = tool_config.mcp.mcp_servers_config_path
    tool_selected_list = tool_config.mcp.tool_selected_list if "tool_selected_list" in tool_config.mcp else None
    await ClientManager.initialize(mcp_servers_config_path, tool_config.config.rate_limit)
    # Wait for MCP client to be ready
    max_retries = 10
    retry_interval = 2  # seconds
    for i in range(max_retries):
        tool_schemas = await ClientManager.fetch_tool_schemas(tool_selected_list)
        if tool_schemas:
            break
        if i < max_retries - 1:
            logger.debug(f"Waiting for MCP client to be ready, attempt {i + 1}/{max_retries}")
            await asyncio.sleep(retry_interval)
    else:
        raise RuntimeError("Failed to initialize MCP tools after maximum retries")
    # mcp registry
    assert len(tool_schemas), "mcp tool is empty"
    for tool_schema_dict in tool_schemas:
        logger.debug(f"tool_schema_dict: {tool_schema_dict}")
        tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
        tool = tool_cls(
            config=OmegaConf.to_container(tool_config.config, resolve=True),
            tool_schema=tool_schema,
        )
        tool_list.append(tool)
    return tool_list


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


def initialize_tools_from_config(tools_config_file, mcp_init_timeout_s: float | None = None):
    """Initialize tools from config file.

    Supports both NATIVE and MCP tool types. For MCP tools, a temporary event loop
    is created only when needed and properly closed after use to prevent memory leaks.
    """
    tools_config = OmegaConf.load(tools_config_file)
    tool_list = []

    # Lazy initialization for MCP support - only create event loop when needed
    tmp_event_loop = None
    thread = None

    def get_mcp_event_loop():
        """Lazily create event loop and thread for MCP tools."""
        nonlocal tmp_event_loop, thread
        if tmp_event_loop is None:
            tmp_event_loop = asyncio.new_event_loop()
            thread = threading.Thread(target=tmp_event_loop.run_forever, name="mcp tool list fetcher", daemon=True)
            thread.start()
        return tmp_event_loop

    def run_coroutine(coroutine):
        """Run coroutine in the MCP event loop."""
        loop = get_mcp_event_loop()
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        try:
            return future.result(timeout=mcp_init_timeout_s)
        except concurrent.futures.TimeoutError as e:
            future.cancel()
            raise TimeoutError(
                f"MCP tool initialization timed out after {mcp_init_timeout_s}s for config {tools_config_file}"
            ) from e

    try:
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
                case ToolType.MCP:
                    mcp_tools = run_coroutine(initialize_mcp_tool(tool_cls, tool_config))
                    tool_list.extend(mcp_tools)
                case _:
                    raise NotImplementedError
    finally:
        # Properly cleanup event loop if it was created
        if tmp_event_loop is not None:
            # stop first and then close
            tmp_event_loop.call_soon_threadsafe(tmp_event_loop.stop)
            if thread is not None and thread.is_alive():
                thread.join(timeout=5.0)
            tmp_event_loop.close()

    return tool_list


class ToolListCache:
    """Process-local cache for initialized tools keyed by tool config path."""

    _cache: dict[str, list] = {}
    _lock = threading.Lock()

    @classmethod
    def _get_init_policy(cls) -> tuple[int, float, float, float]:
        retries = int(os.getenv("VERL_TOOL_CACHE_INIT_RETRIES", "3"))
        timeout_s = float(os.getenv("VERL_TOOL_CACHE_INIT_TIMEOUT_S", "30"))
        backoff_base_s = float(os.getenv("VERL_TOOL_CACHE_INIT_BACKOFF_BASE_S", "1"))
        backoff_max_s = float(os.getenv("VERL_TOOL_CACHE_INIT_BACKOFF_MAX_S", "30"))
        return max(1, retries), max(0.1, timeout_s), max(0.0, backoff_base_s), max(0.0, backoff_max_s)

    @classmethod
    def get_tool_list(cls, tools_config_file: str | None) -> list:
        if not tools_config_file:
            return []

        with cls._lock:
            cached = cls._cache.get(tools_config_file)
            if cached is None:
                retries, timeout_s, backoff_base_s, backoff_max_s = cls._get_init_policy()
                logger.debug(
                    "[ToolListCache] cache miss for %s, initializing tools (retries=%s, timeout_s=%.2f)",
                    tools_config_file,
                    retries,
                    timeout_s,
                )
                last_error = None
                for attempt in range(1, retries + 1):
                    try:
                        cached = initialize_tools_from_config(tools_config_file, mcp_init_timeout_s=timeout_s)
                        break
                    except Exception as e:
                        last_error = e
                        if attempt >= retries:
                            raise RuntimeError(
                                f"[ToolListCache] failed to initialize tools for {tools_config_file} "
                                f"after {retries} attempts"
                            ) from e

                        backoff_s = min(backoff_max_s, backoff_base_s * (2 ** (attempt - 1)))
                        logger.warning(
                            "[ToolListCache] init attempt %s/%s failed for %s: %s. Retrying in %.2fs",
                            attempt,
                            retries,
                            tools_config_file,
                            repr(e),
                            backoff_s,
                        )
                        if backoff_s > 0:
                            time.sleep(backoff_s)

                if cached is None:
                    # Defensive fallback: should be unreachable due to RuntimeError above.
                    raise RuntimeError(
                        f"[ToolListCache] initialization produced no tool list for {tools_config_file}"
                    ) from last_error
                cls._cache[tools_config_file] = cached
            else:
                logger.debug(f"[ToolListCache] cache hit for {tools_config_file}")

            return copy.deepcopy(cached)
