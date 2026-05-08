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
import json
import logging
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import SSETransport

from verl.tools.utils.mcp_clients.utils import TokenBucket, mcp2openai

logger = logging.getLogger(__name__)


class MCPClientManager:
    rootServerName = "mcpServers"
    initialized = False
    clients = []
    tool_client_mapping = {}
    rate_limiter = None

    async def initialize(self, config_path, rate_limit: float = 10.0):
        if self.initialized:
            return
        """Initialize the MCP Client Manager and start all clients"""
        result = self._load_config(config_path)
        servers = result[self.rootServerName]
        exclude_sse_servers = {self.rootServerName: {}}
        for server_name in servers.keys():
            server = servers[server_name]
            if "auth_token" in server:
                transport = SSETransport(url=server["url"], headers={"Authorization": f"Bearer {server['auth_token']}"})
                client = Client(transport)
                self.clients.append(client)
            else:
                exclude_sse_servers[self.rootServerName][server_name] = server

        if exclude_sse_servers[self.rootServerName]:
            self.clients.append(Client(exclude_sse_servers))

        # Initialize rate limiter
        self.rate_limiter = TokenBucket(rate_limit)
        self.initialized = True

    async def call_tool(self, tool_name, parameters, timeout):
        # Apply rate limiting
        while not self.rate_limiter.acquire():
            await asyncio.sleep(0.1)

        client = self.get_client_with_tool_name(tool_name)
        async with client:
            return await asyncio.wait_for(client.call_tool_mcp(tool_name, parameters), timeout=timeout)

    async def fetch_tool_schemas(self, tool_selected_list: list[str]) -> list[dict]:
        tool_schemas = []
        for client in self.clients:
            async with client:
                tools = await client.list_tools_mcp()
                for tool in tools.tools:
                    if not tool_selected_list:
                        self.tool_client_mapping[tool.name] = client
                        tool_schemas.append(mcp2openai(tool))
                    elif tool.name in tool_selected_list:
                        self.tool_client_mapping[tool.name] = client
                        tool_schemas.append(mcp2openai(tool))

        return tool_schemas

    def get_client_with_tool_name(self, tool_name: str):
        return self.tool_client_mapping[tool_name]

    def _load_config(self, file: str) -> dict[str, Any]:
        try:
            with open(file) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f'the "{file}" file was not found')
        except Exception:
            logger.error(f'there was an error reading the "{file}" file')

        return {}


ClientManager = MCPClientManager()
