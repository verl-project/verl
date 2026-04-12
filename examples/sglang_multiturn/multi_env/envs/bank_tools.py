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

"""Bank balance lookup tool — reads account info from a per-instance shared DB.

DB schema: ``{"accounts": {"alice": {"balance": 5000.0}, ...}}``
"""

import json
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


class CheckBalanceTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._dbs: dict[str, dict] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        instance_id = instance_id or str(uuid4())
        self._dbs[instance_id] = kwargs.get("create_kwargs", {}).get("shared_db") or {"accounts": {}}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        acct = parameters.get("account_id", "").strip().lower()
        accounts = self._dbs[instance_id].get("accounts", {})
        if acct not in accounts:
            return ToolResponse(text=json.dumps({"error": f"'{acct}' not found"})), 0.0, {}
        return ToolResponse(text=json.dumps({"account": acct, "balance": accounts[acct].get("balance", 0)})), 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._dbs.pop(instance_id, None)
