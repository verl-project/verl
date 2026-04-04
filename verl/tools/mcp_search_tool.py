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

import json
import logging
import os
import re

from verl.tools.mcp_base_tool import MCPBaseTool
from verl.tools.registry import register_tool

from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register_tool("mcp_search")
class MCPSearchTool(MCPBaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    def _parse_tool_result(self, content: list) -> tuple[str, dict]:
        res = ""
        res_cnt = 0
        query_list = []
        metadata = {
            "api_request_error": "",
            "status": "unknown",
            "total_results": 0,
        }
        try:
            for part in content:
                if part.type != "text":
                    continue
                text = part.text.replace("'", '"')
                query_match = re.search(r'query"\s*:\s*"([^"]+)"', text)
                query = query_match.group(1) if query_match else ""
                query_list.append(query)

                title_matches = re.findall(r'"title"\s*:', text)
                title_count = len(title_matches)

                results_match = re.search(r'"results"\s*:\s*(\[.*?\])', text, re.DOTALL)
                results_content = results_match.group(1) if results_match else ""

                res += results_content
                res_cnt += title_count
        except json.JSONDecodeError:
            err_msg = "json parse error."
            logger.error(err_msg)
            metadata["api_request_error"] = err_msg
            metadata["status"] = "error"

        # update metadata
        metadata["status"] = "success"
        metadata["queries"] = query_list
        metadata["query_count"] = len(query_list)
        metadata["total_results"] = res_cnt
        return res, metadata
