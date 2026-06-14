# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import re

import aiohttp
from transformers.utils import get_json_schema

from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema, ToolResponse


class SandboxTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        # Different model may use different code pattern, e.g. python, py, etc.
        self.code_pattern = re.compile(r"```py(.*?)```", re.DOTALL)

    async def code_interpreter(self, code: str) -> str:
        """Execute the code in the sandbox.

        Args:
            code: The code to be executed.

        Returns:
            str: The output of the code execution.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.get("sandbox_fusion_url"),
                json={"code": code},
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()
                stdout, stderr = result["run_result"]["stdout"], result["run_result"]["stderr"]
                return stdout + stderr

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = get_json_schema(self.code_interpreter)
        return OpenAIFunctionToolSchema(**schema)

    async def execute(self, instance_id: str, parameters: dict, **kwargs) -> tuple[str, float, dict]:
        code = parameters["code"]
        matches = self.code_pattern.findall(code)
        if matches:
            code = matches[0].strip()

        # NOTE: Some script may not explicitly print result, we need to add a print statement to the end of the script.
        # More better way is to SFT the model to make it print result by default, we skip SFT stage in this tutorial.
        lines = code.split("\n")
        for i, line in reversed(list(enumerate(lines))):
            if line == "":
                continue
            if not lines[i].startswith("print"):
                lines[i] = f"print({line})"
            break
        code = "\n".join(lines)

        result = await self.code_interpreter(code)
        return ToolResponse(text=result), 0.0, {}
