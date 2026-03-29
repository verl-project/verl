# Copyright 2025 Individual Contributor: Muhammad Hashmi
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
"""Daytona-backed code interpreter tool for multi-turn rollout."""

import asyncio
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _build_code_interpreter_schema() -> OpenAIFunctionToolSchema:
    """Return the default OpenAI tool schema for code execution."""
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="code_interpreter",
            description="A tool for executing Python code in a Daytona sandbox.",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "code": OpenAIFunctionPropertySchema(
                        type="string",
                        description="The Python code to execute.",
                    )
                },
                required=["code"],
            ),
        ),
    )


def _load_async_daytona_sdk():
    """Import the async Daytona SDK lazily so the backend stays optional."""
    try:
        from daytona import AsyncDaytona, CreateSandboxFromSnapshotParams, DaytonaConfig
    except ImportError as exc:
        raise ImportError(
            "DaytonaSandboxTool requires the optional 'daytona' dependency. "
            "Install it with `pip install -e '.[daytona]'` or `pip install 'daytona>=0.158.0,<0.159.0'`."
        ) from exc

    return AsyncDaytona, DaytonaConfig, CreateSandboxFromSnapshotParams


def _resolve_daytona_auth(config: dict[str, Any]) -> dict[str, str]:
    """Resolve Daytona auth from config or environment."""
    api_key = config.get("api_key") or os.getenv("DAYTONA_API_KEY")
    jwt_token = config.get("jwt_token") or os.getenv("DAYTONA_JWT_TOKEN")

    if not api_key and not jwt_token:
        raise ValueError("DaytonaSandboxTool requires DAYTONA_API_KEY or DAYTONA_JWT_TOKEN")

    auth = {}
    if api_key:
        auth["api_key"] = api_key
    if jwt_token:
        auth["jwt_token"] = jwt_token
    return auth


class DaytonaSandboxTool(BaseTool):
    """A tool for executing Python code in Daytona cloud sandboxes.

    Each rollout trajectory gets its own isolated Daytona sandbox. ``create``
    provisions that sandbox once and stores it under the trajectory's
    ``instance_id``. Subsequent ``execute`` calls reuse the same sandbox, so
    state inside the container persists across tool invocations within the same
    trajectory. ``release`` deletes the remote sandbox and clears the local
    bookkeeping for that instance.

    The tool keeps a single async Daytona client and optionally applies a
    global semaphore so sandbox creation, code execution, and deletion can be
    throttled across concurrent trajectories. Execution outputs are accumulated
    per instance and returned later from ``calc_reward``.

    - ``get_openai_tool_schema``: return the OpenAI function schema exposed to the model.
    - ``create``: create and register a sandbox for one trajectory.
    - ``execute``: run Python code in the trajectory's existing sandbox.
    - ``calc_reward``: return the collected outputs for that trajectory.
    - ``release``: delete the sandbox and drop local tool state.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None):
        tool_schema = tool_schema or _build_code_interpreter_schema()
        super().__init__(config, tool_schema)

        AsyncDaytona, DaytonaConfig, CreateSandboxFromSnapshotParams = _load_async_daytona_sdk()

        auth_config = _resolve_daytona_auth(config)
        self._create_sandbox_params_cls = CreateSandboxFromSnapshotParams

        # Build the async Daytona client.
        client_config = {}
        for key in ("api_key", "api_url", "target", "jwt_token", "organization_id"):
            value = {**config, **auth_config}.get(key)
            if value is not None:
                client_config[key] = value

        if client_config:
            self._daytona = AsyncDaytona(DaytonaConfig(**client_config))
        else:
            self._daytona = AsyncDaytona()

        self._sandboxes: dict[str, Any] = {}
        self._instance_dict: dict[str, dict] = {}

        # Config.
        self.rate_limit = config.get("rate_limit", 32)
        self.default_timeout = config.get("default_timeout", 30)
        self._create_timeout = config.get("create_timeout", 60)
        self._delete_timeout = config.get("delete_timeout", 60)
        self._auto_stop_interval = config.get("auto_stop_interval", 15)
        self._auto_delete_interval = config.get("auto_delete_interval", 30)
        self._name_prefix = config.get("name_prefix", "verl-daytona")
        self._base_labels = dict(config.get("labels") or {})
        self._snapshot = config.get("snapshot")
        self._language = config.get("language", "python")
        self._env_vars = dict(config.get("env_vars") or {})

        if self._language != "python":
            raise ValueError("DaytonaSandboxTool currently supports only language='python'")

        self._semaphore = asyncio.Semaphore(self.rate_limit) if config.get("enable_global_rate_limit", True) else None

        logger.info(
            "Initialized DaytonaSandboxTool with rate_limit=%s default_timeout=%s",
            self.rate_limit,
            self.default_timeout,
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def _rate_limited(self, coro):
        """Await a coroutine under the rate limiter."""
        if self._semaphore is None:
            return await coro
        async with self._semaphore:
            return await coro

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a Daytona sandbox for one rollout trajectory.

        Spins up an isolated cloud sandbox. The sandbox persists until
        ``release`` is called for this instance_id.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        labels = {
            **self._base_labels,
            "framework": "verl",
            "backend": "daytona",
            "tool": self.name,
            "instance_id": instance_id,
        }
        params = self._create_sandbox_params_cls(
            name=f"{self._name_prefix}-{instance_id[:8]}",
            language=self._language,
            snapshot=self._snapshot,
            env_vars=self._env_vars or None,
            labels=labels,
            auto_stop_interval=self._auto_stop_interval,
            auto_delete_interval=self._auto_delete_interval,
        )

        sandbox = await self._rate_limited(self._daytona.create(params, timeout=self._create_timeout))
        self._sandboxes[instance_id] = sandbox
        self._instance_dict[instance_id] = {"reward": []}
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute Python code inside the existing Daytona sandbox.

        Sends the code to the sandbox's code interpreter and returns combined
        stdout/stderr as the tool response.  Structured error details (name,
        value, traceback) are included in the metrics dict when execution fails.
        """
        code = parameters.get("code", "")
        timeout = parameters.get("timeout", self.default_timeout)
        envs = parameters.get("envs")
        language = parameters.get("language")

        if language is not None and language != "python":
            raise ValueError("DaytonaSandboxTool only supports Python code execution")

        if not isinstance(code, str):
            code = str(code)

        if envs is not None and not isinstance(envs, dict):
            raise ValueError("envs must be a dictionary of string environment variables")

        sandbox = self._sandboxes[instance_id]
        result = await self._rate_limited(sandbox.code_interpreter.run_code(code, timeout=timeout, envs=envs))

        output = result.stdout + result.stderr
        if result.error is not None:
            error_text = f"{result.error.name}: {result.error.value}"
            if result.error.traceback:
                error_text = f"{error_text}\n{result.error.traceback}"
            output = f"{output.rstrip()}\n{error_text}".strip()

        self._instance_dict[instance_id]["reward"].append(output)

        metrics = {
            "sandbox_id": sandbox.id,
            "stdout_chars": len(result.stdout),
            "stderr_chars": len(result.stderr),
            "had_error": result.error is not None,
            "error_name": None if result.error is None else result.error.name,
        }
        return ToolResponse(text=output), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> list[str]:
        """Return the collected tool outputs for the trajectory."""
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        """Delete the Daytona sandbox and clean up local state for this trajectory."""
        sandbox = self._sandboxes.pop(instance_id)
        await self._rate_limited(sandbox.delete(timeout=self._delete_timeout))
        del self._instance_dict[instance_id]

    async def close(self) -> None:
        """Close the underlying Daytona client and its HTTP session."""
        await self._daytona.close()
