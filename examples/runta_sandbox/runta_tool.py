# Copyright 2026 Individual Contributor
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
"""Code execution tool backed by `Runta <https://pypi.org/project/runta-sdk/>`_ sandbox runtimes.

``ToolAgentLoop`` calls ``create -> execute -> release`` around every single tool call, so those
hooks cannot own a sandbox that has to survive the turns of one trajectory. This tool keys its
runtimes by trajectory instead (``agent_data.request_id``): the first call of a trajectory creates
the runtime, later calls reuse it, and :meth:`RuntaSandboxTool.release_trajectory` deletes it.
``RuntaToolAgentLoop`` (``runta_agent_loop.py``) calls that method when the trajectory ends.

The Runta SDK is an optional dependency and is imported on first use::

    pip install "runta-sdk>=0.2.6,<0.3"

Credentials are read by the SDK from ``RUNTA_TOKEN`` or ``~/.config/runta/config.toml``.
"""

import asyncio
import logging
import os
import re
from typing import Any
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# How each supported language reads the model-written source from stdin.
_INTERPRETERS = {
    "python": ["python3", "-"],
    "bash": ["bash", "-s"],
}

# Runta kills a command that exceeds its timeout and reports this exit code (as `timeout(1)` does).
_TIMEOUT_EXIT_CODE = 124

# Extra seconds granted on top of the command timeout before the call is abandoned client side.
_EXEC_GRACE_SECS = 10.0


def _load_async_runta():
    """Import the Runta SDK lazily so this module loads without it."""
    try:
        from runta import AsyncRunta
    except ImportError as e:
        raise ImportError(
            "RuntaSandboxTool requires the Runta SDK. Install it with: pip install 'runta-sdk>=0.2.6,<0.3'"
        ) from e
    return AsyncRunta


class RuntaSandboxTool(BaseTool):
    """Run model-written code in a Runta runtime that lives for one trajectory.

    Config keys (all optional except ``type``):

    - ``language``: ``python`` (default) or ``bash``.
    - ``default_timeout``: seconds allowed per execution. Default 30.
    - ``max_output_bytes``: bytes kept per output stream. Default 65536.
    - ``workdir``: working directory inside the runtime. Default: the runtime's own.
    - ``name_prefix``: runtime names are ``<name_prefix>-<8 hex chars>``. Default ``verl-sandbox``.
    - ``vcpus`` / ``memory_mib``: runtime size. Default 1 / 1024.
    - ``checkpoint_id``: create runtimes from this Runta checkpoint, e.g. one with task
      dependencies preinstalled. Default: the base runtime.
    - ``create_timeout``: seconds to wait for a new runtime to become ready. Default 240.
    - ``auto_shutdown_after_secs`` / ``auto_delete_after_secs``: server-side lifetime limits that
      reclaim a runtime if the process dies before deleting it. Default 3600 / 300.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None = None):
        super().__init__(config, tool_schema)
        self.language = config.get("language", "python")
        if self.language not in _INTERPRETERS:
            raise ValueError(f"Unsupported language {self.language!r}, expected one of {sorted(_INTERPRETERS)}")
        self.default_timeout = float(config.get("default_timeout", 30))
        self.max_output_bytes = config.get("max_output_bytes", 65536)
        self.workdir = config.get("workdir")
        self.name_prefix = config.get("name_prefix", "verl-sandbox")
        self.vcpus = int(config.get("vcpus", 1))
        self.memory_mib = int(config.get("memory_mib", 1024))
        self.checkpoint_id = config.get("checkpoint_id")
        self.create_timeout = float(config.get("create_timeout", 240))
        self.auto_shutdown_after_secs = int(config.get("auto_shutdown_after_secs", 3600))
        self.auto_delete_after_secs = int(config.get("auto_delete_after_secs", 300))

        # Models often wrap the source in a markdown fence even inside a tool call.
        self.code_pattern = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)

        self._client = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        # trajectory id -> task that creates the runtime. Storing the task instead of the runtime
        # makes parallel tool calls of one trajectory share a single creation.
        self._runtimes: dict[str, asyncio.Task] = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Schema used when the tool config does not provide ``tool_schema``."""
        language = self.config.get("language", "python")
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="code_interpreter",
                description=(
                    f"Execute {language} code in a sandbox and return its stdout and stderr. "
                    "Files persist between calls, interpreter state does not."
                ),
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "code": OpenAIFunctionPropertySchema(type="string", description="The code to execute."),
                    },
                    required=["code"],
                ),
            ),
        )

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Run ``parameters["code"]`` in the runtime of the calling trajectory.

        Args:
            instance_id: The instance id returned by ``create``. Used as the trajectory id only
                when no ``agent_data`` is passed.
            parameters: Tool call arguments; ``code`` holds the source to run.
            **kwargs: ``agent_data`` is the ``AgentData`` of the trajectory, passed by ``ToolAgentLoop``.

        Returns:
            The combined stdout/stderr as a text ``ToolResponse``, a zero step reward, and
            ``exit_code`` / ``timed_out`` metrics.
        """
        code = parameters.get("code")
        if not isinstance(code, str) or not code.strip():
            return ToolResponse(text="Error: the `code` argument must be a non-empty string."), 0.0, {}
        matches = self.code_pattern.findall(code)
        if matches:
            code = matches[0]

        agent_data = kwargs.get("agent_data")
        trajectory_id = getattr(agent_data, "request_id", None) or instance_id
        runtime = await self._get_runtime(trajectory_id)

        timeout = self.default_timeout
        timeout_text = f"Execution timed out after {timeout:g} seconds."
        try:
            result = await asyncio.wait_for(
                runtime.exec(
                    _INTERPRETERS[self.language],
                    timeout=timeout,
                    stdin=code,
                    cwd=self.workdir,
                    max_output_bytes=self.max_output_bytes,
                ),
                timeout=timeout + _EXEC_GRACE_SECS,
            )
        except (asyncio.TimeoutError, TimeoutError):
            # The call itself never returned. A command killed by the runtime's own timeout
            # comes back as a normal result and is handled below.
            return ToolResponse(text=timeout_text), 0.0, {"timed_out": True}
        except Exception as e:
            logger.warning(f"Runta exec failed for trajectory {trajectory_id}: {e}")
            return ToolResponse(text=f"Sandbox error: {type(e).__name__}: {e}"), 0.0, {}

        metrics = {"exit_code": result.exit_code, "timed_out": result.exit_code == _TIMEOUT_EXIT_CODE}
        if metrics["timed_out"]:
            # Keep what was printed before the kill, drop the runtime's own timeout notice.
            return ToolResponse(text=result.stdout_text + timeout_text), 0.0, metrics
        text = result.stdout_text + result.stderr_text
        if result.exit_code != 0:
            text = f"{text.rstrip()}\n[exit code {result.exit_code}]".lstrip()
        return ToolResponse(text=text or "(no output)"), 0.0, metrics

    async def release(self, instance_id: str, **kwargs) -> None:
        """Delete the runtime keyed by ``instance_id``, if any.

        Runtimes created through ``ToolAgentLoop`` are keyed by ``agent_data.request_id``, so the
        per-call ``release`` of the agent loop does not touch them.
        """
        await self.release_trajectory(instance_id)

    async def release_trajectory(self, trajectory_id: str) -> bool:
        """Delete the runtime of a trajectory. Safe to call when it never created one.

        Args:
            trajectory_id: ``agent_data.request_id`` of the trajectory.

        Returns:
            True if a runtime was deleted.
        """
        task = self._runtimes.pop(trajectory_id, None)
        if task is None:
            return False
        try:
            runtime = await task
        except Exception:
            # Creation failed, and `_create_runtime` already cleaned up after itself.
            return False
        try:
            await runtime.delete()
        except Exception as e:
            logger.warning(f"Failed to delete Runta runtime of trajectory {trajectory_id}: {e}")
            return False
        return True

    async def close(self) -> None:
        """Delete every runtime still held and close the SDK client.

        Training never needs this: ``RuntaToolAgentLoop`` releases per trajectory. It is for
        scripts and notebooks that drive the tool directly.
        """
        for trajectory_id in list(self._runtimes):
            await self.release_trajectory(trajectory_id)
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def _get_runtime(self, trajectory_id: str):
        task = self._runtimes.get(trajectory_id)
        if task is None:
            task = asyncio.ensure_future(self._create_runtime())
            self._runtimes[trajectory_id] = task
        try:
            # Shielded so that a cancelled tool call does not abort a creation that
            # `release_trajectory` still has to see in order to delete the runtime.
            return await asyncio.shield(task)
        except Exception:
            # Let the next tool call of this trajectory retry with a fresh runtime.
            if self._runtimes.get(trajectory_id) is task:
                del self._runtimes[trajectory_id]
            raise

    async def _create_runtime(self):
        client = self._get_client()
        runtime = await client.runtimes.create(
            f"{self.name_prefix}-{uuid4().hex[:8]}",
            checkpoint_id=self.checkpoint_id,
            vcpus=self.vcpus,
            memory_mib=self.memory_mib,
            auto_shutdown_after_secs=self.auto_shutdown_after_secs,
            auto_delete_after_secs=self.auto_delete_after_secs,
            wait=False,
        )
        # Waiting separately keeps a handle on the runtime, so one that never becomes ready can
        # be deleted right away instead of lingering until the server-side limits reclaim it.
        try:
            await runtime.wait_until_ready(timeout=self.create_timeout)
        except BaseException:
            try:
                await runtime.delete()
            except Exception as e:
                logger.warning(f"Failed to delete Runta runtime that did not become ready: {e}")
            raise
        return runtime

    def _get_client(self):
        # AsyncRunta is bound to the event loop it was created on. A worker has a single loop,
        # but a new one (e.g. consecutive `asyncio.run` calls) needs a new client.
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            self._client = _load_async_runta()()
            self._client_loop = loop
        return self._client
