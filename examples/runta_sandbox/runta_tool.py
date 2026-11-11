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
import time
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

# Runta kills a command that exceeds its timeout, reports this exit code (as `timeout(1)` does)
# and appends `_RUNTA_TIMEOUT_NOTICE` to stderr. `CommandResult` has no timed-out flag, and a
# program can exit with 124 on its own, so the exit code only counts as a timeout when the
# command also ran for about the configured timeout.
_TIMEOUT_EXIT_CODE = 124
_TIMEOUT_MIN_FRACTION = 0.95
_RUNTA_TIMEOUT_NOTICE = re.compile(r"Command timed out after \S+ seconds\.\Z")

# Extra seconds granted on top of the command timeout before the call is abandoned client side.
_EXEC_GRACE_SECS = 10.0

# Seconds allowed for asking Runta whether a runtime whose command failed is still running.
_STATUS_CHECK_SECS = 15.0

# Runtime states that never run a command again. A runtime ends up `shutdown` when it reaches
# `auto_shutdown_after_secs`, which limits the running session, not the idle time.
_DEAD_STATUSES = frozenset({"shutdown", "crashed", "error", "deleting"})

# The whole input is one markdown fence: an opening line, a body, a closing line.
_FENCED_INPUT = re.compile(r"\s*```[\w+-]*[ \t]*\n(.*?)^[ \t]*```\s*", re.DOTALL | re.MULTILINE)
_FENCE_LINE = re.compile(r"^[ \t]*```", re.MULTILINE)


def _load_async_runta():
    """Import the Runta SDK lazily so this module loads without it."""
    try:
        from runta import AsyncRunta
    except ImportError as e:
        raise ImportError(
            "RuntaSandboxTool requires the Runta SDK. Install it with: pip install 'runta-sdk>=0.2.6,<0.3'"
        ) from e
    return AsyncRunta


def _strip_fence(code: str) -> str:
    """Unwrap ``code`` if all of it is a single markdown fenced block, else return it unchanged.

    Code that merely contains a fence, e.g. inside a string literal, must run as written.
    """
    match = _FENCED_INPUT.fullmatch(code)
    if match is None or _FENCE_LINE.search(match.group(1)):
        return code
    return match.group(1)


def _account_failure_cause(error: BaseException | None) -> str | None:
    """Name the cause if ``error`` is an authentication or quota failure, which no retry fixes."""
    for _ in range(5):  # the SDK wraps the HTTP / WebSocket error that carries the status
        if error is None:
            break
        if any(cls.__name__ == "ConfigError" for cls in type(error).__mro__):
            return "Runta credentials are missing or unreadable (RUNTA_TOKEN or ~/.config/runta/config.toml)"
        status = getattr(error, "status_code", None) or getattr(getattr(error, "response", None), "status_code", None)
        code = str(getattr(error, "code", None) or "")
        if status == 401 or code == "unauthenticated":
            return "Runta authentication failed: the token is invalid or expired"
        if status == 403 or code == "permission_denied":
            return "Runta denied permission to this token"
        if status == 429 or code == "rate_limited":
            return "Runta quota or rate limit exceeded"
        error = error.__cause__
    return None


class RuntaSandboxTool(BaseTool):
    """Run model-written code in a Runta runtime that lives for one trajectory.

    Config keys (all optional except ``type``):

    - ``language``: ``python`` (default) or ``bash``.
    - ``default_timeout``: seconds allowed per execution. Default 30.
    - ``max_output_bytes``: bytes kept per output stream. Default 65536.
    - ``workdir``: working directory inside the runtime. Default: the runtime's own.
    - ``name_prefix``: runtime names are ``<name_prefix>-<8 hex chars>``. Default ``verl-sandbox``.
    - ``vcpus`` / ``memory_mib``: size of a runtime created from the base runtime. Default
      1 / 1024. Ignored when ``checkpoint_id`` is set.
    - ``checkpoint_id``: create runtimes from this Runta checkpoint, e.g. one with task
      dependencies preinstalled. Such a runtime keeps the size stored in the checkpoint.
      Default: the base runtime.
    - ``create_timeout``: seconds to wait for a new runtime to become ready. Default 240.
    - ``auto_shutdown_after_secs`` / ``auto_delete_after_secs``: server-side lifetime limits that
      reclaim a runtime if the process dies before deleting it. Default 3600 / 300.
      ``auto_shutdown_after_secs`` counts from the start of the runtime, not from its last use.

    A runtime that crashed or was shut down is replaced by a fresh one on the next call, so the
    files of the trajectory are lost at that point.
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
        if self.checkpoint_id is not None and ("vcpus" in config or "memory_mib" in config):
            logger.warning(
                "`vcpus` / `memory_mib` are ignored because `checkpoint_id` is set: a runtime created "
                "from a checkpoint keeps the checkpoint's resources."
            )

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
            ``exit_code`` / ``timed_out`` metrics. ``timed_out`` is set only when the runtime
            killed the program at ``default_timeout``; a call that got no answer from Runta
            reports ``sandbox_timed_out`` instead.
        """
        code = parameters.get("code")
        if not isinstance(code, str) or not code.strip():
            return ToolResponse(text="Error: the `code` argument must be a non-empty string."), 0.0, {}
        # Models often wrap the source in a markdown fence even inside a tool call.
        code = _strip_fence(code)

        agent_data = kwargs.get("agent_data")
        trajectory_id = getattr(agent_data, "request_id", None) or instance_id
        timeout = self.default_timeout

        retried = False
        while True:
            try:
                runtime = await self._get_runtime(trajectory_id)
            except Exception as e:
                # The caller (`ToolAgentLoop`) turns this into tool text and logs a warning.
                self._log_failure("could not create a runtime", trajectory_id, e, otherwise=None)
                raise

            started = time.monotonic()
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
                # The call itself never returned (slow connection, SDK retries, hung runtime).
                # A command killed by the runtime's own timeout comes back as a normal result.
                waited = timeout + _EXEC_GRACE_SECS
                logger.warning(f"Runta exec got no response within {waited:g}s for trajectory {trajectory_id}")
                text = (
                    f"Sandbox error: no response from the sandbox within {waited:g} seconds. This is a sandbox "
                    "or network problem, not a timeout of the program, which may not have run or finished."
                )
                return ToolResponse(text=text), 0.0, {"sandbox_timed_out": True}
            except Exception as e:
                dead_status = await self._dead_status(runtime)
                if dead_status is not None:
                    await self._evict_runtime(trajectory_id, runtime, dead_status)
                    if not retried:
                        # The program did not run to completion, so run it once on a fresh runtime.
                        retried = True
                        continue
                self._log_failure("exec failed", trajectory_id, e, otherwise=logging.WARNING)
                return ToolResponse(text=f"Sandbox error: {type(e).__name__}: {e}"), 0.0, {}
            elapsed = time.monotonic() - started
            break

        # The SDK reports how long the command itself ran, which excludes connection time.
        duration = getattr(result, "duration", None)
        ran_for = min(elapsed, duration) if isinstance(duration, int | float) else elapsed
        timed_out = result.exit_code == _TIMEOUT_EXIT_CODE and ran_for >= timeout * _TIMEOUT_MIN_FRACTION
        metrics = {"exit_code": result.exit_code, "timed_out": timed_out}
        if timed_out:
            # Keep everything the program printed; only the runtime's own notice is replaced.
            text = result.stdout_text + _RUNTA_TIMEOUT_NOTICE.sub("", result.stderr_text)
            if text and not text.endswith("\n"):
                text += "\n"
            return ToolResponse(text=f"{text}Execution timed out after {timeout:g} seconds."), 0.0, metrics
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

    async def _dead_status(self, runtime) -> str | None:
        """Return why ``runtime`` can no longer run commands, or None if that is not established."""
        try:
            info = await asyncio.wait_for(runtime.refresh_info(), timeout=_STATUS_CHECK_SECS)
        except Exception as e:
            return "deleted" if getattr(e, "status_code", None) == 404 else None
        status = getattr(info, "status", None)
        status = str(getattr(status, "value", status))
        return status if status in _DEAD_STATUSES else None

    async def _evict_runtime(self, trajectory_id: str, runtime, dead_status: str) -> None:
        """Forget and delete a dead runtime so the next ``_get_runtime`` creates a fresh one."""
        task = self._runtimes.get(trajectory_id)
        owned = task is not None and task.done() and not task.cancelled() and task.exception() is None
        if not owned or task.result() is not runtime:
            # A parallel tool call of this trajectory already replaced it.
            return
        del self._runtimes[trajectory_id]
        logger.warning(
            f"Runta runtime of trajectory {trajectory_id} is {dead_status}; replacing it with a fresh one. "
            "Files written earlier in the trajectory are lost."
        )
        try:
            await runtime.delete()
        except Exception as e:
            logger.warning(f"Failed to delete dead Runta runtime of trajectory {trajectory_id}: {e}")

    def _log_failure(self, what: str, trajectory_id: str, error: Exception, otherwise: int | None) -> None:
        """Log authentication / quota failures at ERROR, anything else at level ``otherwise``."""
        cause = _account_failure_cause(error)
        detail = f"{type(error).__name__}: {error}"
        if cause is not None:
            logger.error(
                f"{cause}. Runta {what} for trajectory {trajectory_id}; sandbox calls keep failing while "
                f"this lasts, and training continues on the failed tool calls. ({detail})"
            )
        elif otherwise is not None:
            logger.log(otherwise, f"Runta {what} for trajectory {trajectory_id}: {detail}")

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
