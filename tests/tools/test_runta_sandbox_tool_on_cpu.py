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
"""Offline tests for ``examples/runta_sandbox`` against a fake ``runta`` module.

The Runta SDK is an optional dependency that CI does not install, so every test replaces
``sys.modules["runta"]`` and no network is touched.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import hydra
import pytest
from omegaconf import OmegaConf

from examples.runta_sandbox.runta_agent_loop import RuntaToolAgentLoop
from examples.runta_sandbox.runta_tool import RuntaSandboxTool
from verl.experimental.agent_loop.agent_loop import _agent_loop_registry
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.tools.tool_registry import load_all_tools

_EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "runta_sandbox"


@dataclass
class FakeCommandResult:
    exit_code: int = 0
    stdout_text: str = ""
    stderr_text: str = ""


class FakeRuntime:
    def __init__(self, sdk: FakeSdk, name: str, create_kwargs: dict):
        self.sdk = sdk
        self.name = name
        self.create_kwargs = create_kwargs
        self.exec_calls: list[dict] = []
        self.deleted = False

    async def wait_until_ready(self, timeout: float):
        await asyncio.sleep(0)
        if self.sdk.fail_ready:
            raise TimeoutError("runtime did not become ready")

    async def exec(self, command, timeout=None, stdin=None, cwd=None, max_output_bytes=None):
        self.exec_calls.append({"command": command, "timeout": timeout, "stdin": stdin, "cwd": cwd})
        await asyncio.sleep(0)
        return self.sdk.on_exec(stdin)

    async def delete(self):
        self.deleted = True


class FakeSdk:
    """Stands in for the ``runta`` module and records what the tool did with it."""

    def __init__(self):
        self.clients: list = []
        self.runtimes: list[FakeRuntime] = []
        self.fail_ready = False
        self.on_exec = lambda code: FakeCommandResult(stdout_text=f"ran:{code}")

        sdk = self

        class FakeRuntimeManager:
            async def create(self, name, **kwargs):
                await asyncio.sleep(0)
                runtime = FakeRuntime(sdk, name, kwargs)
                sdk.runtimes.append(runtime)
                return runtime

        class AsyncRunta:
            def __init__(self):
                self.runtimes = FakeRuntimeManager()
                self.closed = False
                sdk.clients.append(self)

            async def close(self):
                self.closed = True

        self.module = ModuleType("runta")
        self.module.AsyncRunta = AsyncRunta


@pytest.fixture
def sdk(monkeypatch) -> FakeSdk:
    fake = FakeSdk()
    monkeypatch.setitem(sys.modules, "runta", fake.module)
    return fake


def _make_tool(**config) -> RuntaSandboxTool:
    return RuntaSandboxTool({"type": "native", **config}, None)


def _agent_data(request_id: str) -> SimpleNamespace:
    return SimpleNamespace(request_id=request_id, tools_kwargs={})


async def _call(tool: RuntaSandboxTool, code, agent_data=None):
    """One tool call the way ``ToolAgentLoop._call_tool`` does it: create -> execute -> release."""
    instance_id, _ = await tool.create()
    try:
        return await tool.execute(instance_id, {"code": code}, agent_data=agent_data)
    finally:
        await tool.release(instance_id)


# ---------------------------------------------------------------------------
# Optional dependency + shipped configs
# ---------------------------------------------------------------------------


def test_works_without_sdk_until_first_execution(monkeypatch):
    """Importing the module and building the tool must not need the SDK; running code must say how to get it."""
    monkeypatch.setitem(sys.modules, "runta", None)  # makes `import runta` raise ImportError

    spec = importlib.util.spec_from_file_location("runta_tool_without_sdk", _EXAMPLE_DIR / "runta_tool.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tool = module.RuntaSandboxTool({"type": "native"}, None)
    assert tool.name == "code_interpreter"
    assert tool.tool_schema.function.parameters.required == ["code"]

    with pytest.raises(ImportError, match="pip install 'runta-sdk"):
        asyncio.run(tool.execute("instance", {"code": "print(1)"}))


def test_shipped_tool_config_loads_without_touching_runta(sdk):
    tools = load_all_tools(
        tool_config_path=str(_EXAMPLE_DIR / "config" / "runta_tool_config.yaml"), function_tool_path=None
    )

    assert [type(tool) for tool in tools] == [RuntaSandboxTool]
    assert tools[0].name == "code_interpreter"
    assert tools[0].default_timeout == 30
    assert sdk.clients == []


def test_shipped_agent_loop_config_targets_registered_loop():
    configs = OmegaConf.load(_EXAMPLE_DIR / "config" / "runta_agent_loop_config.yaml")

    assert [config.name for config in configs] == ["runta_tool_agent"]
    assert hydra.utils.get_class(configs[0]._target_) is RuntaToolAgentLoop
    assert _agent_loop_registry["runta_tool_agent"]["_target_"] == configs[0]._target_


def test_unsupported_language_is_rejected():
    with pytest.raises(ValueError, match="Unsupported language"):
        _make_tool(language="ruby")


# ---------------------------------------------------------------------------
# Runtime lifecycle
# ---------------------------------------------------------------------------


def test_one_runtime_per_trajectory_survives_per_call_release(sdk):
    tool = _make_tool(name_prefix="unit", vcpus=2, memory_mib=2048, auto_shutdown_after_secs=120, workdir="/work")

    async def scenario():
        first = _agent_data("trajectory-a")
        await _call(tool, "x = 1", first)
        response, reward, metrics = await _call(tool, "print(x)", first)
        await _call(tool, "other", _agent_data("trajectory-b"))
        return response, reward, metrics

    response, reward, metrics = asyncio.run(scenario())

    assert len(sdk.clients) == 1
    assert len(sdk.runtimes) == 2
    runtime_a, runtime_b = sdk.runtimes
    assert [call["stdin"] for call in runtime_a.exec_calls] == ["x = 1", "print(x)"]
    assert [call["stdin"] for call in runtime_b.exec_calls] == ["other"]
    assert not runtime_a.deleted and not runtime_b.deleted

    assert runtime_a.name.startswith("unit-") and runtime_a.name != runtime_b.name
    assert runtime_a.create_kwargs == {
        "checkpoint_id": None,
        "vcpus": 2,
        "memory_mib": 2048,
        "auto_shutdown_after_secs": 120,
        "auto_delete_after_secs": 300,
        "wait": False,
    }
    assert runtime_a.exec_calls[0] == {"command": ["python3", "-"], "timeout": 30.0, "stdin": "x = 1", "cwd": "/work"}

    assert (response.text, reward, metrics) == ("ran:print(x)", 0.0, {"exit_code": 0, "timed_out": False})


def test_parallel_calls_of_one_trajectory_share_one_runtime(sdk):
    tool = _make_tool()

    async def scenario():
        agent_data = _agent_data("trajectory-a")
        return await asyncio.gather(*(_call(tool, f"job {i}", agent_data) for i in range(4)))

    results = asyncio.run(scenario())

    assert len(sdk.runtimes) == 1
    assert sorted(response.text for response, _, _ in results) == [f"ran:job {i}" for i in range(4)]


def test_release_trajectory_deletes_only_that_runtime(sdk):
    tool = _make_tool()

    async def scenario():
        await _call(tool, "a", _agent_data("trajectory-a"))
        await _call(tool, "b", _agent_data("trajectory-b"))
        released = [
            await tool.release_trajectory("trajectory-a"),
            await tool.release_trajectory("trajectory-a"),
            await tool.release_trajectory("never-used"),
        ]
        # A released trajectory that calls the tool again gets a fresh runtime.
        await _call(tool, "again", _agent_data("trajectory-a"))
        return released

    assert asyncio.run(scenario()) == [True, False, False]
    assert [runtime.deleted for runtime in sdk.runtimes] == [True, False, False]


def test_without_agent_data_release_follows_instance_id(sdk):
    """Used as a plain ``BaseTool`` (no agent loop), the runtime lives from ``create`` to ``release``."""
    tool = _make_tool(language="bash")

    async def scenario():
        instance_id, _ = await tool.create()
        await tool.execute(instance_id, {"code": "echo 1"})
        await tool.execute(instance_id, {"code": "echo 2"})
        await tool.release(instance_id)

    asyncio.run(scenario())

    assert len(sdk.runtimes) == 1
    assert [call["command"] for call in sdk.runtimes[0].exec_calls] == [["bash", "-s"], ["bash", "-s"]]
    assert sdk.runtimes[0].deleted


def test_runtime_that_never_becomes_ready_is_deleted_and_retried(sdk):
    tool = _make_tool()
    agent_data = _agent_data("trajectory-a")

    async def scenario():
        sdk.fail_ready = True
        with pytest.raises(TimeoutError, match="did not become ready"):
            await _call(tool, "print(1)", agent_data)
        sdk.fail_ready = False
        response, _, _ = await _call(tool, "print(1)", agent_data)
        return response

    response = asyncio.run(scenario())

    assert response.text == "ran:print(1)"
    assert [runtime.deleted for runtime in sdk.runtimes] == [True, False]


def test_close_deletes_all_runtimes_and_closes_client(sdk):
    tool = _make_tool()

    async def scenario():
        await _call(tool, "a", _agent_data("trajectory-a"))
        await _call(tool, "b", _agent_data("trajectory-b"))
        await tool.close()

    asyncio.run(scenario())

    assert [runtime.deleted for runtime in sdk.runtimes] == [True, True]
    assert sdk.clients[0].closed


# ---------------------------------------------------------------------------
# What the model sees
# ---------------------------------------------------------------------------


def _hang_forever(code):
    raise TimeoutError()


def _api_down(code):
    raise RuntimeError("HTTP 503: unavailable")


@pytest.mark.parametrize(
    ("on_exec", "expected_text", "expected_metrics"),
    [
        (
            lambda code: FakeCommandResult(0, "4\n", "warning\n"),
            "4\nwarning\n",
            {"exit_code": 0, "timed_out": False},
        ),
        (lambda code: FakeCommandResult(0, "", ""), "(no output)", {"exit_code": 0, "timed_out": False}),
        (
            lambda code: FakeCommandResult(1, "partial\n", "Traceback\nZeroDivisionError: division by zero\n"),
            "partial\nTraceback\nZeroDivisionError: division by zero\n[exit code 1]",
            {"exit_code": 1, "timed_out": False},
        ),
        (lambda code: FakeCommandResult(2, "", ""), "[exit code 2]", {"exit_code": 2, "timed_out": False}),
        # Runta kills the command and reports exit code 124 plus its own notice on stderr.
        (
            lambda code: FakeCommandResult(124, "step 1\n", "Terminated\nCommand timed out after 7.0 seconds.\n"),
            "step 1\nExecution timed out after 7 seconds.",
            {"exit_code": 124, "timed_out": True},
        ),
        # The runtime never answered: the client-side deadline fires instead.
        (_hang_forever, "Execution timed out after 7 seconds.", {"timed_out": True}),
        (_api_down, "Sandbox error: RuntimeError: HTTP 503: unavailable", {}),
    ],
    ids=["ok", "no-output", "nonzero-exit", "nonzero-exit-silent", "timeout", "client-timeout", "sdk-error"],
)
def test_response_text(sdk, on_exec, expected_text, expected_metrics):
    sdk.on_exec = on_exec
    tool = _make_tool(default_timeout=7)

    response, reward, metrics = asyncio.run(_call(tool, "print(2 + 2)", _agent_data("trajectory-a")))

    assert response.text == expected_text
    assert reward == 0.0
    assert metrics == expected_metrics


def test_markdown_fence_is_stripped(sdk):
    tool = _make_tool()

    asyncio.run(_call(tool, "```python\nprint(1)\n```", _agent_data("trajectory-a")))

    assert sdk.runtimes[0].exec_calls[0]["stdin"] == "print(1)\n"


@pytest.mark.parametrize("code", [None, "", "   ", 42])
def test_invalid_code_argument_creates_no_runtime(sdk, code):
    tool = _make_tool()

    response, _, _ = asyncio.run(_call(tool, code, _agent_data("trajectory-a")))

    assert "`code` argument must be a non-empty string" in response.text
    assert sdk.runtimes == []


# ---------------------------------------------------------------------------
# RuntaToolAgentLoop
# ---------------------------------------------------------------------------


def _make_agent_loop(tool: RuntaSandboxTool) -> RuntaToolAgentLoop:
    """A ``RuntaToolAgentLoop`` with only the state ``_call_tool`` and the cleanup need."""
    agent_loop = RuntaToolAgentLoop.__new__(RuntaToolAgentLoop)
    agent_loop.tools = {tool.name: tool}
    agent_loop.max_tool_response_length = 10000
    agent_loop.tool_response_truncate_side = "middle"
    agent_loop._trajectory_ids = set()
    return agent_loop


@pytest.mark.parametrize("fails", [False, True], ids=["finished", "raised"])
def test_agent_loop_deletes_runtime_when_trajectory_ends(sdk, monkeypatch, fails):
    tool = _make_tool()
    agent_loop = _make_agent_loop(tool)
    tool_call = SimpleNamespace(name="code_interpreter", arguments='{"code": "print(1)"}')
    seen = []

    async def fake_run(self, sampling_params, **kwargs):
        """Stands in for the generation state machine: two tool turns, then the end of the trajectory."""
        agent_data = _agent_data("trajectory-a")
        for _ in range(2):
            response, _, _ = await self._call_tool(tool_call, agent_data.tools_kwargs, agent_data)
            seen.append((response.text, sdk.runtimes[0].deleted))
        if fails:
            raise RuntimeError("generation failed")
        return "output"

    monkeypatch.setattr(ToolAgentLoop, "run", fake_run)

    if fails:
        with pytest.raises(RuntimeError, match="generation failed"):
            asyncio.run(agent_loop.run({}))
    else:
        assert asyncio.run(agent_loop.run({})) == "output"

    # Both turns ran in the same, still-alive runtime; it is gone once `run` is over.
    assert seen == [("ran:print(1)", False), ("ran:print(1)", False)]
    assert len(sdk.runtimes) == 1
    assert sdk.runtimes[0].deleted
    assert tool._runtimes == {}


def test_agent_loop_cleanup_failure_does_not_mask_result(sdk, monkeypatch):
    tool = _make_tool()
    agent_loop = _make_agent_loop(tool)
    tool_call = SimpleNamespace(name="code_interpreter", arguments='{"code": "print(1)"}')

    async def fake_run(self, sampling_params, **kwargs):
        agent_data = _agent_data("trajectory-a")
        await self._call_tool(tool_call, agent_data.tools_kwargs, agent_data)
        return "output"

    release_trajectory = tool.release_trajectory

    async def broken_release(trajectory_id):
        # The per-call `release(instance_id)` goes through here too and must keep working.
        if trajectory_id == "trajectory-a":
            raise RuntimeError("delete failed")
        return await release_trajectory(trajectory_id)

    monkeypatch.setattr(ToolAgentLoop, "run", fake_run)
    monkeypatch.setattr(tool, "release_trajectory", broken_release)

    assert asyncio.run(agent_loop.run({})) == "output"
