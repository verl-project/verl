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
import logging
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
    duration: float | None = None  # seconds the command ran inside the runtime, when reported


class FakeApiError(Exception):
    """Shaped like ``runta.ApiError``: an HTTP status plus a Runta error code."""

    def __init__(self, message: str, status_code: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class FakeRuntime:
    def __init__(self, sdk: FakeSdk, name: str, create_kwargs: dict):
        self.sdk = sdk
        self.name = name
        self.create_kwargs = create_kwargs
        self.exec_calls: list[dict] = []
        self.deleted = False
        self.status = "running"

    def kill(self, status: str = "shutdown"):
        """What the tool sees once Runta shut the runtime down, it crashed, or it was deleted."""
        self.status = status

    async def refresh_info(self):
        await asyncio.sleep(0)
        if self.status == "deleted":
            raise FakeApiError("runtime not found", status_code=404, code="not_found")
        return SimpleNamespace(status=self.status)

    async def wait_until_ready(self, timeout: float):
        await asyncio.sleep(0)
        if self.sdk.fail_ready:
            raise TimeoutError("runtime did not become ready")

    async def exec(self, command, timeout=None, stdin=None, cwd=None, max_output_bytes=None):
        self.exec_calls.append({"command": command, "timeout": timeout, "stdin": stdin, "cwd": cwd})
        await asyncio.sleep(self.sdk.exec_secs)
        if self.status != "running":
            raise FakeApiError(f"runtime is {self.status}")
        return self.sdk.on_exec(stdin)

    async def delete(self):
        self.deleted = True


class FakeSdk:
    """Stands in for the ``runta`` module and records what the tool did with it."""

    def __init__(self):
        self.clients: list = []
        self.runtimes: list[FakeRuntime] = []
        self.fail_ready = False
        self.fail_create: Exception | None = None
        self.exec_secs = 0.0  # wall-clock time every `exec` takes
        self.on_exec = lambda code: FakeCommandResult(stdout_text=f"ran:{code}")
        self.on_create = lambda runtime: None

        sdk = self

        class FakeRuntimeManager:
            async def create(self, name, **kwargs):
                await asyncio.sleep(0)
                if sdk.fail_create is not None:
                    raise sdk.fail_create
                runtime = FakeRuntime(sdk, name, kwargs)
                sdk.on_create(runtime)
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


@pytest.mark.parametrize("status", ["shutdown", "crashed", "deleted"])
def test_dead_runtime_is_replaced_and_the_command_retried(sdk, caplog, status):
    """E.g. the runtime reached ``auto_shutdown_after_secs``: the trajectory must not be stuck with it."""
    tool = _make_tool()
    agent_data = _agent_data("trajectory-a")

    async def scenario():
        await _call(tool, "first", agent_data)
        sdk.runtimes[0].kill(status)
        with caplog.at_level(logging.WARNING):
            second = await _call(tool, "second", agent_data)
        third = await _call(tool, "third", agent_data)
        released = await tool.release_trajectory("trajectory-a")
        return second, third, released

    second, third, released = asyncio.run(scenario())

    dead, fresh = sdk.runtimes
    assert [call["stdin"] for call in dead.exec_calls] == ["first", "second"]
    assert [call["stdin"] for call in fresh.exec_calls] == ["second", "third"]
    assert (second[0].text, second[2]) == ("ran:second", {"exit_code": 0, "timed_out": False})
    assert third[0].text == "ran:third"
    assert f"is {status}; replacing it" in caplog.text
    # The dead runtime was deleted on eviction, the fresh one when the trajectory was released.
    assert released and dead.deleted and fresh.deleted
    assert tool._runtimes == {}


def test_runtime_that_keeps_dying_is_replaced_once_per_call(sdk):
    """A program that takes its runtime down must not be rerun forever."""
    sdk.on_create = lambda runtime: runtime.kill("crashed")
    tool = _make_tool()
    agent_data = _agent_data("trajectory-a")

    async def scenario():
        failed = await _call(tool, "boom", agent_data)
        sdk.on_create = lambda runtime: None
        recovered = await _call(tool, "print(1)", agent_data)
        return failed, recovered

    failed, recovered = asyncio.run(scenario())

    assert failed[0].text == "Sandbox error: FakeApiError: runtime is crashed"
    assert recovered[0].text == "ran:print(1)"
    assert [runtime.deleted for runtime in sdk.runtimes] == [True, True, False]
    assert [[call["stdin"] for call in runtime.exec_calls] for runtime in sdk.runtimes] == [
        ["boom"],
        ["boom"],
        ["print(1)"],
    ]


def test_parallel_calls_on_a_dead_runtime_share_one_replacement(sdk):
    tool = _make_tool()
    agent_data = _agent_data("trajectory-a")

    async def scenario():
        await _call(tool, "first", agent_data)
        sdk.runtimes[0].kill()
        return await asyncio.gather(*(_call(tool, f"job {i}", agent_data) for i in range(3)))

    results = asyncio.run(scenario())

    assert sorted(response.text for response, _, _ in results) == [f"ran:job {i}" for i in range(3)]
    assert [runtime.deleted for runtime in sdk.runtimes] == [True, False]


def test_exec_error_on_a_running_runtime_keeps_the_runtime(sdk):
    """A failed call alone is no reason to throw away the files of the trajectory."""
    tool = _make_tool()
    agent_data = _agent_data("trajectory-a")

    async def scenario():
        sdk.on_exec = _api_down
        failed = await _call(tool, "first", agent_data)
        sdk.on_exec = lambda code: FakeCommandResult(stdout_text=f"ran:{code}")
        return failed, await _call(tool, "second", agent_data)

    failed, recovered = asyncio.run(scenario())

    assert failed[0].text == "Sandbox error: RuntimeError: HTTP 503: unavailable"
    assert recovered[0].text == "ran:second"
    assert len(sdk.runtimes) == 1 and not sdk.runtimes[0].deleted


def test_checkpoint_with_explicit_size_warns_that_the_size_is_ignored(sdk, caplog):
    with caplog.at_level(logging.WARNING):
        _make_tool(checkpoint_id="ckpt-1")
        assert caplog.text == ""
        _make_tool(vcpus=4)
        assert caplog.text == ""
        _make_tool(checkpoint_id="ckpt-1", vcpus=4)

    assert "`vcpus` / `memory_mib` are ignored because `checkpoint_id` is set" in caplog.text


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
        # The program itself exits with 124 (`sys.exit(124)`, bash `timeout 1 cmd`) long before the
        # tool's timeout: an ordinary failure whose stderr the model must see.
        (
            lambda code: FakeCommandResult(124, "partial\n", "inner command timed out\n"),
            "partial\ninner command timed out\n[exit code 124]",
            {"exit_code": 124, "timed_out": False},
        ),
        # The runtime never answered: the client-side deadline fires, which says nothing about the program.
        (
            _hang_forever,
            "Sandbox error: no response from the sandbox within 17 seconds. This is a sandbox or network "
            "problem, not a timeout of the program, which may not have run or finished.",
            {"sandbox_timed_out": True},
        ),
        (_api_down, "Sandbox error: RuntimeError: HTTP 503: unavailable", {}),
    ],
    ids=["ok", "no-output", "nonzero-exit", "nonzero-exit-silent", "program-exits-124", "client-timeout", "sdk-error"],
)
def test_response_text(sdk, on_exec, expected_text, expected_metrics):
    sdk.on_exec = on_exec
    tool = _make_tool(default_timeout=7)

    response, reward, metrics = asyncio.run(_call(tool, "print(2 + 2)", _agent_data("trajectory-a")))

    assert response.text == expected_text
    assert reward == 0.0
    assert metrics == expected_metrics


def test_runtime_timeout_is_reported_and_keeps_stderr(sdk):
    """Runta kills the command at the timeout: exit code 124 plus its own notice at the end of stderr."""
    sdk.exec_secs = 0.2
    sdk.on_exec = lambda code: FakeCommandResult(
        124, "step 1\n", "Traceback: KeyboardInterrupt\nCommand timed out after 0.2 seconds."
    )
    tool = _make_tool(default_timeout=0.2)

    response, _, metrics = asyncio.run(_call(tool, "while True: pass", _agent_data("trajectory-a")))

    assert response.text == "step 1\nTraceback: KeyboardInterrupt\nExecution timed out after 0.2 seconds."
    assert metrics == {"exit_code": 124, "timed_out": True}


def test_exit_code_124_after_a_slow_call_is_not_a_timeout(sdk):
    """The call took as long as the timeout, but the SDK says the command itself returned at once."""
    sdk.exec_secs = 0.2
    sdk.on_exec = lambda code: FakeCommandResult(124, "", "boom\n", duration=0.01)
    tool = _make_tool(default_timeout=0.2)

    response, _, metrics = asyncio.run(_call(tool, "import sys; sys.exit(124)", _agent_data("trajectory-a")))

    assert response.text == "boom\n[exit code 124]"
    assert metrics == {"exit_code": 124, "timed_out": False}


@pytest.mark.parametrize(
    ("code", "executed"),
    [
        ("```python\nprint(1)\n```", "print(1)\n"),
        ("\n```\nprint(1)\n```\n", "print(1)\n"),
        ("```python\nprint('```')\n```", "print('```')\n"),
        # A fence inside the code (here a string literal) is part of the program.
        ('doc = """\n```python\nprint(1)\n```\n"""\nprint(len(doc))', None),
        # Neither prose around a block nor several blocks say which code is meant: run as is.
        ("Here it is:\n```python\nprint(1)\n```", None),
        ("```python\nprint(1)\n```\n```python\nprint(2)\n```", None),
    ],
    ids=["fenced", "fenced-no-language", "fenced-with-backticks", "fence-in-string", "prose-and-block", "two-blocks"],
)
def test_fence_is_stripped_only_when_it_wraps_the_whole_input(sdk, code, executed):
    tool = _make_tool()

    asyncio.run(_call(tool, code, _agent_data("trajectory-a")))

    assert sdk.runtimes[0].exec_calls[0]["stdin"] == (code if executed is None else executed)


@pytest.mark.parametrize(
    ("error", "cause"),
    [
        (FakeApiError("token expired", status_code=401, code="unauthenticated"), "Runta authentication failed"),
        (FakeApiError("too many runtimes", status_code=429, code="rate_limited"), "Runta quota or rate limit"),
    ],
    ids=["auth", "quota"],
)
@pytest.mark.parametrize("stage", ["create", "exec"])
def test_auth_and_quota_failures_are_logged_as_errors(sdk, caplog, stage, error, cause):
    """The model only gets tool text either way, so the log must say that the run itself is broken."""
    tool = _make_tool()

    def raise_error(code):
        raise error

    async def scenario():
        if stage == "create":
            sdk.fail_create = error
            with pytest.raises(FakeApiError):
                await _call(tool, "print(1)", _agent_data("trajectory-a"))
        else:
            sdk.on_exec = raise_error
            response, _, _ = await _call(tool, "print(1)", _agent_data("trajectory-a"))
            assert response.text.startswith("Sandbox error: FakeApiError")

    with caplog.at_level(logging.WARNING):
        asyncio.run(scenario())

    errors = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1 and errors[0].startswith(cause)


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


def test_agent_loop_deletes_the_replacement_of_a_dead_runtime(sdk, monkeypatch):
    tool = _make_tool()
    agent_loop = _make_agent_loop(tool)
    tool_call = SimpleNamespace(name="code_interpreter", arguments='{"code": "print(1)"}')

    async def fake_run(self, sampling_params, **kwargs):
        agent_data = _agent_data("trajectory-a")
        await self._call_tool(tool_call, agent_data.tools_kwargs, agent_data)
        sdk.runtimes[0].kill()
        response, _, _ = await self._call_tool(tool_call, agent_data.tools_kwargs, agent_data)
        return response.text

    monkeypatch.setattr(ToolAgentLoop, "run", fake_run)

    assert asyncio.run(agent_loop.run({})) == "ran:print(1)"
    assert [runtime.deleted for runtime in sdk.runtimes] == [True, True]
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
