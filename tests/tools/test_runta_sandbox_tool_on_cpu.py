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
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import hydra
import pytest
from omegaconf import OmegaConf

from examples.runta_sandbox import runta_tool
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
        self.id = f"rt-{len(sdk.runtimes) + 1}"
        self.name = name
        self.create_kwargs = create_kwargs
        self.exec_calls: list[dict] = []
        self.timeout_updates: list[dict] = []  # policies saved by `update_timeout_policies`
        self.update_attempts = 0
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
        try:
            await asyncio.sleep(self.sdk.create_secs)
            if self.sdk.fail_ready:
                raise TimeoutError("runtime did not become ready")
        finally:
            self.sdk.creating -= 1

    async def exec(self, command, timeout=None, stdin=None, cwd=None, max_output_bytes=None):
        self.exec_calls.append({"command": command, "timeout": timeout, "stdin": stdin, "cwd": cwd})
        self.sdk.executing += 1
        self.sdk.max_executing = max(self.sdk.max_executing, self.sdk.executing)
        try:
            await asyncio.sleep(self.sdk.exec_secs)
        finally:
            self.sdk.executing -= 1
        if self.status != "running":
            raise FakeApiError(f"runtime is {self.status}")
        return self.sdk.on_exec(stdin)

    async def update_timeout_policies(self, *, auto_shutdown_after_secs=None, auto_delete_after_secs=None):
        """As in the SDK, None leaves a policy unchanged and saving one restarts its clock."""
        self.update_attempts += 1
        await asyncio.sleep(self.sdk.update_secs)
        if self.sdk.fail_update is not None:
            raise self.sdk.fail_update
        if self.deleted or self.status == "deleted":
            raise FakeApiError("runtime not found", status_code=404, code="not_found")
        policies = {
            "auto_shutdown_after_secs": auto_shutdown_after_secs,
            "auto_delete_after_secs": auto_delete_after_secs,
        }
        self.timeout_updates.append({key: secs for key, secs in policies.items() if secs is not None})

    async def delete(self):
        self.deleted = True


class FakeSdk:
    """Stands in for the ``runta`` module and records what the tool did with it."""

    def __init__(self):
        self.clients: list = []
        self.runtimes: list[FakeRuntime] = []
        self.fail_ready = False
        self.fail_create: Exception | None = None
        self.fail_update: Exception | None = None
        self.create_secs = 0.0  # wall-clock time a runtime takes to become ready
        self.exec_secs = 0.0  # wall-clock time every `exec` takes
        self.update_secs = 0.0  # wall-clock time every `update_timeout_policies` takes
        # Runtimes between `create` and ready, and commands running, right now and at most.
        self.creating = self.max_creating = 0
        self.executing = self.max_executing = 0
        self.on_exec = lambda code: FakeCommandResult(stdout_text=f"ran:{code}")
        self.on_create = lambda runtime: None

        sdk = self

        class FakeRuntimeManager:
            async def create(self, name, **kwargs):
                await asyncio.sleep(0)
                if sdk.fail_create is not None:
                    raise sdk.fail_create
                sdk.creating += 1
                sdk.max_creating = max(sdk.max_creating, sdk.creating)
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


async def _until(condition, timeout: float = 5.0):
    """Wait for something a background task does, without sleeping for a fixed time."""
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition was not reached in time"
        await asyncio.sleep(0.002)


def _tool_logs(caplog, level: int) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == runta_tool.logger.name and record.levelno == level
    ]


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
    assert (tools[0].max_concurrent_creates, tools[0].lease_secs, tools[0].auto_shutdown_after_secs) == (8, 600, 3600)
    assert sdk.clients == []


def test_shipped_agent_loop_config_targets_registered_loop():
    configs = OmegaConf.load(_EXAMPLE_DIR / "config" / "runta_agent_loop_config.yaml")

    assert [config.name for config in configs] == ["runta_tool_agent"]
    assert hydra.utils.get_class(configs[0]._target_) is RuntaToolAgentLoop
    assert _agent_loop_registry["runta_tool_agent"]["_target_"] == configs[0]._target_


def test_unsupported_language_is_rejected():
    with pytest.raises(ValueError, match="Unsupported language"):
        _make_tool(language="ruby")


@pytest.mark.parametrize(
    "config", [{"max_concurrent_creates": 0}, {"lease_secs": -1}, {"auto_shutdown_after_secs": -1}], ids=str
)
def test_invalid_limits_are_rejected(config):
    with pytest.raises(ValueError, match="must"):
        _make_tool(**config)


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
# Logging
# ---------------------------------------------------------------------------


def test_runtime_lifecycle_is_logged_at_info_and_executions_at_debug(sdk, caplog):
    sdk.on_exec = lambda code: FakeCommandResult(3, "secret output\n", "", duration=0.25)
    tool = _make_tool()

    async def scenario():
        await _call(tool, "print('secret code')", _agent_data("trajectory-a"))
        await tool.release_trajectory("trajectory-a")

    with caplog.at_level(logging.DEBUG, logger=runta_tool.logger.name):
        asyncio.run(scenario())

    created, deleted = _tool_logs(caplog, logging.INFO)
    assert re.fullmatch(
        r"Created Runta runtime rt-1 for trajectory trajectory-a in \d+\.\ds "
        r"\(waited \d+\.\ds for a creation slot\)",
        created,
    )
    assert deleted == "Deleted Runta runtime rt-1 of trajectory trajectory-a"
    assert _tool_logs(caplog, logging.DEBUG) == [
        "Runta exec for trajectory trajectory-a on runtime rt-1: exit code 3, 0.00s, 14 output chars"
    ]
    # Neither the code nor what it printed belongs in a log.
    assert "secret" not in caplog.text


def test_success_logs_nothing_at_warning(sdk, caplog):
    tool = _make_tool()

    async def scenario():
        await _call(tool, "print(1)", _agent_data("trajectory-a"))
        await tool.release_trajectory("trajectory-a")

    with caplog.at_level(logging.WARNING, logger=runta_tool.logger.name):
        asyncio.run(scenario())

    assert [record for record in caplog.records if record.name == runta_tool.logger.name] == []


# ---------------------------------------------------------------------------
# Creation cap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("config", "cap"), [({}, 8), ({"max_concurrent_creates": 2}, 2)], ids=["default", "set"])
def test_runtime_creation_is_capped_but_executions_are_not(sdk, config, cap):
    sdk.create_secs = 0.01
    sdk.exec_secs = 0.05
    tool = _make_tool(**config)

    async def scenario():
        for _ in range(2):  # the first round creates the runtimes, the second one only executes
            await asyncio.gather(*(_call(tool, "x", _agent_data(f"trajectory-{i}")) for i in range(12)))

    asyncio.run(scenario())

    assert len(sdk.runtimes) == 12
    assert sdk.max_creating == cap
    assert sdk.max_executing == 12


def test_creation_cap_works_on_every_event_loop_the_tool_is_used_on(sdk):
    """Like the SDK client, an ``asyncio.Semaphore`` that had to wait cannot move to another loop."""
    tool = _make_tool(max_concurrent_creates=1)

    async def scenario(batch: str):
        await asyncio.gather(*(_call(tool, "x", _agent_data(f"{batch}-{i}")) for i in range(3)))

    asyncio.run(scenario("first"))
    asyncio.run(scenario("second"))

    assert len(sdk.clients) == 2
    assert len(sdk.runtimes) == 6
    assert sdk.max_creating == 1


def test_failed_creation_frees_its_slot_and_starts_no_lease(sdk):
    tool = _make_tool(max_concurrent_creates=1)

    async def scenario():
        sdk.fail_ready = True
        failed = await asyncio.gather(
            *(_call(tool, "x", _agent_data(f"trajectory-{i}")) for i in range(3)), return_exceptions=True
        )
        leases = dict(tool._leases)
        sdk.fail_ready = False
        response, _, _ = await asyncio.wait_for(_call(tool, "x", _agent_data("trajectory-0")), timeout=5)
        return failed, leases, response

    failed, leases, response = asyncio.run(scenario())

    assert [type(error) for error in failed] == [TimeoutError] * 3
    assert leases == {}
    assert response.text == "ran:x"
    assert [runtime.deleted for runtime in sdk.runtimes] == [True, True, True, False]


# ---------------------------------------------------------------------------
# Lease
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "created_with", "renews"),
    [
        ({}, 600, True),
        ({"lease_secs": 60, "auto_shutdown_after_secs": 7200}, 60, True),
        ({"auto_shutdown_after_secs": 120}, 120, False),  # the cap is within one lease
        ({"auto_shutdown_after_secs": 0}, 600, True),  # no cap: 0 must not disable the lease
        ({"lease_secs": 0}, 3600, False),  # no lease: the cap alone, as before
        ({"lease_secs": 0, "auto_shutdown_after_secs": 0}, 0, False),
    ],
    ids=["default", "set", "cap-below-lease", "no-cap", "no-lease", "neither"],
)
def test_runtime_is_created_with_the_lease_as_its_shutdown_limit(sdk, config, created_with, renews):
    tool = _make_tool(**config)

    async def scenario():
        await _call(tool, "x", _agent_data("trajectory-a"))
        return list(tool._leases)

    leased = asyncio.run(scenario())

    assert sdk.runtimes[0].create_kwargs["auto_shutdown_after_secs"] == created_with
    assert leased == ([sdk.runtimes[0]] if renews else [])


@pytest.mark.parametrize(
    ("config", "age", "renewed_with", "goes_on"),
    [
        ({}, 200, 600, True),
        ({}, 3000, 600, True),
        ({}, 3300, 300, False),  # the cap leaves less than a lease: renew up to it, then stop
        ({}, 3599.5, None, False),  # less than a second left; 0 would switch the limit off
        ({}, 5000, None, False),
        ({"auto_shutdown_after_secs": 0}, 10**6, 600, True),
    ],
    ids=["young", "one-lease-left", "capped", "cap-reached", "past-cap", "no-cap"],
)
def test_renewal_never_extends_a_runtime_past_the_hard_cap(sdk, config, age, renewed_with, goes_on):
    """Drives one beat directly, for a runtime that was created ``age`` seconds ago."""
    tool = _make_tool(**config)

    async def scenario():
        await _call(tool, "x", _agent_data("trajectory-a"))
        return await tool._renew_lease(sdk.runtimes[0], "trajectory-a", age)

    assert asyncio.run(scenario()) is goes_on
    expected = [] if renewed_with is None else [{"auto_shutdown_after_secs": renewed_with}]
    assert sdk.runtimes[0].timeout_updates == expected


def test_lease_is_renewed_until_the_trajectory_is_released(sdk):
    tool = _make_tool()
    tool._lease_interval = 0.005  # instead of lease_secs / 3

    async def scenario():
        await _call(tool, "x", _agent_data("trajectory-a"))
        runtime = sdk.runtimes[0]
        lease = tool._leases[runtime]
        await _until(lambda: len(runtime.timeout_updates) >= 3)
        released = await tool.release_trajectory("trajectory-a")
        await _until(lease.done)
        renewals = len(runtime.timeout_updates)
        await asyncio.sleep(0.05)  # ten more beats, had the task survived
        return released, lease, renewals

    released, lease, renewals = asyncio.run(scenario())

    assert released and lease.done() and tool._leases == {}
    assert len(sdk.runtimes[0].timeout_updates) == renewals
    assert sdk.runtimes[0].timeout_updates[0] == {"auto_shutdown_after_secs": 600}


def test_lease_task_ends_by_itself_at_the_hard_cap(sdk, monkeypatch):
    tool = _make_tool()
    tool._lease_interval = 0.005
    clock = SimpleNamespace(offset=0.0)  # moves the tool's clock, and only the tool's
    monkeypatch.setattr(runta_tool, "time", SimpleNamespace(monotonic=lambda: time.monotonic() + clock.offset))

    async def scenario():
        await _call(tool, "x", _agent_data("trajectory-a"))
        runtime = sdk.runtimes[0]
        lease = tool._leases[runtime]
        clock.offset = 3300  # 300 s of the 3600 s cap are left
        await _until(lease.done)
        renewals = len(runtime.timeout_updates)
        await asyncio.sleep(0.05)
        return lease, renewals

    lease, renewals = asyncio.run(scenario())

    *before, last = (update["auto_shutdown_after_secs"] for update in sdk.runtimes[0].timeout_updates)
    assert set(before) <= {600} and 290 <= last <= 300
    assert lease.done() and tool._leases == {}
    assert len(sdk.runtimes[0].timeout_updates) == renewals
    assert not sdk.runtimes[0].deleted


def test_failed_renewal_is_logged_and_retried_without_failing_tool_calls(sdk, caplog):
    """E.g. the ``ApiError`` the SDK raises when another writer changed the runtime under it."""
    tool = _make_tool()
    tool._lease_interval = 0.005
    agent_data = _agent_data("trajectory-a")

    async def scenario():
        sdk.fail_update = FakeApiError(
            "runtime revision conflict: expected 3, found 4", status_code=409, code="failed_precondition"
        )
        first, _, _ = await _call(tool, "first", agent_data)
        runtime = sdk.runtimes[0]
        await _until(lambda: runtime.update_attempts >= 2)
        second, _, metrics = await _call(tool, "second", agent_data)
        sdk.fail_update = None
        await _until(lambda: runtime.timeout_updates)
        alive = not tool._leases[runtime].done()
        return first.text, second.text, metrics, alive

    with caplog.at_level(logging.WARNING, logger=runta_tool.logger.name):
        result = asyncio.run(scenario())

    assert result == ("ran:first", "ran:second", {"exit_code": 0, "timed_out": False}, True)
    assert len(sdk.runtimes) == 1 and not sdk.runtimes[0].deleted
    warnings = _tool_logs(caplog, logging.WARNING)
    assert len(warnings) >= 2
    assert all(
        warning.startswith("Failed to renew the lease of Runta runtime rt-1 of trajectory trajectory-a, retrying")
        and "revision conflict" in warning
        for warning in warnings
    )
    assert _tool_logs(caplog, logging.ERROR) == []


def test_stuck_renewal_is_abandoned_before_it_uses_up_the_lease(sdk, caplog):
    sdk.update_secs = 3600
    tool = _make_tool(lease_secs=1)  # a renewal may take a sixth of this

    async def scenario():
        await _call(tool, "x", _agent_data("trajectory-a"))
        return await asyncio.wait_for(tool._renew_lease(sdk.runtimes[0], "trajectory-a", 0), timeout=5)

    with caplog.at_level(logging.WARNING, logger=runta_tool.logger.name):
        assert asyncio.run(scenario()) is True

    assert len(_tool_logs(caplog, logging.WARNING)) == 1


def test_lease_task_stops_even_if_its_cancellation_is_swallowed(sdk, monkeypatch):
    """``asyncio.wait_for`` does that before Python 3.12 when the call ends as the cancellation arrives."""
    tool = _make_tool()
    tool._lease_interval = 0.005
    beats = []

    async def renew_and_swallow_cancellation(runtime, trajectory_id, age):
        beats.append(age)
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        return True

    monkeypatch.setattr(tool, "_renew_lease", renew_and_swallow_cancellation)

    async def scenario():
        await _call(tool, "x", _agent_data("trajectory-a"))
        lease = tool._leases[sdk.runtimes[0]]
        await _until(lambda: beats)
        await tool.release_trajectory("trajectory-a")
        await _until(lease.done)

    asyncio.run(scenario())

    assert len(beats) == 1 and tool._leases == {}


def test_lease_is_stopped_for_a_replaced_runtime_and_on_close(sdk):
    tool = _make_tool()
    tool._lease_interval = 0.005
    agent_data = _agent_data("trajectory-a")

    async def scenario():
        await _call(tool, "first", agent_data)
        dead = sdk.runtimes[0]
        dead_lease = tool._leases[dead]
        dead.kill()
        await _call(tool, "second", agent_data)
        fresh = sdk.runtimes[1]
        await _until(dead_lease.done)
        leased = list(tool._leases)
        fresh_lease = tool._leases[fresh]
        await _until(lambda: fresh.timeout_updates)
        await tool.close()
        await _until(fresh_lease.done)
        return dead_lease, fresh_lease, leased

    dead_lease, fresh_lease, leased = asyncio.run(scenario())

    assert dead_lease.done() and fresh_lease.done()
    assert leased == [sdk.runtimes[1]]
    assert tool._leases == {}


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
    assert tool._runtimes == {} and tool._leases == {}


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


def test_agent_loop_stops_the_lease_of_a_cancelled_trajectory(sdk, monkeypatch):
    tool = _make_tool()
    tool._lease_interval = 0.005
    agent_loop = _make_agent_loop(tool)
    tool_call = SimpleNamespace(name="code_interpreter", arguments='{"code": "print(1)"}')

    async def fake_run(self, sampling_params, **kwargs):
        agent_data = _agent_data("trajectory-a")
        await self._call_tool(tool_call, agent_data.tools_kwargs, agent_data)
        await asyncio.sleep(3600)  # a generation that is still running when the trajectory is cancelled

    monkeypatch.setattr(ToolAgentLoop, "run", fake_run)

    async def scenario():
        run = asyncio.ensure_future(agent_loop.run({}))
        await _until(lambda: sdk.runtimes and sdk.runtimes[0].timeout_updates)
        lease = tool._leases[sdk.runtimes[0]]
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        await _until(lease.done)
        renewals = len(sdk.runtimes[0].timeout_updates)
        await asyncio.sleep(0.05)
        return lease, renewals

    lease, renewals = asyncio.run(scenario())

    assert lease.done() and tool._leases == {}
    assert sdk.runtimes[0].deleted
    assert len(sdk.runtimes[0].timeout_updates) == renewals
