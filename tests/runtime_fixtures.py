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
"""Fixtures shared by Runtime and backend tests."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

BACKENDS = ("ray", "monarch")


@pytest.fixture(scope="session", autouse=True)
def monarch_client_context_lifecycle() -> Iterator[None]:
    """Close Monarch's process-global client context after all Runtime tests."""
    yield
    if "monarch.actor" in sys.modules:
        from monarch.actor import shutdown_context

        shutdown_context().get(timeout=30)


@pytest.fixture(scope="session", autouse=True)
def runtime_test_import_path() -> Iterator[None]:
    """Make Runtime subprocess imports independent of the outer CI command."""
    workspace = str(Path(__file__).resolve().parents[1])
    previous = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (workspace, previous)))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous


def ray_runtime_config() -> dict[str, object]:
    """Build a Ray RuntimeConfig mapping."""
    return {
        "backend": "ray",
        "env_vars": {},
        "ray": {
            "ray_init": {"num_cpus": 4},
        },
        # Unselected section must be ignored by Runtime.from_config.
        "monarch": {"job_mode": "current"},
    }


def monarch_runtime_config() -> dict[str, object]:
    """Build a Monarch RuntimeConfig mapping."""
    monarch: dict[str, object] = {
        "job_mode": "current",
        "worker_ready_timeout_s": 60.0,
        "shutdown_timeout_s": 60.0,
    }
    return {
        "backend": "monarch",
        "env_vars": {},
        "monarch": monarch,
        # Unselected section must be ignored by Runtime.from_config.
        "ray": {"ray_init": {}},
    }


def runtime_config_for(backend: str) -> dict[str, object]:
    if backend == "ray":
        return ray_runtime_config()
    if backend == "monarch":
        return monarch_runtime_config()
    raise ValueError(f"unsupported backend {backend!r}")


@pytest.fixture(params=BACKENDS)
def backend_name(request: pytest.FixtureRequest) -> str:
    name = request.param
    if name == "monarch":
        pytest.importorskip("monarch")
    return name


@pytest.fixture
def monarch_local_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a native local host while worker groups still spawn real processes."""
    pytest.importorskip("monarch")
    from monarch.job import set_current_job

    monkeypatch.chdir(tmp_path)
    set_current_job("tests.single_controller.monarch.local_job.job")


@pytest.fixture
def runtime_config(backend_name: str, request: pytest.FixtureRequest) -> dict[str, object]:
    """Use a local host for RPC contracts; backend unit fakes cover job ownership."""
    if backend_name == "monarch":
        # LocalJob provides a ready local HostMesh. Worker groups still spawn
        # real processes, without racing ProcessJob's external host attach.
        request.getfixturevalue("monarch_local_job")
    return runtime_config_for(backend_name)


@pytest.fixture
def runtime(runtime_config: dict[str, object], backend_name: str) -> Iterator[Any]:
    from verl.runtime import Runtime

    rt = Runtime.from_config(runtime_config)
    try:
        yield rt
    finally:
        rt.close()
        if backend_name == "ray":
            import ray

            if ray.is_initialized():
                ray.shutdown()


@pytest.fixture
def ray_only_runtime() -> Iterator[Any]:
    """Ray Runtime; factory initializes Ray when needed."""
    import ray

    from verl.runtime import Runtime

    rt = Runtime.from_config(ray_runtime_config())
    try:
        yield rt
    finally:
        rt.close()
        if ray.is_initialized():
            ray.shutdown()


@pytest.fixture
def cpu_strategy():
    return {"nnodes": 1, "processes_per_node": 2, "device_type": "cpu"}


@pytest.fixture
def cpu_pool(runtime, cpu_strategy):
    return runtime.create_resource_pool(**cpu_strategy)
