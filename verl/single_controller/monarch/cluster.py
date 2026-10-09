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

"""Process-level view of HostMeshes supplied by the configured Monarch job.

There is at most one live cluster per process. ``Runtime`` initializes and
closes it; WorkerGroup and ResourcePool resolve it via ``get_monarch_cluster()``
instead of carrying a cluster handle.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .config import MonarchRuntimeConfig

if TYPE_CHECKING:
    from monarch._rust_bindings.monarch_hyperactor.pytokio import Handle
    from monarch.actor import HostMesh, ProcMesh
    from monarch.job import JobTrait

_CLUSTER: MonarchCluster | None = None

_DEVICE_BOOTSTRAP = """\
import os
import sys

env_name = sys.argv[1]
device_ordinal = int(sys.argv[2])
program = sys.argv[3]
argv = sys.argv[4:]
visible = [item for item in os.environ.get(env_name, "").split(",") if item]
if visible:
    try:
        selected = visible[device_ordinal]
    except IndexError as exc:
        raise RuntimeError(
            f"device ordinal {device_ordinal} exceeds {env_name}={visible!r}"
        ) from exc
else:
    selected = str(device_ordinal)
os.environ[env_name] = selected
os.execvpe(program, argv, os.environ)
"""


def _bind_bootstrap_to_device(bootstrap_command: Any, env_name: str, device_ordinal: int) -> Any:
    """Return a bootstrap that narrows host visibility before Monarch imports Python actors."""
    program = str(bootstrap_command.program)
    argv0 = bootstrap_command.arg0 or program
    args = [
        "-c",
        _DEVICE_BOOTSTRAP,
        env_name,
        str(device_ordinal),
        program,
        argv0,
        *bootstrap_command.args,
    ]
    wrapper_program = program
    if pythonpath := bootstrap_command.env.get("PYTHONPATH"):
        args = [f"PYTHONPATH={pythonpath}", program, *args]
        wrapper_program = "/usr/bin/env"
    return type(bootstrap_command)(
        wrapper_program,
        None,
        args,
        dict(bootstrap_command.env),
    )


class MonarchCluster:
    """Process-scoped view of one current or Runtime-owned Monarch job."""

    def __init__(
        self,
        config: MonarchRuntimeConfig,
        *,
        job_state: Any | None,
        owned_job: JobTrait | None = None,
    ) -> None:
        self._config = config
        self._job_state = job_state
        self._owned_job = owned_job
        self._closed = False
        self._released_meshes: list[Handle[None]] = []

    @property
    def config(self) -> MonarchRuntimeConfig:
        return self._config

    @property
    def closed(self) -> bool:
        return self._closed

    def host_mesh(self, name: str) -> HostMesh:
        """Return one named HostMesh from the configured Job."""
        self._ensure_open()
        if self._job_state is None:
            raise RuntimeError("attached Monarch Runtime has no Job inventory")
        try:
            return getattr(self._job_state, name)
        except AttributeError as exc:
            raise RuntimeError(f"Monarch Job is missing required HostMesh {name!r}") from exc

    def controller_host_mesh(self) -> HostMesh:
        """Return the current single-controller node as a one-host mesh."""
        self._ensure_open()
        from monarch.actor import this_host

        return this_host()

    def spawn_proc_mesh(
        self,
        host_mesh: HostMesh,
        *,
        processes_per_node: int,
        device_type: str,
        device_indices: tuple[int, ...] = (),
        name: str,
        env_vars: Mapping[str, str],
    ) -> ProcMesh:
        """Spawn a ProcMesh with exact per-node process count."""
        self._ensure_open()
        dimension = "gpus" if device_type == "gpu" else "procs"
        from monarch.actor import default_bootstrap_cmd

        # Match Ray runtime_env semantics: only explicit Runtime environment
        # belongs on remote processes. Host-, device-, and rank-local values
        # must come from the target node or the backend's SPMD setup rather
        # than being copied from the launcher process.
        bootstrap_env = dict(env_vars)
        command = default_bootstrap_cmd()
        bootstrap_command = type(command)(
            command.program,
            command.arg0,
            ["-m", "verl.single_controller.monarch.patches.bootstrap"],
            dict(command.env),
        ).with_env(bootstrap_env)
        if device_type == "gpu":
            base_bootstrap_command = bootstrap_command
            if not device_indices:
                device_indices = tuple(range(processes_per_node))
            if len(device_indices) != processes_per_node:
                raise ValueError(
                    "GPU device count must match processes_per_node, got "
                    f"{device_indices!r} for {processes_per_node} processes"
                )
            from verl.plugin.platform import get_platform

            visible_devices_env = get_platform().visible_devices_envvar()

            def device_bootstrap(point):
                local_rank = int(point[dimension])
                return _bind_bootstrap_to_device(
                    base_bootstrap_command,
                    visible_devices_env,
                    device_indices[local_rank],
                )

            bootstrap_command = device_bootstrap
        elif pythonpath := bootstrap_env.get("PYTHONPATH"):
            # Monarch 0.6 attached HostMeshes do not reliably apply the
            # bootstrap env before importing application actor classes. Keep
            # the full env map, and also pass only the non-secret import path
            # through /usr/bin/env so Python sees it before bootstrap_main.
            args: list[str] = []
            if bootstrap_command.arg0 is not None:
                args.extend(("-a", bootstrap_command.arg0))
            args.extend(
                (
                    f"PYTHONPATH={pythonpath}",
                    str(bootstrap_command.program),
                    *bootstrap_command.args,
                )
            )
            bootstrap_command = type(bootstrap_command)(
                "/usr/bin/env",
                None,
                args,
                dict(bootstrap_command.env),
            )
        host_mesh = host_mesh.with_python_executable(sys.executable)
        return host_mesh.spawn_procs(
            per_host={dimension: processes_per_node},
            bootstrap_command=bootstrap_command,
            name=name,
        )

    def release_mesh(self, mesh: Any) -> None:
        """Start native process termination without blocking a group finalizer."""
        self._released_meshes.append(mesh.stop()._take_inner().spawn_handle())

    def drain_released_meshes(self) -> None:
        """Observe native termination before releasing Runtime dependencies."""
        while self._released_meshes:
            self._released_meshes[-1].get(timeout=self._config.shutdown_timeout_s)
            self._released_meshes.pop()

    def close(self) -> None:
        """Close the process-local view and release a Runtime-owned ProcessJob."""
        if self._closed:
            return
        self.drain_released_meshes()
        if self._owned_job is not None:
            self._owned_job.kill()
            self._owned_job = None
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Monarch cluster is closed")


def init_monarch_cluster(
    config: MonarchRuntimeConfig,
    *,
    local_meshes: Mapping[str, int] | None = None,
) -> MonarchCluster:
    """Initialize the process-level Monarch cluster. Fails if one is already live."""
    global _CLUSTER
    if _CLUSTER is not None and not _CLUSTER.closed:
        raise RuntimeError("Monarch cluster is already initialized in this process")
    if config.job_mode == "current":
        try:
            from monarch.job import load_current_job
        except ImportError as exc:
            raise ImportError(
                "Monarch backend requires the pinned torchmonarch artifact in the project venv; "
                "install it before installing TorchStore with --no-deps"
            ) from exc
        job_state = load_current_job().state()
        _configure_monarch(config)
        _CLUSTER = MonarchCluster(config, job_state=job_state)
    elif config.job_mode == "process":
        from monarch.job import ProcessJob

        _configure_monarch(config)
        meshes = {"hosts": 1} if local_meshes is None else dict(local_meshes)
        job = ProcessJob(meshes, env=dict(config.env_vars))
        try:
            job_state = job.state(cached_path=None)
        except BaseException as state_error:
            try:
                job.kill()
            except BaseException as cleanup_error:
                raise state_error from cleanup_error
            raise
        _CLUSTER = MonarchCluster(config, job_state=job_state, owned_job=job)
    else:
        raise ValueError(f"unsupported Monarch job_mode {config.job_mode!r}")
    return _CLUSTER


def attach_monarch_cluster(config: MonarchRuntimeConfig) -> MonarchCluster:
    """Install a non-owning cluster view in a Monarch actor process."""
    global _CLUSTER
    if _CLUSTER is not None and not _CLUSTER.closed:
        raise RuntimeError("Monarch cluster is already initialized in this process")
    _configure_monarch(config)
    _CLUSTER = MonarchCluster(config, job_state=None)
    return _CLUSTER


def _configure_monarch(config: MonarchRuntimeConfig) -> None:
    """Configure process-global Monarch timeouts before installing Runtime."""
    from monarch.config import configure

    ready_timeout = f"{config.worker_ready_timeout_s:g}s"
    shutdown_timeout = f"{config.shutdown_timeout_s:g}s"
    configure(
        host_spawn_ready_timeout=ready_timeout,
        mesh_proc_spawn_max_idle=ready_timeout,
        actor_spawn_max_idle=ready_timeout,
        get_actor_state_max_idle=ready_timeout,
        process_exit_timeout=shutdown_timeout,
        proc_stop_max_idle=shutdown_timeout,
    )


def get_monarch_cluster() -> MonarchCluster:
    """Return the process-level Monarch cluster."""
    if _CLUSTER is None or _CLUSTER.closed:
        raise RuntimeError("Monarch cluster is not initialized in this process")
    return _CLUSTER


def close_monarch_cluster() -> None:
    """Close and clear the process-level Monarch cluster if present."""
    global _CLUSTER
    cluster = _CLUSTER
    if cluster is not None and not cluster.closed:
        cluster.close()
    _CLUSTER = None
