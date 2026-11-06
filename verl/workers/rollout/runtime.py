# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Runtime selection and remote bootstrap without importing an inference backend."""

import hashlib
import importlib
import logging
import os
import platform
import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import ray
import torch

from verl.utils.ray_utils import merge_runtime_env
from verl.workers.config.rollout import RolloutRuntimeConfig

logger = logging.getLogger(__name__)


def resolve_rollout_runtime_env(
    runtime: RolloutRuntimeConfig, inherited: dict, internal_env: dict[str, str] | None = None
) -> dict:
    """Overlay a role interpreter while retaining Ray code distribution and profiling."""
    result = deepcopy(dict(inherited))
    if runtime.py_executable is not None:
        conflicts = [key for key in ("pip", "conda", "uv") if result.get(key)]
        if conflicts:
            raise ValueError(f"rollout.runtime cannot override inherited package managers: {conflicts}")
        result["py_executable"] = runtime.py_executable
        result = merge_runtime_env(result, {"env_vars": dict(runtime.env_vars)})
    return merge_runtime_env(result, {"env_vars": internal_env or {}})


@lru_cache(maxsize=1)
def _source_fingerprint() -> str:
    """Identify the deployed verl Python sources, including uncommitted checkouts."""
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_identity() -> dict:
    """Return bounded diagnostics and the core compatibility contract; no environment dump."""
    return {
        "python": platform.python_version(),
        "ray": ray.__version__,
        "torch": torch.__version__,
        "verl_source": _source_fingerprint(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "pid": os.getpid(),
        "node": platform.node(),
    }


def check_runtime_identity(actual: dict, expected: dict) -> None:
    """Fail closed on core version or source mismatches before loading weights."""
    mismatches = {
        key: (expected[key], actual.get(key))
        for key in ("python", "ray", "torch", "verl_source")
        if actual.get(key) != expected[key]
    }
    if mismatches:
        raise ValueError(f"Incompatible rollout runtime on {actual.get('node')}: {mismatches}")


@ray.remote(num_cpus=0)
def probe_rollout_runtime(expected: dict, checkpoint_backend: str, custom_backend_module: str | None = None) -> dict:
    """Check the selected node/interpreter and import the backend without loading a model."""
    actual = runtime_identity()
    check_runtime_identity(actual, expected)
    importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_async_server")
    if custom_backend_module:
        importlib.import_module(custom_backend_module)
    from verl.checkpoint_engine import CheckpointEngineRegistry

    CheckpointEngineRegistry.get(checkpoint_backend)
    actual["vllm"] = importlib.import_module("vllm").__version__
    logger.info("Verified rollout runtime: %s", actual)
    return actual


class VLLMServerLauncher:
    """Own one native server for the lifetime of its rollout controller.

    Run this class as a non-detached Ray actor in the selected interpreter.
    A task worker can be reclaimed after returning an actor handle, so it must
    not own a long-running server. The controller retains this launcher's handle;
    controller/job teardown then terminates the launcher and its server through
    Ray ownership, without introducing detached actors or a generic RPC proxy.
    """

    def __init__(self, expected: dict):
        check_runtime_identity(runtime_identity(), expected)
        self.server = None

    def create(self, actor_options: dict, server_kwargs: dict):
        """Create and retain the native actor; callers use its handle directly."""
        if self.server is not None:
            raise RuntimeError("A rollout launcher can own only one server.")
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

        options = {**actor_options, "lifetime": "non_detached"}
        self.server = ray.remote(vLLMHttpServer).options(**options).remote(**server_kwargs)
        return self.server
