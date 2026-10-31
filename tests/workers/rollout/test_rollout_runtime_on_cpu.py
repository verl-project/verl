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
"""Contracts for standalone vLLM interpreter isolation (no inference package required)."""

import asyncio
import os
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.workers.config import CheckpointEngineConfig, RolloutConfig


def test_runtime_is_disabled_by_default():
    config = RolloutConfig(name="vllm")
    assert hasattr(config, "runtime")
    assert config.runtime.py_executable is None


def test_runtime_config_coerces_hydra_mapping():
    config = RolloutConfig(name="vllm", runtime=OmegaConf.create({"py_executable": "/opt/rollout/bin/python"}))
    assert config.runtime.py_executable == "/opt/rollout/bin/python"
    assert config.runtime.env_vars == {}


@pytest.mark.parametrize("path", ["python", "./venv/bin/python", "", "uv run python"])
def test_runtime_requires_absolute_executable(path):
    with pytest.raises(ValueError, match="absolute"):
        RolloutConfig(name="vllm", runtime={"py_executable": path})


@pytest.mark.parametrize("name", ["sglang", "trtllm", "hf"])
def test_runtime_rejects_unsupported_backend(name):
    with pytest.raises(ValueError, match="vllm"):
        RolloutConfig(name=name, runtime={"py_executable": "/opt/rollout/bin/python"})


def test_runtime_rejects_pd():
    with pytest.raises(ValueError, match="disaggregation"):
        RolloutConfig(
            name="vllm",
            runtime={"py_executable": "/opt/rollout/bin/python"},
            disaggregation={"enabled": True},
        )


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_runtime_requires_finite_positive_timeout(timeout):
    with pytest.raises(ValueError, match="timeout"):
        RolloutConfig(name="vllm", runtime={"py_executable": "/opt/rollout/bin/python", "startup_timeout": timeout})


@pytest.mark.parametrize("key", ["CUDA_VISIBLE_DEVICES", "RANK", "MASTER_ADDR", "WG_PREFIX"])
def test_runtime_cannot_override_device_or_rank(key):
    with pytest.raises(ValueError, match="protected"):
        RolloutConfig(name="vllm", runtime={"py_executable": "/opt/python", "env_vars": {key: "0"}})


def test_runtime_merge_preserves_inherited_options_without_mutation():
    from verl.workers.rollout.runtime import resolve_rollout_runtime_env

    inherited = {
        "working_dir": "/work",
        "py_modules": ["gcs://code.zip"],
        "py_executable": "/opt/train/python",
        "env_vars": {"A": "inherited", "B": "keep"},
        "nsight": {"trace": "cuda"},
    }
    before = deepcopy(inherited)
    runtime = RolloutConfig(
        name="vllm", runtime={"py_executable": "/opt/rollout/python", "env_vars": {"A": "role"}}
    ).runtime
    resolved = resolve_rollout_runtime_env(runtime, inherited, {"RANK": "3"})
    assert resolved["py_executable"] == "/opt/rollout/python"
    assert resolved["env_vars"] == {"A": "role", "B": "keep", "RANK": "3"}
    assert resolved["working_dir"] == "/work"
    assert resolved["py_modules"] == ["gcs://code.zip"]
    assert resolved["nsight"] == {"trace": "cuda"}
    assert inherited == before


@pytest.mark.parametrize("manager", ["pip", "conda", "uv"])
def test_runtime_merge_rejects_inherited_package_managers(manager):
    from verl.workers.rollout.runtime import resolve_rollout_runtime_env

    runtime = RolloutConfig(name="vllm", runtime={"py_executable": "/opt/rollout/python"}).runtime
    with pytest.raises(ValueError, match=manager):
        resolve_rollout_runtime_env(runtime, {manager: ["some-package"]})


def test_disabled_runtime_does_not_change_existing_runtime():
    from verl.workers.rollout.runtime import resolve_rollout_runtime_env

    inherited = {"pip": ["a"], "env_vars": {"A": "a"}}
    resolved = resolve_rollout_runtime_env(RolloutConfig(name="vllm").runtime, inherited, {"RANK": "0"})
    assert resolved == {"pip": ["a"], "env_vars": {"A": "a", "RANK": "0"}}
    assert inherited == {"pip": ["a"], "env_vars": {"A": "a"}}


def test_controller_import_and_construction_do_not_import_vllm():
    code = """
import importlib.abc
import sys
from types import SimpleNamespace
class BlockVllm(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'vllm' or fullname.startswith('vllm.'):
            raise AssertionError('vLLM imported on controller: ' + fullname)
sys.meta_path.insert(0, BlockVllm())
from verl.workers.config import RolloutConfig
from verl.workers.rollout.replica import get_rollout_replica_class
cls = get_rollout_replica_class('vllm')
config = RolloutConfig(name='vllm', runtime={'py_executable': sys.executable})
replica = cls(0, config, SimpleNamespace(), gpus_per_node=1)
assert replica.server_class is None
assert not any(k == 'vllm' or k.startswith('vllm.') for k in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("field", ["python", "ray", "torch", "verl_source"])
def test_preflight_rejects_core_or_source_mismatch(field):
    from verl.workers.rollout.runtime import check_runtime_identity

    expected = {"python": "3.12.3", "ray": "2.59.0", "torch": "2.13.0", "verl_source": "abc"}
    actual = {**expected, "node": "rollout-node", field: "different"}
    with pytest.raises(ValueError, match=field):
        check_runtime_identity(actual, expected)


def test_preflight_allows_different_executable_with_matching_core():
    from verl.workers.rollout.runtime import check_runtime_identity, runtime_identity

    expected = runtime_identity()
    actual = {**expected, "executable": "/opt/rollout/python", "pid": os.getpid() + 1}
    check_runtime_identity(actual, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["init_hybrid", "init_hybrid_colocated", "init_colocated"])
async def test_runtime_rejects_shared_gpu_modes_before_startup(mode):
    from verl.workers.rollout.vllm_replica import vLLMReplica

    config = RolloutConfig(name="vllm", runtime={"py_executable": sys.executable})
    replica = vLLMReplica(0, config, SimpleNamespace(), gpus_per_node=1)
    args = (None, None) if mode == "init_hybrid_colocated" else (None,)
    with pytest.raises(ValueError, match="standalone"):
        await getattr(replica, mode)(*args)


@pytest.mark.asyncio
async def test_preflight_preserves_checkpoint_worker_device_visibility(monkeypatch):
    from verl.workers.rollout import vllm_replica

    config = RolloutConfig(
        name="vllm", runtime={"py_executable": sys.executable}, checkpoint_engine=CheckpointEngineConfig(backend="nccl")
    )
    replica = vllm_replica.vLLMReplica(0, config, SimpleNamespace(), gpus_per_node=1)
    replica.resource_pool = SimpleNamespace(world_size=1, get_placement_groups=lambda **kwargs: ["pg"])
    monkeypatch.setattr(vllm_replica.ray, "get_runtime_context", lambda: SimpleNamespace(runtime_env={}))
    monkeypatch.setattr(
        vllm_replica.ray.util, "placement_group_table", lambda pg: {"bundles_to_node_id": {0: "a" * 56}}
    )
    monkeypatch.setattr(vllm_replica, "runtime_identity", lambda: {})

    class Probe:
        def options(self, **kwargs):
            return self

        async def remote(self, *args):
            return {}

    monkeypatch.setattr(vllm_replica, "probe_rollout_runtime", Probe())
    await replica._preflight_standalone()
    # Checkpoint workers need Ray's normal per-actor CUDA visibility: the
    # NCCL collective store key includes the process-local device index.
    env = replica._get_worker_runtime_env().get("env_vars", {})
    assert "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES" not in env


@pytest.mark.asyncio
async def test_startup_failure_cleans_up_only_replica_actors(monkeypatch):
    from verl.workers.rollout import vllm_replica
    from verl.workers.rollout.replica import RolloutReplica

    config = RolloutConfig(name="vllm", runtime={"py_executable": sys.executable})
    replica = vllm_replica.vLLMReplica(0, config, SimpleNamespace(), gpus_per_node=1)
    killed = []
    monkeypatch.setattr(vllm_replica, "get_platform", lambda: SimpleNamespace(device_name="cuda"))
    monkeypatch.setattr(vllm_replica.ray, "kill", lambda actor, **kwargs: killed.append(actor))

    async def fail(self, pool):
        self.workers = ["checkpoint-worker"]
        self.servers = ["server"]
        self._server_launchers = ["launcher"]
        raise RuntimeError("server startup failed")

    monkeypatch.setattr(RolloutReplica, "init_standalone", fail)
    with pytest.raises(RuntimeError, match="server startup failed"):
        await replica.init_standalone(SimpleNamespace())
    assert killed == ["server", "checkpoint-worker", "launcher"]
    assert replica.workers == replica.servers == replica._server_launchers == []


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
async def test_launcher_failure_or_cancellation_kills_only_its_owner(monkeypatch, error):
    from verl.workers.rollout import vllm_replica

    config = RolloutConfig(name="vllm", runtime={"py_executable": sys.executable})
    replica = vllm_replica.vLLMReplica(0, config, SimpleNamespace(), gpus_per_node=1)
    replica.runtime_env = {"py_executable": sys.executable}
    replica.runtime_expected = {}
    killed = []

    async def fail(*args):
        raise error("actor creation failed")

    launcher = SimpleNamespace(create=SimpleNamespace(remote=fail))

    class LauncherClass:
        def options(self, **kwargs):
            assert kwargs["lifetime"] == "non_detached"
            return self

        def remote(self, expected):
            return launcher

    def reject_name_lookup(name):
        raise AssertionError("cleanup must not touch an existing named actor")

    monkeypatch.setattr(vllm_replica.ray, "remote", lambda cls: LauncherClass())
    monkeypatch.setattr(vllm_replica.ray, "get_actor", reject_name_lookup)
    monkeypatch.setattr(vllm_replica.ray, "kill", lambda actor, **kwargs: killed.append(actor))
    with pytest.raises(error, match="actor creation failed"):
        await replica._create_server(
            {"name": "vllm_server_0_0", "scheduling_strategy": "test", "runtime_env": {"env_vars": {}}}, {}
        )
    assert killed == [launcher]
    assert replica._server_launchers == []


def _trainer_config(**updates):
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {"strategy": "fsdp2"},
                "hybrid_engine": True,
                "rollout": {"runtime": {"py_executable": "/opt/rollout/python"}},
            },
            "trainer": {
                "use_v1": True,
                "v1": {
                    "trainer_mode": "separate_async",
                    "separate_async": {"hybrid_rollout": {"enable_switch": False}},
                },
            },
        }
    )
    for path, value in updates.items():
        OmegaConf.update(config, path, value)
    return config


@pytest.mark.parametrize(
    "path,value,match",
    [
        ("trainer.use_v1", False, "V1"),
        ("trainer.v1.trainer_mode", "sync", "separate_async"),
        ("actor_rollout_ref.actor.strategy", "megatron", "fsdp2"),
        ("trainer.v1.separate_async.hybrid_rollout.enable_switch", True, "enable_switch"),
    ],
)
def test_trainer_rejects_unsupported_runtime_topologies(path, value, match):
    from verl.utils.config import _validate_rollout_runtime_config

    with pytest.raises(ValueError, match=match):
        _validate_rollout_runtime_config(_trainer_config(**{path: value}))


def test_trainer_accepts_standalone_and_explicit_no_hybrid_override():
    from verl.utils.config import _validate_rollout_runtime_config

    _validate_rollout_runtime_config(_trainer_config())
    _validate_rollout_runtime_config(
        _trainer_config(
            **{
                "actor_rollout_ref.hybrid_engine": False,
                "trainer.v1.separate_async.hybrid_rollout.enable_switch": True,
            }
        )
    )


def test_selected_interpreter_runs_in_real_virtualenv(tmp_path):
    # Run a separate Ray job so this test owns its processes and shutdown.
    code = r"""
import json
import os
from pathlib import Path
import sys
import sysconfig
import venv
import ray
from verl.workers.config import RolloutConfig
from verl.workers.rollout.runtime import resolve_rollout_runtime_env

target = Path(sys.argv[1]) / 'rollout'
venv.EnvBuilder(with_pip=False, system_site_packages=True, symlinks=True).create(target)
site = target / 'lib' / ('python' + '.'.join(map(str, sys.version_info[:2]))) / 'site-packages'
(site / 'test_dependencies.pth').write_text(sysconfig.get_path('purelib') + '\n')
(site / 'rollout_runtime_marker.py').write_text('VALUE = "rollout-only"\n')

@ray.remote
def inspect_runtime():
    import rollout_runtime_marker
    return sys.prefix, rollout_runtime_marker.VALUE, os.environ['ROLE_VALUE'], os.environ['BASE_VALUE']

ray.init(num_cpus=2, include_dashboard=False, runtime_env={'env_vars': {'BASE_VALUE': 'inherited'}})
try:
    runtime = RolloutConfig(name='vllm', runtime={
        'py_executable': str(target / 'bin/python'), 'env_vars': {'ROLE_VALUE': 'overridden'},
    }).runtime
    options = resolve_rollout_runtime_env(runtime, ray.get_runtime_context().runtime_env)
    prefix, marker, role, base = ray.get(inspect_runtime.options(runtime_env=options).remote(), timeout=90)
    assert prefix == str(target), (prefix, target)
    assert prefix != sys.prefix
    assert (marker, role, base) == ('rollout-only', 'overridden', 'inherited')
    print(json.dumps({'prefix': prefix, 'marker': marker, 'role': role, 'base': base}))
finally:
    ray.shutdown()
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=150)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"marker": "rollout-only"' in result.stdout
