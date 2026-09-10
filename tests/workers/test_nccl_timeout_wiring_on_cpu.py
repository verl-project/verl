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

"""``nccl_timeout`` must reach the worker's torch process group.

``actor_rollout_ref.nccl_timeout`` and ``critic.nccl_timeout`` are documented public config keys
for the torch process-group timeout. The FSDP/Megatron workers read them; when those workers were
replaced by the engine workers every reader disappeared, so ``TrainingWorker.__init__`` created the
process group with ``timeout_second=None`` (torch's own default) and any configured value was
silently ignored.

These tests drive the worker constructor with the process-group init mocked out, so no GPU, no ray
cluster and no real distributed init are needed.
"""

import ast
import inspect
import textwrap
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from verl.utils.distributed import initialize_global_process_group_ray
from verl.workers import engine_workers
from verl.workers.config import (
    FSDPCriticConfig,
    FSDPEngineConfig,
    McoreCriticConfig,
    TrainingWorkerConfig,
)


class _StopInit(Exception):
    """Aborts ``TrainingWorker.__init__`` right after the process group is initialized."""


def _timeout_second_forwarded_by(config: TrainingWorkerConfig):
    """Run ``TrainingWorker.__init__`` up to the process-group init and return its timeout.

    Everything after that point needs a real model/engine, so the recorder raises to stop the
    constructor as soon as the value under test has been observed. ``Worker.__init__`` is stubbed
    out because it expects the ray-injected worker environment.
    """
    recorded = {}

    def _record(timeout_second=None, backend=None):
        recorded["timeout_second"] = timeout_second
        raise _StopInit

    with (
        patch.object(engine_workers.Worker, "__init__", lambda self: None),
        patch.object(engine_workers, "initialize_global_process_group_ray", _record),
        patch.object(engine_workers, "set_numa_affinity", lambda: None),
    ):
        with pytest.raises(_StopInit):
            engine_workers.TrainingWorker(config=config)

    assert "timeout_second" in recorded, "TrainingWorker never initialized a process group"
    return recorded["timeout_second"]


@pytest.mark.parametrize("nccl_timeout", [600, 1800])
def test_training_worker_forwards_configured_nccl_timeout(nccl_timeout):
    """The value the trainer copies out of ``actor_rollout_ref``/``critic`` must be used verbatim."""
    config = TrainingWorkerConfig(model_type="language_model", nccl_timeout=nccl_timeout)

    assert _timeout_second_forwarded_by(config) == nccl_timeout


def test_training_worker_without_nccl_timeout_keeps_torch_default():
    """Callers that configure nothing (e.g. SFT) must keep torch's own default timeout."""
    config = TrainingWorkerConfig(model_type="language_model")

    assert config.nccl_timeout is None
    assert _timeout_second_forwarded_by(config) is None


@pytest.mark.parametrize(
    ("timeout_second", "expected_timeout"),
    [(600, timedelta(seconds=600)), (1800, timedelta(seconds=1800)), (None, None)],
)
def test_initialize_global_process_group_ray_forwards_timeout(monkeypatch, timeout_second, expected_timeout):
    """The seconds the worker passes down land on ``torch.distributed.init_process_group``."""
    calls = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda **kwargs: calls.append(kwargs))

    # An explicit backend keeps this off the accelerator-detection path.
    initialize_global_process_group_ray(timeout_second=timeout_second, backend="gloo")

    assert len(calls) == 1
    assert calls[0]["timeout"] == expected_timeout


# ---------------------------------------------------------------------------
# The trainer-side hop: config key -> TrainingWorkerConfig.nccl_timeout
# ---------------------------------------------------------------------------

# Every place that copies the key out of a user-facing config into a TrainingWorkerConfig.
_CALL_SITES = [
    ("verl.workers.engine_workers", "ActorRolloutRefWorker", "init_model", 2),
    ("verl.trainer.ppo.ray_trainer", "RayPPOTrainer", "init_workers", 1),
    ("verl.trainer.ppo.v1.trainer_base", "PPOTrainer", "_setup", 1),
    ("verl.experimental.separation.ray_trainer", "SeparateRayPPOTrainer", "_create_critic_class", 1),
]


def _nccl_timeout_reads_in(module_name: str, class_name: str, method_name: str):
    """Return one AST call node per ``nccl_timeout=`` keyword argument in a method's body."""
    import importlib

    method = getattr(getattr(importlib.import_module(module_name), class_name), method_name)
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))

    reads = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "nccl_timeout":
                reads.append(keyword.value)
    return reads


@pytest.mark.parametrize(("module_name", "class_name", "method_name", "expected"), _CALL_SITES)
def test_call_sites_fall_back_to_torch_default(module_name, class_name, method_name, expected):
    """No call site may substitute a literal timeout when the config does not carry the key.

    ``critic.nccl_timeout`` only exists on ``McoreCriticConfig``; on an FSDP critic the key is
    absent and ``BaseConfig.get`` quietly hands back whatever default the caller supplied. A
    literal default there is a timeout nobody can configure, and one that *shortens* the effective
    process-group timeout relative to torch's own default - the opposite of what the knob is for.
    So every read must be a plain ``.get("nccl_timeout")`` whose fallback is ``None``.
    """
    reads = _nccl_timeout_reads_in(module_name, class_name, method_name)

    assert len(reads) == expected, f"expected {expected} nccl_timeout read(s) in {class_name}.{method_name}"
    for read in reads:
        rendered = ast.unparse(read)
        assert isinstance(read, ast.Call), f"{rendered} should read the key off the config"
        assert isinstance(read.func, ast.Attribute) and read.func.attr == "get", rendered
        assert len(read.args) == 1 and not read.keywords, (
            f"{rendered} passes a fallback timeout; the fallback must stay None so that an "
            f"unconfigured run keeps torch's own default"
        )


def test_fsdp_critic_config_cannot_carry_an_nccl_timeout():
    """The premise of the test above: only the Megatron critic declares the key.

    ``verl/trainer/config/critic/dp_critic.yaml`` has no ``nccl_timeout`` either, so an FSDP
    critic has no way to set one - which is why the fallback is the *only* value it can ever see.
    """
    assert "nccl_timeout" in McoreCriticConfig.__dataclass_fields__
    assert McoreCriticConfig.__dataclass_fields__["nccl_timeout"].default == 600
    assert "nccl_timeout" not in FSDPCriticConfig.__dataclass_fields__

    fsdp_critic = _critic_config(FSDPCriticConfig)
    assert fsdp_critic.get("nccl_timeout") is None
    # BaseConfig.get swallows the AttributeError, so a literal default is returned verbatim.
    assert fsdp_critic.get("nccl_timeout", 600) == 600


def _critic_config(cls, **kwargs):
    """A minimal critic config; ``model`` only has to expose ``fsdp_config`` to the trainer."""
    return cls(
        ppo_micro_batch_size_per_gpu=1,
        model=SimpleNamespace(fsdp_config=FSDPEngineConfig()),
        **kwargs,
    )


@dataclass
class _CriticConfigWithNcclTimeout(FSDPCriticConfig):
    """An FSDP critic that declares the key, the way ``McoreCriticConfig`` does."""

    nccl_timeout: int = 1234


def _critic_worker_config_built_by_separate_trainer(critic_config):
    """Run the separate-trainer critic wiring and return the ``TrainingWorkerConfig`` it builds."""
    from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
    from verl.trainer.ppo.ray_trainer import Role

    built = {}

    trainer = SimpleNamespace(
        use_critic=True,
        config=SimpleNamespace(critic=critic_config),
        resource_pool_manager=SimpleNamespace(get_resource_pool=lambda role: "pool"),
        resource_pool_to_cls={"pool": {}},
        role_worker_mapping={Role.Critic: object()},
    )

    with (
        patch("verl.experimental.separation.ray_trainer.omega_conf_to_dataclass", lambda cfg: cfg),
        patch(
            "verl.experimental.separation.ray_trainer.RayClassWithInitArgs",
            lambda cls, config: built.setdefault("config", config),
        ),
    ):
        SeparateRayPPOTrainer._create_critic_class(trainer)

    assert "config" in built, "the trainer never built a critic worker config"
    return built["config"]


def test_separate_trainer_keeps_torch_default_for_fsdp_critic():
    """An FSDP critic has no ``nccl_timeout`` key, so the worker must keep torch's own default."""
    worker_config = _critic_worker_config_built_by_separate_trainer(_critic_config(FSDPCriticConfig))

    assert isinstance(worker_config, TrainingWorkerConfig)
    assert worker_config.nccl_timeout is None


def test_separate_trainer_forwards_configured_critic_nccl_timeout():
    """A critic config that does declare the key has it copied over verbatim."""
    worker_config = _critic_worker_config_built_by_separate_trainer(_critic_config(_CriticConfigWithNcclTimeout))

    assert worker_config.nccl_timeout == 1234


# ---------------------------------------------------------------------------
# Colocation: one process group per process, so one timeout per process
# ---------------------------------------------------------------------------


def test_colocated_workers_share_the_first_workers_timeout(monkeypatch):
    """``initialize_global_process_group_ray`` is a no-op once the group exists.

    In a colocated PPO run the actor's and the critic's ``TrainingWorker`` live in the same
    process, so whichever is constructed first decides the timeout for both and the second
    worker's ``nccl_timeout`` is silently dropped. This pins that behavior rather than endorsing
    it - see the PR discussion on which role should win.
    """
    initialized = {"value": False}
    calls = []

    def _init_process_group(**kwargs):
        calls.append(kwargs)
        initialized["value"] = True

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: initialized["value"])
    monkeypatch.setattr(torch.distributed, "init_process_group", _init_process_group)

    def _stop():
        raise _StopInit

    with (
        patch.object(engine_workers.Worker, "__init__", lambda self: None),
        # The real initialize_global_process_group_ray runs; abort on the next statement.
        patch.object(engine_workers, "set_numa_affinity", _stop),
    ):
        for nccl_timeout in (1800, 600):
            config = TrainingWorkerConfig(model_type="language_model", nccl_timeout=nccl_timeout)
            with pytest.raises(_StopInit):
                engine_workers.TrainingWorker(config=config)

    assert len(calls) == 1, "the second colocated worker must not create a second process group"
    assert calls[0]["timeout"] == timedelta(seconds=1800), "the first worker constructed wins"
