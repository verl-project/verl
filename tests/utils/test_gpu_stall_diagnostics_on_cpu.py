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

"""CPU-only tests for the optional GPU stall diagnostic sampler."""

from __future__ import annotations

import builtins
import importlib
import logging
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.profiler.config import GPUStallDiagnosticsConfig
from verl.utils.profiler.gpu_stall_diagnostics import (
    GPUStallDiagnostics,
    _Device,
    _LocalNVMLSampler,
    _NodeGPUStallSampler,
    start_gpu_stall_diagnostics,
)


class _FakeNVML:
    def __init__(self, utilizations: dict[int, object] | None = None):
        self.initializations = 0
        self.shutdowns = 0
        self.utilizations = utilizations or {0: 0.0, 1: 75.0}
        self.bus_ids = {0: b"0000:01:00.0", 1: b"0000:02:00.0"}
        self.calls = {index: 0 for index in self.bus_ids}

    def nvmlInit(self) -> None:
        self.initializations += 1

    def nvmlShutdown(self) -> None:
        self.shutdowns += 1

    def nvmlDeviceGetCount(self) -> int:
        return len(self.bus_ids)

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetPciInfo(self, handle: int) -> SimpleNamespace:
        return SimpleNamespace(busId=self.bus_ids[handle])

    def nvmlDeviceGetUtilizationRates(self, handle: int) -> SimpleNamespace:
        self.calls[handle] += 1
        utilization = self.utilizations[handle]
        if isinstance(utilization, list):
            utilization = utilization.pop(0) if len(utilization) > 1 else utilization[0]
        if isinstance(utilization, Exception):
            raise utilization
        return SimpleNamespace(gpu=utilization)


class _SignalingNVML(_FakeNVML):
    def __init__(self):
        super().__init__({0: 25.0})
        self.bus_ids = {0: b"0000:01:00.0"}
        self.calls = {0: 0}
        self.sampled = threading.Event()

    def nvmlDeviceGetUtilizationRates(self, handle: int) -> SimpleNamespace:
        self.sampled.set()
        return super().nvmlDeviceGetUtilizationRates(handle)


class _ControlledThread:
    def __init__(self):
        self.alive = True
        self.join_timeouts: list[float] = []

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float) -> None:
        self.join_timeouts.append(timeout)


class _FakeRemoteMethod:
    def __init__(self, callback):
        self._callback = callback

    def remote(self, *args):
        return self._callback(*args)


class _FakeActor:
    def __init__(self, index: int):
        self.index = index
        self.begin_phases: list[str] = []
        self.end_phases: list[str] = []
        self.close_calls = 0
        self.end_result = ({}, 0, None)
        self.begin_phase = _FakeRemoteMethod(self._begin_phase)
        self.end_phase = _FakeRemoteMethod(self._end_phase)
        self.close = _FakeRemoteMethod(self._close)

    def _begin_phase(self, phase: str) -> None:
        self.begin_phases.append(phase)

    def _end_phase(self, phase: str):
        self.end_phases.append(phase)
        return self.end_result

    def _close(self) -> None:
        self.close_calls += 1


class _FakeNodeAffinitySchedulingStrategy:
    def __init__(self, node_id: str, soft: bool):
        self.node_id = node_id
        self.soft = soft


class _FakeRemoteClassCall:
    def __init__(self, fake_ray, options: dict[str, object]):
        self._fake_ray = fake_ray
        self._options = options

    def remote(self, *args):
        actor = _FakeActor(len(self._fake_ray.created))
        if self._fake_ray.remote_creation_error_at == len(self._fake_ray.created):
            raise RuntimeError("actor creation failed")
        self._fake_ray.created.append((actor, self._options, args))
        return actor


class _FakeRemoteClass:
    def __init__(self, fake_ray):
        self._fake_ray = fake_ray

    def options(self, **options):
        return _FakeRemoteClassCall(self._fake_ray, options)


class _FakeRay(ModuleType):
    def __init__(self, nodes: list[dict[str, object]] | None = None):
        super().__init__("ray")
        self.__path__ = []
        self._nodes = nodes or []
        self.created: list[tuple[_FakeActor, dict[str, object], tuple[object, ...]]] = []
        self.remote_creation_error_at: int | None = None
        self.get_timeouts: list[float] = []
        self.get_errors: list[Exception] = []
        self.kill_attempts: list[tuple[_FakeActor, bool]] = []
        self.kill_failures: set[_FakeActor] = set()

    def nodes(self):
        return self._nodes

    def remote(self, _target):
        return _FakeRemoteClass(self)

    def get(self, refs, timeout: float):
        self.get_timeouts.append(timeout)
        if self.get_errors:
            raise self.get_errors.pop(0)
        return refs

    def kill(self, actor: _FakeActor, no_restart: bool) -> None:
        self.kill_attempts.append((actor, no_restart))
        if actor in self.kill_failures:
            raise RuntimeError(f"cannot kill actor {actor.index}")


def _install_fake_nvml(monkeypatch: pytest.MonkeyPatch, fake_nvml: _FakeNVML) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)


def _install_fake_ray(monkeypatch: pytest.MonkeyPatch, fake_ray: _FakeRay) -> None:
    util_module = ModuleType("ray.util")
    util_module.__path__ = []
    scheduling_module = ModuleType("ray.util.scheduling_strategies")
    scheduling_module.NodeAffinitySchedulingStrategy = _FakeNodeAffinitySchedulingStrategy
    fake_ray.util = util_module
    util_module.scheduling_strategies = scheduling_module
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.util", util_module)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", scheduling_module)


def test_start_stop_is_idempotent_and_restartable(monkeypatch):
    fake_nvml = _FakeNVML()
    _install_fake_nvml(monkeypatch, fake_nvml)
    sampler = _LocalNVMLSampler(sample_interval_s=0.2, zero_utilization_threshold=1.0)

    assert sampler.start() is None
    assert sampler.start() is None
    assert sampler.started
    sampler.sample_once()
    assert sampler.snapshot_and_reset()["0000:01:00.0"]["sample_count"] == 1.0
    sampler.close()
    sampler.close()

    assert not sampler.started
    assert fake_nvml.initializations == 1
    assert fake_nvml.shutdowns == 1

    assert sampler.start() is None
    sampler.close()
    assert fake_nvml.initializations == 2
    assert fake_nvml.shutdowns == 2


def test_sampler_thread_starts_and_stops_without_sleep(monkeypatch):
    fake_nvml = _SignalingNVML()
    _install_fake_nvml(monkeypatch, fake_nvml)
    sampler = _LocalNVMLSampler(sample_interval_s=0.0, zero_utilization_threshold=1.0)

    assert sampler.start() is None
    assert fake_nvml.sampled.wait(timeout=1.0)
    thread = sampler._thread

    sampler.close()

    assert thread is not None
    assert not thread.is_alive()
    assert not sampler.started
    assert fake_nvml.shutdowns == 1


def test_close_timeout_leaves_nvml_initialized_until_thread_exits(caplog):
    fake_nvml = _FakeNVML()
    thread = _ControlledThread()
    sampler = _LocalNVMLSampler(sample_interval_s=0.2, zero_utilization_threshold=1.0)
    sampler._nvml = fake_nvml
    sampler._thread = thread
    sampler._started = True

    with caplog.at_level(logging.WARNING):
        sampler.close()

    assert thread.join_timeouts == [1.2]
    assert sampler._thread is thread
    assert sampler.started
    assert fake_nvml.shutdowns == 0
    assert "leaving NVML initialized" in caplog.text

    thread.alive = False
    sampler.close()
    sampler.close()

    assert sampler._thread is None
    assert not sampler.started
    assert fake_nvml.shutdowns == 1


def test_missing_pynvml_degrades_without_raising(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    sampler = _LocalNVMLSampler(sample_interval_s=0.2, zero_utilization_threshold=1.0)

    reason = sampler.start()

    assert reason is not None
    assert "NVML is unavailable" in reason
    assert not sampler.started
    sampler.close()


def test_transient_device_failure_is_retried():
    fake_nvml = _FakeNVML({0: [RuntimeError("transient"), 42.0]})
    sampler = _LocalNVMLSampler(sample_interval_s=0.2, zero_utilization_threshold=1.0)
    sampler._nvml = fake_nvml
    sampler._devices = [_Device("0000:01:00.0", 0)]

    sampler.sample_once()
    sampler.sample_once()

    assert sampler.snapshot_and_reset()["0000:01:00.0"]["mean"] == 42.0
    assert sampler._failed_devices == set()
    assert sampler._consecutive_device_failures == {}
    assert fake_nvml.calls[0] == 2


def test_consecutive_device_failures_do_not_drop_healthy_device(caplog):
    fake_nvml = _FakeNVML({0: 42.0, 1: RuntimeError("bad device")})
    sampler = _LocalNVMLSampler(sample_interval_s=0.2, zero_utilization_threshold=1.0)
    sampler._nvml = fake_nvml
    sampler._devices = [
        _Device("0000:01:00.0", 0),
        _Device("0000:02:00.0", 1),
    ]

    with caplog.at_level(logging.WARNING):
        for _ in range(sampler._MAX_CONSECUTIVE_DEVICE_FAILURES):
            sampler.sample_once()
        sampler.sample_once()
    summary = sampler.snapshot_and_reset()

    assert summary == {
        "0000:01:00.0": {
            "mean": 42.0,
            "min": 42.0,
            "max": 42.0,
            "zero_fraction": 0.0,
            "sample_count": 4.0,
        }
    }
    assert sampler._failed_devices == {"0000:02:00.0"}
    assert fake_nvml.calls[1] == sampler._MAX_CONSECUTIVE_DEVICE_FAILURES
    assert "after 3 consecutive NVML failures" in caplog.text


def test_statistics_are_per_pci_device_and_empty_windows_are_safe():
    sampler = _LocalNVMLSampler(sample_interval_s=0.2, zero_utilization_threshold=1.0)
    sampler._samples = [
        ("0000:01:00.0", 0.0),
        ("0000:01:00.0", 2.0),
        ("0000:02:00.0", 50.0),
    ]

    summary = sampler.snapshot_and_reset()

    assert set(summary) == {"0000:01:00.0", "0000:02:00.0"}
    assert summary["0000:01:00.0"] == {
        "mean": 1.0,
        "min": 0.0,
        "max": 2.0,
        "zero_fraction": 0.5,
        "sample_count": 2.0,
    }
    assert summary["0000:02:00.0"]["mean"] == 50.0
    assert sampler.snapshot_and_reset() == {}


def test_nested_phase_windows_do_not_contaminate_each_other(monkeypatch):
    fake_nvml = _FakeNVML()
    _install_fake_nvml(monkeypatch, fake_nvml)
    node_sampler = _NodeGPUStallSampler(sample_interval_s=0.2, zero_utilization_threshold=1.0)
    try:
        node_sampler.begin_phase("training_step")
        node_sampler._sampler._samples.append(("0000:01:00.0", 10.0))
        node_sampler.begin_phase("rollout_generate")
        node_sampler._sampler._samples.append(("0000:02:00.0", 20.0))

        child_summary, _, reason = node_sampler.end_phase("rollout_generate")
        parent_summary, _, _ = node_sampler.end_phase("training_step")

        assert reason is None
        assert set(child_summary) == {"0000:02:00.0"}
        assert set(parent_summary) == {"0000:01:00.0", "0000:02:00.0"}
        assert node_sampler._sampler.snapshot_and_reset() == {}
    finally:
        node_sampler.close()


def test_fake_ray_creates_one_zero_gpu_actor_per_live_gpu_node_and_aggregates(monkeypatch):
    fake_ray = _FakeRay(
        [
            {"NodeID": "gpu-b", "Alive": True, "Resources": {"GPU": 2}},
            {"NodeID": "dead-gpu", "Alive": False, "Resources": {"GPU": 8}},
            {"NodeID": "cpu-only", "Alive": True, "Resources": {"CPU": 4}},
            {"NodeID": "gpu-a", "Alive": True, "Resources": {"GPU": 1}},
        ]
    )
    _install_fake_ray(monkeypatch, fake_ray)

    diagnostics = start_gpu_stall_diagnostics(GPUStallDiagnosticsConfig(enable=True))

    assert diagnostics is not None
    assert diagnostics._node_ids == ["gpu-a", "gpu-b"]
    assert len(fake_ray.created) == 2
    for index, (_, options, args) in enumerate(fake_ray.created):
        assert options["num_cpus"] == 0
        assert options["num_gpus"] == 0
        strategy = options["scheduling_strategy"]
        assert isinstance(strategy, _FakeNodeAffinitySchedulingStrategy)
        assert strategy.node_id == diagnostics._node_ids[index]
        assert strategy.soft is False
        assert args == (0.25, 1.0)

    diagnostics._actors[0].end_result = (
        {
            "0000:01:00.0": {
                "mean": 20.0,
                "min": 10.0,
                "max": 30.0,
                "zero_fraction": 0.25,
                "sample_count": 4.0,
            },
            "0000:02:00.0": {
                "mean": 50.0,
                "min": 50.0,
                "max": 50.0,
                "zero_fraction": 0.0,
                "sample_count": 2.0,
            },
        },
        2,
        None,
    )
    diagnostics._actors[1].end_result = (
        {
            "0000:03:00.0": {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "zero_fraction": 1.0,
                "sample_count": 3.0,
            }
        },
        1,
        None,
    )

    diagnostics.begin_phase("rollout_generate")
    metrics = diagnostics.end_phase("rollout_generate")

    assert [actor.begin_phases for actor in diagnostics._actors] == [
        ["rollout_generate"],
        ["rollout_generate"],
    ]
    assert metrics["gpu_stall/rollout_generate/node_0/device_count"] == 2.0
    assert metrics["gpu_stall/rollout_generate/node_0/sample_count"] == 6.0
    assert metrics["gpu_stall/rollout_generate/node_0/utilization_mean"] == 30.0
    assert metrics["gpu_stall/rollout_generate/node_0/utilization_min"] == 10.0
    assert metrics["gpu_stall/rollout_generate/node_0/utilization_max"] == 50.0
    assert metrics["gpu_stall/rollout_generate/node_0/zero_utilization_fraction"] == pytest.approx(1 / 6)
    assert metrics["gpu_stall/rollout_generate/node_1/sample_count"] == 3.0
    assert metrics["gpu_stall/rollout_generate/node_1/zero_utilization_fraction"] == 1.0

    diagnostics.close()
    diagnostics.close()

    assert [actor.close_calls for actor in diagnostics._actors] == [1, 1]
    assert fake_ray.kill_attempts == [(diagnostics._actors[0], True), (diagnostics._actors[1], True)]


def test_partial_actor_creation_failure_cleans_up_created_actors(monkeypatch):
    fake_ray = _FakeRay(
        [
            {"NodeID": "gpu-b", "Alive": True, "Resources": {"GPU": 1}},
            {"NodeID": "gpu-a", "Alive": True, "Resources": {"GPU": 1}},
        ]
    )
    fake_ray.remote_creation_error_at = 1
    _install_fake_ray(monkeypatch, fake_ray)

    diagnostics = start_gpu_stall_diagnostics(GPUStallDiagnosticsConfig(enable=True))

    assert diagnostics is None
    assert len(fake_ray.created) == 1
    assert fake_ray.kill_attempts == [(fake_ray.created[0][0], True)]


def test_ray_rpc_timeout_is_fail_open(monkeypatch):
    fake_ray = _FakeRay()
    _install_fake_ray(monkeypatch, fake_ray)
    actor = _FakeActor(0)
    diagnostics = GPUStallDiagnostics([actor], ["gpu-node"])
    fake_ray.get_errors = [TimeoutError("begin timeout"), TimeoutError("end timeout")]

    diagnostics.begin_phase("rollout_generate")
    metrics = diagnostics.end_phase("rollout_generate")

    assert metrics == {}
    assert fake_ray.get_timeouts == [diagnostics._RPC_TIMEOUT_S, diagnostics._RPC_TIMEOUT_S]

    diagnostics.close()
    assert fake_ray.kill_attempts == [(actor, True)]


def test_one_actor_kill_failure_does_not_block_other_cleanup(monkeypatch, caplog):
    fake_ray = _FakeRay()
    _install_fake_ray(monkeypatch, fake_ray)
    actors = [_FakeActor(0), _FakeActor(1), _FakeActor(2)]
    fake_ray.kill_failures.add(actors[0])
    diagnostics = GPUStallDiagnostics(actors, ["a", "b", "c"])

    with caplog.at_level(logging.WARNING):
        diagnostics.close()
        diagnostics.close()

    assert fake_ray.kill_attempts == [(actor, True) for actor in actors]
    assert [actor.close_calls for actor in actors] == [1, 1, 1]
    assert "could not destroy node sampler 0" in caplog.text


def test_default_off_path_does_not_import_ray_or_pynvml(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"ray", "pynvml"}:
            raise AssertionError(f"default-off diagnostics imported optional dependency: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.reload(sys.modules["verl.utils.profiler.gpu_stall_diagnostics"])
    assert module.start_gpu_stall_diagnostics(GPUStallDiagnosticsConfig()) is None


def test_gpu_stall_diagnostics_config_hydra_conversion_and_validation():
    config = omega_conf_to_dataclass(
        OmegaConf.create(
            {
                "_target_": "verl.utils.profiler.config.GPUStallDiagnosticsConfig",
                "enable": True,
                "sample_interval_s": 0.2,
                "zero_utilization_threshold": 0.0,
            }
        )
    )

    assert isinstance(config, GPUStallDiagnosticsConfig)
    assert config.enable
    assert config.sample_interval_s == 0.2
    with pytest.raises(AssertionError, match="sample_interval_s"):
        GPUStallDiagnosticsConfig(sample_interval_s=0.1)
    with pytest.raises(AssertionError, match="zero_utilization_threshold"):
        GPUStallDiagnosticsConfig(zero_utilization_threshold=101.0)
