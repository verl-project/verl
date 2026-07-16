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
"""Low-overhead, phase-aware GPU utilization diagnostics.

When enabled, the trainer creates one zero-GPU Ray actor on every live GPU
node. Every actor polls NVML in a daemon thread and returns only aggregate
statistics for explicitly delimited phases. NVML is imported lazily and this
module never calls torch.cuda, so it cannot create a CUDA context.

The metrics include a logical node index and an NVML PCI bus ID. This makes the
multi-node scope explicit and avoids treating CUDA_VISIBLE_DEVICES ordinals as
NVML indices.
"""

from __future__ import annotations

import functools
import logging
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

logger = logging.getLogger(__name__)


def _merge_gpu_stall_metrics(metrics: dict[str, float], updates: dict[str, float]) -> None:
    """Merge diagnostics without replacing metrics already owned by the trainer."""
    collisions = sorted(metrics.keys() & updates.keys())
    if collisions:
        logger.warning("GPU stall diagnostics did not overwrite existing metric key(s): %s", collisions)
    metrics.update({key: value for key, value in updates.items() if key not in metrics})


@contextmanager
def gpu_stall_diagnostics_phase(diagnostics: Any, phase: str, metrics: dict[str, float]) -> Iterator[None]:
    """Run one optional phase without allowing diagnostics failures into training."""
    if diagnostics is None:
        yield
        return

    began = False
    try:
        diagnostics.begin_phase(phase)
        began = True
    except Exception as error:
        logger.warning("GPU stall diagnostics could not begin phase %s: %s", phase, error)

    try:
        yield
    finally:
        if began:
            try:
                updates = diagnostics.end_phase(phase)
            except Exception as error:
                logger.warning("GPU stall diagnostics could not end phase %s: %s", phase, error)
            else:
                _merge_gpu_stall_metrics(metrics, updates)


def close_gpu_stall_diagnostics_on_exit(method):
    """Close an owner's optional diagnostics on every decorated method exit."""

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        finally:
            diagnostics = getattr(self, "_gpu_stall_diagnostics", None)
            if diagnostics is not None:
                try:
                    diagnostics.close()
                except Exception as error:
                    logger.warning("GPU stall diagnostics could not close cleanly: %s", error)
                finally:
                    self._gpu_stall_diagnostics = None

    return wrapped


@dataclass(frozen=True)
class _Device:
    """An NVML device named by its physical PCI bus identifier."""

    bus_id: str
    handle: Any


class _LocalNVMLSampler:
    """Background NVML sampler that never raises into its caller."""

    _MAX_CONSECUTIVE_DEVICE_FAILURES = 3

    def __init__(self, sample_interval_s: float, zero_utilization_threshold: float):
        self._sample_interval_s = sample_interval_s
        self._zero_utilization_threshold = zero_utilization_threshold
        self._samples: list[tuple[str, float]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml: Any = None
        self._devices: list[_Device] = []
        self._failed_devices: set[str] = set()
        self._consecutive_device_failures: dict[str, int] = {}
        self._started = False

    @staticmethod
    def _pci_bus_id(pci_info: Any) -> str:
        bus_id = getattr(pci_info, "busId", "unknown")
        if isinstance(bus_id, bytes):
            bus_id = bus_id.decode("utf-8")
        return str(bus_id).lower()

    @property
    def device_count(self) -> int:
        return len(self._devices)

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> str | None:
        """Start sampling, returning a reason when NVML cannot be used."""
        if self._started:
            return None
        self._stop_event.clear()
        self._failed_devices.clear()
        self._consecutive_device_failures.clear()
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._devices = [
                _Device(
                    bus_id=self._pci_bus_id(pynvml.nvmlDeviceGetPciInfo(handle)),
                    handle=handle,
                )
                for handle in (pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(pynvml.nvmlDeviceGetCount()))
            ]
        except Exception as error:
            if self._nvml is not None:
                try:
                    self._nvml.nvmlShutdown()
                except Exception:
                    pass
            self._nvml = None
            self._devices = []
            return f"NVML is unavailable: {error}"

        if not self._devices:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None
            return "NVML reports no GPU devices"

        self._started = True
        self._thread = threading.Thread(target=self._run, name="verl-gpu-stall-sampler", daemon=True)
        self._thread.start()
        return None

    def _run(self) -> None:
        while not self._stop_event.wait(self._sample_interval_s):
            self.sample_once()

    def mark(self) -> int:
        """Return a boundary that can be summarized later without resetting samples."""
        with self._lock:
            return len(self._samples)

    def summarize_since(self, start: int) -> dict[str, dict[str, float]]:
        """Summarize samples after ``start`` without disturbing overlapping phases."""
        with self._lock:
            samples = list(self._samples[start:])

        grouped: dict[str, list[float]] = defaultdict(list)
        for bus_id, utilization in samples:
            grouped[bus_id].append(utilization)

        summaries: dict[str, dict[str, float]] = {}
        for bus_id, values in grouped.items():
            count = len(values)
            summaries[bus_id] = {
                "mean": sum(values) / count,
                "min": min(values),
                "max": max(values),
                "zero_fraction": sum(value <= self._zero_utilization_threshold for value in values) / count,
                "sample_count": float(count),
            }
        return summaries

    def sample_once(self) -> None:
        """Collect one sample from each healthy NVML device for deterministic tests."""
        for device in self._devices:
            if device.bus_id in self._failed_devices:
                continue
            try:
                utilization = float(self._nvml.nvmlDeviceGetUtilizationRates(device.handle).gpu)
            except Exception as error:
                failure_count = self._consecutive_device_failures.get(device.bus_id, 0) + 1
                self._consecutive_device_failures[device.bus_id] = failure_count
                if failure_count >= self._MAX_CONSECUTIVE_DEVICE_FAILURES:
                    self._failed_devices.add(device.bus_id)
                    logger.warning(
                        "GPU stall diagnostics stopped sampling PCI device %s after %d consecutive NVML failures: %s",
                        device.bus_id,
                        failure_count,
                        error,
                    )
                continue
            self._consecutive_device_failures.pop(device.bus_id, None)
            with self._lock:
                self._samples.append((device.bus_id, utilization))

    def snapshot_and_reset(self) -> dict[str, dict[str, float]]:
        """Summarize samples since the previous boundary without division by zero."""
        with self._lock:
            samples = self._samples
            self._samples = []

        grouped: dict[str, list[float]] = defaultdict(list)
        for bus_id, utilization in samples:
            grouped[bus_id].append(utilization)

        summaries: dict[str, dict[str, float]] = {}
        for bus_id, values in grouped.items():
            count = len(values)
            summaries[bus_id] = {
                "mean": sum(values) / count,
                "min": min(values),
                "max": max(values),
                "zero_fraction": sum(value <= self._zero_utilization_threshold for value in values) / count,
                "sample_count": float(count),
            }
        return summaries

    def close(self) -> None:
        """Stop the daemon thread and shut down NVML. Idempotent."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._sample_interval_s + 1.0)
            if thread.is_alive():
                logger.warning(
                    "GPU stall diagnostics sampler thread did not stop before the join timeout; "
                    "leaving NVML initialized so the thread can finish safely. Call close() again to retry cleanup."
                )
                return
        self._thread = None
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        self._nvml = None
        self._devices = []
        self._failed_devices.clear()
        self._consecutive_device_failures.clear()
        self._started = False


class _NodeGPUStallSampler:
    """Implementation run inside one Ray actor on one GPU node."""

    def __init__(self, sample_interval_s: float, zero_utilization_threshold: float):
        self._sampler = _LocalNVMLSampler(sample_interval_s, zero_utilization_threshold)
        self._reason = self._sampler.start()
        self._phase_starts: dict[str, int] = {}

    def begin_phase(self, phase: str) -> None:
        self._phase_starts[phase] = self._sampler.mark()

    def end_phase(self, phase: str) -> tuple[dict[str, dict[str, float]], int, str | None]:
        start = self._phase_starts.pop(phase, self._sampler.mark())
        summary = self._sampler.summarize_since(start)
        if not self._phase_starts:
            self._sampler.snapshot_and_reset()
        return summary, self._sampler.device_count, self._reason

    def close(self) -> None:
        self._sampler.close()


class GPUStallDiagnostics:
    """Coordinate node-local NVML samplers without affecting normal execution."""

    _RPC_TIMEOUT_S = 5

    def __init__(self, actors: list[Any], node_ids: list[str]):
        self._actors = actors
        self._node_ids = node_ids
        self._closed = False
        self._warned = False
        logger.info("GPU stall diagnostics node index mapping: %s", dict(enumerate(node_ids)))

    @classmethod
    def create(cls, config: Any) -> GPUStallDiagnostics | None:
        """Create a sampler actor per Ray GPU node only when explicitly enabled."""
        if config is None or not getattr(config, "enable", False):
            return None

        actors: list[Any] = []
        ray_module: Any = None
        try:
            import ray
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            ray_module = ray
            nodes = sorted(
                (
                    node
                    for node in ray.nodes()
                    if node.get("Alive") and float(node.get("Resources", {}).get("GPU", 0)) > 0
                ),
                key=lambda node: node["NodeID"],
            )
            if not nodes:
                logger.warning("GPU stall diagnostics disabled: Ray reports no live GPU nodes.")
                return None

            remote_cls = ray.remote(_NodeGPUStallSampler)
            for node in nodes:
                actor = remote_cls.options(
                    num_cpus=0,
                    num_gpus=0,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False),
                ).remote(config.sample_interval_s, config.zero_utilization_threshold)
                actors.append(actor)
            return cls(actors, [node["NodeID"] for node in nodes])
        except Exception as error:
            if ray_module is not None:
                for actor_index, actor in enumerate(actors):
                    try:
                        ray_module.kill(actor, no_restart=True)
                    except Exception as cleanup_error:
                        logger.warning(
                            "GPU stall diagnostics could not destroy partially created node sampler %d: %s",
                            actor_index,
                            cleanup_error,
                        )
            logger.warning("GPU stall diagnostics disabled: could not start node samplers: %s", error)
            return None

    def _warn_once(self, message: str, *args: object) -> None:
        if not self._warned:
            logger.warning(message, *args)
            self._warned = True

    def _begin_phase(self, phase: str) -> None:
        if self._closed:
            return
        try:
            import ray

            ray.get([actor.begin_phase.remote(phase) for actor in self._actors], timeout=self._RPC_TIMEOUT_S)
        except Exception as error:
            self._warn_once("GPU stall diagnostics could not begin a phase: %s", error)

    @staticmethod
    def _metric_key(phase: str, node_index: int, suffix: str) -> str:
        return f"gpu_stall/{phase}/node_{node_index}/{suffix}"

    def _end_phase(self, phase: str) -> dict[str, float]:
        if self._closed:
            return {}
        try:
            import ray

            summaries = ray.get([actor.end_phase.remote(phase) for actor in self._actors], timeout=self._RPC_TIMEOUT_S)
        except Exception as error:
            self._warn_once("GPU stall diagnostics could not finish phase %s: %s", phase, error)
            return {}

        metrics: dict[str, float] = {}
        unavailable_nodes: list[int] = []
        for node_index, (devices, device_count, reason) in enumerate(summaries):
            if reason is not None:
                unavailable_nodes.append(node_index)
            metrics[self._metric_key(phase, node_index, "device_count")] = float(device_count)
            sample_count = sum(value["sample_count"] for value in devices.values())
            metrics[self._metric_key(phase, node_index, "sample_count")] = sample_count
            if sample_count == 0:
                continue

            values = list(devices.values())
            metrics[self._metric_key(phase, node_index, "utilization_mean")] = (
                sum(value["mean"] * value["sample_count"] for value in values) / sample_count
            )
            metrics[self._metric_key(phase, node_index, "utilization_min")] = min(value["min"] for value in values)
            metrics[self._metric_key(phase, node_index, "utilization_max")] = max(value["max"] for value in values)
            metrics[self._metric_key(phase, node_index, "zero_utilization_fraction")] = (
                sum(value["zero_fraction"] * value["sample_count"] for value in values) / sample_count
            )

            for bus_id, value in sorted(devices.items()):
                safe_bus_id = bus_id.replace(":", "_").replace(".", "_")
                metrics[self._metric_key(phase, node_index, f"gpu_{safe_bus_id}/sample_count")] = value["sample_count"]
                metrics[self._metric_key(phase, node_index, f"gpu_{safe_bus_id}/utilization_mean")] = value["mean"]
                metrics[self._metric_key(phase, node_index, f"gpu_{safe_bus_id}/zero_utilization_fraction")] = value[
                    "zero_fraction"
                ]

        if unavailable_nodes:
            self._warn_once("GPU stall diagnostics unavailable on node sampler(s): %s", unavailable_nodes)
        return metrics

    def begin_phase(self, phase: str) -> None:
        """Start a named phase; phases may overlap without cross-contamination."""
        self._begin_phase(phase)

    def end_phase(self, phase: str) -> dict[str, float]:
        """Finish a named phase and return its aggregate metric values."""
        return self._end_phase(phase)

    @contextmanager
    def phase(self, phase: str) -> Iterator[dict[str, float]]:
        """Collect continuous, bounded-rate samples during a named phase."""
        metrics: dict[str, float] = {}
        self._begin_phase(phase)
        try:
            yield metrics
        finally:
            metrics.update(self._end_phase(phase))

    def close(self) -> None:
        """Stop and destroy all Ray actors. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        try:
            import ray
        except Exception as error:
            self._warn_once("GPU stall diagnostics could not cleanly stop: %s", error)
            return

        try:
            ray.get([actor.close.remote() for actor in self._actors], timeout=self._RPC_TIMEOUT_S)
        except Exception as error:
            self._warn_once("GPU stall diagnostics could not cleanly stop: %s", error)

        for actor_index, actor in enumerate(self._actors):
            try:
                ray.kill(actor, no_restart=True)
            except Exception as error:
                logger.warning("GPU stall diagnostics could not destroy node sampler %d: %s", actor_index, error)


def start_gpu_stall_diagnostics(config: Any) -> GPUStallDiagnostics | None:
    """Construct the optional coordinator without importing NVML while disabled."""
    return GPUStallDiagnostics.create(config)
