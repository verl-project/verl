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
"""On-demand shared replica fleet (lazy-imported from ``teacher_model.py``).

::

    MultiTeacherModelManager
      └─ SharedTeacherFleetManager          # launch replicas, build mmap cache
           ├─ TeacherWeightSwapVLLMReplica  # vLLM + weight-swap RPCs
           ├─ teacher_weight_cache          # snapshot / map / H2D
           ├─ teacher_port_isolation        # disjoint gloo / vLLM ports
           └─ SharedTeacherFleet            # Ray actor
                └─ SharedTeacherLeases      # occupancy / donate (slot_scheduler)

    AgentLoopManagerTQ.generate_sequences
      └─ notify_on_demand_occupancy → fleet.set_step_key_counts
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import time
from contextlib import nullcontext
from importlib import import_module
from typing import Any, Optional

import ray
import torch
from omegaconf import OmegaConf

from verl.single_controller.ray.base import RayResourcePool, split_resource_pool
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.ray_utils import auto_await
from verl.utils.tracking import RLInsightLogger
from verl.workers.config import DistillationConfig
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.utils import update_prometheus_config
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

from .teacher_port_isolation import (
    ServerClassWithEnv,
    install_vllm_port_guard,
    replica_gloo_port_range,
    replica_port_kind,
    replica_server_env,
)
from .slot_scheduler import SharedTeacherLeases, occupancy_spawn_keys
from .teacher_weight_cache import (
    apply_shard_to_models,
    cleanup_cache_dir,
    estimate_model_bytes,
    get_tp_rank,
    load_shard_mmap,
    make_job_cache_dir,
    make_teacher_cache_dir,
    prepare_cache_dir,
    register_cleanup_on_exit,
    resolve_cache_root,
    tp_shard_dir,
    write_shard_from_models,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_WORKER_EXTENSION_CLS = "verl.experimental.teacher_loop.teacher_shared_fleet.TeacherWeightSwapWorkerExtension"


# ---------------------------------------------------------------------------
# Cold load: pull a HuggingFace checkpoint into the dummy-booted vLLM worker
# ---------------------------------------------------------------------------
def _with_auto_load_format(load_config: Any) -> Any:
    """Copy ``load_config`` so ``DummyModelLoader`` is not used for a real checkpoint.

    ``LoadFormat`` is not re-exported by ``vllm.config`` in every build, so reuse the
    enum class carried by the configured value instead of importing it.
    """
    current = getattr(load_config, "load_format", None)
    load_format = getattr(type(current), "AUTO", None) or "auto"
    if dataclasses.is_dataclass(load_config):
        return dataclasses.replace(load_config, load_format=load_format)
    return load_config.model_copy(update={"load_format": load_format})


def _vllm_config_context(vllm_config: Any):
    """vLLM CustomOp / EP filters read a thread-local config.

    The entry point is not re-exported from ``vllm.config`` in every build, so probe
    the known locations and degrade to a no-op context instead of failing the load.
    """
    for module_name in ("vllm.config", "vllm.config.vllm"):
        try:
            set_current_vllm_config = getattr(import_module(module_name), "set_current_vllm_config", None)
        except ImportError:
            continue
        if set_current_vllm_config is not None:
            return set_current_vllm_config(vllm_config)
    logger.warning("[teacher_weight_cache] set_current_vllm_config not found; loading without it")
    return nullcontext()


# ---------------------------------------------------------------------------
# Worker extension: snapshot a TP shard, map it, and copy it into the live model
# ---------------------------------------------------------------------------
class TeacherWeightSwapWorkerExtension(vLLMColocateWorkerExtension):
    """Colocate worker that can swap between cached teacher checkpoints."""

    def _teacher_shards(self) -> dict[str, dict]:
        if not hasattr(self, "_teacher_weight_shards"):
            self._teacher_weight_shards = {}
        return self._teacher_weight_shards

    def _ensure_device(self) -> None:
        if self.device is None:
            self.device = torch.device(f"{get_device_name()}:{self.local_rank}")

    def _tp_rank(self) -> int:
        parallel_config = getattr(getattr(self.model_runner, "vllm_config", None), "parallel_config", None)
        return get_tp_rank(self.local_rank, parallel_config)

    def _shard_dir(self, teacher_dir: str) -> str:
        return tp_shard_dir(teacher_dir, self._tp_rank())

    def teacher_prepare_cache_dir(self, cache_dir: str, num_teachers: int) -> None:
        """Create the cache directory and check capacity before any teacher is built."""
        self._ensure_device()
        models = list(self._iter_all_models())
        # The boot model stands in for every teacher; heterogeneous fleets
        # over- or under-estimate here and are re-checked per shard on write.
        per_teacher = estimate_model_bytes(models)
        prepare_cache_dir(cache_dir, per_teacher * max(int(num_teachers), 1))

    def teacher_cleanup_cache_dir(self, cache_dir: str) -> None:
        cleanup_cache_dir(cache_dir)

    def _load_checkpoint(self, models: list[Any], model_path: str) -> None:
        """Load a HuggingFace checkpoint over the ``load_format=dummy`` boot weights."""
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

        for model in models:
            patch_vllm_moe_model_weight_loader(model)

        vllm_config = self.model_runner.vllm_config
        model_config = vllm_config.model_config
        loader = DefaultModelLoader(_with_auto_load_format(vllm_config.load_config))
        original_model_path = model_config.model
        try:
            # The loader reads the checkpoint location from ``model_config``.
            model_config.model = model_path
            with _vllm_config_context(vllm_config):
                init_ep = getattr(loader, "_init_ep_weight_filter", None)
                if init_ep is not None:
                    init_ep(model_config)
                for model in models:
                    loader.load_weights(model, model_config)
                # Non-idempotent: run once after all weights are in, so the
                # snapshot matches the layout the kernels actually use.
                for model in models:
                    process_weights_after_loading(model, model_config, self.device)
        finally:
            model_config.model = original_model_path

    def teacher_build_cache(self, teacher_key: str, model_path: str, teacher_dir: str) -> int:
        """Load ``model_path`` into the dummy boot weights and snapshot one TP shard."""
        self._ensure_device()
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Teacher checkpoint path not found: {model_path}")
        started = time.perf_counter()
        models = list(self._iter_all_models())
        self._load_checkpoint(models, model_path)
        shard_dir = self._shard_dir(teacher_dir)
        nbytes = write_shard_from_models(
            models, shard_dir, min_free_bytes=estimate_model_bytes(models), model_path=model_path
        )
        if os.path.abspath(teacher_dir).startswith("/dev/shm/"):
            register_cleanup_on_exit(os.path.dirname(os.path.abspath(teacher_dir)))
        logger.warning(
            "[teacher_weight_cache] built %r shard=%s bytes=%.2fGiB in %.1fs rank=%s",
            teacher_key,
            shard_dir,
            nbytes / (1 << 30),
            time.perf_counter() - started,
            getattr(self, "local_rank", "?"),
        )
        return nbytes

    def teacher_map_cache(self, teacher_key: str, model_path: str, teacher_dir: str) -> None:
        """Map an already-built shard for this TP rank."""
        shard_dir = self._shard_dir(teacher_dir)
        if not os.path.isfile(os.path.join(shard_dir, "meta.json")):
            raise FileNotFoundError(f"Missing TP shard for teacher {teacher_key!r}: {shard_dir}")
        self._teacher_shards()[teacher_key] = load_shard_mmap(shard_dir, model_path=model_path)
        if os.path.abspath(teacher_dir).startswith("/dev/shm/"):
            register_cleanup_on_exit(os.path.dirname(os.path.abspath(teacher_dir)))
        logger.info(
            "[teacher_weight_cache] mapped %r shard=%s rank=%s",
            teacher_key,
            shard_dir,
            getattr(self, "local_rank", "?"),
        )

    def teacher_apply_from_cache(self, teacher_key: str) -> None:
        """Copy the mapped shard into the live model in place."""
        shard = self._teacher_shards().get(teacher_key)
        if shard is None:
            raise KeyError(f"No mapped weight cache for teacher {teacher_key!r}")
        self._ensure_device()
        started = time.perf_counter()
        params, buffers = apply_shard_to_models(self._iter_all_models(), shard)
        get_torch_device().synchronize()
        logger.warning(
            "[teacher_weight_cache] applied %r params=%d buffers=%d in %.1fs rank=%s",
            teacher_key,
            params,
            buffers,
            time.perf_counter() - started,
            getattr(self, "local_rank", "?"),
        )


# ---------------------------------------------------------------------------
# Server / replica: add teacher weight RPCs on top of the stock vLLM server
# ---------------------------------------------------------------------------
class TeacherWeightSwapVLLMHttpServer(vLLMHttpServer):
    """vLLM HTTP server whose workers can snapshot and swap teacher weights."""

    def __init__(self, *args, **kwargs):
        # The engine spawns from this actor, so the port guard has to be
        # installed here rather than in the driver process that requested it.
        install_vllm_port_guard()
        super().__init__(*args, **kwargs)

    def _get_worker_extension_cls(self) -> str:
        return _WORKER_EXTENSION_CLS

    async def _rank0_rpc(self, method: str, kwargs: Optional[dict] = None) -> Optional[list]:
        # Only node rank 0 owns the engine; the other nodes of a replica run
        # vLLM's headless entry point and have no engine to RPC through. The
        # shared fleet keeps each replica on a single node, so this covers it.
        if self.node_rank != 0:
            return None
        return await self.collective_rpc(method, kwargs=kwargs or {})

    async def teacher_prepare_cache_dir(self, cache_dir: str, num_teachers: int) -> None:
        await self._rank0_rpc(
            "teacher_prepare_cache_dir", {"cache_dir": cache_dir, "num_teachers": num_teachers}
        )

    async def teacher_cleanup_cache_dir(self, cache_dir: str) -> None:
        await self._rank0_rpc("teacher_cleanup_cache_dir", {"cache_dir": cache_dir})

    async def teacher_build_cache(self, teacher_key: str, model_path: str, teacher_dir: str) -> int:
        results = await self._rank0_rpc(
            "teacher_build_cache",
            {"teacher_key": teacher_key, "model_path": model_path, "teacher_dir": teacher_dir},
        )
        return int(results[0]) if results else 0

    async def teacher_map_cache(self, teacher_key: str, model_path: str, teacher_dir: str) -> None:
        await self._rank0_rpc(
            "teacher_map_cache",
            {"teacher_key": teacher_key, "model_path": model_path, "teacher_dir": teacher_dir},
        )

    async def teacher_apply_from_cache(self, teacher_key: str) -> None:
        """Copy a mapped teacher into the live weights and clear stale KV cache."""
        if self.node_rank != 0:
            return
        started = time.perf_counter()
        await self.wait_for_requests_to_drain()
        # No request may observe the old weights afterwards: abort anything that
        # slipped in, and drop the prefix cache keyed by the previous teacher.
        await self.abort_all_requests(reset_prefix_cache=True)
        try:
            await self.collective_rpc("teacher_apply_from_cache", kwargs={"teacher_key": teacher_key})
        finally:
            await self.resume_generation()
        logger.warning(
            "[shared_teacher_fleet] applied %r replica=%s in %.1fs",
            teacher_key,
            self.replica_rank,
            time.perf_counter() - started,
        )


class TeacherWeightSwapVLLMReplica(vLLMReplica):
    """Community vLLM replica wired to the teacher weight-swap server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(TeacherWeightSwapVLLMHttpServer)

    async def init_colocated(self, resource_pool, master_port_range=None):
        """Colocate this replica and pin its gloo MASTER_PORT to ``master_port_range``.

        Copied from ``RolloutReplica.init_colocated`` so the public replica API
        does not grow a teacher-only kwarg. ``RayWorkerGroup`` already accepts
        ``master_port_range``; concurrent fleet replicas pass disjoint windows.
        """
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.rollout.replica import RolloutMode

        self.rollout_mode = RolloutMode.COLOCATED
        self.resource_pool = resource_pool
        use_gpu = self.rollout_worker_use_gpu()
        if self.is_reward_model:
            name_prefix = f"rollout_reward_colocate_{self.replica_rank}{self.name_suffix}"
        elif self.is_teacher_model:
            name_prefix = f"rollout_teacher_colocate_{self.replica_rank}{self.name_suffix}"
        else:
            name_prefix = f"rollout_colocate_{self.replica_rank}{self.name_suffix}"
        worker_group = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=self.get_ray_class_with_init_args(),
            bin_pack=False,
            name_prefix=name_prefix,
            use_gpu=use_gpu,
            device_name=get_device_name(),
            master_port_range=master_port_range,
        )
        self.workers = worker_group.workers
        await self.launch_servers()

    async def _on_servers(self, method: str, *args) -> list:
        return await asyncio.gather(*[getattr(server, method).remote(*args) for server in self.servers])

    async def teacher_prepare_cache_dir(self, cache_dir: str, num_teachers: int) -> None:
        await self._on_servers("teacher_prepare_cache_dir", cache_dir, num_teachers)

    async def teacher_cleanup_cache_dir(self, cache_dir: str) -> None:
        await self._on_servers("teacher_cleanup_cache_dir", cache_dir)

    async def teacher_build_cache(self, teacher_key: str, model_path: str, teacher_dir: str) -> int:
        results = await self._on_servers("teacher_build_cache", teacher_key, model_path, teacher_dir)
        return int(max(results or [0]))

    async def teacher_map_cache(self, teacher_key: str, model_path: str, teacher_dir: str) -> None:
        await self._on_servers("teacher_map_cache", teacher_key, model_path, teacher_dir)

    async def teacher_apply_from_cache(self, teacher_key: str) -> None:
        await self._on_servers("teacher_apply_from_cache", teacher_key)


# ---------------------------------------------------------------------------
# Fleet actor: occupancy / donate leases
# ---------------------------------------------------------------------------
@ray.remote
class SharedTeacherFleet:
    """Ray actor exposing occupancy leases to the agent loop workers."""

    def __init__(self, servers: list[tuple[str, Any]], key_priority: list[str]):
        self._servers = list(servers)
        self._leases = SharedTeacherLeases(self._apply, len(self._servers), key_priority)

    async def _apply(self, slot: int, teacher_key: str) -> None:
        # Same RPC as init-time replica.teacher_apply_from_cache: drain / abort /
        # overlay / resume live on the server. This actor must not run on the
        # CPU head — unpickled handles there have empty method signatures.
        server = self._servers[slot][1]
        started = time.perf_counter()
        await server.teacher_apply_from_cache.remote(teacher_key)
        logger.warning(
            "[shared_teacher_fleet] applied %r slot=%s in %.1fs",
            teacher_key,
            slot,
            time.perf_counter() - started,
        )

    async def acquire(self, teacher_key: str) -> int:
        return await self._leases.acquire(teacher_key)

    async def release(self, slot: int) -> bool:
        return await self._leases.release(slot)

    async def set_step_key_counts(self, counts: dict) -> None:
        """Install this student-generate batch's per-key sample counts."""
        await self._leases.set_step_key_counts(counts)

    def get_status(self) -> dict[str, Any]:
        return self._leases.snapshot()


# ---------------------------------------------------------------------------
# Client: drop-in LLMServerClient that leases a slot per request
# ---------------------------------------------------------------------------
class SharedTeacherLLMServerClient(LLMServerClient):
    """``LLMServerClient`` that routes every request to a leased teacher slot.

    One instance is created per teacher key and remembers its key, so the
    caller's ``generate()`` signature and the teacher manager stay unchanged.
    """

    def __init__(
        self,
        config,
        fleet_handle: Any,
        servers: list[tuple[str, Any]],
        teacher_key: str,
    ):
        super().__init__(config=config, load_balancer_handle=None)
        self._fleet = fleet_handle
        self._servers = list(servers)
        self._teacher_key = teacher_key
        self._slot_by_server = {server_id: slot for slot, (server_id, _) in enumerate(self._servers)}

    async def _acquire_server(self, request_id: str, **extra) -> tuple[str, Any]:
        slot = int(await self._fleet.acquire.remote(self._teacher_key))
        return self._servers[slot]

    def _release_server(self, server_id: str, request_id: str | None = None) -> None:
        slot = self._slot_by_server.get(server_id)
        if slot is None:
            return
        self._fleet.release.remote(slot)


def _shared_fleet_layout(distillation_config: DistillationConfig) -> tuple[int, int]:
    """TP and replica count for a fleet that fills the teacher pool."""
    teachers = distillation_config.teacher_models
    pool = distillation_config.n_gpus_per_node * distillation_config.nnodes
    n_gpus = distillation_config.n_gpus_per_node
    tps = {int(teacher.inference.tensor_model_parallel_size or 0) for teacher in teachers.values()}
    if len(tps) != 1:
        raise ValueError(
            "on_demand needs one inference.tensor_model_parallel_size for all teachers, "
            f"got {sorted(tps)}"
        )
    tp = tps.pop()
    if tp <= 0 or pool % tp != 0:
        raise ValueError(
            f"teacher pool {pool} must be divisible by tensor_model_parallel_size={tp}"
        )
    if n_gpus % tp != 0:
        raise ValueError(
            f"tensor_model_parallel_size={tp} must divide n_gpus_per_node={n_gpus} "
            "(a replica's weight cache is node-local)"
        )
    for key, teacher in teachers.items():
        inference = teacher.inference
        if inference.name != "vllm":
            raise NotImplementedError(f"on_demand only supports vLLM, got {key}={inference.name}")
        if inference.data_parallel_size != 1 or inference.pipeline_model_parallel_size != 1:
            raise ValueError(
                f"on_demand requires dp=1 pp=1, got {key} "
                f"dp={inference.data_parallel_size} pp={inference.pipeline_model_parallel_size}"
            )
    return tp, pool // tp


# ---------------------------------------------------------------------------
# Fleet manager: launch, build the cache, serve clients
# ---------------------------------------------------------------------------
class SharedTeacherFleetManager:
    """Owns the shared teacher replicas and the lease actor."""

    def __init__(self, distillation_config: DistillationConfig, resource_pool: RayResourcePool):
        self.distillation_config = distillation_config
        self.resource_pool = resource_pool
        self.teachers = dict(distillation_config.teacher_models)
        self.initial_key = next(iter(self.teachers))
        self.tp, self.num_replicas = _shared_fleet_layout(distillation_config)

        cache_root = resolve_cache_root(
            distillation_config.teacher_cache_backend, distillation_config.teacher_cache_dir
        )
        self.job_cache_dir = make_job_cache_dir(cache_root)
        self.teacher_cache_dirs = {key: make_teacher_cache_dir(self.job_cache_dir, key) for key in self.teachers}

        self.rollout_replicas: list[TeacherWeightSwapVLLMReplica] = []
        self.replicas_by_node: dict[int, list[TeacherWeightSwapVLLMReplica]] = {}
        self.fleet = None
        self._ready = False
        self._initialize()

    # -- lifecycle ---------------------------------------------------------
    @auto_await
    async def _initialize(self) -> None:
        try:
            await self._launch_replicas()
            await self._prepare_cache_dirs()
            await self._build_caches()
            await self._map_caches()
            await self._apply_initial_key()
            self._create_fleet()
            self._ready = True
            logger.warning(
                "[shared_teacher_fleet] ready replicas=%d teachers=%d tp=%d cache=%s backend=%s",
                self.num_replicas,
                len(self.teachers),
                self.tp,
                self.job_cache_dir,
                self.distillation_config.teacher_cache_backend,
            )
        except BaseException:
            logger.exception("[shared_teacher_fleet] initialization failed; cleaning up weight caches")
            await self._cleanup_caches()
            raise

    async def _launch_replicas(self) -> None:
        shared_inference = self._shared_inference_config()
        # Keep the model path unresolved until the server actor is placed, as in
        # the resident path: each node copies HDFS checkpoints into its own
        # local cache instead of the driver's /tmp.
        model_config = OmegaConf.create(
            {
                "_target_": "verl.workers.config.HFModelConfig",
                "path": self.teachers[self.initial_key].model_path,
            }
        )
        self.rollout_replicas = [
            TeacherWeightSwapVLLMReplica(
                replica_rank=replica_rank,
                config=shared_inference,
                model_config=model_config,
                gpus_per_node=self.distillation_config.n_gpus_per_node,
                is_teacher_model=True,
                name_suffix="shared",
            )
            for replica_rank in range(self.num_replicas)
        ]
        split_pools = split_resource_pool(self.resource_pool, split_size=self.tp)
        assert len(split_pools) == len(self.rollout_replicas), (
            f"{len(split_pools)} sub-pools for {len(self.rollout_replicas)} replicas"
        )
        # The weight cache lives on each node's local filesystem, so group the
        # replicas by node: every node builds its own copy of each teacher.
        gpus_per_node = self.distillation_config.n_gpus_per_node
        for replica, pool in zip(self.rollout_replicas, split_pools, strict=True):
            self.replicas_by_node.setdefault(pool.start_bundle_index // gpus_per_node, []).append(replica)

        # Replicas on one node are created concurrently, so each one gets its own
        # disjoint rendezvous window for both the gloo worker group and the vLLM
        # engine, instead of racing on probe-bind-close allocations. The engine
        # side is enforced by the port guard installed in the server actor.
        launches = []
        for replica, pool in zip(self.rollout_replicas, split_pools, strict=True):
            kind = replica_port_kind(is_teacher_model=replica.is_teacher_model)
            gloo_range = replica_gloo_port_range(replica.replica_rank, kind)
            replica_env = replica_server_env(replica.replica_rank, kind)
            replica.server_class = ServerClassWithEnv(replica.server_class, replica_env)
            logger.warning(
                "[shared_teacher_fleet] replica %d ports: gloo=%s vllm=%s",
                replica.replica_rank,
                gloo_range,
                replica_env["VLLM_PORT"],
            )
            launches.append(replica.init_colocated(pool, master_port_range=gloo_range))
        await asyncio.gather(*launches)
        self._register_metrics(shared_inference)

    def _register_metrics(self, inference_config) -> None:
        """Expose fleet replicas to teacher metrics, as the resident teacher path does."""
        if inference_config.disable_log_stats:
            if inference_config.prometheus.enable or RLInsightLogger.enabled():
                raise ValueError("Metrics monitoring requires disable_log_stats=False, but it is currently True.")
            return
        addresses = self.server_addresses
        if inference_config.prometheus.enable:
            update_prometheus_config(inference_config.prometheus, addresses, inference_config.name)
        if RLInsightLogger.enabled():
            RLInsightLogger.register_rollout_metrics(
                addresses,
                inference_config.name,
                labels=[{"replica": f"teacher_slot_{rank}"} for rank in range(len(addresses))],
            )

    def _shared_inference_config(self):
        """One inference config shared by every teacher: same parallelism, dummy boot."""
        base = self.teachers[self.initial_key].inference
        overrides: dict[str, Any] = {
            "tensor_model_parallel_size": self.tp,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            # Weights start as garbage and every teacher is loaded explicitly
            # from its checkpoint before serving.
            "load_format": "dummy",
        }
        if int(base.expert_parallel_size or 1) > 1:
            overrides["expert_parallel_size"] = self.tp
        return dataclasses.replace(base, **overrides)

    async def _prepare_cache_dirs(self) -> None:
        # Every replica checks the capacity of its own node-local directory, so
        # a too-small /tmp or /dev/shm fails here instead of mid-training.
        await asyncio.gather(
            *[
                replica.teacher_prepare_cache_dir(self.job_cache_dir, len(self.teachers))
                for replica in self.rollout_replicas
            ]
        )

    async def _build_caches(self) -> None:
        # Every node needs its own copy: build all teachers on each node, in
        # parallel across the node's replicas and across nodes.
        await asyncio.gather(*[self._build_node(replicas) for replicas in self.replicas_by_node.values()])

    async def _build_node(self, replicas: list[TeacherWeightSwapVLLMReplica]) -> None:
        keys = list(self.teachers)
        for wave_start in range(0, len(keys), len(replicas)):
            wave = keys[wave_start : wave_start + len(replicas)]
            await asyncio.gather(
                *[self._build_one(replicas[offset], key) for offset, key in enumerate(wave)]
            )

    async def _build_one(self, replica: TeacherWeightSwapVLLMReplica, teacher_key: str) -> None:
        nbytes = await replica.teacher_build_cache(
            teacher_key, self.teachers[teacher_key].model_path, self.teacher_cache_dirs[teacher_key]
        )
        logger.warning(
            "[shared_teacher_fleet] built %r replica=%d shard_bytes=%.2f GiB",
            teacher_key,
            replica.replica_rank,
            nbytes / (1 << 30),
        )

    async def _map_caches(self) -> None:
        await asyncio.gather(
            *[
                replica.teacher_map_cache(teacher_key, self.teachers[teacher_key].model_path, teacher_dir)
                for replica in self.rollout_replicas
                for teacher_key, teacher_dir in self.teacher_cache_dirs.items()
            ]
        )

    async def _apply_initial_key(self) -> None:
        spawn = occupancy_spawn_keys(list(self.teachers), None, self.num_replicas)
        await asyncio.gather(
            *[
                replica.teacher_apply_from_cache(key)
                for replica, key in zip(self.rollout_replicas, spawn, strict=True)
            ]
        )

    def _create_fleet(self) -> None:
        # Pin to this TaskRunner node (already a student GPU worker). Do not
        # request NPU/GPU, do not search teacher nodes, and do not fall back
        # to the CPU head: that is where apply RPCs lose their signatures.
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        node_id = ray.get_runtime_context().get_node_id()
        logger.warning(
            "[shared_teacher_fleet] scheduling SharedTeacherFleet on TaskRunner node_id=%s",
            node_id,
        )
        # Each waiting ``acquire.remote()`` occupies one asyncio slot until a
        # replica is donated. A mixed 2048-sample step can have >1024 later-key
        # waiters; that starves ``release`` and leftover first-wave acquires,
        # so donate never runs. Keep this above train_batch_size.
        self.fleet = SharedTeacherFleet.options(
            max_concurrency=16384,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        ).remote(self.servers, list(self.teachers))

    async def _cleanup_caches(self) -> None:
        tasks = [replica.teacher_cleanup_cache_dir(self.job_cache_dir) for replica in self.rollout_replicas]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # -- clients -----------------------------------------------------------
    @property
    def server_addresses(self) -> list[str]:
        return [replica._server_address for replica in self.rollout_replicas]

    @property
    def servers(self) -> list[tuple[str, Any]]:
        return [(replica._server_address, replica._server_handle) for replica in self.rollout_replicas]

    def get_client(self, config) -> dict[str, SharedTeacherLLMServerClient]:
        if not self._ready:
            raise RuntimeError("Shared teacher fleet is not initialized.")
        servers = self.servers
        return {
            teacher_key: SharedTeacherLLMServerClient(
                config=config,
                fleet_handle=self.fleet,
                servers=servers,
                teacher_key=teacher_key,
            )
            for teacher_key in self.teachers
        }


def notify_on_demand_occupancy(manager, prompts) -> None:
    """Install this generate dispatch's per-key sample counts on the shared fleet.

    ``AgentLoopManagerTQ.generate_sequences`` calls this only when
    ``teacher_serving=on_demand``. Occupancy donate needs remaining counts
    before the batch is split across workers.
    """
    from tensordict import NonTensorData

    distillation = manager.config.get("distillation") if hasattr(manager.config, "get") else None
    if distillation is None:
        return
    if "validate" in prompts.keys():
        validate = prompts["validate"]
        if isinstance(validate, NonTensorData):
            validate = validate.data
        if isinstance(validate, (list, tuple)):
            validate = validate[0] if validate else False
        elif hasattr(validate, "item") and getattr(validate, "ndim", 1) == 0:
            validate = validate.item()
        if bool(validate):
            return
    fleet = None
    for client in (manager.teacher_client or {}).values():
        fleet = getattr(client, "_fleet", None)
        if fleet is not None:
            break
    if fleet is None:
        return
    field = distillation.get("teacher_key") if hasattr(distillation, "get") else "routing_key"
    field = field or "routing_key"
    if field not in prompts.keys() and "routing_key" in prompts.keys():
        field = "routing_key"
    if field not in prompts.keys():
        return
    rollout = manager.config.actor_rollout_ref.rollout
    n = int(rollout.get("n", 1) or 1) if hasattr(rollout, "get") else int(getattr(rollout, "n", 1) or 1)
    column = prompts[field]
    counts: dict[str, int] = {}
    for i in range(len(prompts)):
        item = column[i] if not isinstance(column, NonTensorData) else column.data
        if hasattr(item, "item"):
            try:
                item = item.item()
            except Exception:
                pass
        name = "default" if item is None else str(item)
        counts[name] = counts.get(name, 0) + n
    if counts:
        ray.get(fleet.set_step_key_counts.remote(counts))
        logger.info("[shared_teacher_fleet] occupancy counts=%s", counts)
