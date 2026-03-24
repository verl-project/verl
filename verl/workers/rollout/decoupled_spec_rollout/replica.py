from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional

import ray

from verl.single_controller.ray import RayWorkerGroup
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.decoupled_spec_rollout.layout import DecoupledSpecRole, compute_decoupled_spec_topology
from verl.workers.rollout.replica import RolloutMode, RolloutReplica
from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer, visible_devices_keyword


class _BaseDecoupledSGLangReplica(RolloutReplica):
    server_role: str = ""
    server_name_prefix: str = ""

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(SGLangHttpServer)
        self._base_gpu_id_override = 0

    async def init_hybrid_decoupled(self, worker_group: RayWorkerGroup):
        """Bind this replica to the correct worker slice using decoupled-spec topology (subclass implements slice)."""
        raise NotImplementedError

    def _build_server_name(self, node_rank: int) -> str:
        return f"{self.server_name_prefix}_{self.replica_rank}_{node_rank}"

    def _build_extra_server_kwargs(self) -> dict:
        return {"server_role": self.server_role}

    async def launch_servers(self):
        launch_start = time.perf_counter()
        model_path = getattr(self.model_config, "local_path", None) or getattr(self.model_config, "path", None)
        print(
            "[decoupled_spec][BaseDecoupledSGLangReplica] launch_servers_start "
            f"server_role={self.server_role} replica_rank={self.replica_rank} "
            f"world_size={self.world_size} nnodes={self.nnodes} model_path={model_path}"
        )
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (ray.get_runtime_context().get_node_id(), os.environ[visible_devices_keyword])
                )
                for worker in self.workers
            ]
        )
        worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]
        for worker_idx, (worker_node_id, worker_visible_devices) in enumerate(worker_infos):
            print(
                "[decoupled_spec][BaseDecoupledSGLangReplica] worker_assignment "
                f"server_role={self.server_role} replica_rank={self.replica_rank} "
                f"worker_idx={worker_idx} node_id={worker_node_id} "
                f"worker_visible_devices={worker_visible_devices}"
            )
        base_gpu_id = 0
        if os.environ.get(f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword}", None):
            base_gpu_id = self._base_gpu_id_override
            print(
                "[decoupled_spec][BaseDecoupledSGLangReplica] base_gpu_id_override "
                f"server_role={self.server_role} replica_rank={self.replica_rank} "
                f"base_gpu_id={base_gpu_id} gpus_per_node={self.gpus_per_node}"
            )

        for node_rank in range(self.nnodes):
            start_idx = node_rank * self.gpus_per_replica_node
            end_idx = (node_rank + 1) * self.gpus_per_replica_node
            workers = self.workers[
                start_idx:end_idx
            ]
            node_cuda_visible_devices_set = worker_cuda_visible_devices[
                start_idx:end_idx
            ]
            node_cuda_visible_devices = ",".join(
                map(
                    str,
                    sorted(
                        set(
                            int(device)
                            for worker_devices_set in node_cuda_visible_devices_set
                            for device in worker_devices_set.split(",")
                            if device.strip()
                        )
                    ),
                )
            )
            node_id = worker_node_ids[node_rank * self.gpus_per_replica_node]
            print(
                "[decoupled_spec][BaseDecoupledSGLangReplica] launch_server_actor "
                f"server_role={self.server_role} replica_rank={self.replica_rank} "
                f"node_rank={node_rank} node_id={node_id} worker_slice=[{start_idx}:{end_idx}] "
                f"worker_visible_devices_set={node_cuda_visible_devices_set} "
                f"cuda_visible_devices={node_cuda_visible_devices} base_gpu_id={base_gpu_id} "
                f"model_path={model_path}"
            )
            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": {f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword}": "1"}},
                name=self._build_server_name(node_rank),
                max_concurrency=self.max_concurrency,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                nnodes=self.nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
                base_gpu_id=base_gpu_id,
                **self._build_extra_server_kwargs(),
            )
            self.servers.append(server)

        master_address, master_port = None, None
        if self.nnodes > 1:
            master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )
        print(
            "[decoupled_spec][BaseDecoupledSGLangReplica] launch_servers_done "
            f"server_role={self.server_role} replica_rank={self.replica_rank} "
            f"server_address={self._server_address} elapsed_s={time.perf_counter() - launch_start:.6f}"
        )


class DraftSGLangReplica(_BaseDecoupledSGLangReplica):
    server_role = "draft"
    server_name_prefix = "sglang_draft_server"

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        topology_rollout_config: RolloutConfig | None = None,
    ):
        """Args:
        topology_rollout_config: Full rollout config (verify mesh + draft.ngpus) for GPU layout.
            Required for ``init_hybrid_decoupled``; defaults to ``config`` if unset.
        """
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self._topology_rollout_config = topology_rollout_config

    async def init_hybrid_decoupled(self, worker_group: RayWorkerGroup):
        rollout_cfg = self._topology_rollout_config if self._topology_rollout_config is not None else self.config
        topo = compute_decoupled_spec_topology(rollout_cfg, world_size=worker_group.world_size)
        start, end = topo.get_replica_worker_range(DecoupledSpecRole.DRAFT, self.replica_rank)
        self._base_gpu_id_override = topo.get_replica_base_gpu_id(
            DecoupledSpecRole.DRAFT, self.replica_rank, self.gpus_per_node
        )
        print(
            "[decoupled_spec][DraftSGLangReplica] init_hybrid_decoupled "
            f"server_role={self.server_role} replica_rank={self.replica_rank} "
            f"start={start} end={end} draft_world_size={topo.draft_world_size} "
            f"verify_gpu_count={topo.verify_gpu_count} base_gpu_id={self._base_gpu_id_override}"
        )
        self.rollout_mode = RolloutMode.HYBRID
        self.workers = worker_group.workers[start:end]
        await self.launch_servers()

    @property
    def draft_actor_handle(self):
        """Ray actor handle for this draft replica's HTTP server (node 0)."""
        return self._server_handle


class VerifySGLangReplica(_BaseDecoupledSGLangReplica):
    server_role = "verify"
    server_name_prefix = "sglang_server"

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        draft_actor_handles: Optional[list[Any]] = None,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.draft_actor_handles = list(draft_actor_handles or [])

    async def init_hybrid_decoupled(self, worker_group: RayWorkerGroup):
        topo = compute_decoupled_spec_topology(self.config, world_size=worker_group.world_size)
        start, end = topo.get_replica_worker_range(DecoupledSpecRole.VERIFY, self.replica_rank)
        self._base_gpu_id_override = topo.get_replica_base_gpu_id(
            DecoupledSpecRole.VERIFY, self.replica_rank, self.gpus_per_node
        )
        print(
            "[decoupled_spec][VerifySGLangReplica] init_hybrid_decoupled "
            f"server_role={self.server_role} replica_rank={self.replica_rank} "
            f"start={start} end={end} verify_world_size={topo.verify_world_size} "
            f"num_draft_handles={len(self.draft_actor_handles)} base_gpu_id={self._base_gpu_id_override}"
        )
        self.rollout_mode = RolloutMode.HYBRID
        self.workers = worker_group.workers[start:end]
        await self.launch_servers()

    def _build_extra_server_kwargs(self) -> dict:
        extra_kwargs = super()._build_extra_server_kwargs()
        extra_kwargs["draft_actor_handles"] = self.draft_actor_handles
        return extra_kwargs
