from __future__ import annotations

import asyncio
import os
from typing import Optional

import ray

from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.decoupled_spec_rollout.protocol import DraftServerEndpoint
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

    async def init_hybrid_with_workers(self, workers: list):
        self.rollout_mode = RolloutMode.HYBRID
        self.workers = list(workers)
        await self.launch_servers()

    def _build_server_name(self, node_rank: int) -> str:
        return f"{self.server_name_prefix}_{self.replica_rank}_{node_rank}"

    def _build_extra_server_kwargs(self) -> dict:
        return {"server_role": self.server_role}

    async def launch_servers(self):
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
        base_gpu_id = 0
        replica_world_size = self.world_size
        if os.environ.get(f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword}", None):
            base_gpu_id = (0 + self.replica_rank * replica_world_size) % self.gpus_per_node

        for node_rank in range(self.nnodes):
            workers = self.workers[
                node_rank * self.gpus_per_replica_node : (node_rank + 1) * self.gpus_per_replica_node
            ]
            node_cuda_visible_devices_set = worker_cuda_visible_devices[
                node_rank * self.gpus_per_replica_node : (node_rank + 1) * self.gpus_per_replica_node
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


class DraftSGLangReplica(_BaseDecoupledSGLangReplica):
    server_role = "draft"
    server_name_prefix = "sglang_draft_server"

    def as_endpoint(self) -> DraftServerEndpoint:
        return DraftServerEndpoint(
            replica_rank=self.replica_rank,
            server_address=self._server_address,
            actor_handle=self._server_handle,
            node_rank=0,
        )


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
        draft_server_endpoints: Optional[list[DraftServerEndpoint]] = None,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.draft_server_endpoints = list(draft_server_endpoints or [])

    def _build_extra_server_kwargs(self) -> dict:
        extra_kwargs = super()._build_extra_server_kwargs()
        extra_kwargs["draft_server_endpoints"] = self.draft_server_endpoints
        return extra_kwargs
