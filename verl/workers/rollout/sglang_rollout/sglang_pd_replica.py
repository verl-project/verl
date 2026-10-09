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
"""SGLang PD-disaggregated replica: 1 prefill + N decode servers per replica,
asymmetric TP supported. MVP: prefill_replicas=1, whole replica on one node."""

import asyncio
import logging
from dataclasses import replace as _dc_replace
from typing import Optional

from omegaconf import DictConfig

from verl.plugin.platform import get_platform
from verl.runtime import ClassWithInitArgs, RemoteCall, RemoteWorkerGroup, ResourcePool
from verl.utils.device import get_visible_devices_keyword, is_torch_npu_available
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.workers.config import RolloutConfig
from verl.workers.rollout.sglang_rollout.async_sglang_server import (
    SGLangHttpServer,
    SGLangReplica,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class SGLangPDReplica(SGLangReplica):
    """Replica that runs SGLang in prefill-decode disaggregated mode."""

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: DictConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
    ):
        super().__init__(
            replica_rank,
            config,
            model_config,
            gpus_per_node,
            is_reward_model,
            is_teacher_model,
        )
        disagg = self.config.disaggregation
        assert disagg.enabled, "SGLangPDReplica requires rollout.disaggregation.enabled=True"

        if disagg.prefill_replicas != 1:
            raise NotImplementedError(f"prefill_replicas=1 only (got {disagg.prefill_replicas})")
        self._n_prefill = disagg.prefill_replicas
        self._n_decode = disagg.decode_replicas

        self._prefill_tp = self.config.tensor_model_parallel_size
        # Inline decode_tp default: OmegaConf/Ray serialization drops dataclass methods.
        self._decode_tp = (
            disagg.decode_tensor_model_parallel_size
            if disagg.decode_tensor_model_parallel_size is not None
            else self._prefill_tp
        )

        pd_world_size = self._prefill_tp + self._n_decode * self._decode_tp
        if pd_world_size > gpus_per_node:
            raise NotImplementedError(
                f"PD replica needs {pd_world_size} GPUs but gpus_per_node={gpus_per_node}; "
                f"use more replicas to span nodes"
            )

        if self.config.data_parallel_size != 1:
            raise NotImplementedError(f"data_parallel_size=1 only (got {self.config.data_parallel_size})")
        self.world_size = pd_world_size
        self.gpus_per_replica_node = min(self.gpus_per_node, self.world_size)
        assert self.world_size % self.gpus_per_replica_node == 0
        self.nnodes = self.world_size // self.gpus_per_replica_node

        self._prefill_servers: list[RemoteWorkerGroup] = []
        self._decode_servers: list[RemoteWorkerGroup] = []
        self._prefill_server_address: Optional[str] = None
        self._decode_server_addresses: list[str] = []
        self._bootstrap_port: Optional[int] = None

    async def launch_servers(self):
        if self._worker_group is None or self.resource_pool is None:
            raise RuntimeError("rollout worker placement is not initialized")
        if self._worker_group.world_size != self.world_size:
            raise RuntimeError(f"worker count {self._worker_group.world_size} != PD world size {self.world_size}")
        assert not is_torch_npu_available(check_device=False), "PD on NPU not validated"

        worker_devices, worker_ips = await asyncio.gather(
            RemoteCall.gather(self._worker_group.execute_all_async("get_assigned_device_id")),
            RemoteCall.gather(self._worker_group.execute_all_async("get_node_ip_address")),
        )

        # Hold the bootstrap socket open until prefill binds it; closing earlier
        # opens a TOCTOU window where another process can grab the port.
        bootstrap_port = self.config.disaggregation.bootstrap_port
        self._bootstrap_sock = None
        if bootstrap_port is None:
            prefill_host_ip = worker_ips[0]
            bootstrap_port, self._bootstrap_sock = get_free_port(prefill_host_ip, with_alive_sock=True)
        self._bootstrap_port = bootstrap_port

        prefill_end = self._prefill_tp
        prefill_pool = self.resource_pool.slice(slice(0, prefill_end))
        if self._bootstrap_sock is not None:
            self._bootstrap_sock.close()
            self._bootstrap_sock = None

        [prefill_server] = await self._launch_one(
            role="prefill",
            source_pool=prefill_pool,
            bootstrap_port=self._bootstrap_port,
            tp=self._prefill_tp,
            worker_devices=worker_devices[: self._prefill_tp],
        )
        self._prefill_servers = [prefill_server]

        prefill_address, prefill_port = await prefill_server.submit("get_server_address")

        def _fmt(addr, port):
            return f"[{addr}]:{port}" if is_valid_ipv6_address(addr) else f"{addr}:{port}"

        self._prefill_server_address = _fmt(prefill_address, prefill_port)

        self._decode_servers = []
        self._decode_server_addresses = []
        for i in range(self._n_decode):
            start = self._prefill_tp + i * self._decode_tp
            end = start + self._decode_tp
            source_pool_i = self.resource_pool.slice(slice(start, end))
            [decode_server] = await self._launch_one(
                role="decode",
                source_pool=source_pool_i,
                bootstrap_port=self._bootstrap_port,
                tp=self._decode_tp,
                worker_devices=worker_devices[start:end],
            )
            self._decode_servers.append(decode_server)

            d_addr, d_port = await decode_server.submit("get_server_address")
            self._decode_server_addresses.append(_fmt(d_addr, d_port))

        self._server_address = self._prefill_server_address
        self._server_handle = prefill_server
        self.servers = list(self._prefill_servers) + list(self._decode_servers)
        await self._set_server_endpoints(
            [prefill_server] * self._prefill_tp
            + [server for server in self._decode_servers for _ in range(self._decode_tp)]
        )

        await prefill_server.submit("set_pd_peer", args=(list(self._decode_servers), prefill_address))

        logger.info(
            f"SGLangPDReplica rank={self.replica_rank} launched: "
            f"prefill={self._prefill_server_address}, "
            f"decodes=[{', '.join(self._decode_server_addresses)}], "
            f"bootstrap_port={self._bootstrap_port}"
        )

    async def _launch_one(
        self,
        role: str,
        source_pool: ResourcePool,
        bootstrap_port: int,
        tp: int,
        worker_devices: list[str],
    ) -> list[RemoteWorkerGroup]:
        pool_config = _dc_replace(self.config, tensor_model_parallel_size=tp)
        server_env = {
            **get_platform().rollout_env_vars(),
            get_visible_devices_keyword(): self._merge_cuda_visible_devices(worker_devices, expected_count=tp),
        }

        group = await self._create_server_worker_group(
            ClassWithInitArgs(
                SGLangHttpServer,
                config=pool_config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                replica_rank=self.replica_rank,
                node_rank=0,
                nnodes=1,
                env_vars=server_env,
                disaggregation_role=role,
                disaggregation_bootstrap_port=bootstrap_port,
            ),
            source_pool=source_pool,
        )
        server = group.remote()
        await server.submit("launch_server", kwargs={"master_address": None, "master_port": None})
        return [server]
