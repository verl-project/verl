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
"""vLLM PD-disaggregated replica: 1 prefill + N decode servers per replica,
asymmetric TP supported. MVP: prefill_replicas=1, single-node only."""

import asyncio
import logging
import os
import uuid
from dataclasses import replace as _dc_replace
from typing import Optional

from verl.runtime import ClassWithInitArgs, RemoteCall, RemoteWorkerGroup, ResourcePool
from verl.utils.device import get_device_name, is_torch_npu_available
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class vLLMPDReplica(vLLMReplica):
    """Replica that runs vLLM in prefill-decode disaggregated mode."""

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ):
        super().__init__(
            replica_rank,
            config,
            model_config,
            gpus_per_node,
            is_reward_model,
            is_teacher_model,
            name_suffix,
        )

        disagg = self.config.disaggregation
        assert disagg.enabled, "vLLMPDReplica requires rollout.disaggregation.enabled=True"

        if disagg.transfer_backend not in ("nixl", "mooncake"):
            raise NotImplementedError(
                f"vLLMPDReplica supports transfer_backend in ('nixl', 'mooncake') in this "
                f"revision; got {disagg.transfer_backend!r}. mori/ascend/fake are reserved "
                f"in DisaggregationConfig and will land in follow-ups."
            )
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
                f"single-node only in this revision (use more replicas to span nodes once "
                f"multi-node lands)"
            )
        if self.config.data_parallel_size != 1:
            raise NotImplementedError(f"data_parallel_size=1 only (got {self.config.data_parallel_size})")
        if self.config.pipeline_model_parallel_size != 1:
            raise NotImplementedError(
                f"pipeline_model_parallel_size=1 only "
                f"(got {self.config.pipeline_model_parallel_size}); PD path does not model PP yet"
            )

        self.world_size = pd_world_size
        self.gpus_per_replica_node = min(self.gpus_per_node, self.world_size)
        assert self.world_size % self.gpus_per_replica_node == 0
        self.nnodes = self.world_size // self.gpus_per_replica_node

        self._prefill_servers: list[RemoteWorkerGroup] = []
        self._decode_servers: list[RemoteWorkerGroup] = []

    async def launch_servers(self):
        if self._worker_group is None or self.resource_pool is None:
            raise RuntimeError("rollout worker placement is not initialized")
        if self._worker_group.world_size != self.world_size:
            raise RuntimeError(f"worker count {self._worker_group.world_size} != PD world size {self.world_size}")
        assert not is_torch_npu_available(check_device=False), "vLLM PD on NPU not validated"

        worker_devices, worker_ips = await asyncio.gather(
            RemoteCall.gather(self._worker_group.execute_all_async("get_assigned_device_id")),
            RemoteCall.gather(self._worker_group.execute_all_async("get_node_ip_address")),
        )

        # Bind the side-channel on the prefill worker's node.
        prefill_host_ip = worker_ips[0]
        prefill_engine_id = uuid.uuid4().hex

        prefill_end = self._prefill_tp
        prefill_pool = self.resource_pool.slice(slice(0, prefill_end))
        prefill_devs = self._merge_cuda_visible_devices(worker_devices[0:prefill_end], expected_count=self._prefill_tp)

        # Keep side-channel sockets reserved until all actors bind.
        reserved_socks = []
        prefill_side_channel_port, prefill_sock = get_free_port(prefill_host_ip, with_alive_sock=True)
        reserved_socks.append(prefill_sock)
        try:
            prefill_kv_cfg = self._build_kv_transfer_config(
                role="prefill",
                engine_id=prefill_engine_id,
                transfer_backend=self.config.disaggregation.transfer_backend,
                mooncake_protocol=self.config.disaggregation.mooncake_protocol,
            )
            prefill_server = await self._spawn_pd_server(
                role="prefill",
                source_pool=prefill_pool,
                worker_count=self._prefill_tp,
                cuda_visible_devices=prefill_devs,
                tp=self._prefill_tp,
                kv_transfer_config=prefill_kv_cfg,
                side_channel_host=prefill_host_ip,
                side_channel_port=prefill_side_channel_port,
                mooncake_bootstrap_port=prefill_side_channel_port,
                zmq_base_trainer_rank=0,
            )
            self._prefill_servers = [prefill_server]

            for i in range(self._n_decode):
                start = self._prefill_tp + i * self._decode_tp
                end = start + self._decode_tp
                source_pool_i = self.resource_pool.slice(slice(start, end))
                devs_i = self._merge_cuda_visible_devices(worker_devices[start:end], expected_count=self._decode_tp)

                decode_side_channel_port, decode_sock = get_free_port(prefill_host_ip, with_alive_sock=True)
                reserved_socks.append(decode_sock)
                decode_kv_cfg = self._build_kv_transfer_config(
                    role="decode",
                    engine_id=uuid.uuid4().hex,
                    transfer_backend=self.config.disaggregation.transfer_backend,
                    mooncake_protocol=self.config.disaggregation.mooncake_protocol,
                )
                self._decode_servers.append(
                    await self._spawn_pd_server(
                        role="decode",
                        source_pool=source_pool_i,
                        worker_count=self._decode_tp,
                        cuda_visible_devices=devs_i,
                        tp=self._decode_tp,
                        kv_transfer_config=decode_kv_cfg,
                        side_channel_host=prefill_host_ip,
                        side_channel_port=decode_side_channel_port,
                        mooncake_bootstrap_port=prefill_side_channel_port,
                        zmq_base_trainer_rank=start,
                    )
                )

            await asyncio.gather(
                *[
                    server.submit(
                        "launch_server",
                        kwargs={"master_address": None, "master_port": None, "dp_rpc_port": None},
                    )
                    for server in self._prefill_servers + self._decode_servers
                ]
            )
        finally:
            for sock in reserved_socks:
                sock.close()

        await self._prefill_servers[0].submit(
            "set_pd_peer",
            args=(self._decode_servers, prefill_side_channel_port, prefill_engine_id),
        )

        self.servers = list(self._prefill_servers) + list(self._decode_servers)
        await self._set_server_endpoints(
            [self._prefill_servers[0]] * self._prefill_tp
            + [server for server in self._decode_servers for _ in range(self._decode_tp)]
        )
        prefill_address, prefill_port = await self._prefill_servers[0].submit("get_server_address")
        self._server_handle = self._prefill_servers[0]
        self._server_address = (
            f"[{prefill_address}]:{prefill_port}"
            if is_valid_ipv6_address(prefill_address)
            else f"{prefill_address}:{prefill_port}"
        )

        logger.info(
            "vLLMPDReplica rank=%s launched: prefill=%s (engine_id=%s, side_channel=%s:%d), decodes=%d",
            self.replica_rank,
            self._server_address,
            prefill_engine_id,
            prefill_host_ip,
            prefill_side_channel_port,
            len(self._decode_servers),
        )

    @staticmethod
    def _build_kv_transfer_config(
        role: str,
        engine_id: str,
        transfer_backend: str,
        mooncake_protocol: Optional[str] = None,
    ) -> dict:
        """Assemble vLLM's ``--kv-transfer-config`` payload."""
        role_to_kv_role = {
            "prefill": "kv_producer",
            "decode": "kv_consumer",
        }
        connector = {
            "nixl": "NixlConnector",
            "mooncake": "MooncakeConnector",
        }[transfer_backend]
        cfg: dict = {
            "kv_connector": connector,
            "kv_role": role_to_kv_role[role],
            "engine_id": engine_id,
            "kv_buffer_device": get_device_name(),
        }
        if transfer_backend == "mooncake" and mooncake_protocol:
            cfg["kv_connector_extra_config"] = {"mooncake_protocol": mooncake_protocol}
        return cfg

    async def _spawn_pd_server(
        self,
        role: str,
        source_pool: ResourcePool,
        worker_count: int,
        cuda_visible_devices: str,
        tp: int,
        kv_transfer_config: dict,
        side_channel_host: str,
        side_channel_port: int,
        mooncake_bootstrap_port: int,
        zmq_base_trainer_rank: int = 0,
    ) -> RemoteWorkerGroup:
        """Construct one PD ``vLLMHttpServer`` WorkerGroup."""
        per_role_config = _dc_replace(self.config, tensor_model_parallel_size=tp)

        env_vars = {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            "NCCL_CUMEM_ENABLE": "0",
            "VLLM_NIXL_SIDE_CHANNEL_HOST": side_channel_host,
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(side_channel_port),
            "VLLM_MOONCAKE_BOOTSTRAP_PORT": str(mooncake_bootstrap_port),
            # Avoid Mooncake TCP port exhaustion under validation concurrency.
            "MC_TCP_ENABLE_CONNECTION_POOL": os.environ.get("MC_TCP_ENABLE_CONNECTION_POOL", "1"),
            "VERL_ZMQ_BASE_TRAINER_RANK": str(zmq_base_trainer_rank),
        }

        group = await self._create_server_worker_group(
            ClassWithInitArgs(
                vLLMHttpServer,
                config=per_role_config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                worker_count=worker_count,
                replica_rank=self.replica_rank,
                node_rank=0,
                gpus_per_node=self.gpus_per_replica_node,
                nnodes=1,
                cuda_visible_devices=cuda_visible_devices,
                env_vars=env_vars,
                disaggregation_role=role,
                disaggregation_kv_transfer_config=kv_transfer_config,
            ),
            source_pool=source_pool,
        )
        return group.remote()
