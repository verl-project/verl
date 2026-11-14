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
"""vLLM PD-disaggregated replica with one prefill and N decode servers.

Asymmetric TP is supported, and the complete PD replica must fit on one node.
GPU keeps vLLM's native NIXL/Mooncake connectors; Ascend uses
vLLM-Ascend's MooncakeConnectorV1.
"""

import asyncio
import logging
import os
import socket
import uuid
from dataclasses import replace as _dc_replace
from typing import Any, Optional

import ray
from ray.actor import ActorHandle

from verl.utils.device import get_device_name, get_resource_name
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

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

        self._prefill_servers: list[ActorHandle] = []
        self._decode_servers: list[ActorHandle] = []
        self._prefill_server_addresses: list[str] = []
        self._decode_server_addresses: list[str] = []

    async def launch_servers(self):
        assert len(self.workers) == self.world_size, (
            f"worker count {len(self.workers)} != PD world size {self.world_size}"
        )
        use_ascend_mooncake_v1 = self._is_ascend_platform()
        transfer_backend = self.config.disaggregation.transfer_backend
        if use_ascend_mooncake_v1 and transfer_backend == "nixl":
            logger.warning("NixlConnector is not supported for Ascend PD; falling back to MooncakeConnectorV1")
            transfer_backend = "mooncake"

        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
                        ray.util.get_node_ip_address().strip("[]"),
                    )
                )
                for worker in self.workers
            ]
        )

        prefill_host_ip = worker_infos[0][2]
        prefill_engine_id = uuid.uuid4().hex
        prefill_end = self._prefill_tp
        prefill_workers = self.workers[:prefill_end]
        prefill_node_id = worker_infos[0][0]
        prefill_devs = self._collect_cuda_devices(worker_infos[:prefill_end])

        (
            prefill_side_channel_port,
            bootstrap_port,
            decode_side_channel_ports,
            reserved_socks,
            close_sockets_before_launch,
        ) = self._reserve_pd_ports(prefill_host_ip, use_ascend_mooncake_v1)
        if close_sockets_before_launch:
            for sock in reserved_socks:
                sock.close()
            reserved_socks.clear()
        try:
            prefill_kv_cfg = self._build_kv_transfer_config(
                role="prefill",
                engine_id=prefill_engine_id,
                transfer_backend=transfer_backend,
                mooncake_protocol=self.config.disaggregation.mooncake_protocol,
                use_ascend_mooncake_v1=use_ascend_mooncake_v1,
                kv_port=prefill_side_channel_port,
                prefill_tp=self._prefill_tp,
                decode_tp=self._decode_tp,
            )
            self._prefill_servers = [
                self._spawn_pd_server(
                    role="prefill",
                    workers=prefill_workers,
                    node_id=prefill_node_id,
                    cuda_visible_devices=prefill_devs,
                    tp=self._prefill_tp,
                    kv_transfer_config=prefill_kv_cfg,
                    side_channel_host=prefill_host_ip,
                    side_channel_port=prefill_side_channel_port,
                    mooncake_bootstrap_port=bootstrap_port,
                    actor_name=f"vllm_server_{self.replica_rank}_0{self.name_suffix}",
                    zmq_base_trainer_rank=0,
                )
            ]

            for i in range(self._n_decode):
                start = self._prefill_tp + i * self._decode_tp
                end = start + self._decode_tp
                workers_i = self.workers[start:end]
                node_id_i = worker_infos[start][0]
                devs_i = self._collect_cuda_devices(worker_infos[start:end])

                decode_side_channel_port = decode_side_channel_ports[i]
                decode_kv_cfg = self._build_kv_transfer_config(
                    role="decode",
                    engine_id=uuid.uuid4().hex,
                    transfer_backend=transfer_backend,
                    mooncake_protocol=self.config.disaggregation.mooncake_protocol,
                    use_ascend_mooncake_v1=use_ascend_mooncake_v1,
                    kv_port=decode_side_channel_port,
                    prefill_tp=self._prefill_tp,
                    decode_tp=self._decode_tp,
                )
                self._decode_servers.append(
                    self._spawn_pd_server(
                        role="decode",
                        workers=workers_i,
                        node_id=node_id_i,
                        cuda_visible_devices=devs_i,
                        tp=self._decode_tp,
                        kv_transfer_config=decode_kv_cfg,
                        side_channel_host=prefill_host_ip,
                        side_channel_port=decode_side_channel_port,
                        mooncake_bootstrap_port=bootstrap_port,
                        actor_name=f"vllm_server_decode_{self.replica_rank}_{i}{self.name_suffix}",
                        zmq_base_trainer_rank=start,
                    )
                )

            await asyncio.gather(
                *[
                    server.launch_server.remote(master_address=None, master_port=None, dp_rpc_port=None)
                    for server in self._prefill_servers + self._decode_servers
                ]
            )
        finally:
            for sock in reserved_socks:
                sock.close()

        await self._prefill_servers[0].set_pd_peer.remote(
            decode_peers=self._decode_servers,
            prefill_side_channel_port=prefill_side_channel_port,
            prefill_engine_id=prefill_engine_id,
        )

        self.servers = list(self._prefill_servers) + list(self._decode_servers)
        self._server_handle = self._prefill_servers[0]
        if use_ascend_mooncake_v1:
            prefill_addresses = await asyncio.gather(
                *[server.get_server_address.remote() for server in self._prefill_servers]
            )
            decode_addresses = await asyncio.gather(
                *[server.get_server_address.remote() for server in self._decode_servers]
            )
            self._prefill_server_addresses = [
                self._format_server_address(host, port) for host, port in prefill_addresses
            ]
            self._decode_server_addresses = [self._format_server_address(host, port) for host, port in decode_addresses]
            self._server_address = self._prefill_server_addresses[0]
        else:
            prefill_address, prefill_port = await self._prefill_servers[0].get_server_address.remote()
            self._server_address = self._format_server_address(prefill_address, prefill_port)

        logger.info(
            "vLLMPDReplica rank=%s launched: prefill=%s (engine_id=%s, side_channel=%s:%d), decodes=%d",
            self.replica_rank,
            self._server_address,
            prefill_engine_id,
            prefill_host_ip,
            prefill_side_channel_port,
            len(self._decode_servers),
        )

    def get_metrics_server_endpoints(self) -> list[tuple[str, dict[str, Any]]]:
        """Return all Ascend P/D metrics endpoints with role labels."""
        if len(self._prefill_server_addresses) != len(self._prefill_servers):
            raise RuntimeError("PD prefill metrics endpoints are not ready")
        if len(self._decode_server_addresses) != len(self._decode_servers):
            raise RuntimeError("PD decode metrics endpoints are not ready")
        return [
            *[
                (address, {"pd_role": "prefill", "pd_index": index})
                for index, address in enumerate(self._prefill_server_addresses)
            ],
            *[
                (address, {"pd_role": "decode", "pd_index": index})
                for index, address in enumerate(self._decode_server_addresses)
            ],
        ]

    async def resume_generation(self):
        """Resume every independent P/D engine before opening any admission gate."""
        await asyncio.gather(*[server.resume_engine_generation.remote() for server in self.servers])
        await asyncio.gather(*[server.open_submission_gate.remote() for server in self.servers])

    @staticmethod
    def _is_ascend_platform() -> bool:
        """Follow Verl's existing NPU availability convention."""
        return get_device_name() == "npu"

    @staticmethod
    def _format_server_address(host: str, port: int) -> str:
        return f"[{host}]:{port}" if is_valid_ipv6_address(host) else f"{host}:{port}"

    def _reserve_pd_ports(
        self,
        address: str,
        contiguous: bool,
    ) -> tuple[int, int, list[int], list[socket.socket], bool]:
        """Reserve P/D ports using either upstream or Ascend allocation semantics."""
        reserved_socks: list[socket.socket] = []
        decode_ports: list[int] = []
        try:
            if contiguous:
                prefill_port, prefill_socks = self._reserve_port_range(address, self._prefill_tp)
                reserved_socks.extend(prefill_socks)
                bootstrap_port, bootstrap_socks = self._reserve_port_range(address, self.world_size)
                reserved_socks.extend(bootstrap_socks)
                for _ in range(self._n_decode):
                    decode_port, decode_socks = self._reserve_port_range(address, self._decode_tp)
                    decode_ports.append(decode_port)
                    reserved_socks.extend(decode_socks)
                return prefill_port, bootstrap_port, decode_ports, reserved_socks, True

            prefill_port, prefill_sock = get_free_port(address, with_alive_sock=True)
            reserved_socks.append(prefill_sock)
            for _ in range(self._n_decode):
                decode_port, decode_sock = get_free_port(address, with_alive_sock=True)
                decode_ports.append(decode_port)
                reserved_socks.append(decode_sock)
            return prefill_port, prefill_port, decode_ports, reserved_socks, False
        except Exception:
            # Reservation may fail before launch_servers enters its try/finally.
            for sock in reserved_socks:
                sock.close()
            raise

    @staticmethod
    def _reserve_port_range(address: str, count: int) -> tuple[int, list[socket.socket]]:
        """Reserve ``count`` consecutive TCP ports on ``address``.

        Mooncake builds its per-rank handshake endpoints as ``base + rank``.
        Reserving only the base port lets two independently selected engines
        claim overlapping ranges and makes PD startup intermittent.
        """
        if count < 1:
            raise ValueError(f"port range count must be positive, got {count}")
        family = socket.AF_INET6 if is_valid_ipv6_address(address) else socket.AF_INET
        last_error: OSError | None = None
        for _ in range(128):
            probe = socket.socket(family=family, type=socket.SOCK_STREAM)
            socks = [probe]
            try:
                probe.bind((address, 0))
                base = probe.getsockname()[1]
                if base + count - 1 > 65535:
                    probe.close()
                    continue
                for port in range(base + 1, base + count):
                    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
                    socks.append(sock)
                    sock.bind((address, port))
                return base, socks
            except OSError as exc:
                last_error = exc
                for sock in socks:
                    sock.close()
        raise RuntimeError(f"failed to reserve {count} consecutive TCP ports on {address}: {last_error}")

    @staticmethod
    def _collect_cuda_devices(worker_infos) -> str:
        return ",".join(worker_info[1] for worker_info in worker_infos)

    @staticmethod
    def _build_kv_transfer_config(
        role: str,
        engine_id: str,
        transfer_backend: str,
        mooncake_protocol: Optional[str] = None,
        use_ascend_mooncake_v1: bool = False,
        kv_port: Optional[int] = None,
        prefill_tp: Optional[int] = None,
        decode_tp: Optional[int] = None,
    ) -> dict:
        """Assemble vLLM's ``--kv-transfer-config`` payload."""
        role_to_kv_role = {
            "prefill": "kv_producer",
            "decode": "kv_consumer",
        }
        if use_ascend_mooncake_v1:
            if transfer_backend != "mooncake":
                raise ValueError("Ascend PD requires transfer_backend='mooncake'")
            if kv_port is None or prefill_tp is None or decode_tp is None:
                raise ValueError("MooncakeConnectorV1 requires kv_port, prefill_tp, and decode_tp")
            connector = "MooncakeConnectorV1"
        else:
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
        if use_ascend_mooncake_v1:
            cfg["kv_port"] = kv_port
            cfg["kv_connector_extra_config"] = {
                "prefill": {"dp_size": 1, "tp_size": prefill_tp},
                "decode": {"dp_size": 1, "tp_size": decode_tp},
            }
        elif transfer_backend == "mooncake" and mooncake_protocol:
            cfg["kv_connector_extra_config"] = {"mooncake_protocol": mooncake_protocol}
        return cfg

    def _spawn_pd_server(
        self,
        role: str,
        workers: list[ActorHandle],
        node_id: str,
        cuda_visible_devices: str,
        tp: int,
        kv_transfer_config: dict,
        side_channel_host: str,
        side_channel_port: int,
        mooncake_bootstrap_port: int,
        actor_name: str,
        zmq_base_trainer_rank: int = 0,
    ) -> ActorHandle:
        """Construct one PD ``vLLMHttpServer`` actor."""
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
            "VERL_RAY_JOB_ID": ray.get_runtime_context().get_job_id(),
        }

        return self.server_class.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
            runtime_env={"env_vars": env_vars},
            name=actor_name,
            max_concurrency=self.config.ray_actor_max_concurrency,
        ).remote(
            config=per_role_config,
            model_config=self.model_config,
            rollout_mode=self.rollout_mode,
            workers=workers,
            replica_rank=self.replica_rank,
            node_rank=0,
            gpus_per_node=self.gpus_per_replica_node,
            nnodes=1,
            cuda_visible_devices=cuda_visible_devices,
            disaggregation_role=role,
            disaggregation_kv_transfer_config=kv_transfer_config,
        )
