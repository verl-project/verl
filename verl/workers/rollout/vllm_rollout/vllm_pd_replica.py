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
import zlib
from dataclasses import replace as _dc_replace
from typing import Optional

import ray
from ray.actor import ActorHandle

from verl.utils.device import get_device_name, get_resource_name, is_torch_npu_available
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# MoRI-IO control ports: fixed blocks below the kernel ephemeral range (32768+), see _reserve_mori_ports.
_MORI_PORT_BASE = 20000
_MORI_PORT_END = 30000
_MORI_PORTS_PER_SERVER = 32
_MORI_SERVERS_PER_REPLICA = 16


def _bind_port(host_ip: str, port: int):
    """Bind ``port`` on ``host_ip`` and return the open socket (raises OSError if taken)."""
    import socket

    family = socket.AF_INET6 if is_valid_ipv6_address(host_ip) else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.bind((host_ip, port))
    except OSError:
        sock.close()
        raise
    return sock


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

        if disagg.transfer_backend not in ("nixl", "mooncake", "mori"):
            raise NotImplementedError(
                f"vLLMPDReplica supports transfer_backend in ('nixl', 'mooncake', 'mori') in this "
                f"revision; got {disagg.transfer_backend!r}. ascend/fake are reserved "
                f"in DisaggregationConfig and will land in follow-ups."
            )
        if disagg.transfer_backend == "mori" and self.config.enable_sleep_mode:
            # vLLM sleep mode backs the KV cache with VMM (hipMemCreate) allocations, which
            # MoRI-IO cannot export (hipIpcGetMemHandle / ibv_reg_mr fail).
            raise ValueError(
                "transfer_backend=mori requires rollout.enable_sleep_mode=False (and rollout.free_cache_engine=False)"
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

    async def launch_servers(self):
        assert len(self.workers) == self.world_size, (
            f"worker count {len(self.workers)} != PD world size {self.world_size}"
        )
        assert not is_torch_npu_available(check_device=False), "vLLM PD on NPU not validated"

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

        # Bind the side-channel on the prefill worker's node.
        prefill_host_ip = worker_infos[0][2]
        prefill_engine_id = uuid.uuid4().hex

        prefill_end = self._prefill_tp
        prefill_workers = self.workers[0:prefill_end]
        prefill_node_id = worker_infos[0][0]
        prefill_devs = self._collect_cuda_devices(worker_infos[0:prefill_end])

        # Keep side-channel sockets reserved until all actors bind.
        reserved_socks = []
        # MoRI-IO binds its handshake/notify ports while the engine starts, so these
        # reservations are released right before launch (not held like the side channel).
        mori_socks = []
        prefill_side_channel_port, prefill_sock = get_free_port(prefill_host_ip, with_alive_sock=True)
        reserved_socks.append(prefill_sock)
        try:
            prefill_mori = self._reserve_mori_ports(prefill_host_ip, self._prefill_tp, mori_socks, server_index=0)
            prefill_kv_cfg = self._build_kv_transfer_config(
                role="prefill",
                engine_id=prefill_engine_id,
                transfer_backend=self.config.disaggregation.transfer_backend,
                mooncake_protocol=self.config.disaggregation.mooncake_protocol,
                mori_info=prefill_mori,
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
                    mooncake_bootstrap_port=prefill_side_channel_port,
                    actor_name=f"vllm_server_{self.replica_rank}_0{self.name_suffix}",
                    zmq_base_trainer_rank=0,
                )
            ]

            decode_mori_infos: list[Optional[dict]] = []
            for i in range(self._n_decode):
                start = self._prefill_tp + i * self._decode_tp
                end = start + self._decode_tp
                workers_i = self.workers[start:end]
                node_id_i = worker_infos[start][0]
                devs_i = self._collect_cuda_devices(worker_infos[start:end])

                decode_side_channel_port, decode_sock = get_free_port(prefill_host_ip, with_alive_sock=True)
                reserved_socks.append(decode_sock)
                decode_host_ip = worker_infos[start][2]
                decode_mori = self._reserve_mori_ports(decode_host_ip, self._decode_tp, mori_socks, server_index=1 + i)
                decode_mori_infos.append(decode_mori)
                decode_kv_cfg = self._build_kv_transfer_config(
                    role="decode",
                    engine_id=uuid.uuid4().hex,
                    transfer_backend=self.config.disaggregation.transfer_backend,
                    mooncake_protocol=self.config.disaggregation.mooncake_protocol,
                    mori_info=decode_mori,
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
                        mooncake_bootstrap_port=prefill_side_channel_port,
                        actor_name=f"vllm_server_decode_{self.replica_rank}_{i}{self.name_suffix}",
                        zmq_base_trainer_rank=start,
                    )
                )

            for sock in mori_socks:
                sock.close()
            mori_socks.clear()
            await asyncio.gather(
                *[
                    server.launch_server.remote(master_address=None, master_port=None, dp_rpc_port=None)
                    for server in self._prefill_servers + self._decode_servers
                ]
            )
        finally:
            for sock in reserved_socks + mori_socks:
                sock.close()

        await self._prefill_servers[0].set_pd_peer.remote(
            self._decode_servers,
            prefill_side_channel_port,
            prefill_engine_id,
            prefill_mori_info=prefill_mori,
            decode_mori_infos=decode_mori_infos,
            min_remote_prefill_tokens=self.config.disaggregation.min_remote_prefill_tokens,
        )

        self.servers = list(self._prefill_servers) + list(self._decode_servers)
        prefill_address, prefill_port = await self._prefill_servers[0].get_server_address.remote()
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
    def _collect_cuda_devices(worker_infos) -> str:
        return ",".join(worker_info[1] for worker_info in worker_infos)

    @staticmethod
    def _build_kv_transfer_config(
        role: str,
        engine_id: str,
        transfer_backend: str,
        mooncake_protocol: Optional[str] = None,
        mori_info: Optional[dict] = None,
    ) -> dict:
        """Assemble vLLM's ``--kv-transfer-config`` payload."""
        role_to_kv_role = {
            "prefill": "kv_producer",
            "decode": "kv_consumer",
        }
        connector = {
            "nixl": "NixlConnector",
            "mooncake": "MooncakeConnector",
            "mori": "MoRIIOConnector",
        }[transfer_backend]
        cfg: dict = {
            "kv_connector": connector,
            "kv_role": role_to_kv_role[role],
            "engine_id": engine_id,
            "kv_buffer_device": get_device_name(),
        }
        if transfer_backend == "mooncake" and mooncake_protocol:
            cfg["kv_connector_extra_config"] = {"mooncake_protocol": mooncake_protocol}
        if transfer_backend == "mori":
            assert mori_info is not None, "mori transfer_backend needs reserved MoRI-IO ports"
            # No proxy: verl dispatches both legs itself and passes the peer address in
            # kv_transfer_params, so proxy_ip stays empty (disables the discovery heartbeat).
            cfg["kv_connector_extra_config"] = {
                "proxy_ip": "",
                "http_port": "0",
                "host_ip": mori_info["host"],
                "handshake_port": str(mori_info["handshake_port"]),
                "notify_port": str(mori_info["notify_port"]),
                "backend": mori_info["backend"],
                "read_mode": "true" if mori_info["mode"] == "read" else "false",
            }
        return cfg

    def _reserve_mori_ports(self, host_ip: str, tp: int, reserved_socks: list, server_index: int) -> Optional[dict]:
        """Pick MoRI-IO handshake/notify ports for one PD server (None for other backends).

        MoRI binds these ports (``tcp://*:port``) only after the engine has loaded the model, minutes
        after this call, and its listener thread dies silently if the bind fails (requests then hang).
        Ports from the kernel's ephemeral range can be taken by any outgoing connection in that window,
        so use a fixed block below it instead, one per (replica, server). MoRI uses ``handshake_port +
        tp_rank`` and ``notify_port + tp_rank`` (DP=1), so each gets its own half of the block.
        """
        if self.config.disaggregation.transfer_backend != "mori":
            return None
        half = _MORI_PORTS_PER_SERVER // 2
        assert tp <= half, f"tp={tp} needs more than {_MORI_PORTS_PER_SERVER} MoRI-IO ports"
        assert server_index < _MORI_SERVERS_PER_REPLICA, f"too many PD servers per replica: {server_index + 1}"
        slots = (_MORI_PORT_END - _MORI_PORT_BASE) // (_MORI_PORTS_PER_SERVER * _MORI_SERVERS_PER_REPLICA)
        first_slot = self.replica_rank + zlib.crc32(self.name_suffix.encode())
        wildcard = "::" if is_valid_ipv6_address(host_ip) else "0.0.0.0"
        for attempt in range(slots):
            slot = (first_slot + attempt) % slots
            base = _MORI_PORT_BASE + (slot * _MORI_SERVERS_PER_REPLICA + server_index) * _MORI_PORTS_PER_SERVER
            socks = []
            try:
                for port in [*range(base, base + tp), *range(base + half, base + half + tp)]:
                    socks.append(_bind_port(wildcard, port))
            except OSError:
                for sock in socks:
                    sock.close()
                continue
            reserved_socks.extend(socks)
            return {
                "host": host_ip,
                "handshake_port": base,
                "notify_port": base + half,
                "tp_size": tp,
                "backend": self.config.disaggregation.mori_backend,
                "mode": self.config.disaggregation.mori_mode,
            }
        raise RuntimeError(
            f"no free MoRI-IO port block for PD server {server_index} of replica {self.replica_rank} "
            f"in [{_MORI_PORT_BASE}, {_MORI_PORT_END})"
        )

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
            # Platform noset/rollout env (e.g. ROCm HIP/ROCR_VISIBLE_DEVICES), same as non-PD replicas.
            **self._get_server_env_vars([node_id]),
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
