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
"""vLLM PD-disaggregated replica: 1 prefill + N decode servers per replica.

MVP scope (mirrors verl PR #6117 for SGLang):
  * NIXL transfer backend only; mooncake/mori/ascend/fake reserved in
    ``DisaggregationConfig`` but unimplemented here.
  * ``prefill_replicas == 1``, ``data_parallel_size == 1``.
  * Whole replica fits on one node.

NIXL pull-mode handshake is lazy on first request; only the per-engine
``VLLM_NIXL_SIDE_CHANNEL_HOST/PORT`` need broadcasting, and only the decodes
need the prefill's coordinates (recorded in ``set_pd_peer``).

The per-request prefill→decode dispatch (sequential ``kv_transfer_params``
round-trip, matching ``vllm-project/router``'s ``vllm_pd_router.rs``) is
wired in Phase 2 of the verl-vllm-pd-disagg series; this file only handles
config validation and server launch.
"""

import asyncio
import logging
import uuid
from dataclasses import replace as _dc_replace
from typing import Optional

import ray
from ray.actor import ActorHandle

from verl.utils.device import get_resource_name, is_torch_npu_available
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

        if disagg.transfer_backend != "nixl":
            raise NotImplementedError(
                f"vLLMPDReplica supports transfer_backend='nixl' only in this revision; "
                f"got {disagg.transfer_backend!r}. mooncake/mori/ascend/fake are reserved "
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

        # Override the values RolloutReplica.__init__ computed from a colocated
        # topology — under PD the world is asymmetric.
        self.world_size = pd_world_size
        self.gpus_per_replica_node = min(self.gpus_per_node, self.world_size)
        assert self.world_size % self.gpus_per_replica_node == 0
        self.nnodes = self.world_size // self.gpus_per_replica_node

        self._prefill_servers: list[ActorHandle] = []
        self._decode_servers: list[ActorHandle] = []
        self._prefill_engine_id: Optional[str] = None
        self._prefill_side_channel_host: Optional[str] = None
        self._prefill_side_channel_port: Optional[int] = None

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
                    )
                )
                for worker in self.workers
            ]
        )

        prefill_host_ip = ray.util.get_node_ip_address().strip("[]")
        self._prefill_engine_id = uuid.uuid4().hex

        # Launch prefill (tp first slice of workers).
        prefill_workers = self.workers[0 : self._prefill_tp]
        prefill_node_id = worker_infos[0][0]
        prefill_devs = ",".join(worker_infos[0 + i][1] for i in range(self._prefill_tp))

        # Hold the bootstrap socket open until the prefill actor binds it; closing
        # earlier opens a TOCTOU window for another process to grab the port.
        prefill_side_channel_port, prefill_sock = get_free_port(prefill_host_ip, with_alive_sock=True)
        try:
            prefill_kv_cfg = self._build_kv_transfer_config(
                role="prefill",
                engine_id=self._prefill_engine_id,
                ib_device=self.config.disaggregation.ib_device,
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
                    actor_name=f"vllm_server_{self.replica_rank}_prefill{self.name_suffix}",
                )
            ]
        finally:
            prefill_sock.close()

        self._prefill_side_channel_host = prefill_host_ip
        self._prefill_side_channel_port = prefill_side_channel_port

        # Launch N decode replicas.
        for i in range(self._n_decode):
            start = self._prefill_tp + i * self._decode_tp
            end = start + self._decode_tp
            workers_i = self.workers[start:end]
            node_id_i = worker_infos[start][0]
            devs_i = ",".join(worker_infos[start + j][1] for j in range(self._decode_tp))

            # Single-node MVP: every actor's side channel binds the same host as
            # prefill. Multi-node will need per-node IP discovery.
            decode_side_channel_port, decode_sock = get_free_port(prefill_host_ip, with_alive_sock=True)
            try:
                decode_kv_cfg = self._build_kv_transfer_config(
                    role="decode",
                    engine_id=uuid.uuid4().hex,
                    ib_device=self.config.disaggregation.ib_device,
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
                        actor_name=f"vllm_server_{self.replica_rank}_decode_{i}{self.name_suffix}",
                    )
                )
            finally:
                decode_sock.close()

        # Boot every actor's HTTP server. ``vLLMHttpServer.launch_server`` inherits
        # the colocated path here; PD-only pieces (kv_transfer_config injection)
        # are already wired through the actor kwargs.
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=None, master_port=None, dp_rpc_port=None)
                for server in self._prefill_servers + self._decode_servers
            ]
        )

        # Wire the prefill server with its decode peers so vLLMHttpServer.generate
        # can fan out per request. NIXL is pull-mode: only the decode side reads
        # the prefill's side-channel coordinates from kv_transfer_params, but we
        # still record them on prefill so dispatch logging stays self-contained.
        await self._prefill_servers[0].set_pd_peer.remote(
            self._decode_servers,
            self._prefill_side_channel_host,
            self._prefill_side_channel_port,
            self._prefill_engine_id,
        )

        # Externally, the replica looks like a single server fronting the
        # prefill; the prefill's generate() override fans out to decode peers.
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
            self._prefill_engine_id,
            self._prefill_side_channel_host,
            self._prefill_side_channel_port,
            len(self._decode_servers),
        )

    @staticmethod
    def _build_kv_transfer_config(
        role: str,
        engine_id: str,
        ib_device: Optional[str],
    ) -> dict:
        """Assemble the JSON payload for vLLM's ``--kv-transfer-config``.

        See ``vllm/config/kv_transfer.py`` for the full schema. ``kv_role`` maps
        verl's ``"prefill"|"decode"`` 1:1 to vLLM's ``"kv_producer"|"kv_consumer"``.
        """
        kv_role = "kv_producer" if role == "prefill" else "kv_consumer"
        cfg: dict = {
            "kv_connector": "NixlConnector",
            "kv_role": kv_role,
            "engine_id": engine_id,
            "kv_buffer_device": "cuda",
        }
        if ib_device:
            cfg["kv_connector_extra_config"] = {"ib_device": ib_device}
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
        actor_name: str,
    ) -> ActorHandle:
        """Construct one ``vLLMHttpServer`` Ray actor pinned to ``node_id`` with
        the right NIXL env vars and TP override. Mirrors the surrounding
        scheduling-strategy / runtime_env block of ``vLLMReplica.launch_servers``
        but adds the side-channel host/port and the PD role kwargs."""
        per_role_config = _dc_replace(self.config, tensor_model_parallel_size=tp)

        env_vars = {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            # Same NCCL note as the colocated path: avoid hangs / crashes
            # during weight-sync collectives (see vllm troubleshooting docs).
            "NCCL_CUMEM_ENABLE": "0",
            # NIXL side-channel listener: each engine binds its own port; the
            # connector reads these on import (nixl_connector.py:556-558).
            "VLLM_NIXL_SIDE_CHANNEL_HOST": side_channel_host,
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(side_channel_port),
        }

        prefix = self._get_server_name_prefix()
        if not actor_name.startswith(prefix):
            actor_name = f"{prefix}{actor_name}"

        return self.server_class.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
            runtime_env={"env_vars": env_vars},
            name=actor_name,
            max_concurrency=self.max_concurrency,
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
