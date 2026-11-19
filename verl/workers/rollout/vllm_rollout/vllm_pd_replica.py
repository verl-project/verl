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
"""vLLM PD-disaggregated replica with N prefill and M decode servers.

Asymmetric TP is supported, and the complete PD replica must fit on one node.
GPU keeps vLLM's native NIXL/Mooncake connectors; Ascend uses
vLLM-Ascend's MooncakeConnectorV1.
"""

import asyncio
import copy
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


# 【讲解入口 3：一个 PD 组的启动与生命周期】
# LLMServerManager 管全部组；本类只管一组；vLLMHttpServer 管其中一个 P 或 D 引擎。
# 统一例子：2P2D/TP4，组内 worker 顺序为 P0[0:4]、P1[4:8]、D0[8:12]、D1[12:16]。
# “单机 PD”限制的是一组 P/D 必须同节点，不表示整个任务只允许一个 replica。
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
        self._n_prefill = disagg.prefill_replicas
        self._n_decode = disagg.decode_replicas

        self._prefill_tp = self.config.tensor_model_parallel_size
        # Inline decode_tp default: OmegaConf/Ray serialization drops dataclass methods.
        self._decode_tp = (
            disagg.decode_tensor_model_parallel_size
            if disagg.decode_tensor_model_parallel_size is not None
            else self._prefill_tp
        )

        # 一组设备数 = P引擎数×P的TP + D引擎数×D的TP。
        # 例如 1P2D，P TP8、D TP4：8 + 2×4 = 16；TP 不同并不改变这套切片公式。
        # 布局还要符合底层模型/Connector 的 TP 约束，这个公式只计算需要多少 worker。
        pd_world_size = self._n_prefill * self._prefill_tp + self._n_decode * self._decode_tp
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

        # servers 列表保存 Ray 句柄，供 RPC 调用；addresses 列表保存启动后查到的 HTTP 地址。
        # P/D 分开保存，便于“全部 P 注册为请求入口”“全部 D 先休眠”等按角色处理。
        self._prefill_servers: list[ActorHandle] = []
        self._decode_servers: list[ActorHandle] = []
        self._prefill_server_addresses: list[str] = []
        self._decode_server_addresses: list[str] = []

    async def launch_servers(self):
        assert len(self.workers) == self.world_size, (
            f"worker count {len(self.workers)} != PD world size {self.world_size}"
        )
        # 【启动步骤1：确定实际 Connector 路径】这里按 NPU 设备选择 Ascend V1。
        # 配置写 nixl 的 NPU 路径也会回退到 Mooncake V1，所以后面不能只看原配置字符串。
        use_ascend_mooncake_v1 = self._is_ascend_platform()
        transfer_backend = self.config.disaggregation.transfer_backend
        if use_ascend_mooncake_v1 and transfer_backend == "nixl":
            logger.warning("NixlConnector is not supported for Ascend PD; falling back to MooncakeConnectorV1")
            transfer_backend = "mooncake"

        # 【启动步骤2：询问真实 worker】__ray_call__.remote 内的 lambda 在各 worker 进程执行。
        # 每项返回 (Ray节点ID, 设备编号字符串, 节点IP)，顺序与 self.workers 保持一致。
        # 设备可见列表与 worker 切片必须一致，后续权重 IPC 才能对应到正确设备。
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
        # 集合去重后只有一个节点 ID，才说明这一组 worker 真正在同一台机器上。
        # 这里只检查 P/D 所在节点，没有检查 Manager 自己是否也在该节点。
        if len({info[0] for info in worker_infos}) != 1:
            raise ValueError("All workers of a PD replica must be placed on the same node")

        # 【启动步骤3：预留端口】以下调用仍是 Manager 所在进程的普通 Python 调用。
        # 传入 worker IP 不会把代码迁移到 worker 节点；Manager 异机时可能 bind 非本机IP失败。
        # 这是当前实现边界，注释版没有把预留逻辑改成 worker-side RPC。
        # 五个返回值依次是：P基础端口列表、bootstrap基础端口、D基础端口列表、占位socket列表、提前关闭标记。
        prefill_host_ip = worker_infos[0][2]
        (
            prefill_side_channel_ports,
            bootstrap_port,
            decode_side_channel_ports,
            reserved_socks,
            close_sockets_before_launch,
        ) = self._reserve_pd_ports(prefill_host_ip, use_ascend_mooncake_v1)
        # Ascend 的占位 socket 要先关闭，后端才能自己绑定这些连续端口。
        # 释放占位到真实后端绑定之间仍有被抢占的窗口；这不是可跨进程移交的端口租约。
        # GPU 路径保留原有清理时机，下面的 finally 负责关闭尚未释放的占位 socket。
        if close_sockets_before_launch:
            for sock in reserved_socks:
                sock.close()
            reserved_socks.clear()
        prefill_peers = []
        try:
            # 【启动步骤4：逐个创建 P】每个 P 都有自己的 worker 切片、engine_id、端口和 actor 名称。
            # start=i×P_TP；end=start+P_TP。例子中 i=1 时切片 [4:8] 正好是 P1 的四个 worker。
            for i in range(self._n_prefill):
                start = i * self._prefill_tp
                end = start + self._prefill_tp
                prefill_engine_id = uuid.uuid4().hex
                prefill_side_channel_port = prefill_side_channel_ports[i]
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
                # Keep the existing actor name for 1P callers and weight adapters.
                actor_name = (
                    f"vllm_server_{self.replica_rank}_0{self.name_suffix}"
                    if self._n_prefill == 1
                    else f"vllm_server_prefill_{self.replica_rank}_{i}_0{self.name_suffix}"
                )
                # 把引擎在“整组”中的起点传给权重接收端。例如 P1 的起点为4，内部TP-local2对应整组rank6。
                server = self._spawn_pd_server(
                    role="prefill",
                    workers=self.workers[start:end],
                    node_id=worker_infos[start][0],
                    cuda_visible_devices=self._collect_cuda_devices(worker_infos[start:end]),
                    tp=self._prefill_tp,
                    kv_transfer_config=prefill_kv_cfg,
                    side_channel_host=prefill_host_ip,
                    side_channel_port=prefill_side_channel_port,
                    # Native GPU Mooncake starts one bootstrap server per P.
                    mooncake_bootstrap_port=(bootstrap_port if use_ascend_mooncake_v1 else prefill_side_channel_port),
                    actor_name=actor_name,
                    zmq_base_trainer_rank=start,
                )
                self._prefill_servers.append(server)
                prefill_peers.append((server, prefill_side_channel_port, prefill_engine_id))

            # 【启动步骤5：逐个创建 D】先跳过全部 P，再在 D 区域中切片。
            # 2P2D/TP4 时：D0 start=8，D1 start=12；每个 D 只得到自己那四个 worker。
            # 所有 P 会持有本组全部 D 的句柄，但不会持有其他 replica 的 D。
            for i in range(self._n_decode):
                start = self._n_prefill * self._prefill_tp + i * self._decode_tp
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

            # 先创建 actor，再发 launch_server RPC 启动各自的推理引擎；gather 等待全部启动完成。
            # 前面的 Ray actor 句柄存在，不代表模型、Connector 和 HTTP 服务都已经就绪。
            await asyncio.gather(
                *[
                    server.launch_server.remote(master_address=None, master_port=None, dp_rpc_port=None)
                    for server in self._prefill_servers + self._decode_servers
                ]
            )
        finally:
            for sock in reserved_socks:
                sock.close()

        await asyncio.gather(
            *[
                # 【启动步骤6：建立组内关系】每个 P 获得全部 D 句柄及自己 P 的交接地址信息。
                # set_pd_peer 在 P 侧初始化轮询或可选 UniAgent router；每个 P 各有独立实例和账本。
                server.set_pd_peer.remote(
                    decode_peers=self._decode_servers,
                    prefill_side_channel_port=port,
                    prefill_engine_id=engine_id,
                )
                for server, port, engine_id in prefill_peers
            ]
        )

        # 【启动步骤7：收集端点】servers 用于组级生命周期调用；旧单入口属性仍指向 P0。
        # 下面重新读取所有引擎的 HTTP 地址；这些地址与前面的 KV side-channel 端口用途不同。
        self.servers = list(self._prefill_servers) + list(self._decode_servers)
        self._server_handle = self._prefill_servers[0]
        # Every independent engine exposes metrics on both GPU and Ascend.
        addresses = await asyncio.gather(*[server.get_server_address.remote() for server in self.servers])
        self._prefill_server_addresses = [
            self._format_server_address(host, port) for host, port in addresses[: len(self._prefill_servers)]
        ]
        self._decode_server_addresses = [
            self._format_server_address(host, port) for host, port in addresses[len(self._prefill_servers) :]
        ]
        self._server_address = self._prefill_server_addresses[0]

        logger.info(
            "vLLMPDReplica rank=%s launched: prefills=%s, decodes=%s",
            self.replica_rank,
            self._prefill_server_addresses,
            self._decode_server_addresses,
        )

    def get_request_server_endpoints(self) -> list[tuple[str, ActorHandle]]:
        """Expose every prefill engine, using addresses collected at startup."""
        # 【交给外层路由器的名单】返回全部 P 的 (地址,句柄)；P 负责完成后续 P→D 调用。
        # zip 按下标配对，strict=True 保证数量相等；只返回 P0 会让其他 P 接不到外层请求。
        return list(zip(self._prefill_server_addresses, self._prefill_servers, strict=True))

    def get_metrics_server_endpoints(self) -> list[tuple[str, dict[str, Any]]]:
        # 【交给监控系统的名单】启动未收齐地址时直接报错；P 与 D 都要监控。
        # 标签如 0-prefill1 / 0-decode1 用来区别同组的独立引擎，不是额外的 replica。
        if len(self._prefill_server_addresses) != len(self._prefill_servers):
            raise RuntimeError("PD prefill metrics endpoints are not ready")
        if len(self._decode_server_addresses) != len(self._decode_servers):
            raise RuntimeError("PD decode metrics endpoints are not ready")
        return [
            *[
                (
                    address,
                    {
                        "replica": (
                            f"{self.replica_rank}-prefill"
                            if len(self._prefill_servers) == 1
                            else f"{self.replica_rank}-prefill{index}"
                        ),
                        "pd_role": "prefill",
                        "pd_index": index,
                    },
                )
                for index, address in enumerate(self._prefill_server_addresses)
            ],
            *[
                (address, {"replica": f"{self.replica_rank}-decode{index}", "pd_role": "decode", "pd_index": index})
                for index, address in enumerate(self._decode_server_addresses)
            ],
        ]

    async def resume_generation(self):
        """Resume every independent P/D engine before opening any admission gate."""
        # 【恢复屏障】先恢复所有 P/D 引擎的生成能力，全部成功后才开放新请求入口。
        # 两个 gather 分成两阶段，避免 P 已经接请求而目标 D 仍未恢复。
        await asyncio.gather(*[server.resume_engine_generation.remote() for server in self.servers])
        await asyncio.gather(*[server.open_submission_gate.remote() for server in self.servers])

    async def sleep(self):
        """Drain all prefills, then sleep Mooncake consumers before producers.

        The trainer must quiesce generation before calling this method. Worker
        sleep owns transport unregistration; no connector disconnect RPC is used.
        """
        # 【组级 sleep】本函数是编排者，vllm_async_server.sleep 才执行某个引擎的实际休眠。
        # trainer 需先停新请求；这里等待全部 P 的排空调用完成，再安排引擎休眠。
        # P 排空并不单独证明所有远端传输都已结束，完整 transfer barrier 仍依赖底层与调用方协调。
        await asyncio.gather(*[server.wait_for_requests_to_drain.remote() for server in self._prefill_servers])
        # Ascend also uses Mooncake when the configured NIXL backend falls back.
        if self._is_ascend_platform() or self.config.disaggregation.transfer_backend == "mooncake":
            # Mooncake 按 D 后 P 的依赖顺序收尾：先等全部消费者 D 休眠成功，再让生产者 P 休眠。
            # 第一个 gather 抛错时不会进入下一行；传输内存注销由 worker 生命周期负责。
            await asyncio.gather(*[server.sleep.remote() for server in self._decode_servers])
            await asyncio.gather(*[server.sleep.remote() for server in self._prefill_servers])
        else:
            # 原生 GPU NIXL 保留各引擎并行休眠；仍是在前面的 P 排空之后执行。
            # 这里的判断是生命周期策略分支，不表示非 Mooncake 的 D 可以跳过 sleep。
            await asyncio.gather(*[server.sleep.remote() for server in self.servers])

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
    ) -> tuple[list[int], int, list[int], list[socket.socket], bool]:
        """Reserve P/D ports using either upstream or Ascend allocation semantics."""
        # 【端口分配总管】端口号是数字，socket.bind 才是暂时占位；关闭 socket 才释放占位。
        # prefill_ports/decode_ports 只保存每个引擎的基础端口；reserved_socks 保存所有占位对象。
        # contiguous=True 是当前 Ascend V1 的分支，False 是保留的 GPU 分配分支。
        reserved_socks: list[socket.socket] = []
        prefill_ports: list[int] = []
        decode_ports: list[int] = []
        try:
            # 例：1P2D，P TP8/D TP4：给 P 预留8个连续端口，给每个D预留4个，另给bootstrap预留16个。
            # 例如基础端口 P=30000，D0=32000，D1=33000；列表记起点，rank端口由base+rank派生。
            # append 加一个基础端口；extend 把一整段的全部 socket 加进总清理列表。
            if contiguous:
                for _ in range(self._n_prefill):
                    prefill_port, prefill_socks = self._reserve_port_range(address, self._prefill_tp)
                    prefill_ports.append(prefill_port)
                    reserved_socks.extend(prefill_socks)
                # bootstrap 是引导通信，不是“另一个 D”；当前实现额外预留与整组 worker 数相同的连续范围。
                # 上例 world_size=16，总共暂占8+16+4+4=32个端口。
                bootstrap_port, bootstrap_socks = self._reserve_port_range(address, self.world_size)
                reserved_socks.extend(bootstrap_socks)
                for _ in range(self._n_decode):
                    decode_port, decode_socks = self._reserve_port_range(address, self._decode_tp)
                    decode_ports.append(decode_port)
                    reserved_socks.extend(decode_socks)
                return prefill_ports, bootstrap_port, decode_ports, reserved_socks, True

            for _ in range(self._n_prefill):
                # GPU 分支每个引擎只找一个端口，不在本层展开成完整的每TP连续区间。
                # with_alive_sock=True 返回尚未关闭的占位 socket，供外层统一管理其生命周期。
                prefill_port, prefill_sock = get_free_port(address, with_alive_sock=True)
                prefill_ports.append(prefill_port)
                reserved_socks.append(prefill_sock)
            for _ in range(self._n_decode):
                decode_port, decode_sock = get_free_port(address, with_alive_sock=True)
                decode_ports.append(decode_port)
                reserved_socks.append(decode_sock)
            # 返回第一个 P 端口作为该分支的 bootstrap 返回值；GPU Mooncake 的每个 P 启动时用各自端口。
            # 末尾 False 表示不走“创建actor之前提前关闭”分支，外层 finally 会清理剩余占位。
            return prefill_ports, prefill_ports[0], decode_ports, reserved_socks, False
        # 整组分配中途失败，要释放前面已成功预留的 P/bootstrap/D 端口段。
        # 当前失败段自身的清理由 _reserve_port_range 完成，两层清理负责不同范围。
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
        # 【连续区间的底层分配】一次尝试找一个起点，逐个绑定后续端口；失败就整段释放后重试。
        # 最多试128次，不保证一定找到；family 按 IPv4/IPv6 选择 socket 类型。
        for _ in range(128):
            probe = socket.socket(family=family, type=socket.SOCK_STREAM)
            socks = [probe]
            try:
                # 端口0的含义是“让操作系统挑一个空闲端口”，不是让服务固定使用端口0。
                # getsockname 读取实际挑中的端口；系统只保证这个起点当下可绑定，后续端口仍需逐个检查。
                probe.bind((address, 0))
                base = probe.getsockname()[1]
                # 区间末尾不能超过65535；例如 base=65534、count=4 就需要换起点。
                if base + count - 1 > 65535:
                    probe.close()
                    continue
                # base 已被 probe 占住；count=4 时这里只再绑定 base+1、base+2、base+3。
                # 如果中间某个端口被其他程序占用，本次占到的所有 socket 都要关闭，不能留下半段。
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
        # 【Connector 配置】这里只组装传给 vLLM 的配置，不执行 KV 数据复制。
        # P 是 kv_producer（产出KV），D 是 kv_consumer（消费KV）；两端角色必须与实际切片一致。
        role_to_kv_role = {
            "prefill": "kv_producer",
            "decode": "kv_consumer",
        }
        if use_ascend_mooncake_v1:
            if transfer_backend != "mooncake":
                raise ValueError("Ascend PD requires transfer_backend='mooncake'")
            if kv_port is None or prefill_tp is None or decode_tp is None:
                raise ValueError("MooncakeConnectorV1 requires kv_port, prefill_tp, and decode_tp")
            # V1 是当前 Ascend 使用的 Connector 类名；GPU 的 MooncakeConnector 是另一个实现。
            # 不能按“名字带V1就比GPU旧”理解，两者应分别匹配各自的 vLLM/Ascend/Mooncake 运行栈。
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
            # kv_port 给当前引擎每TP端点提供基础端口；prefill/decode 的 tp_size 描述两端 KV 切片布局。
            # 即使当前 actor 是 D，也要知道 P 的 TP；它需要据此解释远端块中每份 KV 的组织方式。
            # GPU 路径使用原生 Connector 的配置协议，不能因为这里没显式填TP就断言它不需要布局信息。
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
        disagg = self.config.disaggregation
        memory_utilization = (
            disagg.prefill_gpu_memory_utilization if role == "prefill" else disagg.decode_gpu_memory_utilization
        )
        role_overrides = disagg.prefill_engine_kwargs if role == "prefill" else disagg.decode_engine_kwargs
        # 【按角色复制配置】deepcopy 防止 P 的覆盖影响 D，或一个引擎改动嵌套配置污染其他引擎。
        # 先复制公共 kwargs，再合并角色中非None的项；缺失/null 的 vllm 子字典按空字典处理。
        # 这里是按键覆盖，不做递归合并：角色给了一个嵌套对象，会替换公共的整个同名对象。
        engine_kwargs = copy.deepcopy(self.config.engine_kwargs)
        engine_kwargs["vllm"] = engine_kwargs.get("vllm") or {}
        engine_kwargs["vllm"].update({key: value for key, value in role_overrides.items() if value is not None})
        # 显式角色显存字段最高优先。例如公共0.8、D显式0.5，最终引擎必须收到0.5。
        # 启动器最后会把 engine_kwargs 覆盖到参数字典上，所以这里也要写入它，并去掉横杠别名。
        # 否则只改 per_role_config.gpu_memory_utilization，仍可能被 kwargs 中的0.8覆盖回去。
        if memory_utilization is not None:
            # Explicit role fractions must also win in the final vLLM CLI args.
            engine_kwargs["vllm"].pop("gpu-memory-utilization", None)
            engine_kwargs["vllm"]["gpu_memory_utilization"] = memory_utilization
        per_role_config = _dc_replace(
            self.config,
            tensor_model_parallel_size=tp,
            gpu_memory_utilization=(
                self.config.gpu_memory_utilization if memory_utilization is None else memory_utilization
            ),
            engine_kwargs=engine_kwargs,
        )

        env_vars = {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            "NCCL_CUMEM_ENABLE": "0",
            "VLLM_NIXL_SIDE_CHANNEL_HOST": side_channel_host,
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(side_channel_port),
            "VLLM_MOONCAKE_BOOTSTRAP_PORT": str(mooncake_bootstrap_port),
            # Avoid Mooncake TCP port exhaustion under validation concurrency.
            "MC_TCP_ENABLE_CONNECTION_POOL": os.environ.get("MC_TCP_ENABLE_CONNECTION_POOL", "1"),
            # 权重接收端用“引擎起点 + TP内部编号”还原整组编号，例如 D1:12+1=13。
            # 任务ID、replica号与该编号共同组成 IPC 路径；这不是 KV side-channel 的 TCP 端口。
            "VERL_ZMQ_BASE_TRAINER_RANK": str(zmq_base_trainer_rank),
            "VERL_RAY_JOB_ID": ray.get_runtime_context().get_job_id(),
        }

        # NodeAffinitySchedulingStrategy(soft=False) 把这个服务 actor 固定到其 worker 所在节点。
        # 这个定位约束作用于新 actor，不会把前面已经在 Manager 执行的端口预留操作搬过去。
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
