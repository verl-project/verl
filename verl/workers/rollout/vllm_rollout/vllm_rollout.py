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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import time
from typing import Any, Generator, Optional

import ray
import torch
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL
from verl.utils.device import is_support_ipc
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class ServerAdapter(BaseRollout):
    """
    vLLM server adapter used in native async mode, serve as a client to request vLLM server
    to resume/release/update weights and kv_cache.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        replica_rank: int = -1,
    ):
        super().__init__(config, model_config, device_mesh)
        self.server_handle: ray.actor.ActorHandle = None

        # 【讲解入口 5：训练侧怎样找到同设备的推理worker】本适配器负责权重同步/生命周期控制。
        # 统一例子2P2D/TP4：每组16个worker，P0[0:4]、P1[4:8]、D0[8:12]、D1[12:16]。
        # rank是全局训练编号；RAY_LOCAL_WORLD_SIZE是节点内worker数，不是某个引擎的TP数。
        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        # PD asymmetric layout inflates per-replica footprint; must match
        # llm_server.py:_initialize_llm_servers or trainer-to-replica mapping breaks.
        prefill_tp = self.config.tensor_model_parallel_size
        disagg = getattr(self.config, "disaggregation", None)
        if disagg is not None and getattr(disagg, "enabled", False):
            decode_tp = (
                disagg.decode_tensor_model_parallel_size
                if disagg.decode_tensor_model_parallel_size is not None
                else prefill_tp
            )
            # 这里也要用整组P/D设备数，必须与Manager分组、PD启动器切片相同，否则训练发给错误的引擎。
            # PD当前要求DP=PP=1；普通路径仍按原有TP×DP×PP计算。
            per_replica = prefill_tp * disagg.prefill_replicas + decode_tp * disagg.decode_replicas
        else:
            decode_tp = prefill_tp
            per_replica = prefill_tp
        rollout_world_size = per_replica * self.config.data_parallel_size * self.config.pipeline_model_parallel_size
        # //取整除的商，回答“第几组”；%取余，回答“组内第几个”。编号都从0开始。
        # 每组16个worker：全局rank29 → 29//16=1即replica1，29%16=13即组内rank13。
        # 显式传了replica_rank时保留调用方组号，常用于避免不同资源池的actor名称冲突。
        if replica_rank == -1:
            self.replica_rank = rank // rollout_world_size
        else:
            self.replica_rank = replica_rank
        self.rollout_rank = rank % rollout_world_size
        # 按每节点worker数算“当前replica内的节点编号”；不是整个集群的物理节点ID。
        # 本例local_world_size=16，13//16=0；当前有效单节点PD组的这一编号为0。
        self.node_rank = self.rollout_rank // local_world_size

        # Match launch_servers: all prefill engines, then all decode engines.
        # Each engine's TP-rank-0 controls that engine's weight-update RPC.
        self._pd_role: Optional[str] = None
        self._pd_server_index: Optional[int] = None
        self._pd_tp_local_rank: Optional[int] = None
        if disagg is not None and getattr(disagg, "enabled", False):
            # 【先找P/D区域】全部P占prefill_span=2×4=8个位置；footprint=8+2×4=16是整组大小。
            # local<8属于P；local>=8属于D。%footprint把编号限制到这一组布局范围内。
            prefill_span = prefill_tp * disagg.prefill_replicas
            footprint = prefill_span + disagg.decode_replicas * decode_tp
            local = self.rollout_rank % footprint
            if local < prefill_span:
                self._pd_role = "prefill"
                # 【P区域内分组】比如local=6：6//4=1是P1，6%4=2是P1的TP-local2。
                # server_index表示第几个P引擎，tp_local_rank表示该引擎内部第几个worker。
                self._pd_server_index = local // prefill_tp
                self._pd_tp_local_rank = local % prefill_tp
            else:
                # 【D区域内分组】先减掉全部P占的8个位置，再每4个划为一个D。
                # local=13 → off=5 → 5//4=1即D1，5%4=1即TP-local1。
                # 不能直接13//4当D下标，那会把前面的两个P也算进去。
                off = local - prefill_span
                self._pd_role = "decode"
                self._pd_server_index = off // decode_tp
                self._pd_tp_local_rank = off % decode_tp
            # Each role's TP-rank-0 owns the adapter (vs colocated where every
            # rollout_rank=0 owns it). One log line per PD rank at startup so
            # a deadlock report can be traced back to the role mapping.
            # 每个引擎只由其TP-local0发控制RPC，避免四个rank重复唤醒/清缓存/启动同一次权重接收。
            # 其他rank仍有自己的权重payload发送端，不是被排除在权重同步之外。
            self._has_server = self._pd_tp_local_rank == 0
            logger.info(
                "vllm PD ServerAdapter: rank=%d replica=%d rollout=%d role=%s server_idx=%s tp_local=%s has_server=%s",
                rank,
                self.replica_rank,
                self.rollout_rank,
                self._pd_role,
                self._pd_server_index,
                self._pd_tp_local_rank,
                self._has_server,
            )
        else:
            self._has_server = self.rollout_rank == 0

        if config.layered_summon:
            logger.warning("Setting the sleep level to 1 may cause a memory overflow.")
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

        # PD receivers add VERL_ZMQ_BASE_TRAINER_RANK to the engine-local rank.
        # Ordinary replicas retain node-local IPC ranks. Job/replica identifiers
        # keep independent engines and jobs out of each other's IPC namespace.
        # 【IPC地址里的编号】PD用整组编号，例如D1/TP-local1用13，而不是各引擎都从0起算的1。
        # 接收端在utils._get_zmq_handle用base12+内部rank1也算出13，发送和接收才能匹配。
        # 普通路径保留节点内取余规则。若单机PD中local_world_size>=footprint，两种写法数值本来就相同；
        # 这里明确的是编号约定，不能把“改前在这个例子也得到13”说成必然存在冲突。
        local_rank = self.rollout_rank if self._pd_role is not None else self.rollout_rank % local_world_size
        job_id = ray.get_runtime_context().get_job_id()
        # 路径同时包含任务ID、组号、组内worker号；两个组都用rank13也因replica不同而互不混淆。
        # 这是本机IPC的命名路径，不是前面的HTTP地址或Mooncake TCP通信端口。
        self.zmq_handle = f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{self.replica_rank}-rank-{local_rank}.sock"

        self.use_shm = not is_support_ipc()
        self._delta_weight_transfer_engine_initialized = False
        if self.use_shm:
            logger.warning(
                "IPC is not supported on your devices. Falling back to shared memory for weight transfer, "
                "which may cause performance degradation. If you are using Ascend NPUs, please ensure that "
                "your software and CANN toolkit versions meet the requirements for IPC support. (Ascend HDK version "
                ">= 25.3.rc1 and CANN toolkit version >= 8.3.RC1)"
            )

    def _ensure_server_handle(self) -> bool:
        """Lazy-init server handle. Returns False if this rank should not proceed."""
        # 【按名字找actor】只有控制rank会继续；HTTP服务可能晚于adapter创建，所以第一次使用时再找。
        # 名字必须与vLLMPDReplica.launch_servers创建actor时完全一致。
        if not self._has_server:
            return False
        # Lazy init http server adapter because http server is launched after hybrid engine.
        if self.server_handle is None:
            prefix = self._get_server_name_prefix()
            if self._pd_role == "prefill":
                # 1P保留旧名server_{组号}_0；多P增加prefill_{组号}_{P下标}_0，以区分同组多个P。
                # 比如replica1的P1对应server_prefill_1_1_0；D1对应server_decode_1_1。
                actor_name = (
                    f"{prefix}server_{self.replica_rank}_0"
                    if self.config.disaggregation.prefill_replicas == 1
                    else f"{prefix}server_prefill_{self.replica_rank}_{self._pd_server_index}_0"
                )
            elif self._pd_role == "decode":
                actor_name = f"{prefix}server_decode_{self.replica_rank}_{self._pd_server_index}"
            else:
                actor_name = f"{prefix}server_{self.replica_rank}_{self.node_rank}"
            self.server_handle = ray.get_actor(actor_name)
        return True

    async def _execute_method(
        self,
        method: str,
        non_block: bool = False,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ) -> Any:
        """Execute method on inference engine via ray.

        Args:
            method: The method name to execute on the server.
            non_block: If True, execute the method asynchronously and return immediately.
            timeout: Timeout for the collective_rpc call.
            args: Positional arguments for the method.
            kwargs: Keyword arguments for the method.

        Returns:
            The result of the method execution, or None if non_block=True.
        """
        if not self._ensure_server_handle():
            return None

        future = self.server_handle.collective_rpc.remote(method, timeout=timeout, args=args, kwargs=kwargs)
        return future if non_block else await future

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if self.config.free_cache_engine and self._ensure_server_handle():
            await self.server_handle.wake_up.remote(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine and self._ensure_server_handle():
            await self.server_handle.sleep.remote()

    @torch.no_grad()
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int = None,
        wire_format: str = "named_tensors",
        **kwargs,
    ):
        """Update rollout weights from full tensors or DeltaFlush payloads.

        Full-tensor updates may fall back to shared memory; delta updates
        require colocated CUDA IPC.
        """
        if wire_format == "delta_flush":
            return await self._update_delta_weights(weights, global_steps=global_steps)
        assert wire_format == "named_tensors", (
            f"vLLM rollout supports wire_format='named_tensors' or 'delta_flush'; got {wire_format!r}"
        )
        start_time = time.time()

        # 【权重同步两层动作】控制rank先让引擎各TP worker进入接收方法，non_block=True先取RPC引用。
        # 下面每个训练rank都创建自己的sender并发送自己的权重分片；不能只让TP-local0发全部payload。
        future = await self._execute_method(
            "update_weights_from_ipc",
            non_block=True,
            kwargs={**kwargs, "use_shm": self.use_shm},
        )

        bucket_size_mb = self.config.checkpoint_engine.update_weights_bucket_megabytes
        sender = BucketedWeightSender(
            zmq_handle=self.zmq_handle,
            bucket_size_mb=bucket_size_mb,
            use_shm=self.use_shm,
        )
        # sender使用上面计算的IPC路径，与同设备vLLM接收worker对接；分桶协议本身没有因多P而重写。
        # 权重传输和P→D的KV传输是两条不同的数据通路。
        await sender.async_send_weights(weights)

        if future is not None:
            await future

        # reset caches after updating weights
        # 传完后仍只由每引擎的控制rank发清缓存RPC，避免旧权重KV被继续复用；其他rank不重复发控制命令。
        if self._has_server:
            await self.server_handle.clear_kv_cache.remote()
            if global_steps is not None:
                await self.server_handle.set_global_steps.remote(global_steps)

        if self.replica_rank == 0 and self.rollout_rank == 0:
            logger.info(f"update_weights done, time cost: {time.time() - start_time:.2f}s")

    async def _update_delta_weights(
        self,
        weights,
        *,
        global_steps: int | None,
    ) -> None:
        """Send one delta weight update as a stream of DeltaFlush payloads."""

        from verl.workers.rollout.utils import ensure_async_iterator

        if self.use_shm:
            raise NotImplementedError("delta_sharded with vLLM requires colocated CUDA IPC")

        flushes = ensure_async_iterator(weights)
        try:
            first_item = await anext(flushes)
        except StopAsyncIteration:
            # A steady sync with no changed BF16 values sends only a terminal
            # marker: weights are unchanged, so advance the step tag without
            # touching the transfer engine or the KV cache.
            if global_steps is not None and self._ensure_server_handle():
                await self.server_handle.set_global_steps.remote(global_steps)
            return

        first_named_tensors, saw_last = first_item
        if not self._delta_weight_transfer_engine_initialized:
            await self._execute_method(
                "init_weight_transfer_engine",
                kwargs={"init_info": {}},
            )
            self._delta_weight_transfer_engine_initialized = True

        await self._execute_method("start_weight_update")

        async def send_flush(flush_tensors: list[tuple[str, torch.Tensor]]) -> None:
            receiver_future = await self._execute_method(
                "update_verl_delta_weights",
                non_block=True,
                kwargs={"update_info": {}},
            )
            sender = BucketedWeightSender(
                zmq_handle=self.zmq_handle,
                bucket_size_mb=self.config.checkpoint_engine.update_weights_bucket_megabytes,
                use_shm=False,
            )
            await sender.async_send_weights(iter(flush_tensors))
            if receiver_future is not None:
                await receiver_future

        await send_flush(list(first_named_tensors))
        async for named_tensors, is_last in flushes:
            if saw_last:
                raise ValueError("DeltaFlush stream yielded data after is_last=True")
            saw_last = is_last
            await send_flush(list(named_tensors))

        if not saw_last:
            raise ValueError("DeltaFlush stream ended without is_last=True")

        await self._execute_method("finish_weight_update")

        if self._has_server:
            await self.server_handle.clear_kv_cache.remote()
            if global_steps is not None:
                await self.server_handle.set_global_steps.remote(global_steps)

    def _get_server_name_prefix(self) -> str:
        """Return the Ray actor name prefix matching the rollout type (e.g. 'vllm_')."""
        return f"{self.config.get('name', 'vllm')}_"

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode.

        Note: ServerAdapter uses async server mode and does not support synchronous
        generation. Since SPMD mode was retired (PR #4411), the generation workflow
        should use the async server interface instead.

        Raises:
            NotImplementedError: Always raised as sync generation is not supported.
        """
        raise NotImplementedError(
            "ServerAdapter does not support synchronous generate_sequences(). "
            "The vLLM SPMD mode was retired in PR #4411. For batch generation, "
            "please use the async server interface via vLLMReplica and LLMServerClient, "
            "or use HFRollout for synchronous generation. "
            "See https://github.com/verl-project/verl/issues/4682 for more details."
        )
