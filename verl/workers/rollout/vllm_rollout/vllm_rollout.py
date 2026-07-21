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
from packaging import version as vs
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL, get_version
from verl.utils.device import get_device_id, is_support_ipc
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender
from verl.workers.rollout.vllm_rollout.utils import get_device_uuid

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _check_vllm_version_for_sleep_level():
    # https://github.com/vllm-project/vllm/issues/25171
    minver = "0.11.0"
    current_version = get_version("vllm")
    if not current_version:
        logger.warning("Could not determine vLLM version, assuming an older version for sleep_level configuration.")
        return False
    return vs.parse(current_version) >= vs.parse(minver)


def _should_expand_vllm_moe_params() -> bool:
    current_version = get_version("vllm")
    if not current_version:
        return False

    try:
        return vs.parse(current_version) <= vs.parse("0.24.0")
    except vs.InvalidVersion:
        return False


async def _iter_vllm_compatible_moe_params(weights):
    """Expand Transformers 5 packed MoE expert tensors to vLLM checkpoint keys.

    Transformers 5 stores Qwen-style MoE experts as packed 3D parameters:
    ``mlp.experts.gate_up_proj`` with shape
    ``[num_experts, 2 * intermediate_size, hidden_size]`` and
    ``mlp.experts.down_proj`` with shape
    ``[num_experts, hidden_size, intermediate_size]``. vLLM's Qwen MoE reload
    path still accepts the original per-expert checkpoint keys during live
    weight sync, so stream those keys without materializing a full dict.
    """
    from verl.workers.rollout.utils import ensure_async_iterator

    async for name, tensor in ensure_async_iterator(weights):
        if name.endswith(".mlp.experts.gate_up_proj") and tensor.dim() == 3:
            gate, up = tensor.chunk(2, dim=1)
            base = name.removesuffix(".gate_up_proj")
            for expert_id in range(tensor.size(0)):
                yield f"{base}.{expert_id}.gate_proj.weight", gate[expert_id].contiguous()
                yield f"{base}.{expert_id}.up_proj.weight", up[expert_id].contiguous()
            continue

        if name.endswith(".mlp.experts.down_proj") and tensor.dim() == 3:
            base = name.removesuffix(".down_proj")
            for expert_id in range(tensor.size(0)):
                yield f"{base}.{expert_id}.down_proj.weight", tensor[expert_id].contiguous()
            continue

        yield name, tensor


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
            per_replica = prefill_tp * disagg.prefill_replicas + decode_tp * disagg.decode_replicas
        else:
            decode_tp = prefill_tp
            per_replica = prefill_tp
        rollout_world_size = per_replica * self.config.data_parallel_size * self.config.pipeline_model_parallel_size
        if replica_rank == -1:
            self.replica_rank = rank // rollout_world_size
        else:
            self.replica_rank = replica_rank
        self.rollout_rank = rank % rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size

        # Map each trainer rank to its co-located vLLM server so weight-update
        # IPC handles stay on the GPU where they were created. Offset math
        # assumes prefill_replicas == 1 (enforced by vLLMPDReplica); if that
        # ever lifts, update both this block and vLLMPDReplica.launch_servers.
        self._pd_role: Optional[str] = None
        self._pd_server_index: Optional[int] = None
        self._pd_tp_local_rank: Optional[int] = None
        if disagg is not None and getattr(disagg, "enabled", False):
            footprint = prefill_tp + disagg.decode_replicas * decode_tp
            local = self.rollout_rank % footprint
            if local < prefill_tp:
                self._pd_role = "prefill"
                self._pd_server_index = 0
                self._pd_tp_local_rank = local
            else:
                off = local - prefill_tp
                self._pd_role = "decode"
                self._pd_server_index = off // decode_tp
                self._pd_tp_local_rank = off % decode_tp
            # Each role's TP-rank-0 owns the adapter (vs colocated where every
            # rollout_rank=0 owns it). One log line per PD rank at startup so
            # a deadlock report can be traced back to the role mapping.
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

        if config.layered_summon or (config.expert_parallel_size > 1 and not _check_vllm_version_for_sleep_level()):
            logger.warning("Setting the sleep level to 1 may cause a memory overflow.")
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

        self.device_uuid = get_device_uuid(get_device_id())
        # Use replica_rank + node-local rank to form ZMQ handle instead of GPU UUID,
        # because CheckpointEngineWorker and vLLM worker may see different GPU UUIDs
        # when CUDA_VISIBLE_DEVICES differs between processes (common on ROCm/AMD).
        # Must use node-local rank (not rollout_rank) so it matches vLLM worker's
        # local_rank on every node. Include replica_rank to avoid collisions when
        # multiple replicas share a node, and the Ray job id so two independent
        # verl jobs on the same host (or a new run after a crashed one with a
        # stale socket file) cannot collide on the shared /tmp namespace.
        local_rank = self.rollout_rank % local_world_size
        job_id = ray.get_runtime_context().get_job_id()
        self.zmq_handle = f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{self.replica_rank}-rank-{local_rank}.sock"

        self.use_shm = not is_support_ipc()
        if self.use_shm:
            logger.warning(
                "IPC is not supported on your devices. Falling back to shared memory for weight transfer, "
                "which may cause performance degradation. If you are using Ascend NPUs, please ensure that "
                "your software and CANN toolkit versions meet the requirements for IPC support. (Ascend HDK version "
                ">= 25.3.rc1 and CANN toolkit version >= 8.3.RC1)"
            )

    def _ensure_server_handle(self) -> bool:
        """Lazy-init server handle. Returns False if this rank should not proceed."""
        if not self._has_server:
            return False
        # Lazy init http server adapter because http server is launched after hybrid engine.
        if self.server_handle is None:
            prefix = self._get_server_name_prefix()
            if self._pd_role == "prefill":
                actor_name = f"{prefix}server_{self.replica_rank}_0"
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
        """Update model weights via CUDA IPC (fallback to shared memory if IPC not supported) to inference workers."""
        assert wire_format == "named_tensors", (
            f"vLLM rollout only consumes full named tensors; got wire_format={wire_format!r}"
        )
        start_time = time.time()

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
        # With LoRA enabled (peft_config), vLLM wraps FusedMoE so the per-expert
        # checkpoint keys no longer resolve (`experts.w13_weight` becomes
        # `experts.base_layer.w13_weight`); only the packed `experts.gate_up_proj`
        # / `experts.down_proj` load path is wrapper-aware, so keep tensors packed.
        if _should_expand_vllm_moe_params() and kwargs.get("peft_config") is None:
            weights = _iter_vllm_compatible_moe_params(weights)
        await sender.async_send_weights(weights)

        if future is not None:
            await future

        # reset caches after updating weights
        if self._has_server:
            await self.server_handle.clear_kv_cache.remote()
            if global_steps is not None:
                await self.server_handle.set_global_steps.remote(global_steps)

        if self.replica_rank == 0 and self.rollout_rank == 0:
            logger.info(f"update_weights done, time cost: {time.time() - start_time:.2f}s")

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
