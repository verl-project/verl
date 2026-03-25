from __future__ import annotations

import logging
import os
import time
from typing import Generator

import ray
import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.decoupled_spec_rollout.layout import DecoupledSpecRole, resolve_server_adapter_layout
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter
from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets
from verl.workers.rollout.utils import ensure_async_iterator

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DecoupledSGLangServerAdapter(BaseRollout):
    """Dedicated server adapter for decoupled-spec SGLang rollout.

    It resolves verify/draft ownership from global rank and the decoupled-spec topology
    (``compute_decoupled_spec_topology`` / ``resolve_server_adapter_layout``) instead of
    assuming all hybrid workers belong to the verifier rollout mesh.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        replica_rank: int = -1,
    ):
        if config.get("quantization", None) == "fp8":
            import sglang
            from packaging import version

            assert version.parse(sglang.__version__) >= version.parse("0.5.5"), (
                "sglang>=0.5.5 is required for FP8 quantization"
            )
            fp8_block_quant_kwargs = {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
            model_config.hf_config.quantization_config = dict(fp8_block_quant_kwargs)

        super().__init__(config, model_config, device_mesh)
        self._engine: AsyncHttpServerAdapter = None

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        world_size = int(os.environ["WORLD_SIZE"])

        adapter_layout = resolve_server_adapter_layout(
            self.config,
            global_rank=rank,
            local_world_size=local_world_size,
            world_size=world_size,
        )
        self.active_in_decoupled_spec = adapter_layout is not None
        if adapter_layout is None:
            self.decoupled_spec_role = None
            self.replica_rank = replica_rank if replica_rank != -1 else -1
            self.rollout_rank = -1
            self.node_rank = -1
            self.local_rank = -1
            self.server_actor_name = None
            self.is_leader_rank = False
            self.server_actor = None
            print(
                "[decoupled_spec][server_adapter] inactive_rank "
                f"global_rank={rank} world_size={world_size} assigned_ranks_skipped=True"
            )
            return

        self.decoupled_spec_role = adapter_layout.role
        self.replica_rank = adapter_layout.replica_rank if replica_rank == -1 else replica_rank
        self.rollout_rank = adapter_layout.rollout_rank
        self.node_rank = adapter_layout.node_rank
        self.local_rank = adapter_layout.local_rank
        self.server_actor_name = adapter_layout.server_actor_name
        self.is_leader_rank = self.local_rank == 0 and self.decoupled_spec_role != DecoupledSpecRole.DRAFT
        self.server_actor = None

    async def _drain_weights_for_collective_alignment(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        *,
        role_label: str,
        global_steps: int | None,
    ) -> None:
        drain_start = time.perf_counter()
        print(
            "[decoupled_spec][server_adapter] drain_weights_start "
            f"role={role_label} replica_rank={self.replica_rank} "
            f"node_rank={self.node_rank} local_rank={self.local_rank} global_steps={global_steps}"
        )

        drained_count = 0
        sample_names: list[str] = []
        async for name, _tensor in ensure_async_iterator(weights):
            drained_count += 1
            if len(sample_names) < 5:
                sample_names.append(name)

        print(
            "[decoupled_spec][server_adapter] drain_weights_done "
            f"role={role_label} replica_rank={self.replica_rank} "
            f"node_rank={self.node_rank} local_rank={self.local_rank} global_steps={global_steps} "
            f"drained_count={drained_count} sample_names={sample_names} "
            f"elapsed_s={time.perf_counter() - drain_start:.6f}"
        )

    def _ensure_device_mesh(self) -> None:
        if self.device_mesh is None:
            assert torch.distributed.is_initialized(), "torch distributed must be initialized"
            infer_tp = self.config.tensor_model_parallel_size * self.config.data_parallel_size
            infer_pp = self.config.pipeline_model_parallel_size
            infer_world_size = infer_tp * infer_pp
            dp = torch.distributed.get_world_size() // infer_world_size
            self.device_mesh = init_device_mesh(
                "cpu", mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )

    def _should_control_server(self) -> bool:
        # Each HTTP server actor is launched per node, so only one local rank
        # should issue control-plane requests for that node.
        return self.local_rank == 0

    async def _init_server_adapter(self):
        if not self.active_in_decoupled_spec:
            return

        if self._engine is not None:
            return

        if not self._should_control_server():
            return

        init_start = time.perf_counter()
        print(
            "[decoupled_spec][server_adapter] init_server_adapter_start "
            f"role={self.decoupled_spec_role} replica_rank={self.replica_rank} "
            f"node_rank={self.node_rank} local_rank={self.local_rank} server_actor_name={self.server_actor_name}"
        )
        self.server_actor = ray.get_actor(self.server_actor_name)
        server_address, server_port = await self.server_actor.get_server_address.remote()
        logger.debug(
            f"replica_rank={self.replica_rank} node_rank={self.node_rank}, "
            f"server address: {server_address}, port: {server_port}"
        )
        host = f"[{server_address}]" if is_valid_ipv6_address(server_address) else server_address
        model_path = self.model_config.local_path or self.model_config.path
        self._engine = AsyncHttpServerAdapter(
            model_path=model_path,
            host=host,
            port=server_port,
            launch_server=False,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        print(
            "[decoupled_spec][server_adapter] init_server_adapter_done "
            f"role={self.decoupled_spec_role} replica_rank={self.replica_rank} "
            f"node_rank={self.node_rank} host={host} port={server_port} "
            f"elapsed_s={time.perf_counter() - init_start:.6f}"
        )

    async def resume(self, tags: list[str]):
        if not self.active_in_decoupled_spec:
            return
        resume_start = time.perf_counter()
        await self._init_server_adapter()
        if self._should_control_server() and self.config.free_cache_engine:
            print(
                "[decoupled_spec][server_adapter] resume_start "
                f"role={self.decoupled_spec_role} replica_rank={self.replica_rank} "
                f"node_rank={self.node_rank} tags={tags}"
            )
            await self._engine.resume_memory_occupation(tags=tags)
            print(
                "[decoupled_spec][server_adapter] resume_done "
                f"role={self.decoupled_spec_role} replica_rank={self.replica_rank} "
                f"node_rank={self.node_rank} tags={tags} "
                f"elapsed_s={time.perf_counter() - resume_start:.6f}"
            )

    async def release(self):
        if not self.active_in_decoupled_spec:
            return
        release_start = time.perf_counter()
        await self._init_server_adapter()
        if self._should_control_server() and self.config.free_cache_engine:
            print(
                "[decoupled_spec][server_adapter] release_start "
                f"role={self.decoupled_spec_role} replica_rank={self.replica_rank} "
                f"node_rank={self.node_rank} tags=['kv_cache', 'weights']"
            )
            await self._engine.release_memory_occupation(tags=["kv_cache", "weights"])
            print(
                "[decoupled_spec][server_adapter] release_done "
                f"role={self.decoupled_spec_role} replica_rank={self.replica_rank} "
                f"node_rank={self.node_rank} elapsed_s={time.perf_counter() - release_start:.6f}"
            )

    async def update_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None], global_steps: int = None, **kwargs
    ):
        from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights

        update_weights_bucket_bytes = int(self.config.checkpoint_engine.update_weights_bucket_megabytes) << 20
        if self.config.get("quantization", None) == "fp8":
            from verl.utils.sglang.sglang_fp8_utils import SGLangFP8QuantizerHelper

            print("[decoupled_spec][server_adapter] convert bf16 weights to fp8 format before loading")

            logger.info("Convert bf16 weights to fp8 format before loading")
            quant_start = time.perf_counter()
            fp8_quantizer_helper = SGLangFP8QuantizerHelper(self.model_config.hf_config.quantization_config)
            weights = fp8_quantizer_helper.quant_weights_by_name(
                weights,
                dtype=self.model_config.hf_config.dtype,
            )
            print(
                "[decoupled_spec][server_adapter] update_weights_quantized "
                f"role={self.decoupled_spec_role} replica_rank={self.replica_rank} "
                f"elapsed_s={time.perf_counter() - quant_start:.6f}"
            )

        if not self.active_in_decoupled_spec:
            await self._drain_weights_for_collective_alignment(
                weights,
                role_label="inactive",
                global_steps=global_steps,
            )
            return

        if self.decoupled_spec_role == DecoupledSpecRole.DRAFT:
            await self._drain_weights_for_collective_alignment(
                weights,
                role_label=DecoupledSpecRole.DRAFT.value,
                global_steps=global_steps,
            )
            return

        total_start = time.perf_counter()
        print(
            "[decoupled_spec][server_adapter] update_weights-start_ensure_device_mesh"
            f"role={self.decoupled_spec_role} replica_rank={self.replica_rank} "
        )
        self._ensure_device_mesh()
        print(
            "[decoupled_spec][server_adapter] update_weights-ensure_device_mesh_done"
            f"role={self.decoupled_spec_role} replica_rank={self.replica_rank} "
        )
        await self._init_server_adapter()
        print(
            "[decoupled_spec][server_adapter] update_weights_start "
            f"role={self.decoupled_spec_role} replica_rank={self.replica_rank} "
            f"node_rank={self.node_rank} local_rank={self.local_rank} global_steps={global_steps}"
        )


        print(
            "[decoupled_spec][server_adapter] start updating weights by buckets..."
            f"role={self.decoupled_spec_role} replica_rank={self.replica_rank} "
        )

        batch_idx = 0
        async for params_batch in get_named_tensor_buckets(weights, update_weights_bucket_bytes):
            batch_idx += 1
            batch_start = time.perf_counter()
            param_count = len(params_batch) if hasattr(params_batch, "__len__") else -1
            print(
                "[decoupled_spec][server_adapter] update_weights_bucket_start "
                f"role={self.decoupled_spec_role} replica_rank={self.replica_rank} "
                f"batch_idx={batch_idx} param_count={param_count}"
            )
            await sgl_update_weights(
                engine=self._engine,
                params_batch=params_batch,
                device_mesh_key="infer_tp",
                device_mesh=self.device_mesh,
            )
            print(
                "[decoupled_spec][server_adapter] update_weights_bucket_done "
                f"role={self.decoupled_spec_role} replica_rank={self.replica_rank} "
                f"batch_idx={batch_idx} param_count={param_count} "
                f"elapsed_s={time.perf_counter() - batch_start:.6f}"
            )

        if self._should_control_server():
            flush_start = time.perf_counter()
            await self._engine.flush_cache()
            print(
                "[decoupled_spec][server_adapter] update_weights_flush_cache_done "
                f"role={self.decoupled_spec_role} replica_rank={self.replica_rank} "
                f"elapsed_s={time.perf_counter() - flush_start:.6f}"
            )
            if global_steps is not None:
                global_steps_start = time.perf_counter()
                await self.server_actor.set_global_steps.remote(global_steps)
                print(
                    "[decoupled_spec][server_adapter] update_weights_set_global_steps_done "
                    f"role={self.decoupled_spec_role} replica_rank={self.replica_rank} "
                    f"global_steps={global_steps} elapsed_s={time.perf_counter() - global_steps_start:.6f}"
                )
        print(
            "[decoupled_spec][server_adapter] update_weights_done "
            f"role={self.decoupled_spec_role} replica_rank={self.replica_rank} "
            f"batches={batch_idx} total_elapsed_s={time.perf_counter() - total_start:.6f}"
        )
