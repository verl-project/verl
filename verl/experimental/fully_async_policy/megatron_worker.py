# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
# Copyright 2025 NVIDIA Ltd. and/or its affiliates
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

import logging
import os
import time

import torch
import torch.distributed
from omegaconf import DictConfig

from verl.experimental.fully_async_policy.base_detach_sync import BaseDetachNcclSync
from verl.experimental.fully_async_policy.megatron_utils import (
    copy_megatron_model_to_cpu,
    restore_megatron_model_from_cpu,
)
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import (
    get_device_name,
    get_torch_device,
)
from verl.utils.megatron_utils import load_megatron_model_to_gpu, offload_megatron_model_to_cpu, per_tensor_generator
from verl.workers.megatron_workers import AsyncActorRolloutRefWorker, CriticWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

__all__ = ["DetachActorWorker", "DetachAsyncRolloutWorker", "CriticWorker"]


class DetachNcclSync(BaseDetachNcclSync, AsyncActorRolloutRefWorker):
    def __init__(self, config: DictConfig, role: str):
        BaseDetachNcclSync.__init__(self, config, role)

        AsyncActorRolloutRefWorker.__init__(self, config, role)

    def _get_actor_params(self):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self, sync_group_name="actor_rollout"):
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None
        if self._is_actor and self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module, False)
        params_generator = self._get_actor_params_generator() if self._is_actor else None
        params = {key: tensor for key, tensor in params_generator} if params_generator is not None else None

        rollout_name = self.config.rollout.name
        inference_model = None
        if self._is_rollout and (not self._is_actor):
            if rollout_name == "vllm":
                inference_model = BaseDetachNcclSync.get_inference_model(self.rollout)
                from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

                patch_vllm_moe_model_weight_loader(inference_model)
            elif rollout_name == "sglang":
                inference_model = self.rollout._engine
                if inference_model is None:
                    print("[sync_rollout_weights] Initialize server adapter engine")

                    async def init_engine():
                        if hasattr(self.rollout, "_init_server_adapter"):
                            await self.rollout._init_server_adapter()
                        else:
                            print("[sync_rollout_weights] No _init_server_adapter method found")
                        return self.rollout._engine

                    inference_model = self._run_async_safely(init_engine())
                    # For ServerAdapter, only TP rank 0 initializes the engine
                    # TP rank != 0 can safely have inference_model as None
                    from verl.workers.rollout.sglang_rollout.sglang_rollout import ServerAdapter

                    is_server_adapter = isinstance(self.rollout, ServerAdapter)
                    is_non_tp_rank = False
                    if (
                        is_server_adapter
                        and hasattr(self.rollout, "device_mesh")
                        and self.rollout.device_mesh is not None
                    ):
                        try:
                            is_non_tp_rank = self.rollout.device_mesh["infer_tp"].get_local_rank() != 0
                        except Exception:
                            pass

                    if inference_model is None and not (is_server_adapter and is_non_tp_rank):
                        raise RuntimeError(
                            f"Failed to initialize rollout engine. "
                            f"rollout type: {type(self.rollout)}, "
                            f"has _init_server_adapter: {hasattr(self.rollout, '_init_server_adapter')}"
                        )
            else:
                raise NotImplementedError(f"Unknown rollout name: {rollout_name}")

        if rollout_name == "sglang" and self._is_rollout:
            self._sync_sglang_weights(inference_model, params, sync_group_name)
        else:
            self._sync_vllm_weights(inference_model, params, sync_group_name)

        if self._is_actor and self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            get_torch_device().empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_model_to_cpu(self, n):
        if not hasattr(self, "cpu_saved_models"):
            self.cpu_saved_models = {}
        self.cpu_saved_models[n] = copy_megatron_model_to_cpu(self.actor.actor_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_model_from_cpu(self, n):
        if n in self.cpu_saved_models:
            restore_megatron_model_from_cpu(self.actor.actor_module, self.cpu_saved_models[n])

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def clear_cpu_model(self, n):
        if n in self.cpu_saved_models:
            del self.cpu_saved_models[n]

    def cache_actor_weights_to_cpu(self):
        self.cpu_named_params = {}
        if self._is_actor:
            params_generator = self._get_actor_params_generator()
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            print(f"cache_actor_weights_to_cpu, local_rank:{local_rank}, world_size:{world_size}")
            for tensor_idx, (key, tensor) in enumerate(params_generator):
                if tensor_idx % world_size == local_rank:
                    self.cpu_named_params[key] = tensor.to("cpu", non_blocking=True)
            get_torch_device().synchronize()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights_by_checkpoint(self, sync_group_name="actor_rollout"):
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None

        # Load model to GPU
        load_start_time = time.time()
        if self._is_actor and self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module, False)
        load_duration = time.time() - load_start_time

        from ray.util.collective import collective

        # Cache actor weights to CPU and measure the time taken
        cache_start_time = time.time()
        self.cache_actor_weights_to_cpu()
        cache_end_time = time.time()
        cache_duration = cache_end_time - cache_start_time

        # Register the cached weights into the checkpoint engine
        self.checkpoint_engine.register_checkpoint(self._weights_info, self.cpu_named_params)
        register_end_time = time.time()
        register_duration = register_end_time - cache_end_time
        self.cpu_named_params = {}

        collective.barrier(group_name=sync_group_name)
        update_start_time = time.time()

        rollout_name = self.config.rollout.name
        inference_model = None
        if self._is_rollout and (not self._is_actor):
            if rollout_name == "vllm":
                inference_model = BaseDetachNcclSync.get_inference_model(self.rollout)
                from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

                patch_vllm_moe_model_weight_loader(inference_model)
            elif rollout_name == "sglang":
                inference_model = self.rollout._engine
                # For ServerAdapter, _engine might be None and needs async initialization
                if inference_model is None:
                    # Initialize the server adapter engine
                    print("[sync_rollout_weights] Initialize server adapter engine")

                    async def init_engine():
                        if hasattr(self.rollout, "_init_server_adapter"):
                            await self.rollout._init_server_adapter()
                        else:
                            print("[sync_rollout_weights] No _init_server_adapter method found")
                        return self.rollout._engine

                    inference_model = self._run_async_safely(init_engine())
                    # For ServerAdapter, only TP rank 0 initializes the engine
                    # TP rank != 0 can safely have inference_model as None
                    from verl.workers.rollout.sglang_rollout.sglang_rollout import ServerAdapter

                    is_server_adapter = isinstance(self.rollout, ServerAdapter)
                    is_non_tp_rank = False
                    if (
                        is_server_adapter
                        and hasattr(self.rollout, "device_mesh")
                        and self.rollout.device_mesh is not None
                    ):
                        try:
                            is_non_tp_rank = self.rollout.device_mesh["infer_tp"].get_local_rank() != 0
                        except Exception:
                            pass

                    if inference_model is None and not (is_server_adapter and is_non_tp_rank):
                        raise RuntimeError(
                            f"Failed to initialize rollout engine. "
                            f"rollout type: {type(self.rollout)}, "
                            f"has _init_server_adapter: {hasattr(self.rollout, '_init_server_adapter')}"
                        )
            else:
                raise NotImplementedError(f"Unknown rollout name: {rollout_name}")
        # Update the checkpoint with the inference model and broadcast weights
        self.checkpoint_engine.update_checkpoint(
            inference_model=inference_model,
            group_name=sync_group_name,
            overlap_broadcast_and_consume=self.config.checkpoint_engine.overlap_broadcast_and_consume,
        )

        update_end_time = time.time()
        update_duration = update_end_time - update_start_time

        collective.barrier(group_name=sync_group_name)
        offload_start_time = time.time()
        if self._is_actor and self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        offload_duration = time.time() - offload_start_time

        print(
            f"sync_rollout_weights_by_checkpoint finish!, rank:{torch.distributed.get_rank()},"
            f" is_actor:{self._is_actor}, is_rollout:{self._is_rollout},"
            f" total cost:{update_end_time - cache_start_time} seconds, while cache cost {cache_duration} seconds, "
            f" register cost {register_duration} seconds, update cost {update_duration} seconds"
        )

        if self._is_actor and self._is_offload_param:
            print(
                f"sync_rollout_weights_by_checkpoint load model to gpu cost {load_duration} seconds,"
                f" offload model to cpu cost {offload_duration} seconds"
            )


class DetachActorWorker(DetachNcclSync):
    def __init__(self, config: DictConfig, role: str):
        print("[DetachAsyncRolloutWorker] Initializing via DetachNcclSync...")
        DetachNcclSync.__init__(self, config, role)

    def _get_actor_params_generator(self):
        assert self._is_actor
        if self.bridge is not None:
            if self.vanilla_bridge:
                generator = self.bridge.export_weights(self.actor.actor_module)
            else:
                generator = self.bridge.export_hf_weights(self.actor.actor_module)
        else:
            generator = per_tensor_generator(
                self.actor.actor_module,
                self.actor_model_config,
                self.weight_converter,
                self.tf_config,
                self.layer_name_mapping,
            )

        return generator

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        assert self._is_actor
        if hasattr(self, "_weights_info"):
            return self._weights_info
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module, False)
        params_generator = self._get_actor_params_generator()
        ret = []
        for key, tensor in params_generator:
            ret.append((key, tensor.size(), tensor.dtype))

        self._weights_info = ret
        # Here, we only call this function at the beginning,
        # and immediately afterwards we call sync_rollout_weights.
        # So we no longer call offload in this.
        return ret


class DetachAsyncRolloutWorker(DetachNcclSync):
    def __init__(self, config: DictConfig, role: str):
        print(f"[DetachAsyncRolloutWorker] {DetachAsyncRolloutWorker.__mro__}")
        DetachNcclSync.__init__(self, config, role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info):
        assert self._is_rollout
        self._weights_info = weights_info
