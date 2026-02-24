# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import asyncio
import logging
import os
import threading

import torch
from omegaconf import DictConfig
from ray.util.collective import collective

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_torch_device, is_npu_available
from verl.utils.distributed import stateless_init_process_group

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class BaseDetachNcclSync:
    _bucket_size_mb = 1024.0
    _sync_history = []
    _max_history_size = 20
    _last_avg_bucket_size = 1024.0

    def __init__(self, config: DictConfig, role: str):
        self._bg_loop = asyncio.new_event_loop()
        self._bg_thread = threading.Thread(
            target=self._start_background_loop, args=(self._bg_loop,), name="rollout_actor_async_worker", daemon=True
        )
        self._bg_thread.start()
        logger.info(f"[DetachNcclSync] Background thread for SGLang sync started. PID: {os.getpid()}")

    @classmethod
    def get_bucket_size_mb(cls):
        return cls._bucket_size_mb

    @classmethod
    def get_last_avg_bucket_size(cls):
        return cls._last_avg_bucket_size

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def get_last_avg_bucket_size_remote(self):
        return BaseDetachNcclSync._last_avg_bucket_size

    @classmethod
    def record_sync_metrics(cls, bucket_size_mb, sync_time):
        """Dynamically adjust the bucket size based on past synchronization times."""
        bucket_size_mb_value = bucket_size_mb[0] if isinstance(bucket_size_mb, list) else bucket_size_mb
        print(f"[DetachNcclSync] sync_metrics: bucket_size_mb={bucket_size_mb_value:.2f}MB, sync_time={sync_time:.2f}s")
        cls._sync_history.append((bucket_size_mb_value, sync_time))
        if len(cls._sync_history) > cls._max_history_size:
            cls._sync_history.pop(0)

        MIN_BUCKET_SIZE_MB = 512
        MAX_BUCKET_SIZE_MB = 8192  # 8GB

        if len(cls._sync_history) < 4:
            cls._bucket_size_mb = min(MAX_BUCKET_SIZE_MB, cls._bucket_size_mb * 1.5)
        else:
            times = [t for _, t in cls._sync_history]
            buckets = [b for b, _ in cls._sync_history]
            recent_avg_time = sum(times[-2:]) / 2
            previous_avg_time = sum(times[-4:-2]) / 2
            recent_avg_bucket = sum(buckets[-2:]) / 2
            previous_avg_bucket = sum(buckets[-4:-2]) / 2

            performance_improved = recent_avg_time < previous_avg_time
            bucket_increased = recent_avg_bucket > previous_avg_bucket
            time_change_ratio = (
                abs(recent_avg_time - previous_avg_time) / previous_avg_time if previous_avg_time > 0 else 0.0
            )

            if time_change_ratio > 0.2:
                increase_step, decrease_step = 1.2, 0.8
            elif time_change_ratio > 0.1:
                increase_step, decrease_step = 1.1, 0.9
            elif time_change_ratio > 0.05:
                increase_step, decrease_step = 1.05, 0.95
            else:
                increase_step, decrease_step = 1.02, 0.98

            should_increase = (performance_improved and bucket_increased) or (
                not performance_improved and not bucket_increased
            )
            step = increase_step if should_increase else decrease_step
            new_size = cls._bucket_size_mb * step
            cls._bucket_size_mb = min(MAX_BUCKET_SIZE_MB, max(MIN_BUCKET_SIZE_MB, new_size))

    def _start_background_loop(self, loop):
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        except Exception as e:
            logger.error(f"[DetachNcclSync] Background loop crashed: {e}")

    def _run_async_safely(self, coro):
        if not self._bg_thread.is_alive():
            raise RuntimeError("Background thread for SGLang sync is not running!")

        future = asyncio.run_coroutine_threadsafe(coro, self._bg_loop)
        return future.result()

    def __del__(self):
        if hasattr(self, "_bg_loop") and self._bg_loop.is_running():
            self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
        if hasattr(self, "_bg_thread") and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=1.0)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def init_checkpoint_engine(self, rank_offset: int, actor_num: int, rollout_num: int):
        from .checkpoint_engine import CheckpointEngine

        current_rank = torch.distributed.get_rank() + rank_offset
        actor_ranks = list(range(actor_num))
        rollout_ranks = [rank + actor_num for rank in range(rollout_num)]
        assert rank_offset == 0 or rank_offset == actor_num

        self.checkpoint_engine = CheckpointEngine(
            current_rank, actor_ranks, rollout_ranks, self.config.checkpoint_engine.device_buffer_size_M
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def create_weight_sync_group(self, master_address, master_port, rank_offset, world_size):
        rank = torch.distributed.get_rank() + rank_offset
        self._weight_sync_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            get_torch_device().current_device(),
        )

    @staticmethod
    def get_inference_model(rollout):
        """
        Get models according to different types of inference_engine
        Args:
            rollout: rollout object
        Returns:
            model: model object (for vllm) or rollout object itself (for sglang)
        """
        inference_engine = rollout.inference_engine
        if hasattr(inference_engine, "llm_engine"):
            inference_model = inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        elif hasattr(inference_engine, "worker"):
            inference_model = inference_engine.worker.model_runner.model
        else:
            raise AttributeError(
                f"Unsupported inference_engine type: {type(inference_engine)}. "
                f"Expected LLM (with llm_engine attribute) or WorkerWrapperBase (with worker attribute)."
            )
        return inference_model

    def _sync_sglang_weights(self, inference_model, params, sync_group_name):
        bucket_size_bytes = int(self.get_bucket_size_mb() * 1024 * 1024)
        actual_bucket_sizes = []
        current_batch = []
        current_batch_size = 0

        def flush_batch():
            if current_batch:
                actual_bucket_sizes.append(current_batch_size / (1024 * 1024))
                self._run_async_safely(self.update_weights(inference_model, iter(current_batch)))
                get_torch_device().synchronize()
                current_batch.clear()

        for key, shape, dtype in self._weights_info:
            tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
            if self._is_actor:
                assert key in params
                origin_data = params[key]
                if hasattr(origin_data, "full_tensor"):
                    origin_data = origin_data.full_tensor()
                if torch.distributed.get_rank() == 0:
                    tensor.copy_(origin_data)
            collective.broadcast(tensor, src_rank=0, group_name=sync_group_name)

            tensor_size = tensor.numel() * tensor.element_size()
            current_batch.append((key, tensor))
            current_batch_size += tensor_size

            if current_batch_size >= bucket_size_bytes:
                flush_batch()
                current_batch_size = 0

        flush_batch()
        cls = type(self)
        cls._last_avg_bucket_size = (
            sum(actual_bucket_sizes) / len(actual_bucket_sizes) if actual_bucket_sizes else self.get_bucket_size_mb()
        )

        # Resume kv_cache after weights sync to restore GPU memory released during pause
        if self._is_rollout and self.rollout_device_mesh["infer_tp"].get_local_rank() == 0:
            self._run_async_safely(inference_model.resume_memory_occupation(tags=["kv_cache"]))

    def _sync_vllm_weights(self, inference_model, params, sync_group_name):
        for key, shape, dtype in self._weights_info:
            tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
            if self._is_actor:
                assert key in params
                origin_data = params[key]
                if hasattr(origin_data, "full_tensor"):
                    origin_data = origin_data.full_tensor()
                if torch.distributed.get_rank() == 0:
                    tensor.copy_(origin_data)
            if is_npu_available:
                self._weight_sync_group.broadcast(tensor, src=0, stream=get_torch_device().current_stream())
            else:
                collective.broadcast(tensor, src_rank=0, group_name=sync_group_name)
            if self._is_rollout:
                inference_model.load_weights([(key, tensor)])

    async def update_weights(self, inference_engine, params):
        from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights

        await sgl_update_weights(
            engine=inference_engine,
            params_batch=params,
            device_mesh_key="infer_tp",
            device_mesh=self.rollout_device_mesh,
        )

        if self.rollout_device_mesh["infer_tp"].get_local_rank() == 0:
            await inference_engine.flush_cache()
