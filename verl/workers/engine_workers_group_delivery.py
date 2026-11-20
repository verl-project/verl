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
"""Megatron workers for synchronous RL with group-level delivery."""

import torch
from tensordict import TensorDict

from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_torch_device, is_device_available
from verl.utils.profiler import DistProfiler
from verl.workers.engine_workers import _with_routing_replay_flag
from verl.workers.engine_workers_tinker import TinkerActorRolloutRefWorker, TinkerTrainingWorker
from verl.workers.utils.group_gradient_accumulator import GroupGradientAccumulator, megatron_gradient_buffers


class GroupDeliveryTrainingWorker(TinkerTrainingWorker):
    """Finalize each group's gradients independently, then perform one global update."""

    def begin_group_batch(self, group_ids: list[str]) -> None:
        """Start a fixed-policy accumulation window, checking the actual model config."""
        if getattr(self, "group_accumulator", None) is not None:
            raise RuntimeError("A group batch is already active")
        engine = self.engine
        if self.engine_config.strategy != "megatron":
            raise ValueError("group_delivery currently requires Megatron")
        if self.engine_config.use_megatron_fsdp or self.optimizer_config.use_precision_aware_optimizer:
            raise ValueError("group_delivery requires native Megatron DDP and the conventional optimizer")
        optimizers = getattr(engine.optimizer, "chained_optimizers", [engine.optimizer])
        for optimizer in optimizers:
            config = optimizer.config
            if (
                config.optimizer not in ("adam", "sgd")
                or getattr(config, "use_precision_aware_optimizer", False)
                or getattr(config, "use_layer_wise_distributed_optimizer", False)
            ):
                raise ValueError("group_delivery requires the conventional Megatron Adam/SGD optimizer")
        if (
            self.engine_config.dtype != "bfloat16"
            or getattr(engine.tf_config, "fp16", False)
            or self.engine_config.dynamic_context_parallel
        ):
            raise ValueError("group_delivery currently requires BF16 and static context parallelism")
        if self.engine_config.virtual_pipeline_model_parallel_size not in (None, 1):
            raise ValueError("group_delivery does not yet support virtual pipeline parallelism")
        if self.engine_config.param_offload or self.engine_config.optimizer_offload:
            raise ValueError("group_delivery requires resident model, optimizer and gradient buffers")
        if getattr(engine, "_qat_enabled", False) or any(
            getattr(engine.tf_config, name, None) for name in ("fp8", "fp4")
        ):
            raise ValueError("group_delivery does not yet support quantized training")
        if self.model_config.mtp.enable:
            raise ValueError("group_delivery does not yet support MTP auxiliary losses")
        # These losses/state updates have a different normalizer or batch boundary
        # from the response-token / sequence mean used by the RL objective.
        for name in ("moe_aux_loss_coeff", "moe_z_loss_coeff", "moe_router_enable_expert_bias"):
            value = getattr(engine.tf_config, name, None)
            enabled = any(value) if isinstance(value, list | tuple) else bool(value)
            if enabled:
                raise ValueError(f"group_delivery requires {name}=0/False")
        if getattr(engine.tf_config, "moe_router_load_balancing_type", None) in ("sinkhorn", "quantile_balancing"):
            raise ValueError("group_delivery does not support batch-dependent routing or router state updates")
        # Validate the storage before generation begins, rather than failing mid-batch.
        buffers = megatron_gradient_buffers(engine.module)
        if any(str(grad.dtype) != "torch.float32" for grad in buffers.values()):
            raise ValueError("group_delivery requires FP32 gradient reduction")
        # Allocate the FP32 accumulator once and reuse it for every batch. A
        # per-batch allocation of the full owned gradient shard can fail on the
        # second step, after the allocator is fragmented by the first update.
        # It lives in host memory: at 4B/TP2 the 8 GiB it would occupy on the
        # GPU is exactly what the backward needs for activations, and the sum is
        # only touched between backward calls (one D2H per group, one H2D per
        # step). FP32 precision is unchanged.
        if getattr(self, "_accumulator_sums", None) is None:
            self._accumulator_sums = {name: torch.zeros_like(grad, device="cpu") for name, grad in buffers.items()}
        if getattr(self, "_accumulator_staging", None) is None:
            elements = GroupGradientAccumulator.STAGING_BYTES // buffers[next(iter(buffers))].element_size()
            self._accumulator_staging = torch.empty(elements, dtype=torch.float32, pin_memory=True)
        self.group_accumulator = GroupGradientAccumulator(
            group_ids, sums=self._accumulator_sums, staging=self._accumulator_staging
        )
        self.optimizer_zero_grad()

    def backward_group(self, data: TensorDict) -> TensorDict | None:
        """Run one group without updating weights, retaining only its gradient and metrics."""
        accumulator = self.group_accumulator
        group_id = tu.get_non_tensor_data(data, "group_delivery_id", default=None)
        weight = tu.get_non_tensor_data(data, "group_delivery_weight", default=None)
        accumulator.check_group(group_id, weight)
        tu.assign_non_tensor(data, disable_auto_offload=True, return_model_output=False)
        # Reuse the upstream split primitives, but isolate each finalized group
        # from the next Megatron schedule's reduction/scaling and bucket state.
        self.optimizer_zero_grad()
        # A long packed sequence can require a contiguous attention buffer larger
        # than the allocator's free block while cached-but-unused segments still
        # sit in the pool. Release them before the schedule needs the big buffer.
        if is_device_available() and get_torch_device().mem_get_info()[0] < 9 * 1024**3:
            get_torch_device().empty_cache()
        try:
            output = self.forward_backward(data)
            # The accumulator allocates a full FP32 copy of this rank's gradient
            # shard on the first group of each batch (~7.5 GiB for 4B/TP2). The
            # schedule's freed activation blocks sit in the allocator cache, so
            # release them immediately before that single large allocation.
            if is_device_available() and get_torch_device().mem_get_info()[0] < 9 * 1024**3:
                get_torch_device().empty_cache()
            accumulator.add(group_id, megatron_gradient_buffers(self.engine.module), weight)
        finally:
            self.optimizer_zero_grad()
        return output

    def _postprocess_output(self, output, **kwargs):
        # Reference/old-policy inference still needs its token-level outputs.
        if not kwargs["forward_only"]:
            output.pop("model_output", None)
        return super()._postprocess_output(output, **kwargs)

    def finish_group_batch(self) -> dict:
        """Clip and update once, only when all expected groups have contributed."""
        accumulator = self.group_accumulator
        with self.engine.train_mode(disable_auto_offload=True, zero_grad_on_exit=False):
            accumulator.restore(megatron_gradient_buffers(self.engine.module))
            metrics = self.optimizer_step()
            metrics["lr"] = self.engine.lr_scheduler_step()
        self.group_accumulator = None
        # The weight sync that follows this update converts TP shards to HF
        # tensors and needs a few hundred MB of scratch space on the training
        # GPUs, which at 4B are otherwise fully committed. The step's activation
        # blocks are dead now; release them before the sync asks for room.
        if is_device_available():
            get_torch_device().empty_cache()
        return metrics

    def abort_group_batch(self) -> None:
        """Discard a failed batch without changing parameters or the scheduler."""
        self.group_accumulator = None
        self.optimizer_zero_grad()


class GroupDeliveryActorWorker(TinkerActorRolloutRefWorker):
    """Ray actor exposing a fixed-batch begin/backward/finish protocol."""

    actor_worker_cls = GroupDeliveryTrainingWorker

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def begin_group_batch(self, group_ids: list[str]) -> None:
        self.actor.begin_group_batch(group_ids)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    @_with_routing_replay_flag(enabled=True)
    def backward_group(self, data: TensorDict) -> TensorDict | None:
        return self.actor.backward_group(data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def finish_group_batch(self) -> dict:
        return self.actor.finish_group_batch()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def abort_group_batch(self) -> None:
        self.actor.abort_group_batch()
