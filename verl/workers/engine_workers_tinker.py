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
from codetiming import Timer
from tensordict import TensorDict

from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.utils.profiler import DistProfiler
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker


class TinkerTrainingWorker(TrainingWorker):
    """
    Training worker exposing Tinker-style split training primitives.

    Unlike TrainingWorker.train_batch(), these APIs let a caller explicitly separate gradient
    clearing, forward/backward, and optimizer stepping.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def optimizer_zero_grad(self) -> None:
        with self.engine.train_mode(zero_grad_on_exit=False):
            self.engine.optimizer_zero_grad()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    @DistProfiler.annotate(color="red", role="forward_backward")
    def forward_backward(self, data: TensorDict) -> TensorDict:
        assert self.loss_fn is not None, "loss function can't be None when calling forward_backward"
        assert not self.engine_config.forward_only, (
            "Can't run `forward_backward` when forward_only is in the engine config."
        )
        global_token_num = tu.get(data, key="global_token_num")
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        maybe_fix_3d_position_ids(data)

        with (
            self.engine.train_mode(
                disable_auto_offload=disable_auto_offload,
                zero_grad_on_exit=False,
            ),
            Timer(name="forward_backward", logger=None) as timer,
        ):
            output = self.engine.forward_backward_batch(data, loss_function=self.loss_fn, forward_only=False)
        delta_time = timer.last

        if self.engine.is_mp_src_rank_with_outputs():
            output.pop("model_output")
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=False,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def optimizer_step(self, update_lr_scheduler: bool = True) -> dict:
        with self.engine.train_mode(zero_grad_on_exit=True):
            grad_norm = self.engine.optimizer_step()
            lr = self.engine.lr_scheduler_step() if update_lr_scheduler else None

        metrics = {}
        if grad_norm is not None and self.engine.is_mp_src_rank_with_outputs():
            metrics["grad_norm"] = grad_norm
        if lr is not None and self.engine.is_mp_src_rank_with_outputs():
            metrics["lr"] = lr
        return metrics


class TinkerActorRolloutRefWorker(ActorRolloutRefWorker):
    """Actor-rollout-ref worker exposing Tinker-style split training primitives for the actor."""

    actor_worker_cls = TinkerTrainingWorker

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def optimizer_zero_grad(self) -> None:
        assert "actor" in self.role, "optimizer_zero_grad only support actor role"
        return self.actor.optimizer_zero_grad()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def forward_backward(self, data: TensorDict) -> TensorDict:
        assert "actor" in self.role, "forward_backward only support actor role"
        output = self.actor.forward_backward(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def optimizer_step(self, update_lr_scheduler: bool = True) -> dict:
        assert "actor" in self.role, "optimizer_step only support actor role"
        return self.actor.optimizer_step(update_lr_scheduler=update_lr_scheduler)
