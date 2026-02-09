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


import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from veomni.distributed import parallel_state
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models.auto import build_foundation_model
from veomni.optim import build_lr_scheduler, build_optimizer

import verl.utils.torch_functional as verl_F
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import fsdp_version
from verl.utils.model import convert_weight_keys
from verl.utils.profiler import log_gpu_memory_usage
from verl.workers.config import HFModelConfig, VeOmniEngineConfig, VeOmniOptimizerConfig
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from ..base import BaseEngineCtx, EngineRegistry
from ..fsdp.transformer_impl import FSDPEngine, FSDPEngineWithLMHead
from ..utils import enable_full_determinism, postprocess_batch_func, prepare_micro_batches
from .utils import (
    MOE_PARAM_HANDERS,
    VL_TYPE2INDEX,
    load_veomni_model_to_gpu,
    load_veomni_optimizer,
    offload_veomni_model_to_cpu,
    offload_veomni_optimizer,
)

logger = logging.getLogger(__file__)


class VeOmniEngine(FSDPEngine):
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: VeOmniEngineConfig,
        optimizer_config: VeOmniOptimizerConfig,
        checkpoint_config: CheckpointConfig,
        **kwargs,
    ):
        """
        Initialize the VeOmniEngine.

        Sets up distributed device meshes, LoRA, and offload policies based on config.

        Args:
            config: Configuration object with VeOmni and model settings.
        """

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        assert self.engine_config.data_parallel_mode == "fsdp2", "VeOmniEngine only supports fsdp2."

        self.rank = dist.get_rank()

        parallel_state.init_parallel_state(
            dp_size=self.engine_config.data_parallel_size,
            dp_replicate_size=self.engine_config.data_parallel_replicate_size,
            dp_shard_size=self.engine_config.data_parallel_shard_size,
            tp_size=self.engine_config.tensor_parallel_size,
            ep_size=self.engine_config.expert_parallel_size,
            pp_size=self.engine_config.pipeline_parallel_size,
            cp_size=self.engine_config.context_parallel_size,
            ulysses_size=self.engine_config.ulysses_parallel_size,
            dp_mode=self.engine_config.data_parallel_mode,
        )

        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)

        self.use_remove_padding = self.model_config.use_remove_padding

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self._is_lora = self.model_config.lora_rank > 0

        self.use_ulysses_sp = parallel_state.get_parallel_state().sp_enabled
        self.ulysses_sequence_parallel_size = self.engine_config.ulysses_parallel_size

        if self.use_ulysses_sp:
            self.ulysses_sharding_manager = FSDPUlyssesShardingManager(parallel_state.get_parallel_state().device_mesh)
        else:
            self.ulysses_sharding_manager = FSDPUlyssesShardingManager(None)

        if self.engine_config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.engine_config.use_torch_compile  #  use torch compile by default
            else entropy_from_logits
        )

    def initialize(self):
        """
        Build the model, optimizer, and learning rate scheduler under VeOmni.

        Applies device, dtype, and precision configurations, including mixed precision.
        Sets up checkpoint manager and FLOPs counter.
        """
        self._build_model_optimizer()

        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.module,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.model_config.get_processor(),
            checkpoint_config=self.checkpoint_config,
            trust_remote_code=self.model_config.trust_remote_code,
        )

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_optimizer,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def _build_optimizer(self, module):
        optimizer = build_optimizer(
            module,
            lr=self.optimizer_config.lr,
            betas=self.optimizer_config.betas,
            weight_decay=self.optimizer_config.weight_decay,
            optimizer_type=self.optimizer_config.optimizer,
        )
        get_optimizer_pre_hook = getattr(module, "get_optimizer_pre_hook", None)
        if get_optimizer_pre_hook is not None:
            optimizer_pre_hook = get_optimizer_pre_hook(module, module.config, self.engine_config.data_parallel_mode)
            optimizer.register_step_pre_hook(optimizer_pre_hook)

        return optimizer

    def _build_lr_scheduler(self, optimizer):
        optim_config = self.optimizer_config
        lr_scheduler = build_lr_scheduler(
            optimizer,
            train_steps=optim_config.total_training_steps,
            lr=optim_config.lr,
            lr_min=optim_config.lr_min,
            lr_decay_style=optim_config.lr_scheduler_type,
            lr_decay_ratio=optim_config.lr_decay_ratio,
            lr_warmup_ratio=optim_config.lr_warmup_steps_ratio,
            lr_start=optim_config.lr_start,
        )

        return lr_scheduler

    def _build_model_optimizer(self):
        # Load base model with specified configuration and dtype
        module = build_foundation_model(
            config_path=self.model_config.hf_config_path,
            weights_path=self.model_config.path,
            torch_dtype="float32" if self.engine_config.mixed_precision else "bfloat16",
            attn_implementation=self.engine_config.attn_implementation,
            moe_implementation=self.engine_config.moe_implementation,
            init_device=self.engine_config.init_device,
        )
        log_gpu_memory_usage("After load base model", logger=logger)

        # Applies parallel strategies to the model.
        log_gpu_memory_usage("Before parallelize model", logger=logger)
        module = build_parallelize_model(
            module,
            init_device=self.engine_config.init_device,
            weights_path=self.model_config.path,
            enable_full_shard=self.engine_config.enable_full_shard,
            enable_mixed_precision=self.engine_config.mixed_precision,
            enable_gradient_checkpointing=self.model_config.enable_gradient_checkpointing,
            enable_fsdp_offload=self.engine_config.enable_fsdp_offload,
            basic_modules=module._no_split_modules + self.engine_config.basic_modules,
            enable_reentrant=self.engine_config.enable_reentrant,
            enable_forward_prefetch=self.engine_config.forward_prefetch,
        )
        log_gpu_memory_usage("After parallelize model", logger=logger)

        if not self.engine_config.forward_only:
            # Initialize optimizer with model parameters and config settings
            optimizer = self._build_optimizer(module)
            # Create learning rate scheduler with warmup and decay settings
            lr_scheduler = self._build_lr_scheduler(optimizer)
        else:
            optimizer = None
            lr_scheduler = None

        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model_fwd_context, self.model_bwd_context = build_activation_offloading_context(
            self.model_config.enable_activation_offload,
            self.model_config.enable_gradient_checkpointing,
            self.engine_config.activation_gpu_limit,
        )

    def optimizer_step(self):
        """
        Perform an optimization step using the optimizer.
        """
        if hasattr(self.module, "clip_grad_norm_"):
            grad_norm = self.module.clip_grad_norm_(self.optimizer_config.clip_grad)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.optimizer_config.clip_grad)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        return grad_norm.item()

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
        """
        Perform a forward pass and optionally a backward pass on a batch of data.

        Args:
            data: The input data for the forward pass, typically containing tensors and metadata.
            loss_function: The loss function to optimize. See `verl.workers.roles.utils.losses` for examples.
            forward_only: If True, perform only the forward pass. If False, perform forward and backward pass.

        Returns:
            Any: The output of the forward pass, which can be used for loss computation or other purposes.
        """
        tu.assign_non_tensor(data, sp_size=parallel_state.get_parallel_state().ulysses_size)

        # compute num_tokens in global batch for loss normalization
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True
        )

        output_lst = []

        for micro_batch in micro_batches:
            with self.model_fwd_context:
                loss, meta_info = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)
            if not forward_only:
                with self.model_bwd_context:
                    loss.backward()

            output_lst.append(meta_info)

        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def get_data_parallel_rank(self):
        return parallel_state.get_parallel_state().device_mesh.get_local_rank("dp")

    def get_data_parallel_size(self):
        return torch.distributed.get_world_size() // parallel_state.get_parallel_state().ulysses_size

    def get_data_parallel_group(self):
        if parallel_state.get_parallel_state().ulysses_size > 1:
            return parallel_state.get_parallel_state().device_mesh.get_group(mesh_dim="dp")
        else:
            return torch.distributed.group.WORLD

    def is_mp_src_rank_with_outputs(self):
        """
        Whether the current rank is the first rank in model parallel group that contains model outputs
        """
        if parallel_state.get_parallel_state().ulysses_size > 1:
            is_collect = parallel_state.get_parallel_state().device_mesh["ulysses"].get_local_rank() == 0
        else:
            is_collect = True
        return is_collect

    def train_mode(self, **kwargs):
        """
        Return a context manager that switches to training mode with VeOmni-specific handling.

        Includes parameter and optimizer offload entry/exit.
        """
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """
        Return a context manager that switches to evaluation mode with VeOmni-specific handling.

        Includes activation offload entry/exit.
        """
        return EngineEvalModeCtx(self, **kwargs)

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.
        Note that this function executes irrespective of offload config. It serves as manual control.

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        super(FSDPEngine, self).to(device=device, model=model, optimizer=optimizer, grad=grad)

        device_name = get_device_name()

        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                load_veomni_model_to_gpu(self.module)
            if optimizer and self.optimizer is not None:
                load_veomni_optimizer(self.optimizer, device)
        elif device == "cpu":
            if model:
                offload_veomni_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_veomni_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Save VeOmni checkpoint, handling parameter offload as needed.
        """
        origin_module_device = next(self.module.parameters()).device.type
        if self._is_offload_param or origin_module_device == "cpu":
            load_veomni_model_to_gpu(self.module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_veomni_model_to_cpu(self.module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
    ) -> None:
        """
        Load VeOmni checkpoint, restoring parameters and optimizer state.
        """
        if self._is_offload_param:
            load_veomni_model_to_gpu(self.module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_veomni_model_to_cpu(self.module)

        if self._is_offload_optimizer:
            offload_veomni_optimizer(self.optimizer)

    def get_per_tensor_param(self, **kwargs):
        load_veomni_model_to_gpu(self.module)

        params = self.module.state_dict()
        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

        if self._is_offload_param:
            offload_veomni_model_to_cpu(self.module)

        device = get_device_id()
        ps = parallel_state.get_parallel_state()
        model_type = getattr(self.module.config, "model_type", "default")
        process_func = MOE_PARAM_HANDERS.get(model_type, lambda n, t: iter([(n, t)]))

        def param_generator():
            for name, param in params.items():
                unsharded_tensor = param.full_tensor() if isinstance(param, DTensor) else param

                is_expert_layer = "mlp.experts." in name
                is_proj = any(p in name for p in ["down_proj", "gate_proj", "up_proj", "gate_up_proj"])

                if is_expert_layer and is_proj and ps.ep_enabled:
                    output_shape = list(unsharded_tensor.shape)
                    output_shape[0] *= ps.ep_size
                    stacked_tensor = torch.empty(output_shape, dtype=unsharded_tensor.dtype, device=device)

                    # all gather expert tensors [32, H, I] -> [128, H, I]
                    torch.distributed.all_gather_into_tensor(stacked_tensor, unsharded_tensor, group=ps.ep_group)
                    yield from process_func(name, stacked_tensor)

                    del stacked_tensor
                else:
                    if is_expert_layer:
                        yield from process_func(name, unsharded_tensor)
                    else:
                        yield name, unsharded_tensor

        # TODO: support VeOmni LoRA
        return param_generator(), None


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: VeOmniEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, VeOmniEngine)
        super().__enter__()
        self.engine.ulysses_sharding_manager.__enter__()
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, VeOmniEngine)
        self.engine.ulysses_sharding_manager.__exit__(exc_type, exc_value, traceback)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if parallel_state.get_parallel_state().dp_shard_size > 1:
            if fsdp_version(self.engine.module) == 1:
                self.engine.module._handle.reshard(True)
            elif fsdp_version(self.engine.module) == 2:
                self.engine.module.reshard()

        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: VeOmniEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, VeOmniEngine)
        super().__enter__()
        self.engine.ulysses_sharding_manager.__enter__()
        # TODO: Switch to eval mode after Integrating the CI environment
        # VeOmni (ref: https://github.com/ByteDance-Seed/VeOmni/pull/421)
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, VeOmniEngine)
        self.engine.ulysses_sharding_manager.__exit__(exc_type, exc_value, traceback)
        self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@dataclass
class OmniSequenceShardCollator:
    """
    Data collator to chunk inputs along the sequence length.
    """

    # features to slice sequence dimension
    sp_slice_features: dict[str, int] = field(
        default_factory=lambda: {
            "input_ids": -1,
            "labels": -1,
            "pixel_values": 0,
            "pixel_values_videos": 0,
        },
        metadata={"help": "features to slice sequence dimension."},
    )

    def __post_init__(self):
        self.sp_size = parallel_state.get_parallel_state().sp_size
        self.sp_rank = parallel_state.get_parallel_state().sp_rank

    def sp_slice(self, feature: torch.Tensor, dim: int = -1) -> dict[str, "torch.Tensor"]:
        seq_length = feature.size(dim)
        sp_chunk_size = (seq_length + self.sp_size - 1) // self.sp_size
        return feature.narrow(dim, self.sp_rank * sp_chunk_size, sp_chunk_size)

    def __call__(self, batch: Sequence[dict[str, "torch.Tensor"]]) -> dict[str, "torch.Tensor"]:
        # sp slice
        for key in batch.keys():
            if key in self.sp_slice_features.keys():
                batch[key] = self.sp_slice(batch[key], dim=self.sp_slice_features[key])

        return batch


@EngineRegistry.register(model_type="language_model", backend=["veomni"], device=["cuda", "npu"])
class VeOmniEngineWithLMHead(VeOmniEngine, FSDPEngineWithLMHead):
    def prepare_model_inputs(self, micro_batch: TensorDict):
        # TODO: Cannot work properly for qwen_vl ulysses
        model_inputs, output_args = super().prepare_model_inputs(micro_batch)
        input_ids_rmpad = model_inputs["input_ids"]
        if self.module.config.model_type in VL_TYPE2INDEX.keys():
            image_mask = input_ids_rmpad == VL_TYPE2INDEX[self.module.config.model_type]["IMAGE_INPUT_INDEX"]
            video_mask = input_ids_rmpad == VL_TYPE2INDEX[self.module.config.model_type]["VIDEO_INPUT_INDEX"]
            model_inputs.update({"image_mask": image_mask, "video_mask": video_mask})

            if parallel_state.get_parallel_state().sp_enabled:
                omni_sequence_shard_collator = OmniSequenceShardCollator()
                omni_sequence_shard_collator(model_inputs)

        return model_inputs, output_args
