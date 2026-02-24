# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Automodel (nemo_automodel) engine for verl SFT training.

This engine delegates model building, parallelization, optimizer sharding,
LR scheduling, gradient clipping, and checkpointing to Automodel's
infrastructure while using verl's training loop, data pipeline, and loss function.
"""

import gc
import logging
import os
from contextlib import nullcontext
from typing import Any, Callable, Optional

import torch
import torch.distributed
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.components.training.utils import (
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from tensordict import TensorDict
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.model import convert_weight_keys, extract_multi_modal_inputs
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.config import AutomodelEngineConfig, AutomodelOptimizerConfig, HFModelConfig

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import enable_full_determinism, postprocess_batch_func, prepare_micro_batches
from .utils import (
    build_automodel_model,
    build_model_wrapper_from_engine_config,
    get_dp_group_size,
    get_dp_rank,
    get_pp_rank,
    get_tp_rank,
    load_automodel_model_to_gpu,
    load_automodel_optimizer,
    maybe_fully_shard_optimizer,
    offload_automodel_model_to_cpu,
    offload_automodel_optimizer,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AutomodelEngine(BaseEngine):
    """Engine implementation using Automodel for distributed training."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: AutomodelEngineConfig,
        optimizer_config: AutomodelOptimizerConfig,
        checkpoint_config: CheckpointConfig,
        **kwargs,
    ):
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        self.mode = None
        self.rank = torch.distributed.get_rank()

        world_size = torch.distributed.get_world_size()
        self.model_wrapper = build_model_wrapper_from_engine_config(self.engine_config, world_size)
        self.device_mesh = getattr(self.model_wrapper, "device_mesh", None)
        self.moe_mesh = getattr(self.model_wrapper, "moe_mesh", None)

        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload

        if self.engine_config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.engine_config.use_torch_compile
            else entropy_from_logits
        )

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def initialize(self):
        """Build the model, optimizer, LR scheduler, and checkpointer using Automodel infrastructure."""
        self.module = build_automodel_model(self.model_config, self.engine_config, self.model_wrapper)
        log_gpu_memory_usage("After Automodel model build", logger=logger)

        if not self.engine_config.forward_only:
            self.optimizer = self._build_optimizer(self.module)
            # maybe shard optimizer for MegatronFSDP
            maybe_fully_shard_optimizer(self.module, self.optimizer, self.model_wrapper)
            self.lr_scheduler = self._build_lr_scheduler(self.optimizer)
        else:
            self.optimizer = None
            self.lr_scheduler = None
        self._build_checkpointer()

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def _build_optimizer(self, module):
        from verl.workers.config.optimizer import build_optimizer

        trainable_params = [p for p in module.parameters() if p.requires_grad]
        return build_optimizer(trainable_params, self.optimizer_config)

    def _build_lr_scheduler(self, optimizer):
        """Build Automodel's OptimizerParamScheduler (matching train_ft.py build_lr_scheduler)."""
        cfg = self.optimizer_config
        total_steps = cfg.total_training_steps
        num_warmup_steps = cfg.lr_warmup_steps

        if num_warmup_steps <= 0:
            num_warmup_steps = int(cfg.lr_warmup_steps_ratio * total_steps)

        base_lr = cfg.lr
        init_lr_ratio = cfg.init_lr_ratio if cfg.init_lr_ratio is not None else 0.1
        min_lr_ratio = cfg.min_lr_ratio if cfg.min_lr_ratio is not None else 0.01

        if self.rank == 0:
            print(
                f"Automodel LR Scheduler: total_steps={total_steps}, warmup={num_warmup_steps}, "
                f"decay_style={cfg.lr_scheduler_type}, init_lr={base_lr * init_lr_ratio:.2e}, "
                f"max_lr={base_lr:.2e}, min_lr={base_lr * min_lr_ratio:.2e}"
            )

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            init_lr=base_lr * init_lr_ratio,
            max_lr=base_lr,
            min_lr=base_lr * min_lr_ratio,
            lr_warmup_steps=num_warmup_steps,
            lr_decay_steps=total_steps,
            lr_decay_style=cfg.lr_scheduler_type,
            start_wd=cfg.weight_decay,
            end_wd=cfg.weight_decay,
            wd_incr_steps=total_steps,
            wd_incr_style=getattr(cfg, "wd_incr_style", "constant"),
        )
        return scheduler

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
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
        ctx = torch.no_grad() if forward_only else nullcontext()

        if not forward_only:
            prepare_for_grad_accumulation([self.module])

        num_micro_batches = len(micro_batches)
        for i, micro_batch in enumerate(micro_batches):
            # Signal final backward for MoE support (matching train_ft.py)
            if not forward_only and i == num_micro_batches - 1:
                prepare_for_final_backward([self.module])

            with ctx:
                loss, meta_info = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)
                if not forward_only:
                    loss.backward()
            output_lst.append(meta_info)

        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def optimizer_step(self):
        grad_norm = scale_grads_and_clip_grad_norm(
            max_grad_norm=self.optimizer_config.clip_grad,
            model_parts=[self.module],
            norm_type=2.0,
            pp_enabled=False,
            device_mesh=self.device_mesh,
            moe_mesh=self.moe_mesh,
            ep_axis_name="ep" if self.moe_mesh is not None and "ep" in self.moe_mesh.mesh_dim_names else None,
            pp_axis_name=None,
            foreach=True,
            num_label_tokens=None,
            dp_group_size=get_dp_group_size(self.device_mesh, include_cp=True),
        )

        if isinstance(grad_norm, torch.Tensor):
            grad_norm_val = grad_norm.item()
        else:
            grad_norm_val = float(grad_norm)

        # If grad_norm is not finite, skip the update
        if not torch.isfinite(torch.tensor(grad_norm_val)):
            print(f"WARN: grad_norm is not finite: {grad_norm_val}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
            if hasattr(self.module, "update_moe_gate_bias"):
                self.module.update_moe_gate_bias()

        return grad_norm_val

    def lr_scheduler_step(self):
        """Step Automodel's OptimizerParamScheduler and return current LR."""
        self.lr_scheduler.step(increment=1)
        lr = self.optimizer.param_groups[0]["lr"]
        return lr

    def get_data_parallel_rank(self):
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank("dp")
        return torch.distributed.get_rank()

    def get_data_parallel_size(self):
        if self.device_mesh is not None:
            return self.device_mesh["dp"].size()
        return torch.distributed.get_world_size()

    def get_data_parallel_group(self):
        if self.device_mesh is not None:
            return self.device_mesh.get_group(mesh_dim="dp")
        return torch.distributed.group.WORLD

    def is_mp_src_rank_with_outputs(self):
        if self.device_mesh is not None and "tp" in self.device_mesh.mesh_dim_names:
            if self.device_mesh["tp"].size() > 1:
                return self.device_mesh.get_local_rank("tp") == 0
        return True

    def train_mode(self, **kwargs):
        return AutomodelTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        return AutomodelEvalModeCtx(self, **kwargs)

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        if self.engine_config.forward_only:
            return

        device_name = get_device_name()
        assert device in (device_name, "cpu")

        if device == device_name:
            if model:
                load_automodel_model_to_gpu(self.module)
            if optimizer and self.optimizer is not None:
                load_automodel_optimizer(self.optimizer, get_device_id())
            gc.collect()
        elif device == "cpu":
            if model:
                offload_automodel_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_automodel_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def _build_checkpointer(self):
        ckpt_config = CheckpointingConfig(
            enabled=True,
            checkpoint_dir="checkpoints/",
            model_save_format="safetensors",
            model_cache_dir=os.environ.get("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface/hub")),
            model_repo_id=self.model_config.path,
            save_consolidated=True,
            is_peft=False,
        )
        self.checkpointer = Checkpointer(
            config=ckpt_config,
            dp_rank=get_dp_rank(self.device_mesh, include_cp=True),
            tp_rank=get_tp_rank(self.device_mesh),
            pp_rank=get_pp_rank(self.device_mesh),
            moe_mesh=self.moe_mesh,
        )

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Save model, optimizer, and LR scheduler using Automodel's Checkpointer."""
        origin_module_device = next(self.module.parameters()).device.type
        if self._is_offload_param or origin_module_device == "cpu":
            load_automodel_model_to_gpu(self.module)

        # Save model weights
        self.checkpointer.save_model(self.module, local_path)

        # Save optimizer and LR scheduler state
        if self.optimizer is not None:
            scheduler_list = [self.lr_scheduler] if self.lr_scheduler is not None else None
            self.checkpointer.save_optimizer(self.optimizer, self.module, local_path, scheduler=scheduler_list)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_automodel_model_to_cpu(self.module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
    ) -> None:
        """Load model, optimizer, and LR scheduler using Automodel's Checkpointer."""
        if self._is_offload_param:
            load_automodel_model_to_gpu(self.module)

        model_path = os.path.join(local_path, "model")
        if not os.path.isdir(model_path):
            model_path = local_path
        self.checkpointer.load_model(self.module, model_path)

        if self.optimizer is not None:
            scheduler_list = [self.lr_scheduler] if self.lr_scheduler is not None else None
            self.checkpointer.load_optimizer(self.optimizer, self.module, local_path, scheduler=scheduler_list)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_automodel_model_to_cpu(self.module)

        if self._is_offload_optimizer and self.optimizer is not None:
            offload_automodel_optimizer(self.optimizer)

    def get_per_tensor_param(self, **kwargs):
        load_automodel_model_to_gpu(self.module)

        params = self.module.state_dict()
        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

        if self._is_offload_param:
            offload_automodel_model_to_cpu(self.module)

        def param_generator():
            for name, param in params.items():
                unsharded_tensor = param.full_tensor() if isinstance(param, DTensor) else param
                yield name, unsharded_tensor

        return param_generator(), None


class AutomodelEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: AutomodelEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, AutomodelEngine)
        super().__enter__()
        # Set model to train mode (same as VeOmni pattern for FSDP2 compatibility)
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, AutomodelEngine)
        # Reshard the root FSDP module
        if hasattr(self.engine.module, "reshard"):
            self.engine.module.reshard()
        super().__exit__(exc_type, exc_value, traceback)


class AutomodelTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: AutomodelEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, AutomodelEngine)
        super().__enter__()
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, AutomodelEngine)
        self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@EngineRegistry.register(model_type="language_model", backend=["automodel"], device=["cuda"])
class AutomodelEngineWithLMHead(AutomodelEngine):
    """Automodel engine for language model with LM head training."""

    def prepare_model_inputs(self, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        temperature = micro_batch["temperature"]
        temperature_item = temperature
        if use_fused_kernels:
            assert not isinstance(temperature, torch.Tensor), (
                "use_fused_kernels does not support per sample temperature yet"
            )
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        multi_modal_inputs = extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", []))
        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]

        if not isinstance(temperature, torch.Tensor):
            temperature = torch.tensor([temperature] * input_ids.shape[0], device=input_ids.device)

        temperature = temperature.to(torch.float32)
        assert temperature.shape[0] == input_ids.shape[0]

        output_args = {}

        if use_remove_padding:
            temperature_rmpad = verl_F.expand_as_nested(temperature, input_ids).values()
            temperature_rmpad = temperature_rmpad.unsqueeze(0)

            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids_rmpad = input_ids.values().unsqueeze(0)
                if position_ids.dim() == 3:
                    position_ids_rmpad = position_ids.values().unsqueeze(1)
                else:
                    position_ids_rmpad = position_ids.values().unsqueeze(0)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
            temperature_rmpad = temperature_rmpad.squeeze(0)
            output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled
            output_args["temperature_rmpad"] = temperature_rmpad

            model_inputs = {
                "input_ids": input_ids_rmpad,
                "attention_mask": None,
                "position_ids": position_ids_rmpad,
            }

        else:
            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids = micro_batch["input_ids"]
                position_ids = micro_batch["position_ids"]
                loss_mask = micro_batch["loss_mask"]

                pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
                batch_size = micro_batch.batch_size[0]
                seq_len_effective = input_ids.offsets().diff()
                max_seq_len = max(seq_len_effective)

                input_ids_rmpad_rolled = torch.roll(input_ids.values(), shifts=-1, dims=0)
                output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled
                output_args["temperature"] = temperature

                input_ids = torch.nested.to_padded_tensor(
                    input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len)
                )

                if position_ids.dim() == 3:
                    position_ids = torch.nested.to_padded_tensor(
                        position_ids, padding=0, output_size=(batch_size, 4, max_seq_len)
                    ).transpose(0, 1)
                else:
                    position_ids = torch.nested.to_padded_tensor(
                        position_ids, padding=0, output_size=(batch_size, max_seq_len)
                    )

                attention_mask_list = [torch.ones_like(t, dtype=torch.int32) for t in loss_mask]
                attention_mask = torch.nested.as_nested_tensor(attention_mask_list, layout=torch.jagged)
                attention_mask = torch.nested.to_padded_tensor(
                    attention_mask, padding=0, output_size=(batch_size, max_seq_len)
                )

                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }

            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        extra_args = {}
        if use_fused_kernels:
            extra_args["temperature"] = temperature_item
            extra_args["return_dict"] = True

        model_inputs.update(multi_modal_inputs)
        model_inputs.update(extra_args)

        return model_inputs, output_args

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)

        model_output = {}
        input_ids = micro_batch["input_ids"]

        if use_remove_padding:
            input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
            temperature_rmpad = output_args["temperature_rmpad"]

            if use_fused_kernels:
                log_probs = output.log_probs.squeeze(0)
                entropy_rmpad = output.entropy.squeeze(0)
            else:
                logits_rmpad = output.logits.squeeze(0)
                # With TP, logits are DTensors sharded on vocab dim; gather for log_softmax.
                if isinstance(logits_rmpad, DTensor):
                    logits_rmpad = logits_rmpad.full_tensor()
                logits_rmpad = logits_rmpad / temperature_rmpad.clamp(min=1e-8).unsqueeze(-1).to(logits_rmpad.dtype)

                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=input_ids_rmpad_rolled,
                    inplace_backward=inplace_backward,
                )

                if calculate_entropy:
                    if not self.engine_config.entropy_checkpointing:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                    else:
                        entropy_rmpad = torch.utils.checkpoint.checkpoint(
                            self.compute_entropy_from_logits, logits_rmpad
                        )

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
                if calculate_entropy:
                    entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        else:
            response_length = tu.get_non_tensor_data(data=micro_batch, key="max_response_length", default=1024)
            if use_fused_kernels:
                log_probs = output.log_probs[:, -response_length - 1 : -1]
                entropy = output.entropy[:, -response_length - 1 : -1]
            else:
                logits = output.logits
                # With TP, logits are DTensors sharded on vocab dim; gather for log_softmax.
                if isinstance(logits, DTensor):
                    logits = logits.full_tensor()
                temperature = output_args["temperature"]
                temperature = temperature.unsqueeze(-1).unsqueeze(-1)
                logits = logits / temperature.clamp(min=1e-8).to(logits.dtype)

                if calculate_entropy:
                    if not self.engine_config.entropy_checkpointing:
                        entropy = verl_F.entropy_from_logits(logits)
                    else:
                        entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

                if pad_mode == DatasetPadMode.NO_PADDING:
                    cu_seqlens = input_ids.offsets()
                    seq_lengths = cu_seqlens.diff()
                    starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
                    logits = torch.nested.narrow(logits, 1, starts, seq_lengths, layout=torch.jagged)
                    logits_rmpad = torch.cat([t for t in logits.unbind()])
                    input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
                    log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)
                    log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
                    if calculate_entropy:
                        entropy = torch.nested.narrow(entropy, 1, starts, seq_lengths, layout=torch.jagged)
                        entropy_rmpad = torch.cat([t for t in entropy.unbind()])
                        entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
                else:
                    raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        model_output["log_probs"] = log_probs
        if calculate_entropy:
            model_output["entropy"] = entropy

        return model_output

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        """Run forward pass, compute loss, and return outputs."""
        device_name = get_device_name()
        micro_batch = micro_batch.to(get_device_id())
        model_inputs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            raw_output = self.module(
                **model_inputs,
                use_cache=False,
            )

            model_output = self.prepare_model_outputs(
                output=raw_output, output_args=output_args, micro_batch=micro_batch
            )

            if loss_function is not None:
                loss, metrics = loss_function(
                    model_output=model_output, data=micro_batch, dp_group=self.get_data_parallel_group()
                )
            else:
                assert forward_only, "forward_only must be True when loss_function is None"
                loss = torch.tensor(1.0, device=device_name)
                metrics = {}

            output = {
                "model_output": model_output,
                "loss": loss.detach().item(),
                "metrics": metrics,
            }

            return loss, output
