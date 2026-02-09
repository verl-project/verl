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
DeepSpeed engine implementation aligned with the BaseEngine interface.

Supported features:
    - ZeRO-1/2/3
    - Optional CPU/NVMe offload for ZeRO-3 parameters and ZeRO-2/3 optimizer states
    - Activation checkpointing
    - Activation offload handled by the HF model when configured
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Callable, ContextManager, Optional

import torch
import torch.distributed
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.deepspeed_checkpoint_manager import DeepSpeedCheckpointManager
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.deepspeed_utils import (
    deepspeed_gather_params,
    get_deepspeed_config,
    get_global_grad_norm,
    initialize_deepspeed_engine,
    load_deepspeed_model_to_gpu,
    offload_deepspeed_model_to_cpu,
)
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import collect_lora_params, replace_lora_wrapper
from verl.utils.model import convert_weight_keys, extract_multi_modal_inputs, print_model_size
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import postprocess_batch_func, prepare_micro_batches

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class DeepSpeedEngine(BaseEngine):
    """
    Base DeepSpeed engine implementing the common BaseEngine contract.
    """

    def __init__(
        self,
        model_config,
        engine_config,
        optimizer_config,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__()
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        self.rank = torch.distributed.get_rank()
        self.mode = None
        self.ds_engine = None
        self.module = None
        self.optimizer = None
        self.lr_scheduler = None

        self.use_remove_padding = self.model_config.use_remove_padding
        self._is_lora = self.model_config.lora_rank > 0

        self._is_offload_param = False
        self._is_offload_optimizer = False

        self.compute_entropy_from_logits = verl_F.entropy_from_logits
        self.autocast_dtype = PrecisionType.to_dtype(self.engine_config.dtype)

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def is_mp_src_rank_with_outputs(self):
        return self.get_data_parallel_rank() == 0

    def initialize(self):
        self._build_model_optimizer()
        self.checkpoint_manager = DeepSpeedCheckpointManager(self.ds_engine)

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.ds_engine)

    def train_mode(self, **kwargs):
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        return EngineEvalModeCtx(self, **kwargs)

    def get_data_parallel_rank(self):
        return torch.distributed.get_rank()

    def get_data_parallel_size(self):
        return torch.distributed.get_world_size()

    def get_data_parallel_group(self):
        return torch.distributed.group.WORLD

    def optimizer_zero_grad(self):
        if self.ds_engine is not None:
            try:
                if getattr(self.ds_engine, "optimizer", None) is not None:
                    self.ds_engine.optimizer.zero_grad(set_to_none=True)
                else:
                    self.ds_engine.zero_grad()
            except Exception as exc:  # pragma: no cover - best-effort fallback
                logger.warning("DeepSpeed zero_grad failed: %s", exc)
        elif self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self):
        grad_norm = get_global_grad_norm(self.ds_engine) if self.ds_engine is not None else None
        if self.ds_engine is not None:
            self.ds_engine.step()
        elif self.optimizer is not None:
            self.optimizer.step()
        return grad_norm if grad_norm is not None else 0.0

    def lr_scheduler_step(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            return lr
        return None

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False):
        tu.assign_non_tensor(data, sp_size=1)

        # DeepSpeed forward pre-hooks expect the underlying module to be attached.
        self._ensure_ds_engine_module()

        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True
        )

        if self.ds_engine is not None and hasattr(self.ds_engine, "set_gradient_accumulation_steps"):
            self.ds_engine.set_gradient_accumulation_steps(max(1, len(micro_batches)))

        output_lst = []
        ctx = torch.no_grad() if forward_only else nullcontext()

        for micro_batch in micro_batches:
            with ctx:
                loss, meta_info = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)
                if not forward_only:
                    if self.ds_engine is not None:
                        self.ds_engine.backward(loss)
                    else:
                        loss.backward()
            output_lst.append(meta_info)

        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)
        assert device in (device_name, "cpu")

        if device == device_name:
            if model and self.ds_engine is not None:
                load_deepspeed_model_to_gpu(self.ds_engine)
        else:
            if model and self.ds_engine is not None:
                offload_deepspeed_model_to_cpu(self.ds_engine)

    def save_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, global_step: int = 0, max_ckpt_to_keep: Optional[int] = None
    ) -> None:
        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.ds_engine)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.ds_engine)

    def load_checkpoint(self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: bool = True, **kwargs):
        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.ds_engine)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.ds_engine)

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False):
        load_deepspeed_model_to_gpu(self.ds_engine)
        module = self.ds_engine.module if self.ds_engine is not None else self.module

        peft_config = None
        peft_model = getattr(module, "_fsdp_wrapped_module", module)
        if hasattr(peft_model, "peft_config"):
            peft_config = peft_model.peft_config.get("default", None)
            params = collect_lora_params(
                module=module,
                layered_summon=layered_summon,
                base_sync_done=base_sync_done,
            )
            if not base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        else:
            with deepspeed_gather_params(self.ds_engine):
                params = {k: v.detach().cpu() for k, v in module.state_dict().items()}

        params = convert_weight_keys(params, module)

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.ds_engine)

        device = get_device_id()
        per_tensor_param = ((name, param.to(device, non_blocking=True)) for name, param in params.items())
        return per_tensor_param, peft_config

    def disable_adapter(self) -> ContextManager:
        if hasattr(self.module, "disable_adapter"):
            return self.module.disable_adapter()
        return nullcontext()

    # ----- helpers -----
    def _ensure_ds_engine_module(self):
        """DeepSpeed forward pre-hooks require engine.module to exist; guard against it being None."""
        if self.ds_engine is None:
            return
        try:
            if getattr(self.ds_engine, "module", None) is None:
                self.ds_engine.module = self.module
        except Exception:  # pragma: no cover - defensive
            self.ds_engine.module = self.module

    def _build_model_optimizer(self):
        module = self._build_module()

        if self.model_config.enable_activation_offload:
            enable_activation_offloading(
                module, "deepspeed", enable_gradient_checkpointing=self.model_config.enable_gradient_checkpointing
            )

        if self._is_lora:
            module = self._build_lora_module(module)

        torch.distributed.barrier()

        ds_config = self._build_deepspeed_config()

        if self.rank == 0:
            print_model_size(module)
            logger.info(
                "[DeepSpeed] zero_stage=%s offload_param=%s offload_optimizer=%s dtype=%s world_size=%s",
                self.engine_config.zero_stage,
                self._is_offload_param,
                self._is_offload_optimizer,
                self.engine_config.dtype,
                torch.distributed.get_world_size(),
            )

        ds_engine, optimizer, _, lr_scheduler = initialize_deepspeed_engine(
            model=module,
            config=ds_config,
            model_parameters=module.parameters(),
        )

        # Ensure module attribute is present for older DeepSpeed versions.
        try:
            if getattr(ds_engine, "module", None) is None:
                ds_engine.module = module
        except Exception:
            ds_engine.module = module

        self.ds_engine = ds_engine
        self.module = ds_engine.module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.ds_engine)

    def _build_deepspeed_config(self):
        world_size = torch.distributed.get_world_size()
        micro_bsz = self.engine_config.micro_batch_size_per_gpu or 1
        grad_accum = max(1, self.engine_config.gradient_accumulation_steps)
        train_batch_size = micro_bsz * grad_accum * max(1, world_size)

        # offload
        offload_target = self.engine_config.offload
        offload_device = None
        offload_optimizer_device = None
        offload_param_nvme_path = None
        offload_optimizer_nvme_path = None
        if offload_target in {"cpu", "nvme", "auto"}:
            offload_device = "cpu" if offload_target != "nvme" else "nvme"
            offload_optimizer_device = offload_device if self.engine_config.zero_stage >= 2 else None
            offload_param_nvme_path = self.engine_config.offload_dir if offload_device == "nvme" else None
            offload_optimizer_nvme_path = offload_param_nvme_path

        self._is_offload_param = offload_device is not None and self.engine_config.zero_stage >= 3
        self._is_offload_optimizer = offload_optimizer_device is not None

        # precision
        mixed_precision_cfg = self.engine_config.mixed_precision or {}
        param_dtype = mixed_precision_cfg.get("param_dtype", None)
        reduce_dtype = mixed_precision_cfg.get("reduce_dtype", None)

        if param_dtype is None:
            bf16_enabled = self.engine_config.dtype == "bfloat16"
            fp16_enabled = self.engine_config.dtype == "float16"
        else:
            bf16_enabled = str(param_dtype).lower() in {"bf16", "bfloat16"}
            fp16_enabled = str(param_dtype).lower() in {"fp16", "float16"}

        optim_type = getattr(self.optimizer_config, "optimizer", "AdamW")
        betas = getattr(self.optimizer_config, "betas", (0.9, 0.999))
        clip_grad = getattr(self.optimizer_config, "clip_grad", None)

        ds_config = get_deepspeed_config(
            optimizer_type=optim_type,
            train_batch_size=train_batch_size,
            train_micro_batch_size_per_gpu=micro_bsz,
            gradient_accumulation_steps=grad_accum,
            zero_stage=self.engine_config.zero_stage,
            lr=self.optimizer_config.lr,
            betas=betas,
            eps=getattr(self.optimizer_config, "eps", 1e-8),
            weight_decay=self.optimizer_config.weight_decay,
            bf16_enabled=bf16_enabled,
            fp16_enabled=fp16_enabled,
            offload_param_device=offload_device if self.engine_config.zero_stage >= 3 else None,
            offload_optimizer_device=offload_optimizer_device,
            offload_param_nvme_path=offload_param_nvme_path,
            offload_optimizer_nvme_path=offload_optimizer_nvme_path,
            gradient_clipping=clip_grad,
        )

        # allow advanced overrides passed via engine_config.mixed_precision
        if reduce_dtype is not None:
            ds_config.setdefault("zero_optimization", {})
            ds_config["zero_optimization"]["reduce_dtype"] = reduce_dtype

        return ds_config

    def _build_module(self):
        from verl.utils.model import get_hf_auto_model_class

        torch_dtype = self.engine_config.model_dtype
        if torch_dtype is None:
            torch_dtype = "fp32" if not self.engine_config.forward_only else "bfloat16"
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        auto_class = get_hf_auto_model_class(hf_config=self.model_config.hf_config)
        module = auto_class.from_pretrained(
            pretrained_model_name_or_path=self.model_config.local_path,
            torch_dtype=torch_dtype,
            config=self.model_config.hf_config,
            trust_remote_code=self.model_config.trust_remote_code,
        )

        if self.model_config.use_liger:
            from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

            _apply_liger_kernel_to_instance(model=module)

        fused_kernel_options = self.model_config.fused_kernel_options
        fused_kernels_backend = (
            fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
        )

        apply_monkey_patch(
            model=module,
            use_remove_padding=self.use_remove_padding,
            ulysses_sp_size=1,
            use_fused_kernels=self.model_config.use_fused_kernels,
            fused_kernels_backend=fused_kernels_backend,
        )

        module.to(torch_dtype)

        if self.model_config.enable_gradient_checkpointing or self.engine_config.activation_checkpointing:
            module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        return module

    def _build_lora_module(self, module):
        module.enable_input_require_grads()

        lora_adapter_path = getattr(self.model_config, "lora_adapter_path", None)
        if lora_adapter_path is not None:
            from peft import PeftModel

            from verl.utils.fs import copy_to_local

            print(f"Loading pre-trained LoRA adapter from: {lora_adapter_path}")
            local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.model_config.use_shm)

            module = PeftModel.from_pretrained(module, local_adapter_path, is_trainable=True)
            peft_config = module.peft_config["default"]
            if isinstance(peft_config.task_type, str):
                peft_config.task_type = TaskType.CAUSAL_LM
        else:
            lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": self.model_config.lora_rank,
                "lora_alpha": self.model_config.lora_alpha,
                "target_modules": convert_to_regular_types(self.model_config.target_modules),
                "exclude_modules": convert_to_regular_types(self.model_config.exclude_modules),
                "bias": "none",
            }
            module = get_peft_model(module, LoraConfig(**lora_config))

        return module


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: DeepSpeedEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        super().__enter__()
        self.engine.module.eval()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: DeepSpeedEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        super().__enter__()
        self.engine.module.train()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_val, exc_tb)


class DeepSpeedEngineWithLMHead(DeepSpeedEngine):
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
            temperature_rmpad = verl_F.expand_as_nested(temperature, input_ids).values()  # (total_nnz,)
            temperature_rmpad = temperature_rmpad.unsqueeze(0)  # (1, total_nnz)

            input_ids_rmpad = input_ids.values().unsqueeze(0)  # (1, total_nnz)
            if position_ids.dim() == 3:
                position_ids_rmpad = position_ids.values().unsqueeze(1)  # (4, 1, total_nnz)
            else:
                position_ids_rmpad = position_ids.values().unsqueeze(0)  # (1, total_nnz)

            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

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
                position_ids = torch.nested.to_padded_tensor(position_ids, padding=0, output_size=(batch_size, max_seq_len))

            attention_mask_list = [torch.ones_like(t, dtype=torch.int32) for t in loss_mask]
            attention_mask = torch.nested.as_nested_tensor(attention_mask_list, layout=torch.jagged)
            attention_mask = torch.nested.to_padded_tensor(attention_mask, padding=0, output_size=(batch_size, max_seq_len))

            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }

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
                entropy_rmpad = output.entropy.squeeze(0) if calculate_entropy else None
            else:
                logits_rmpad = output.logits.squeeze(0)
                logits_rmpad.div_(temperature_rmpad.clamp(min=1e-8).unsqueeze(-1).to(logits_rmpad.dtype))

                inplace_backward = not calculate_entropy
                log_probs = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=input_ids_rmpad_rolled,
                    inplace_backward=inplace_backward,
                )

                entropy_rmpad = None
                if calculate_entropy:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
                if calculate_entropy and entropy_rmpad is not None:
                    entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")
        else:
            response_length = tu.get_non_tensor_data(data=micro_batch, key="max_response_length", default=1024)
            if use_fused_kernels:
                log_probs = output.log_probs[:, -response_length - 1 : -1]
                entropy = output.entropy[:, -response_length - 1 : -1] if calculate_entropy else None
            else:
                logits = output.logits
                temperature = output_args["temperature"]
                temperature = temperature.unsqueeze(-1).unsqueeze(-1)
                logits.div_(temperature.clamp(min=1e-8).to(logits.dtype))

                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)

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
        micro_batch = micro_batch.to(get_device_id())
        model_inputs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        self._ensure_ds_engine_module()

        with torch.autocast(device_type=device_name, dtype=self.autocast_dtype):
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


@EngineRegistry.register(model_type="value_model", backend="deepspeed", device="cuda")
class DeepSpeedEngineWithValueHead(DeepSpeedEngineWithLMHead):
    """Value head variant (critic)."""

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)

        input_ids = micro_batch["input_ids"]
        if use_remove_padding:
            if hasattr(self.module, "v_head"):
                values_rmpad = output[2].squeeze(0).unsqueeze(-1)
            else:
                values_rmpad = output.logits
                values_rmpad = values_rmpad.squeeze(0)
                values_rmpad = values_rmpad.squeeze(-1)

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                values = torch.nested.nested_tensor_from_jagged(values_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        else:
            if hasattr(self.module, "v_head"):
                values = output[2]
            else:
                values = output.logits

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                seq_lengths = cu_seqlens.diff()
                starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
                values = torch.nested.narrow(values, 1, starts, seq_lengths, layout=torch.jagged)
                values_rmpad = torch.cat([t for t in values.unbind()])
                values = torch.nested.nested_tensor_from_jagged(values_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        return {"values": values}


@EngineRegistry.register(model_type="language_model", backend="deepspeed", device="cuda")
class DeepSpeedEngineWithLMHeadRegistered(DeepSpeedEngineWithLMHead):
    """Registration wrapper for language model."""

    pass
