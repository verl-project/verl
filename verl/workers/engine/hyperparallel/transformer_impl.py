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

import functools
import logging
import os
from packaging import version
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Generator, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from tensordict import TensorDict
from hyper_parallel.core.dtensor.dtensor import DTensor

from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy, MixedPrecisionPolicy

import verl.utils.torch_functional as verl_F
from verl.utils import tensordict_utils as tu
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.device import get_device_id, get_device_name
from verl.utils.model import convert_weight_keys, extract_multi_modal_inputs
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.profiler.performance import log_gpu_memory_usage
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
from verl.workers.engine.base import BaseEngineCtx, EngineRegistry
from verl.workers.engine.utils import postprocess_batch_func, prepare_micro_batches
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.workers.engine.hyperparallel.utils import (
    apply_hp_fsdp,
    fsdp2_load_full_state_dict,
    load_hp_model_to_gpu,
    offload_hp_model_to_cpu,
    load_hp_optimizer,
    offload_hp_optimizer,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class HyperParallelEngine(FSDPEngine):
    """HyperParallel training engine.

    通过BaseEngine接口与verl上层RL控制流解耦。
    支持FSDP/HSDP、TP、PP、CP、EP等混合并行策略。

    注意：此引擎使用HyperParallel原生的DeviceMesh（而非PyTorch DeviceMesh），
    因此不能复用verl的create_device_mesh工具函数。
    """

    def __init__(self, model_config, engine_config, optimizer_config=None, checkpoint_config=None):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        # State flags
        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def initialize(self):
        """
        Build the model, optimizer, and learning rate scheduler under FSDP.

        Applies device, dtype, and precision configurations, including mixed precision.
        Sets up checkpoint manager and FLOPs counter.
        """
        # This is used to import external_lib into the huggingface systems
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
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)


    def _init_device_mesh(self):
        """创建HyperParallel DeviceMesh"""
        from hyper_parallel import init_device_mesh
        world_size = torch.distributed.get_world_size()
        fsdp_size = self.engine_config.fsdp_size
        device_name = get_device_name()
        
        if fsdp_size < 0 or fsdp_size >= world_size:
            self.device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
        else:
            self.device_mesh = init_device_mesh(
                device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
            )
        self.ulysses_device_mesh = None
        self.ulysses_parallel_group = None
        self.ulysses_sequence_parallel_size = self.engine_config.ulysses_sequence_parallel_size
        dp_size = self.get_data_parallel_size()
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp_size, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )
            self.ulysses_parallel_group = self.ulysses_device_mesh["sp"].get_group()

        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1


    def _build_fsdp_module(self, module):
        from verl.utils.torch_dtypes import PrecisionType

        mixed_precision_config = self.engine_config.mixed_precision
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32

        # - actor: offload_policy
        # - critic: offload_policy
        # - ref: CPUOffloadPolicy(pin_memory=True)
        assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
        )
        offload_policy = None
        if self.engine_config.offload_policy or self.engine_config.forward_only:
            self._is_offload_param = False
            self._is_offload_optimizer = False
            offload_policy = CPUOffloadPolicy(pin_memory=True)

        fsdp_kwargs = {
            "mesh": self.device_mesh,
            "mp_policy": mp_policy,
            "offload_policy": offload_policy,
            "reshard_after_forward": self.engine_config.reshard_after_forward,
        }
        # if not self.engine_config.forward_only:
        full_state = module.state_dict()
        apply_hp_fsdp(module, fsdp_kwargs, self.engine_config)
        fsdp2_load_full_state_dict(module, full_state, self.device_mesh, offload_policy)
        # else:
        #     # Forward-only (e.g. ref model): no sharding needed, place model on device directly.
        #     self._is_offload_param = False
        #     self._is_offload_optimizer = False
        #     module = module.to(get_device_id())

        # enable_activation_offload默认是false，不会走到
        # enable_activation_offloading内判断了FSDP，hp需要适配
        # if self.model_config.enable_activation_offload:
        #     enable_gradient_checkpointing = self.model_config.enable_gradient_checkpointing
        #     enable_activation_offloading(module, self.engine_config.strategy, enable_gradient_checkpointing)

        return module

    def _build_model_optimizer(self):
        from verl.utils.model import print_model_size

        # Load base model with specified configuration and dtype
        module = self._build_module()
        # Synchronize all distributed processes before proceeding
        torch.distributed.barrier()
        if self.rank == 0:
            print_model_size(module)
        log_gpu_memory_usage("After init model from HF AutoModel", logger=logger)

        # Wrap model with FSDP for distributed training (shaerding, mixed precision, etc.)
        log_gpu_memory_usage("Before FSDP", logger=None)
        module = self._build_fsdp_module(module)
        log_gpu_memory_usage("After FSDP", logger=None)

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

    def train_mode(self, **kwargs):
        return HyperParallelEngineTrainCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        return HyperParallelEngineEvalCtx(self, **kwargs)

    # ===== BaseEngine Interface: Optimizer =====

    def optimizer_zero_grad(self):
        if self.optimizer is not None:
            # HyperParallel fully_shard 将梯度变为 DTensor，
            # optimizer.zero_grad() 内部操作的 grad 是 DTensor，
            # 需在 SkipDTensorDispatch 中执行以跳过 DTensor dispatch。
            with SkipDTensorDispatch():
                self.optimizer.zero_grad()

    def optimizer_step(self):
        """
        Clip gradients, skip update if non-finite, and step optimizer.

        Returns:
            grad_norm (float): Norm of gradients before clipping.
        """
        from hyper_parallel.core.utils import clip_grad_norm_ as hp_clip_grad_norm_
        grad_norm = hp_clip_grad_norm_(self.module, max_norm=self.optimizer_config.clip_grad, norm_type=2.0)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            with SkipDTensorDispatch():
                self.optimizer.zero_grad()
        else:
            with SkipDTensorDispatch():
                self.optimizer.step()

        return grad_norm.item()
    
    def lr_scheduler_step(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            return self.lr_scheduler.get_last_lr()[0]
        return 0.0

    # ===== BaseEngine Interface: Forward / Backward =====

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False):
        """执行micro-batch循环的前向+反向传播

        遵循FSDPEngine的标准模式：
        1. prepare_micro_batches 分割数据
        2. 逐 micro-batch 调用 forward_step
        3. postprocess_batch_func 聚合结果
        """
        # Assign metadata for micro-batch preparation
        tu.assign_non_tensor(data, sp_size=1)

        # Compute global num_tokens for loss normalization.
        # 注意：loss_mask 是 nested tensor（非 DTensor），但为防止意外，
        # 使用 .item() 转为 float 再构造普通 tensor 进行 all_reduce，
        # 避免 HyperParallel DTensor layout 推断问题。
        local_num_tokens = data["loss_mask"].sum().item()
        batch_num_tokens = torch.tensor(local_num_tokens, dtype=torch.float32, device=get_device_id())
        dp_group = self.get_data_parallel_group()
        if dp_group is not None:
            torch.distributed.all_reduce(
                batch_num_tokens,
                op=torch.distributed.ReduceOp.SUM,
                group=dp_group,
            )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            same_micro_num_in_dp=True,
        )

        output_lst = []
        ctx = torch.no_grad() if forward_only else nullcontext()

        # HyperParallel's internal all_gather buffers carry view state from
        # prior no_grad invocations (e.g. inference).  When we later run a
        # training forward (grad mode), PyTorch's view-safety check rejects
        # the mixed no_grad / grad-mode view usage.  HyperParallel's own
        # tests avoid this by running the entire training loop inside
        # SkipDTensorDispatch, so we do the same here.
        for micro_batch in micro_batches:
            with ctx:
                loss, meta_info = self.forward_step(
                    micro_batch, loss_function=loss_function, forward_only=forward_only,
                )
                if not forward_only:
                    loss.backward()
            output_lst.append(meta_info)

        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.
        Note that this function executes irrespective of offload config. It serves as manual control.

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        # if self.engine_config.forward_only:
        #     # force cpu_offload
        #     return
        device_name = get_device_name()

        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                load_hp_model_to_gpu(self.module)
            if optimizer and self.optimizer is not None:
                load_hp_optimizer(self.optimizer, device)
        elif device == "cpu":
            if model:
                offload_hp_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_hp_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def get_per_tensor_param(self, **kwargs):
        load_hp_model_to_gpu(self.module)

        device = get_device_id()
        params = self.module.state_dict()
        params = convert_weight_keys(params, self.module)
        per_tensor_param = (
            (
                name,
                param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
                if isinstance(param, DTensor)
                else param,
            )
            for name, param in params.items()
        )

        # Offload model to CPU AFTER materializing all params
        if self._is_offload_param:
            offload_hp_model_to_cpu(self.module)

        return per_tensor_param, None

class HyperParallelEngineTrainCtx(BaseEngineCtx):
    """训练模式上下文管理器"""

    def __init__(self, engine: HyperParallelEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        super().__enter__()
        self.engine.module.train()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_val, exc_tb)


class HyperParallelEngineEvalCtx(BaseEngineCtx):
    """评估模式上下文管理器"""

    def __init__(self, engine: HyperParallelEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        super().__enter__()
        self.engine.module.eval()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # ToDO: test reshard
        if self.engine.engine_config.fsdp_size > 1:
            self.engine.module.reshard()
        super().__exit__(exc_type, exc_val, exc_tb)

@EngineRegistry.register(model_type="language_model", backend="hyperparallel", device=["cuda", "npu"])
class HyperParallelEngineWithLMHead(HyperParallelEngine):
    """HyperParallel engine with LM head support.

    单继承 HyperParallelEngine（不继承 FSDPEngineWithLMHead），
    避免 MRO 冲突。prepare_model_inputs/outputs 由本类自行实现。
    """
    def prepare_model_inputs(self, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        temperature = micro_batch["temperature"]
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        multi_modal_inputs = extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", []))
        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]

        if not isinstance(temperature, torch.Tensor):
            temperature = torch.tensor([temperature] * input_ids.shape[0], device=input_ids.device)

        temperature = temperature.to(torch.float32)
        assert temperature.shape[0] == input_ids.shape[0]

        # args used to get outputs
        output_args = {}

        if use_remove_padding:
            # support per sample temperature
            # temperature (bsz,)
            # input_ids (bsz, j1)
            temperature_rmpad = verl_F.expand_as_nested(temperature, input_ids).values()  # (total_nnz,)
            temperature_rmpad = temperature_rmpad.unsqueeze(0)  # (1, total_nnz)

            input_ids_rmpad = input_ids.values().unsqueeze(0)  # (1, total_nnz)
            if position_ids.dim() == 3:
                position_ids_rmpad = position_ids.values().unsqueeze(1)  # (4, 1, total_nnz)
            else:
                position_ids_rmpad = position_ids.values().unsqueeze(0)  # (1, total_nnz)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
            temperature_rmpad = temperature_rmpad.squeeze(0)
            output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled
            output_args["temperature_rmpad"] = temperature_rmpad

            # only pass input_ids and position_ids to enable flash_attn_varlen
            # _rng_state = torch.random.get_rng_state()
            # torch.manual_seed(42)
            # input_ids_rmpad = torch.randint(0, 151936, input_ids_rmpad.shape, device=input_ids_rmpad.device)
            # torch.random.set_rng_state(_rng_state)
            model_inputs = {
                "input_ids": input_ids_rmpad,
                "attention_mask": None,        #ToDo: support attention_mask for remove_padding in the future
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
            # we store the per sample temperature
            output_args["temperature"] = temperature

            input_ids = torch.nested.to_padded_tensor(
                input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len)
            )

            if position_ids.dim() == 3:
                position_ids = torch.nested.to_padded_tensor(
                    position_ids, padding=0, output_size=(batch_size, 4, max_seq_len)
                ).transpose(0, 1)  # (4, batch_size, max_seq_len)
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

        model_inputs.update(multi_modal_inputs)

        return model_inputs, output_args

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict, logits_processor_func):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)
        distillation_use_topk = tu.get_non_tensor_data(data=micro_batch, key="distillation_use_topk", default=False)
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"
        model_output = {}

        input_ids = micro_batch["input_ids"]

        if use_remove_padding:
            input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
            temperature_rmpad = output_args["temperature_rmpad"]

            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature_rmpad.clamp(min=1e-8).unsqueeze(-1).to(logits_rmpad.dtype))

            # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
            inplace_backward = True
            if calculate_entropy:
                inplace_backward = False
            log_probs = logprobs_from_logits(
                logits=logits_rmpad,
                labels=input_ids_rmpad_rolled,
                inplace_backward=inplace_backward,
            )
            # compute entropy
            if calculate_entropy:
                if not self.engine_config.entropy_checkpointing:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                else:
                    entropy_rmpad = torch.utils.checkpoint.checkpoint(
                        self.compute_entropy_from_logits, logits_rmpad
                    )

            # logits_processor_func return tensors with shape (1, total_nnz/sp_size)
            if distillation_use_topk:
                outputs = logits_processor_func(student_logits=logits_rmpad.unsqueeze(0), data=micro_batch)
                cu_seqlens = input_ids.offsets()
                for k, v in outputs.items():
                    v = v.squeeze(0)
                    assert v.shape == log_probs.shape, f"log_probs shape: {log_probs.shape}, {k} shape: {v.shape}"
                    if self.use_ulysses_sp:
                        pad_size = output_args["pad_size"]
                        v = gather_outputs_and_unpad(v, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    model_output[k] = torch.nested.nested_tensor_from_jagged(v, cu_seqlens)

            cu_seqlens = input_ids.offsets()
            # (bsz, j1), for each sample, is the length of each sample: [real_prompt length + real_response length]
            log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
            if calculate_entropy:
                entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
        else:  # not using rmpad and no ulysses sp
            logits = output.logits  # (bsz, response_length, vocab_size)
            temperature = output_args["temperature"]  # (bsz,)
            temperature = temperature.unsqueeze(-1).unsqueeze(-1)
            logits.div_(temperature.clamp(min=1e-8).to(logits.dtype))

            if calculate_entropy:
                if not self.engine_config.entropy_checkpointing:
                    entropy = verl_F.entropy_from_logits(logits)
                else:
                    entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            cu_seqlens = input_ids.offsets()
            seq_lengths = cu_seqlens.diff()
            starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
            logits = torch.nested.narrow(logits, 1, starts, seq_lengths, layout=torch.jagged)
            logits_rmpad = torch.cat([t for t in logits.unbind()])
            input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
            log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)
            # (bsz, j1), for each sample, length of each sample: [real_prompt_length + real_response_length]
            log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
            if calculate_entropy:
                entropy = torch.nested.narrow(entropy, 1, starts, seq_lengths, layout=torch.jagged)
                entropy_rmpad = torch.cat([t for t in entropy.unbind()])
                entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)

        model_output["log_probs"] = log_probs
        if calculate_entropy:
            model_output["entropy"] = entropy

        return model_output

    
    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        device_name = get_device_name()
        micro_batch = micro_batch.to(get_device_id())

        model_inputs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)
        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            raw_output = self.module(**model_inputs, use_cache=False)
            model_output = self.prepare_model_outputs(
                output=raw_output, output_args=output_args, micro_batch=micro_batch, logits_processor_func=loss_function
            )

            if loss_function is not None:
                loss, metrics = loss_function(
                    model_output=model_output, data=micro_batch, dp_group=self.get_data_parallel_group(),
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
