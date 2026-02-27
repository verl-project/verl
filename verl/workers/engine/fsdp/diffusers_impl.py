# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
The concrete Engine implementation using PyTorch FullyShardedDataParallel (FSDP)
"""

import gc
import json
import logging
import os
import warnings
from contextlib import contextmanager, nullcontext
from typing import Callable, Optional

import torch
import torch.distributed
from peft import LoraConfig
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.tensor import DTensor

from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    apply_fsdp2,
    collect_lora_params,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    merged_lora_context,
    normalize_peft_param_name,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from verl.utils.model import convert_weight_keys
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.ulysses import get_ulysses_sequence_parallel_group, set_ulysses_sequence_parallel_group
from verl.workers.config import DiffusersModelConfig, FSDPEngineConfig, FSDPOptimizerConfig

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import enable_full_determinism, prepare_micro_batches
from .utils import create_device_mesh, get_sharding_strategy

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


@EngineRegistry.register(model_type="diffusion_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class DiffusersFSDPEngine(BaseEngine):
    """
    Concrete Diffusers Engine implementation using PyTorch FullyShardedDataParallel (FSDP).

    Supports model sharding, activation/optimizer offloading, LoRA, and sequence parallelism.
    """

    def __init__(
        self,
        model_config: DiffusersModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        """
        Initialize the DiffusersFSDPEngine.

        Sets up distributed device meshes, LoRA, and offload policies based on config.

        Args:
            config: Configuration object with FSDP and model settings.
        """
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        self.mode = None

        self.rank = torch.distributed.get_rank()

        # Apply NPU patches for FSDP backend
        from .utils import apply_npu_fsdp_patches

        apply_npu_fsdp_patches()

        # build device mesh for Ulysses Sequence Parallel

        self._init_device_mesh()

        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)

        # set FSDP offload params
        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self._is_lora = self.model_config.lora_rank > 0
        self._guidance_scale = self.model_config.guidance_scale

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def is_mp_src_rank_with_outputs(self):
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
        else:
            is_collect = True
        return is_collect

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
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.engine_config.fsdp_size

        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)
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

    def _build_module(self):
        from diffusers import AutoModel

        from verl.utils.torch_dtypes import PrecisionType

        # for checkpoint saving
        def save_config(self, save_directory: str | os.PathLike):
            output_config_file = os.path.join(save_directory, "config.json")
            with open(output_config_file, "w", encoding="utf-8") as f:
                json.dump(self, f, indent=4, sort_keys=True)

        torch_dtype = self.engine_config.model_dtype

        if torch_dtype is None:
            # if it is training, we force torch_dtype to fp32
            torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16

        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        init_context = get_init_weight_context_manager(use_meta_tensor=True, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            module = AutoModel.from_pretrained(
                self.model_config.local_path,
                torch_dtype=torch_dtype,
                trust_remote_code=self.model_config.trust_remote_code,
                subfolder="transformer",
            )

            use_liger = self.model_config.use_liger
            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                raise NotImplementedError("Liger kernel is not supported yet.")

            use_fused_kernels = self.model_config.use_fused_kernels
            if use_fused_kernels:
                module.fuse_qkv_projections()

            # some parameters may not in torch_dtype
            module.to(torch_dtype)

            if self.model_config.enable_gradient_checkpointing:
                module.enable_gradient_checkpointing()

            # for checkpoint saving
            module.can_generate = lambda: False
            module.config.save_pretrained = save_config.__get__(module.config)

        return module

    def _build_lora_module(self, module):
        lora_adapter_path = getattr(self.model_config, "lora_adapter_path", None)
        if lora_adapter_path is not None:
            from verl.utils.fs import copy_to_local

            print(f"Loading pre-trained LoRA adapter to from: {lora_adapter_path}")
            # Copy adapter to local if needed
            local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.model_config.use_shm)

            module.load_lora_adapter(local_adapter_path)
        else:
            # Convert config to regular Python types before creating PEFT model
            lora_config = {
                "r": self.model_config.lora_rank,
                "lora_alpha": self.model_config.lora_alpha,
                "init_lora_weights": "gaussian",
                "target_modules": convert_to_regular_types(self.model_config.target_modules),
                "target_parameters": convert_to_regular_types(self.model_config.target_parameters),
                "exclude_modules": convert_to_regular_types(self.model_config.exclude_modules),
                "bias": "none",
            }
            module.add_adapter(LoraConfig(**lora_config))

        return module

    def _build_fsdp_module(self, module):
        # TODO(ziheng): need to improve
        from torch.distributed.fsdp import CPUOffload, MixedPrecision

        from verl.utils.torch_dtypes import PrecisionType

        mixed_precision_config = self.engine_config.mixed_precision
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=module,
            config=self.engine_config.wrap_policy,
            is_lora=self.model_config.lora_rank > 0,
        )

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Note: We force turn off CPUOffload because it causes incorrect results when using grad accumulation
        if self.engine_config.strategy == "fsdp":
            # cpu_offload:
            # - actor: None
            # - critic: None
            # - ref: CPUOffload(offload_params=True)

            # We force reference policy to use CPUOffload to save memory.
            # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
            cpu_offload = None
            if self.engine_config.forward_only:
                cpu_offload = CPUOffload(offload_params=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False

            module = FSDP(
                module,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=self.engine_config.forward_prefetch,
                use_orig_params=self.engine_config.use_orig_params,
                cpu_offload=cpu_offload,
            )
        elif self.engine_config.strategy == "fsdp2":
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
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": self.engine_config.reshard_after_forward,
            }
            full_state = module.state_dict()
            apply_fsdp2(module, fsdp_kwargs, self.engine_config)
            fsdp2_load_full_state_dict(module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {self.engine_config.strategy}")

        if self.model_config.enable_activation_offload:
            enable_gradient_checkpointing = self.model_config.enable_gradient_checkpointing
            enable_activation_offloading(module, self.engine_config.strategy, enable_gradient_checkpointing)

        if torch.distributed.get_world_size() == 1 and fsdp_version(module) == 1:
            FSDP.set_state_dict_type(
                module,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(module) == 1:
            FSDP.set_state_dict_type(
                module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        return module

    def _build_scheduler(self):
        # TODO (mike): generalize to other diffusers scheduler later
        from verl.utils.diffusers.schedulers import FlowMatchSDEDiscreteScheduler
        from verl.utils.diffusers.utils import set_timesteps

        scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path=self.model_config.local_path, subfolder="scheduler"
        )
        set_timesteps(scheduler, self.model_config)
        return scheduler

    def _build_optimizer(self, module):
        from verl.workers.config.optimizer import build_optimizer

        optimizer = build_optimizer(module.parameters(), self.optimizer_config)

        return optimizer

    def _build_lr_scheduler(self, optimizer):
        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        optim_config = self.optimizer_config

        total_steps = optim_config.total_training_steps
        num_warmup_steps = optim_config.lr_warmup_steps
        lr_scheduler_type = optim_config.lr_scheduler_type
        min_lr_ratio = optim_config.min_lr_ratio
        num_cycles = optim_config.num_cycles
        if num_warmup_steps <= 0:
            num_warmup_steps_ratio = optim_config.lr_warmup_steps_ratio
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        if lr_scheduler_type == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
        elif lr_scheduler_type == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
            )
        else:
            raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")
        return lr_scheduler

    def _build_model_optimizer(self):
        from verl.utils.model import print_model_size

        # Load base model with specified configuration and dtype
        module = self._build_module()
        scheduler = self._build_scheduler()
        # Apply LoRA adapters if low-rank adaptation is enabled
        if self._is_lora:
            module = self._build_lora_module(module)

        # Synchronize all distributed processes before proceeding
        torch.distributed.barrier()
        if self.rank == 0:
            print_model_size(module)
        log_gpu_memory_usage("After init model from Diffusers AutoModel", logger=logger)

        # Wrap model with FSDP for distributed training (sharding, mixed precision, etc.)
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
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def train_mode(self, **kwargs):
        """
        Return a context manager that switches to training mode with FSDP-specific handling.

        Includes parameter and optimizer offload entry/exit.
        """
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """
        Return a context manager that switches to evaluation mode with FSDP-specific handling.

        Includes activation offload entry/exit.
        """
        return EngineEvalModeCtx(self, **kwargs)

    def get_data_parallel_rank(self):
        if self.ulysses_device_mesh is not None:
            return self.ulysses_device_mesh["dp"].get_local_rank()
        else:
            return torch.distributed.get_rank()

    def get_data_parallel_size(self):
        return torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size

    def get_data_parallel_group(self):
        if self.ulysses_device_mesh is not None:
            return self.ulysses_device_mesh.get_group(mesh_dim="dp")
        else:
            return torch.distributed.group.WORLD

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> list[TensorDict]:
        # note that the global_batch_size should include data on all the dp
        tu.assign_non_tensor(data, sp_size=self.ulysses_sequence_parallel_size)

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

        ctx = torch.no_grad() if forward_only else nullcontext()

        for micro_batch in micro_batches:
            meta_info_lst = {
                "model_output": [],
                "loss": [],
                "metrics": [],
            }
            for step in range(micro_batch["all_timesteps"].shape[1]):
                with ctx:
                    loss, meta_info = self.forward_step(
                        micro_batch, loss_function=loss_function, forward_only=forward_only, step=step
                    )

                    if not forward_only:
                        loss.backward()
                for key, val in meta_info.items():
                    meta_info_lst[key].append(val)

            output_lst.append(meta_info_lst)

        # postprocess and return
        return self.postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def postprocess_batch_func(self, output_lst, indices, data: TensorDict):
        """postprocess the output of a forward_backward_batch.
        output_lst is a list of dict containing outputs for each micro-batch
        reorder entropy and outputs. Return None for other pp ranks
        only on last rank. It should be on every tp rank

        each losses_reduced contains 1. model_output, 2. loss, 3. metrics.
        """

        from verl.utils.py_functional import append_to_dict
        # from verl.utils.seqlen_balancing import restore_dynamic_batch

        # use_dynamic_bsz = tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=True)

        # losses_reduced is a list of dict containing outputs for each micro-batch
        # reorder entropy and outputs. Return None for other pp ranks
        # only on last rank. It should be on every tp rank

        # losses_reduced contains 1. model_output, 2. loss, 3. metrics.
        # We perform reverse

        model_output = {}
        losses = []
        aggregated_metrics = {}

        for o in output_lst:
            # model output
            model_output_lst = {}
            if "model_output" in o:
                for model_output_dict in o["model_output"]:
                    for key, val in model_output_dict.items():
                        if key not in model_output_lst:
                            model_output_lst[key] = []
                        model_output_lst[key].append(val)
                for key, val in model_output_lst.items():
                    if key not in model_output:
                        model_output[key] = []
                    model_output[key].append(torch.stack(val, dim=1))  # (bsz, steps, ...)
            # loss
            if "loss" in o:
                losses.append(o["loss"])

            # metrics
            if "metrics" in o:  # TODO: (susan) not sure
                for metrics in o["metrics"]:
                    append_to_dict(aggregated_metrics, metrics)

        # concat results from micro batches

        for key, val in model_output.items():
            model_output[key] = torch.concat(val, dim=0)  # (global_bsz, steps, ...)
            # reverse with dynamic bsz
            # if use_dynamic_bsz:
            #     model_output[key] = restore_dynamic_batch(model_output[key], indices)

        output = {
            "model_output": model_output,  # a dict of tensors in shape (global_bsz, steps, ...)
            "loss": losses,  # micro-batch step-wise losses
            "metrics": aggregated_metrics,
        }

        return output

    def prepare_model_inputs(self, micro_batch: TensorDict, step: int):
        latents = micro_batch["all_latents"]
        timesteps = micro_batch["all_timesteps"]
        prompt_embeds = micro_batch["prompt_embeds"]
        prompt_embeds_mask = micro_batch["prompt_embeds_mask"]
        negative_prompt_embeds = micro_batch["negative_prompt_embeds"]
        negative_prompt_embeds_mask = micro_batch["negative_prompt_embeds_mask"]

        if prompt_embeds.is_nested:
            batch_size = prompt_embeds.size(0)
            seq_len_effective = prompt_embeds.offsets().diff()
            max_seq_len = max(seq_len_effective)
            embed_dim = prompt_embeds.size(-1)
            prompt_embeds = torch.nested.to_padded_tensor(
                prompt_embeds, padding=0, output_size=(batch_size, max_seq_len, embed_dim)
            )
            prompt_embeds_mask = torch.nested.to_padded_tensor(
                prompt_embeds_mask, padding=0, output_size=(batch_size, max_seq_len)
            )
        if isinstance(negative_prompt_embeds, torch.Tensor) and negative_prompt_embeds.is_nested:
            batch_size = negative_prompt_embeds.size(0)
            seq_len_effective = negative_prompt_embeds.offsets().diff()
            max_seq_len = max(seq_len_effective)
            embed_dim = negative_prompt_embeds.size(-1)
            negative_prompt_embeds = torch.nested.to_padded_tensor(
                negative_prompt_embeds, padding=0, output_size=(batch_size, max_seq_len, embed_dim)
            )
            negative_prompt_embeds_mask = torch.nested.to_padded_tensor(
                negative_prompt_embeds_mask, padding=0, output_size=(batch_size, max_seq_len)
            )

        height = tu.get_non_tensor_data(data=micro_batch, key="height", default=None)
        width = tu.get_non_tensor_data(data=micro_batch, key="width", default=None)
        vae_scale_factor = tu.get_non_tensor_data(data=micro_batch, key="vae_scale_factor", default=None)
        img_shapes = [[(1, height // vae_scale_factor // 2, width // vae_scale_factor // 2)]]

        if getattr(self.module.config, "guidance_embeds", False):
            guidance = torch.full([1], self._guidance_scale, dtype=torch.float32)
        else:
            guidance = None

        hidden_states = latents[:, step]
        timestep = timesteps[:, step] / 1000.0

        # TODO (mike): in diffusers main branch, it no longer accept txt_seq_lens
        txt_seq_lens = torch.ones_like(prompt_embeds_mask).sum(dim=1).tolist()

        if isinstance(negative_prompt_embeds_mask, torch.Tensor):
            negative_txt_seq_lens = torch.ones_like(negative_prompt_embeds_mask).sum(dim=1).tolist()
        else:
            negative_txt_seq_lens = None

        model_inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "guidance": guidance,
            "encoder_hidden_states_mask": prompt_embeds_mask,
            "encoder_hidden_states": prompt_embeds,
            "img_shapes": img_shapes,
            "txt_seq_lens": txt_seq_lens,
            "return_dict": False,
        }

        negative_model_inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "guidance": guidance,
            "encoder_hidden_states_mask": negative_prompt_embeds_mask,
            "encoder_hidden_states": negative_prompt_embeds,
            "img_shapes": img_shapes,
            "txt_seq_lens": negative_txt_seq_lens,
            "return_dict": False,
        }

        return model_inputs, negative_model_inputs

    def prepare_model_outputs(self, output, micro_batch: TensorDict):
        log_prob, prev_sample_mean, std_dev_t = output
        model_output = {}
        model_output["log_probs"] = log_prob
        model_output["prev_sample_mean"] = prev_sample_mean
        model_output["std_dev_t"] = std_dev_t
        return model_output

    def forward_model_with_scheduler(self, model_inputs, negative_model_inputs, micro_batch, step):
        latents = micro_batch["all_latents"]
        timesteps = micro_batch["all_timesteps"]

        noise_pred = self.module(**model_inputs)[0]
        if self._guidance_scale > 1.0:
            neg_noise_pred = self.module(**negative_model_inputs)[0]
            comb_pred = neg_noise_pred + self._guidance_scale * (noise_pred - neg_noise_pred)
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            noise_pred = comb_pred * (cond_norm / noise_norm)

        _, log_prob, prev_sample_mean, std_dev_t = self.scheduler.sample_previous_step(
            sample=latents[:, step].float(),
            model_output=noise_pred,
            timestep=timesteps[:, step],
            noise_level=self.model_config.noise_level,
            prev_sample=latents[:, step + 1].float(),
            sde_type=self.model_config.sde_type,
        )
        return log_prob, prev_sample_mean, std_dev_t

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only, step):
        device_name = get_device_name()
        # actually, we should avoid assigning like this...
        micro_batch = micro_batch.to(get_device_id())
        model_inputs, negative_model_inputs = self.prepare_model_inputs(micro_batch=micro_batch, step=step)
        raw_output = self.forward_model_with_scheduler(
            model_inputs=model_inputs, negative_model_inputs=negative_model_inputs, micro_batch=micro_batch, step=step
        )
        model_output = self.prepare_model_outputs(output=raw_output, micro_batch=micro_batch)

        if loss_function is not None:
            data = tu.get_tensordict(
                {
                    "old_log_probs": micro_batch["old_log_probs"][:, step],
                    "advantages": micro_batch["advantages"][:, step],
                    "response_mask": micro_batch["response_mask"][:, step],
                },
                {
                    "dp_size": tu.get_non_tensor_data(micro_batch, "dp_size", None),
                    "batch_num_tokens": tu.get_non_tensor_data(micro_batch, "batch_num_tokens", None),
                    "global_batch_size": tu.get_non_tensor_data(micro_batch, "global_batch_size", None),
                },
            )
            if micro_batch.get("ref_log_prob", None) is not None:
                data["ref_log_prob"] = micro_batch["ref_log_prob"][:, step]

            if micro_batch.get("ref_prev_sample_mean", None) is not None:
                data["ref_prev_sample_mean"] = micro_batch["ref_prev_sample_mean"][:, step]

            loss, metrics = loss_function(model_output=model_output, data=data, dp_group=self.get_data_parallel_group())
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

    def optimizer_zero_grad(self):
        """
        Zero gradients and enforce FSDP grad-clipping logic.
        """
        self.optimizer.zero_grad()

    def optimizer_step(self):
        """
        Clip gradients, skip update if non-finite, and step optimizer.

        Returns:
            grad_norm (float): Norm of gradients before clipping.
        """
        assert self.optimizer_config.clip_grad is not None

        if isinstance(self.module, FSDP):
            grad_norm = self.module.clip_grad_norm_(self.optimizer_config.clip_grad)
        elif isinstance(self.module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.module.parameters(), max_norm=self.optimizer_config.clip_grad)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.module.parameters(), max_norm=self.optimizer_config.clip_grad
            )

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        return grad_norm.item()

    def lr_scheduler_step(self):
        """
        Advance FSDP scheduler and return updated learning rate.
        """
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]  # only return the first group
        return lr

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """
        Move FSDP model and/or optimizer to CPU or GPU with offload support.
        Note that this function executes irrespective of offload config. It serves as manual control
        """
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        if self.engine_config.forward_only:
            # force cpu_offload
            return

        device_name = get_device_name()

        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                load_fsdp_model_to_gpu(self.module)
            if optimizer and self.optimizer is not None:
                load_fsdp_optimizer(self.optimizer, device)
            gc.collect()
        elif device == "cpu":
            if model:
                offload_fsdp_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_fsdp_optimizer(self.optimizer)
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
        Save FSDP checkpoint, handling parameter offload as needed.
        """
        origin_module_device = next(self.module.parameters()).device.type
        if self._is_offload_param or origin_module_device == "cpu":
            load_fsdp_model_to_gpu(self.module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
    ) -> None:
        """
        Load FSDP checkpoint, restoring parameters and optimizer state.
        """
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.optimizer)

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)

        load_fsdp_model_to_gpu(self.module)

        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        merge_lora = self.model_config.lora.get("merge", False)

        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if hasattr(peft_model, "peft_config"):  # LoRA
            if not merge_lora:
                peft_config = peft_model.peft_config.get("default", None)
                params = collect_lora_params(
                    module=self.module,
                    layered_summon=layered_summon,
                    base_sync_done=base_sync_done,
                    is_diffusers=True,
                )
                if not base_sync_done:
                    params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
            else:  # merge lora
                with merged_lora_context(self.module, backup_adapters=True):
                    params = self.module.state_dict()
                    params = normalize_peft_param_name(params)
        else:
            params = self.module.state_dict()

        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        if peft_config is not None and base_sync_done:
            per_tensor_param = params.items()
        else:
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            # TODO: cast fp32 to bf16 to reduce weight sync overhead, need more fine-grained control, e.g MoE gate
            per_tensor_param = (
                (
                    name,
                    param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
                    if isinstance(param, DTensor)
                    else param,
                )
                for name, param in params.items()
            )
        # return per_tensor_param, peft_config
        # Convert peft_config to dict for vLLM compatibility (PEFTHelper.from_dict expects dict)

        # diffusers: transformer backbone only
        # vllm-omni: whole pipeline
        # thus we need to add the prefix
        per_tensor_param = ((f"transformer.{name}", tensor) for name, tensor in per_tensor_param)
        peft_config_dict = peft_config.to_dict() if peft_config is not None else None
        return per_tensor_param, peft_config_dict

    @contextmanager
    def disable_adapter(self):
        try:
            self.module.disable_adapters()
            yield
        finally:
            self.module.enable_adapters()


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: DiffusersFSDPEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, DiffusersFSDPEngine)
        super().__enter__()
        self.prev_sp_group = get_ulysses_sequence_parallel_group()
        set_ulysses_sequence_parallel_group(self.engine.ulysses_parallel_group)
        self.engine.module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, DiffusersFSDPEngine)
        set_ulysses_sequence_parallel_group(self.prev_sp_group)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.engine.engine_config.fsdp_size > 1:
            if fsdp_version(self.engine.module) == 1:
                self.engine.module._handle.reshard(True)
            elif fsdp_version(self.engine.module) == 2:
                self.engine.module.reshard()

        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: DiffusersFSDPEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, DiffusersFSDPEngine)
        super().__enter__()
        self.prev_sp_group = get_ulysses_sequence_parallel_group()
        set_ulysses_sequence_parallel_group(self.engine.ulysses_parallel_group)
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, DiffusersFSDPEngine)
        set_ulysses_sequence_parallel_group(self.prev_sp_group)
        self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)
