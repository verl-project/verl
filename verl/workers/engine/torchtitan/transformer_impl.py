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
The concrete Engine implementation using PyTorch TorchTitan parallelism (FSDP2 + TP + PP)
"""

import gc
import logging
import os
import re
from contextlib import nullcontext
from typing import Any, Callable, Optional

import torch
import torch.distributed
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from torchtitan.config.job_config import (
    Checkpoint,
    Compile,
    JobConfig,
    LRScheduler,
    Model,
    Optimizer,
    Parallelism,
    Training,
)
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.context_parallel import prepare_context_parallel_input
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.train import Trainer

import verl.utils.torch_functional as verl_F
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.model import extract_multi_modal_inputs
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.config import HFModelConfig, TorchtitanEngineConfig, TorchtitanOptimizerConfig
from verl.workers.engine.torchtitan.utils import (
    derive_torchtitan_name_and_flavor,
    enable_fsdp_gradient_division,
    get_attention_masks,
)

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import enable_full_determinism, postprocess_batch_func, prepare_micro_batches

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class TorchTitanEngine(BaseEngine):
    """
    Concrete Engine implementation using PyTorch TorchTitan parallelism.

    Supports model sharding with FSDP2, tensor parallelism, activation/optimizer offloading,
    LoRA, and sequence parallelism following the TorchTitan design.
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: TorchtitanEngineConfig,
        optimizer_config: TorchtitanOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        """
        Initialize the TorchTitanEngine.

        Sets up distributed device meshes for tensor and data parallelism, LoRA, and offload policies.

        Args:
            model_config: Configuration for HuggingFace model.
            engine_config: Configuration for FSDP/TorchTitan engine (uses FSDP2).
            optimizer_config: Configuration for optimizer.
            checkpoint_config: Configuration for checkpointing.
        """
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        # Disable torchtitan's dataloader since verl has its own data loading
        # Ideally torchtitan trainer init should not initialize dataloader
        import torchtitan.protocols.train_spec as train_spec_module

        original_get_train_spec = train_spec_module.get_train_spec

        def _get_train_spec_without_dataloader(model_name):
            train_spec = original_get_train_spec(model_name)
            train_spec.build_dataloader_fn = None
            return train_spec

        train_spec_module.get_train_spec = _get_train_spec_without_dataloader

        # Derive torchtitan model name and flavor from HF config
        torchtitan_name, torchtitan_flavor = derive_torchtitan_name_and_flavor(self.model_config.hf_config)

        # Get train_spec and directly override model_args before Trainer init
        train_spec = train_spec_module.get_train_spec(torchtitan_name)
        model_args = train_spec.model_args.get(torchtitan_flavor)
        if model_args is not None:
            if hasattr(model_args, "attn_type"):
                model_args.attn_type = self.engine_config.attn_type

        model = Model(
            name=torchtitan_name,
            flavor=torchtitan_flavor,
            hf_assets_path=self.model_config.path,
        )
        optimizer = Optimizer(
            name=self.optimizer_config.name,
            lr=self.optimizer_config.lr,
            eps=self.optimizer_config.eps,
            beta1=self.optimizer_config.betas[0],
            beta2=self.optimizer_config.betas[1],
            weight_decay=self.optimizer_config.weight_decay,
        )

        total_steps = self.optimizer_config.total_training_steps
        lr_warmup_steps = self.optimizer_config.lr_warmup_steps
        if lr_warmup_steps is None or lr_warmup_steps <= 0:
            lr_warmup_steps = int(self.optimizer_config.lr_warmup_steps_ratio * total_steps)

        lr_scheduler = LRScheduler(
            warmup_steps=lr_warmup_steps,
            decay_type=self.optimizer_config.decay_type,
            min_lr_factor=self.optimizer_config.min_lr_factor,
        )
        parallelism = Parallelism(
            data_parallel_replicate_degree=self.engine_config.data_parallel_replicate_size,
            data_parallel_shard_degree=self.engine_config.data_parallel_shard_size,
            fsdp_reshard_after_forward=self.engine_config.reshard_after_forward,
            tensor_parallel_degree=self.engine_config.tensor_parallel_size,
            pipeline_parallel_degree=self.engine_config.pipeline_parallel_size,
            context_parallel_degree=self.engine_config.context_parallel_size,
            expert_parallel_degree=self.engine_config.expert_parallel_size,
            expert_tensor_parallel_degree=self.engine_config.expert_tensor_parallel_size,
        )
        checkpoint = Checkpoint(
            enable=True,
            initial_load_in_hf=True,
            initial_load_model_only=True,
            initial_load_path=model_config.path,
        )
        compile = Compile(enable=self.engine_config.use_torch_compile)
        if self.engine_config.offload_policy or self.engine_config.forward_only:
            training = Training(enable_cpu_offload=True)
        else:
            training = Training()

        # Construct Torchtitan's JobConfig
        self.config = JobConfig(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            parallelism=parallelism,
            checkpoint=checkpoint,
            compile=compile,
            training=training,
        )
        self.trainer = Trainer(self.config)

        self._init_device_mesh()

        # Re-enable FSDP's gradient division for verl's loss scaling.
        # TorchTitan disables gradient division by default (for global token normalization),
        # but verl's loss function multiplies by dp_size to compensate for gradient averaging.
        if self.engine_config.data_parallel_shard_size > 1:
            dp_size = self.get_data_parallel_size()
            for model_part in self.trainer.model_parts:
                enable_fsdp_gradient_division(model_part, dp_size)

        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)

        # set FSDP offload params
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

    def is_mp_src_rank_with_outputs(self):
        """
        Whether the current rank is the first rank in model parallel group that contains model outputs
        """
        is_collect = True
        # TP: outputs are on TP rank 0
        if self.parallel_dims.tp > 1:
            tp_mesh = self.parallel_dims.get_optional_mesh("tp")
            is_collect = is_collect and (tp_mesh.get_local_rank() == 0)
        # PP: outputs are on the last PP rank
        if self.parallel_dims.pp > 1:
            pp_mesh = self.parallel_dims.get_optional_mesh("pp")
            is_collect = is_collect and (pp_mesh.get_local_rank() == self.parallel_dims.pp - 1)
        # CP: outputs are on CP rank 0
        if self.parallel_dims.cp > 1:
            cp_mesh = self.parallel_dims.get_optional_mesh("cp")
            is_collect = is_collect and (cp_mesh.get_local_rank() == 0)
        return is_collect

    def initialize(self):
        """
        Build the model, optimizer, and learning rate scheduler with TorchTitan parallelism.

        Applies device, dtype, and precision configurations, including mixed precision.
        Sets up checkpoint manager.
        """
        self.module = self.trainer.model_parts
        self.checkpointer = self.trainer.checkpointer
        # load initial HF weights
        self.checkpointer.load()

        if not self.engine_config.forward_only:
            self.optimizer = self.trainer.optimizers
            self.lr_scheduler = self.trainer.lr_schedulers
        else:
            self.optimizer = None
            self.lr_scheduler = None

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def _init_device_mesh(self):
        """Initialize the device mesh for TorchTitan style parallelism."""
        world_size = torch.distributed.get_world_size()
        self.parallel_dims = ParallelDims(
            dp_shard=self.engine_config.data_parallel_shard_size,
            dp_replicate=self.engine_config.data_parallel_replicate_size,
            cp=self.engine_config.context_parallel_size,
            tp=self.engine_config.tensor_parallel_size,
            pp=self.engine_config.pipeline_parallel_size,
            ep=self.engine_config.expert_parallel_size,
            etp=self.engine_config.expert_tensor_parallel_size,
            world_size=world_size,
        )
        self.device_mesh = self.parallel_dims.build_mesh()

    def train_mode(self, **kwargs):
        """Return a context manager for training mode."""
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """Return a context manager for evaluation mode."""
        return EngineEvalModeCtx(self, **kwargs)

    def get_data_parallel_rank(self):
        mesh = self._get_data_parallel_mesh()
        return 0 if mesh is None else mesh.get_local_rank()

    def get_data_parallel_size(self):
        return self.engine_config.data_parallel_shard_size * self.engine_config.data_parallel_replicate_size

    def get_data_parallel_group(self):
        mesh = self._get_data_parallel_mesh()
        return mesh.get_group() if mesh is not None else None

    def _get_data_parallel_mesh(self):
        """Get the data parallel mesh, handling hybrid/fully/replicate shard modes."""
        mesh = self.parallel_dims.get_optional_mesh(["dp_replicate", "fsdp"])
        if mesh is None:
            mesh = self.parallel_dims.get_optional_mesh("fsdp")
        if mesh is None:
            mesh = self.parallel_dims.get_optional_mesh("dp_replicate")
        return mesh

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False):
        """Perform forward and optionally backward pass on a batch."""
        tu.assign_non_tensor(data, sp_size=self.engine_config.tensor_parallel_size)

        # Compute num_tokens in global batch for loss normalization
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        dp_group = self.get_data_parallel_group()
        if dp_group is not None:
            torch.distributed.all_reduce(batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=dp_group)
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True
        )

        output_lst = []

        ctx = torch.no_grad() if forward_only else nullcontext()

        for micro_batch in micro_batches:
            with ctx:
                loss, output = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)
                if not forward_only:
                    loss.backward()
            output_lst.append(output)

        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def model_forward_step(
        self,
        *,
        inputs: torch.Tensor,
        extra_inputs: dict[str, torch.Tensor] | None = None,
        extra_kwargs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the trainer model without backward.
        """
        model_parts = self.module
        parallel_dims = self.parallel_dims

        if parallel_dims.pp_enabled:
            raise NotImplementedError(
                "Pipeline parallelism is not yet supported in model_forward_step. "
                "This will be implemented in a follow-up PR."
            )
        else:
            # Non-PP forward
            assert len(model_parts) == 1
            with self.trainer.train_context():
                with self.trainer.maybe_enable_amp:
                    pred = model_parts[0](inputs, **extra_inputs, **extra_kwargs)

        if isinstance(pred, DTensor):
            pred = pred.full_tensor()
        return pred

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def optimizer_zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()

    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.module for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
            ep_enabled=self.parallel_dims.ep_enabled,
        )

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            logger.warning(f"grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        return grad_norm.item()

    def lr_scheduler_step(self):
        """Advance learning rate scheduler."""
        self.lr_scheduler.step()
        lr = self.lr_scheduler.schedulers[0].get_last_lr()[0]
        return lr

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """Move model and/or optimizer to CPU or GPU."""
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        if self.engine_config.forward_only:
            return

        device_name = get_device_name()
        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                for module in self.module:
                    load_fsdp_model_to_gpu(module)
            if optimizer and self.optimizer is not None:
                load_fsdp_optimizer(self.optimizer, device)
            gc.collect()
        elif device == "cpu":
            if model:
                for module in self.module:
                    offload_fsdp_model_to_cpu(module)
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
        """Save checkpoint."""
        if self._is_offload_param:
            for module in self.module:
                load_fsdp_model_to_gpu(module)

        # Override TorchTitan's folder to use verl's path
        parent_dir = os.path.dirname(local_path)
        self.checkpointer.folder = parent_dir

        if max_ckpt_to_keep is not None:
            self.checkpointer.keep_latest_k = max_ckpt_to_keep

        self.checkpointer.save(curr_step=global_step)

        torch.distributed.barrier()
        if self._is_offload_param:
            for module in self.module:
                offload_fsdp_model_to_cpu(module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
    ) -> None:
        """Load checkpoint."""
        if self._is_offload_param:
            for module in self.module:
                load_fsdp_model_to_gpu(module)

        # Override TorchTitan's folder to use verl's path
        parent_dir = os.path.dirname(local_path)
        self.checkpointer.folder = parent_dir

        # Extract step number from path (verl uses global_step_N format)
        match = re.search(r"global_step_(\d+)", local_path)
        if match:
            step = int(match.group(1))
            self.checkpointer.load(step=step)
        else:
            # Fallback to latest
            self.checkpointer.load(step=-1)

        torch.distributed.barrier()
        if self._is_offload_param:
            for module in self.module:
                offload_fsdp_model_to_cpu(module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.optimizer)


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: TorchTitanEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, TorchTitanEngine)
        super().__enter__()
        for module in self.engine.module:
            module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, TorchTitanEngine)

        # Reshard the root FSDP module
        if self.engine.engine_config.data_parallel_shard_size > 1:
            for module in self.engine.module:
                module.reshard()

        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: TorchTitanEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, TorchTitanEngine)
        super().__enter__()
        for module in self.engine.module:
            module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, TorchTitanEngine)
        self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@EngineRegistry.register(model_type="language_model", backend=["torchtitan"], device=["cuda", "npu"])
class TorchTitanEngineWithLMHead(TorchTitanEngine):
    """TorchTitan engine implementation for language models with LM head."""

    def prepare_model_inputs(self, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        multi_modal_inputs = extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", []))
        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]
        output_args = {}

        if use_remove_padding:
            input_ids = input_ids.values().unsqueeze(0)
            if position_ids.dim() == 3:
                position_ids = position_ids.values().unsqueeze(1)
            else:
                position_ids = position_ids.values().unsqueeze(0)

            labels = torch.roll(input_ids, shifts=-1, dims=1)
            attn_type = self.trainer.model_args.attn_type
            attention_mask = get_attention_masks(
                input_batch=input_ids,
                positions=position_ids,
                attn_type=attn_type,
            )
        else:
            loss_mask = micro_batch["loss_mask"]
            pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
            batch_size = micro_batch.batch_size[0]
            max_seq_len = max(input_ids.offsets().diff())

            labels = torch.roll(input_ids.values(), shifts=-1, dims=0)
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

        extra_inputs = {
            "positions": position_ids,
        }
        # For arguments, like attention_masks, we have to put them in a separate
        # dict as extra_inputs are not forwarded to other stages in PP, but
        # extra_kwargs are.
        extra_kwargs: dict[str, Any] = {"attention_masks": attention_mask}
        if self.parallel_dims.cp_enabled:
            input_ids, labels, extra_kwargs = prepare_context_parallel_input(
                input_ids,
                labels,
                extra_kwargs,
                self.parallel_dims.get_mesh("cp"),
                self.trainer.device,
                self.trainer.job_config.parallelism.context_parallel_load_balancer,
            )

        # TODO(jessicazhong): multimodal is not yet supported for Torchtitan engine
        extra_inputs.update(multi_modal_inputs)
        output_args["labels"] = labels
        return input_ids, extra_inputs, extra_kwargs, output_args

    def prepare_model_outputs(self, logits, output_args, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        temperature = micro_batch["temperature"]
        calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)
        labels = output_args["labels"]
        model_output = {}

        input_ids = micro_batch["input_ids"]
        cu_seqlens = input_ids.offsets()
        if use_remove_padding:
            labels = labels.squeeze(0)
            logits_rmpad = logits.squeeze(0)
            # PyTorch's autograd doesn't allow in-place modification of views when gradients need to flow back
            logits_rmpad = logits_rmpad / temperature

            inplace_backward = True
            if calculate_entropy:
                inplace_backward = False
            log_probs = logprobs_from_logits(
                logits=logits_rmpad,
                labels=labels,
                inplace_backward=inplace_backward,
            )

            if calculate_entropy:
                if not self.engine_config.entropy_checkpointing:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                else:
                    entropy_rmpad = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)

            log_probs = torch.nested.nested_tensor_from_jagged(log_probs.squeeze(0), cu_seqlens)
            if calculate_entropy:
                entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
        else:
            logits.div_(temperature)
            if calculate_entropy:
                if not self.engine_config.entropy_checkpointing:
                    entropy = verl_F.entropy_from_logits(logits)
                else:
                    entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            seq_lengths = cu_seqlens.diff()
            starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
            logits = torch.nested.narrow(logits, 1, starts, seq_lengths, layout=torch.jagged)
            logits_rmpad = torch.cat([t for t in logits.unbind()])
            log_probs = logprobs_from_logits(logits=logits_rmpad, labels=output_args["labels"])
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
        input_ids, extra_inputs, extra_kwargs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            logits = self.model_forward_step(inputs=input_ids, extra_inputs=extra_inputs, extra_kwargs=extra_kwargs)

            model_output = self.prepare_model_outputs(logits=logits, output_args=output_args, micro_batch=micro_batch)

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
