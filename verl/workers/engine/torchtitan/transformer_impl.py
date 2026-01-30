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
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Callable, Optional

import torch
import torch.distributed
import verl.utils.torch_functional as verl_F
from tensordict import TensorDict
from torchtitan.config.job_config import (
    Compile,
    JobConfig,
    LRScheduler,
    Model,
    Optimizer,
    Parallelism,
)
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.train import Trainer
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
from verl.workers.config import (
    TorchtitanEngineConfig,
    TorchtitanModelConfig,
    TorchtitanOptimizerConfig,
)

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import (
    enable_full_determinism,
    postprocess_batch_func,
    prepare_micro_batches,
)

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
        model_config: TorchtitanModelConfig,
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

        # Patch torchtitan's init_distributed to skip if verl has already
        # called initialize_global_process_group()
        # This avoids "trying to initialize the default process group twice"
        original_init_distributed = dist_utils.init_distributed

        def patched_init_distributed(*args, **kwargs):
            if torch.distributed.is_initialized():
                return torch.distributed.get_world_size()
            return original_init_distributed(*args, **kwargs)

        dist_utils.init_distributed = patched_init_distributed

        # Patch train_spec to return a dummy dataloader since verl has its own data pipeline
        import torchtitan.protocols.train_spec as train_spec_module
        original_get_train_spec = train_spec_module.get_train_spec

        def patched_get_train_spec(model_name):
            train_spec = original_get_train_spec(model_name)

            # Create a dummy dataloader function that returns None
            # verl uses its own dataloader, not TorchTitan's
            def dummy_build_dataloader_fn(*args, **kwargs):
                return None

            # Replace the dataloader builder with our dummy
            train_spec.build_dataloader_fn = dummy_build_dataloader_fn
            return train_spec

        train_spec_module.get_train_spec = patched_get_train_spec

        self.model_config = Model(
            name=model_config.name, flavor=model_config.flavor, hf_assets_path=model_config.hf_assets_path
        )
        self.engine_config = engine_config
        self.optimizer_config = Optimizer(name=optimizer_config.name, lr=optimizer_config.lr, eps=optimizer_config.eps)
        self.lr_scheduler_config = LRScheduler(
            warmup_steps=optimizer_config.lr_warmup_steps,
            decay_ratio=optimizer_config.weight_decay,
            decay_type=optimizer_config.decay_type,
            min_lr_factor=optimizer_config.min_lr_factor,
        )
        self.parallelism_config = Parallelism(
            data_parallel_replicate_degree=engine_config.data_parallel_replicate_size,
            data_parallel_shard_degree=engine_config.data_parallel_shard_size,
            fsdp_reshard_after_forward=engine_config.reshard_after_forward,
            tensor_parallel_degree=engine_config.tensor_parallel_size,
            pipeline_parallel_degree=engine_config.pipeline_parallel_size,
            context_parallel_degree=engine_config.context_parallel_size,
            expert_parallel_degree=engine_config.expert_parallel_size,
            expert_tensor_parallel_degree=engine_config.expert_tensor_parallel_size,
        )
        self.checkpoint_config = checkpoint_config
        self.compile_config = Compile(enable=engine_config.use_torch_compile)

        # Construct Torchtitan's JobConfig
        self.config = JobConfig(
            model=self.model_config,
            optimizer=self.optimizer_config,
            lr_scheduler=self.lr_scheduler_config,
            parallelism=self.parallelism_config,
            # checkpoint=self.checkpoint_config,
            compile=self.compile_config,
        )
        self.trainer = Trainer(self.config)

        self._init_device_mesh()

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
        mesh = self.parallel_dims.get_optional_mesh(["dp_replicate", "fsdp"])
        if mesh is None:
            # No data parallelism, return rank 0
            return 0
        return mesh.get_local_rank()

    def get_data_parallel_size(self):
        mesh = self.parallel_dims.get_optional_mesh(["dp_replicate", "fsdp"])
        if mesh is None:
            # No data parallelism, return size 1
            return 1
        return mesh.size()

    def get_data_parallel_group(self):
        mesh = self.parallel_dims.get_optional_mesh(["dp_replicate", "fsdp"])
        if mesh is None:
            # No data parallelism, return None or world group
            return None
        return mesh

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> list[TensorDict]:
        """Perform forward and optionally backward pass on a batch."""
        tu.assign_non_tensor(data, sp_size=self.engine_config.tensor_parallel_size)

        # Compute num_tokens in global batch for loss normalization
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
            with ctx:
                loss, output = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)

                # TODO(jessicazhong): enable forward only in Torchtitan
                # if not forward_only:
                #     loss.backward()
            output_lst.append(output)

        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def optimizer_zero_grad(self):
        """Zero gradients."""
        dist_utils.clip_grad_norm_(
            [p for m in self.module for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
            ep_enabled=self.parallel_dims.ep_enabled,
        )
        self.optimizer.zero_grad()

    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        self.optimizer.step()

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
        """Save checkpoint."""
        if self._is_offload_param:
            for module in self.module:
                load_fsdp_model_to_gpu(module)

        # self.checkpoint_manager.save_checkpoint(
        #     local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        # )
        # Torchtitan's checkpoint save
        self.checkpointer.save(curr_step=global_step)

        # TODO(jessicazhong): figure out if this is still needed
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

        # self.checkpoint_manager.load_checkpoint(
        #     local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        # )
        # Torchtitan's checkpoint load
        self.checkpointer.load()

        # TODO(jessicazhong): figure out if this is still needed
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
            # TODO(jessicazhong): figure out FSDP reshard correctly
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
            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids_rmpad = input_ids.values().unsqueeze(0)
                if position_ids.dim() == 3:
                    position_ids_rmpad = position_ids.values().unsqueeze(1)
                else:
                    position_ids_rmpad = position_ids.values().unsqueeze(0)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
            output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled

            model_inputs = {
                "input_ids": input_ids_rmpad,
                "attention_mask": None,
                "position_ids": position_ids_rmpad,
            }
        else:
            if pad_mode == DatasetPadMode.NO_PADDING:
                loss_mask = micro_batch["loss_mask"]

                pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
                batch_size = micro_batch.batch_size[0]
                seq_len_effective = input_ids.offsets().diff()
                max_seq_len = max(seq_len_effective)

                input_ids_rmpad_rolled = torch.roll(input_ids.values(), shifts=-1, dims=0)
                output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled

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

        model_inputs.update(multi_modal_inputs)
        return model_inputs, output_args

    def prepare_model_outputs(self, logits, output_args, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        temperature = micro_batch["temperature"]
        calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)

        model_output = {}

        input_ids = micro_batch["input_ids"]
        if use_remove_padding:
            input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
            logits_rmpad = logits.squeeze(0)
            logits_rmpad.div_(temperature)

            inplace_backward = True
            if calculate_entropy:
                inplace_backward = False
            # logits_rmpad torch.Size([1912, 151936]), input_ids_rmpad_rolled torch.Size([1, 1912])
            log_probs = logprobs_from_logits(
                logits=logits_rmpad.unsqueeze(0),
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

            # TODO(jessicazhong): how to handle this with TorchTitan SP
            # if self.use_ulysses_sp:
            #     pad_size = output_args["pad_size"]

            #     log_probs = gather_outputs_and_unpad(
            #         log_probs,
            #         gather_dim=0,
            #         unpad_dim=0,
            #         padding_size=pad_size,
            #     )
            #     if calculate_entropy:
            #         entropy_rmpad = gather_outputs_and_unpad(
            #             entropy_rmpad,
            #             gather_dim=0,
            #             unpad_dim=0,
            #             padding_size=pad_size,
            #         )

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                log_probs = torch.nested.nested_tensor_from_jagged(log_probs.squeeze(0), cu_seqlens)
                if calculate_entropy:
                    entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        else:
            logits.div_(temperature)
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
        device_name = get_device_name()
        micro_batch = micro_batch.to(get_device_id())
        model_inputs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            # shifted_input shape torch.Size([1, 1912]), labels shape torch.Size([1, 1912])
            input_dict = {
                "input": model_inputs["input_ids"],
                "positions": model_inputs["position_ids"],
                "attention_masks": model_inputs["attention_mask"],
            }

            loss, logits = self.trainer.forward_backward_step(input_dict=input_dict, labels=output_args["input_ids_rmpad_rolled"])

            model_output: dict[str, Unknown] = self.prepare_model_outputs(
                logits=logits, output_args=output_args, micro_batch=micro_batch
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
