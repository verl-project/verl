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

import json
import logging
import os
import random
from collections.abc import Callable
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed
from megatron.core import mpu, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.transformer.enums import AttnBackend
from transformers import GenerationConfig

from verl.models.weight_loader_registry import get_weight_saver
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.fs import is_non_local, local_mkdir_safe
from verl.utils.logger import log_with_rank
from verl.utils.megatron.dist_checkpointing import load_dist_checkpointing, save_dist_checkpointing
from verl.utils.megatron_utils import (
    get_dist_checkpoint_path,
    get_hf_model_checkpoint_path,
    get_transformer_config_checkpoint_path,
)

from .checkpoint_manager import BaseCheckpointManager

# Setup logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class MegatronCheckpointManager(BaseCheckpointManager):
    """
    Checkpoint manager for Megatron-LM distributed training.

    This class manages the saving and loading of model checkpoints in a Megatron-LM
    distributed training environment. It handles various aspects of checkpointing
    including model states, optimizer states, learning rate schedulers, and random
    number generator states, ensuring compatibility with HuggingFace formats.

    Key features:
    - Distributed checkpoint saving and loading using Megatron's dist_checkpointing
    - Support for tensor parallel, pipeline parallel, and data parallel configurations
    - Automatic handling of model state dictionaries across multiple pipeline stages
    - Integration with HuggingFace model configurations and tokenizers
    - Random number generator state management for reproducibility
    - Support for both synchronous and asynchronous checkpoint operations

    The manager automatically handles:
    - Directory structure creation based on global steps and process ranks
    - Model configuration and tokenizer saving in HuggingFace format
    - Optimizer and scheduler state persistence
    - CUDA RNG state management for deterministic training
    - Checkpoint cleanup and retention policies

    Args:
        model: The Megatron model instance to checkpoint
        optimizer: The optimizer instance (optional)
        lr_scheduler: The learning rate scheduler instance (optional)

    Attributes:
        model: Reference to the Megatron model being checkpointed
        optimizer: Reference to the optimizer (if provided)
        lr_scheduler: Reference to the learning rate scheduler (if provided)
        rank: Current process rank in the distributed setup

    Example:
        ```python
        checkpoint_manager = MegatronCheckpointManager(
            model=megatron_model,
            optimizer=optimizer,
            lr_scheduler=scheduler
        )

        checkpoint_manager.save_checkpoint(
            local_path="checkpoints/step_1000",
            global_step=1000
        )

        checkpoint_manager.load_checkpoint(
            local_path="checkpoints/step_1000"
        )
        ```
    """

    def __init__(
        self,
        config,
        checkpoint_config,
        model_config,
        transformer_config,
        role,
        model: torch.nn.ModuleList,
        arch: str,
        hf_config,
        param_dtype: torch.dtype,
        share_embeddings_and_output_weights: bool,
        processing_class,
        optimizer,
        optimizer_scheduler,
        use_distributed_optimizer: bool,
        use_checkpoint_opt_param_scheduler: bool = False,
        use_dist_checkpointing: bool = True,
        bridge=None,
        provider=None,
        peft_cls=None,
        **kwargs,
    ):
        super().__init__(
            model,
            optimizer=optimizer,
            lr_scheduler=optimizer_scheduler,
            processing_class=processing_class,
            checkpoint_config=checkpoint_config,
        )
        self.arch = arch
        self.config = config
        self.transformer_config = transformer_config
        self.role = role
        self.is_value_model = False
        if self.role in ["reward", "critic"]:
            self.is_value_model = True
        self.model_config = model_config
        self.hf_config = hf_config
        self.param_dtype = param_dtype
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.model_path = self.config.model.path
        self.use_distributed_optimizer = use_distributed_optimizer
        self.use_checkpoint_opt_param_scheduler = use_checkpoint_opt_param_scheduler
        self.bridge = bridge
        self.provider = provider
        self.vanilla_bridge = self.provider is None
        self.peft_cls = peft_cls
        self.rank = torch.distributed.get_rank()
        # Megatron-Bridge is Okay to load/save HF checkpoint for value model as well
        self.use_dist_checkpointing = (
            use_dist_checkpointing or not self.bridge or (self.vanilla_bridge and self.is_value_model)
        )
        self.use_hf_checkpoint = not self.use_dist_checkpointing

        self.weight_saver = None
        if self.bridge is None:
            self.weight_saver = get_weight_saver(self.arch)

    def get_rng_state(self, use_dist_ckpt: bool = True, data_parallel_random_init: bool = False):
        """collect rng state across data parallel ranks"""
        rng_state = {
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
        }

        if get_device_name() != "cpu":
            rng_state[f"{get_device_name()}_rng_state"] = get_torch_device().get_rng_state()

        rng_state_list = None
        if torch.distributed.is_initialized() and mpu.get_data_parallel_world_size() > 1 and data_parallel_random_init:
            rng_state_list = [None for i in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather_object(rng_state_list, rng_state, group=mpu.get_data_parallel_group())
        else:
            rng_state_list = [rng_state]

        if use_dist_ckpt:
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            pp_size = mpu.get_pipeline_model_parallel_world_size()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            tp_size = mpu.get_tensor_model_parallel_world_size()
            rng_state_list = ShardedObject(
                "rng_state",
                rng_state_list,
                (pp_size, tp_size),
                (pp_rank, tp_rank),
                replica_id=mpu.get_data_parallel_rank(with_context_parallel=True),
            )

        return rng_state_list

    def get_checkpoint_name(
        self,
        checkpoints_path,
        pipeline_parallel=None,
        tensor_rank=None,
        pipeline_rank=None,
        cp_rank=None,
        expert_parallel=None,
        expert_rank=None,
        return_base_dir=True,
        basename="model.pt",
    ):
        """Determine the directory name for this rank's checkpoint."""
        # Use both the tensor and pipeline MP rank.
        if pipeline_parallel is None:
            pipeline_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
        if tensor_rank is None:
            tensor_rank = mpu.get_tensor_model_parallel_rank()
        if pipeline_rank is None:
            pipeline_rank = mpu.get_pipeline_model_parallel_rank()
        if cp_rank is None:
            cp_rank = mpu.get_context_parallel_rank()
        if expert_parallel is None:
            expert_parallel = mpu.get_expert_model_parallel_world_size() > 1
        if expert_rank is None:
            expert_rank = mpu.get_expert_model_parallel_rank()

        # Use both the tensor and pipeline MP rank. If using the distributed
        # optimizer, then the optimizer's path must additionally include the
        # data parallel rank.

        # due to the fact that models are identical across cp ranks, cp rank is not used in the checkpoint path
        if not pipeline_parallel:
            common_path = os.path.join(checkpoints_path, f"mp_rank_{tensor_rank:02d}")
        else:
            common_path = os.path.join(checkpoints_path, f"mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}")

        if expert_parallel:
            common_path = common_path + f"_{expert_rank:03d}"

        os.makedirs(common_path, exist_ok=True)

        if return_base_dir:
            return common_path
        return os.path.join(common_path, basename)

    def generate_state_dict(
        self,
        generate_model: bool = True,
        generate_optimizer: bool = True,
        generate_extra: bool = True,
        is_loading: bool = False,
    ):
        # For save dist checkpointing
        state_dict = {}

        # Should always generate model state dict
        # All ranks Save Model to reduce memory pressure
        # Get sharded state dict, notice that state_dict will collect among dp groups, causing memory pressure
        for vpp_rank, model in enumerate(self.model):
            if len(self.model) > 1:
                mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
                key = f"model{vpp_rank}" if len(self.model) > 1 else "model"
            else:
                key = "model"
            if hasattr(model, "module"):
                model = model.module

            # GPTModel's sharded_state_dict function when having mtp requires metadata['dp_cp_group']
            kwargs = {"metadata": {"dp_cp_group": mpu.get_data_parallel_group(with_context_parallel=True)}}
            state_dict[key] = model.sharded_state_dict(**kwargs)

        # Optimizer State Dict
        if generate_optimizer:
            torch.distributed.barrier()
            optimizer_sharded_states = self.optimizer.sharded_state_dict(state_dict, is_loading=is_loading)
            state_dict["optimizer"] = optimizer_sharded_states

            if self.lr_scheduler is not None:
                lr_state_dict = self.lr_scheduler.state_dict()
                state_dict["lr_scheduler"] = lr_state_dict

        if not generate_model:
            state_dict.pop("model", None)

        # RNG States State Dict
        if generate_extra:
            torch.distributed.barrier()
            rng_state = self.get_rng_state()
            state_dict["rng_state"] = rng_state

        return state_dict

    def load_rng_states(self, rng_states, data_parallel_random_init=False, use_dist_ckpt=True):
        # access rng_state for data parallel rank
        if data_parallel_random_init:
            rng_states = rng_states[mpu.get_data_parallel_rank()]
        else:
            rng_states = rng_states[0]
        random.setstate(rng_states["random_rng_state"])
        np.random.set_state(rng_states["np_rng_state"])
        torch.set_rng_state(rng_states["torch_rng_state"])

        if get_device_name() != "cpu":
            get_torch_device().set_rng_state(rng_states[f"{get_device_name()}_rng_state"])

        # Check for empty states array
        if not rng_states["rng_tracker_states"]:
            raise KeyError
        tensor_parallel.get_cuda_rng_tracker().set_states(rng_states["rng_tracker_states"])

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        if local_path is not None:
            assert os.path.exists(local_path), f"Checkpoint path {local_path} does not exist."

        # For load optimizer dist_ckpt
        try:
            import transformer_engine

            torch.serialization.add_safe_globals([torch.optim.AdamW])
            torch.serialization.add_safe_globals([transformer_engine.pytorch.optimizers.fused_adam.FusedAdam])
        except Exception:
            pass

        dist_checkpoint_path = get_dist_checkpoint_path(local_path)

        # Get State Dict for loading
        sharded_state_dict = self.generate_state_dict(
            self.should_load_model and self.use_dist_checkpointing,
            self.should_load_optimizer,
            self.should_load_extra,
            is_loading=True,
        )
        log_with_rank(f"Generated state dict for loading: {sharded_state_dict.keys()}", rank=self.rank, logger=logger)

        # Load Dist Checkpointing
        state_dict = load_dist_checkpointing(
            sharded_state_dict=sharded_state_dict,
            ckpt_dir=dist_checkpoint_path,
        )

        if self.should_load_model and self.use_dist_checkpointing:
            assert "model" in state_dict or any(
                f"model{vpp_rank}" in state_dict for vpp_rank in range(len(self.model))
            ), f"Model state dict not found in {state_dict.keys()}. Please check the checkpoint file {local_path}."
            for vpp_rank, model in enumerate(self.model):
                if len(self.model) == 1:
                    model_state_dict = state_dict["model"]
                else:
                    assert f"model{vpp_rank}" in state_dict, f"model{vpp_rank} not found in state_dict"
                    model_state_dict = state_dict[f"model{vpp_rank}"]
                mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
                self.model[vpp_rank].load_state_dict(model_state_dict)
            log_with_rank(f"Loaded sharded model checkpoint from {local_path}", rank=self.rank, logger=logger)

        # Skip HF checkpoint loading if PEFT is used
        elif self.should_load_model and self.use_hf_checkpoint and self.peft_cls is None:
            hf_model_path = get_hf_model_checkpoint_path(local_path)
            if self.vanilla_bridge:
                self.bridge.load_weights(self.model, hf_model_path)
            else:
                self.bridge.load_hf_weights(self.model, hf_model_path)
            log_with_rank(f"Loaded HF model checkpoint from {hf_model_path} with bridge", rank=self.rank, logger=logger)
        # Load PEFT adapter checkpoint if available
        if self.should_load_model and self.peft_cls is not None:
            adapter_ckpt_path = os.path.join(local_path, "adapter_checkpoint")
            if os.path.exists(adapter_ckpt_path):
                from verl.utils.megatron_peft_utils import load_adapter_checkpoint

                # TODO: a better format for adapter checkpoint, waiting megatron-bridge support

                load_adapter_checkpoint(
                    self.model,
                    adapter_ckpt_path,
                )
                log_with_rank(
                    f"Loaded adapter checkpoint from {adapter_ckpt_path}",
                    rank=self.rank,
                    logger=logger,
                )
            else:
                log_with_rank(
                    f"PEFT config is set but no adapter checkpoint found at {adapter_ckpt_path}",
                    rank=self.rank,
                    logger=logger,
                )

        if self.should_load_optimizer:
            assert "optimizer" in state_dict, (
                f"Optimizer state dict not found in {state_dict.keys()}. Please check the checkpoint file {local_path}."
            )
            optimizer_state_dict = state_dict["optimizer"]
            self.optimizer.load_state_dict(optimizer_state_dict)
            log_with_rank(f"Loaded optimizer checkpoint from {local_path}", rank=self.rank, logger=logger)
            if self.use_checkpoint_opt_param_scheduler:
                assert "lr_scheduler" in state_dict, (
                    f"LR scheduler state dict not found in {state_dict.keys()}. Please check the checkpoint file "
                    f"{local_path}."
                )
                lr_scheduler_state_dict = state_dict["lr_scheduler"]
                if self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                    log_with_rank(f"Loaded LR scheduler checkpoint from {local_path}", rank=self.rank, logger=logger)

        if self.should_load_extra:
            assert "rng_state" in state_dict, (
                f"RNG state dict not found in {state_dict.keys()}. Please check the checkpoint file {local_path}."
            )
            rng_state = state_dict["rng_state"]
            self.load_rng_states(rng_state)
            log_with_rank(f"Loaded RNG states from {local_path}", rank=self.rank, logger=logger)

        if del_local_after_load:
            try:
                os.remove(local_path) if is_non_local(local_path) else None
            except Exception as e:
                log_with_rank(
                    f"remove local resume ckpt file after loading failed, exception {e} will be ignored",
                    rank=self.rank,
                    logger=logger,
                )

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        # record the previous global step
        self.previous_global_step = global_step

        if not self.checkpoint_config.async_save:
            self.ensure_checkpoint_capacity(max_ckpt_to_keep)

        local_path = local_mkdir_safe(local_path)
        dist_checkpoint_path = get_dist_checkpoint_path(local_path)

        # Note that model weights, optimizer states, and extra states are generated
        # together in a state dict, we save them in one time
        if self.use_dist_checkpointing:
            # Generate state dict for saving
            state_dict = self.generate_state_dict(
                self.should_save_model, self.should_save_optimizer, self.should_save_extra
            )
            log_with_rank(f"Generated state dict for saving: {state_dict.keys()}", rank=self.rank, logger=logger)
            for vpp_rank, model in enumerate(self.model):
                if len(self.model) > 1:
                    model_i_keys = state_dict[f"model{vpp_rank}"].keys()
                    log_with_rank(f"Generated state dict for saving: {model_i_keys}", rank=self.rank, logger=logger)
                else:
                    log_with_rank(
                        f"Generated state dict for saving: {state_dict['model'].keys()}", rank=self.rank, logger=logger
                    )
            # Start Async save if enabled
            async_save_request = save_dist_checkpointing(
                sharded_state_dict=state_dict,
                ckpt_path=dist_checkpoint_path,
                async_save=self.checkpoint_config.async_save,
            )

            # Synchronize all async save requests
            if not self.checkpoint_config.async_save:
                assert async_save_request is None, "Async save request should be None when not using async save."
                torch.distributed.barrier()
        else:
            assert self.use_hf_checkpoint, "When not using distributed checkpointing, use_hf_checkpoint should be True."
            # Generate optimizer and exra state dicts
            state_dict = self.generate_state_dict(
                generate_model=False,
                generate_optimizer=self.should_save_optimizer,
                generate_extra=self.should_save_extra,
            )
            # Save optimizer and extra states to local path
            # Start Async save if enabled
            async_save_request = save_dist_checkpointing(
                sharded_state_dict=state_dict,
                ckpt_path=dist_checkpoint_path,
                async_save=self.checkpoint_config.async_save,
            )

            # Synchronize all async save requests
            if not self.checkpoint_config.async_save:
                assert async_save_request is None, "Async save request should be None when not using async save."
                torch.distributed.barrier()

        if self.should_save_model:
            # Save adapter-only checkpoint if PEFT is enabled
            if self.peft_cls is not None:
                from verl.utils.megatron_peft_utils import save_adapter_checkpoint

                adapter_ckpt_path = os.path.join(local_path, "adapter_checkpoint")

                # Save adapter weights only (much smaller than full model)
                save_adapter_checkpoint(
                    self.model,
                    adapter_ckpt_path,
                    self.rank,
                )

                log_with_rank(
                    f"Saved adapter-only checkpoint to {adapter_ckpt_path}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
            elif self.use_hf_checkpoint:
                # Use mbridge to save HF model checkpoint
                log_with_rank(f"Saving HF model checkpoint to {local_path} with bridge", rank=self.rank, logger=logger)
                hf_ckpt_path = get_hf_model_checkpoint_path(local_path)
                if self.vanilla_bridge:
                    self.bridge.save_weights(
                        self.model, hf_ckpt_path, distributed_filesystem=True, memory_efficient=True
                    )
                else:
                    self.bridge.save_hf_weights(self.model, hf_ckpt_path)

                log_with_rank(f"Saved bridge checkpoint to {hf_ckpt_path}", rank=self.rank, logger=logger)

            # Only rank 0 saves the hf config and tokenizer to huggingface path
            # No matter whether we save hf model or not
            if self.rank == 0:
                # Save tokenizer
                hf_config_tokenizer_path = get_hf_model_checkpoint_path(local_path)
                if self.processing_class is not None:
                    self.processing_class.save_pretrained(hf_config_tokenizer_path)
                # Save huggingface config
                self.hf_config.save_pretrained(hf_config_tokenizer_path)
                if hasattr(self.hf_config, "name_or_path") and self.hf_config.name_or_path:
                    try:
                        generation_config = GenerationConfig.from_pretrained(self.hf_config.name_or_path)
                        generation_config.save_pretrained(hf_config_tokenizer_path)
                    except Exception:
                        # if the generation config isn't available, we don't save it
                        pass
                log_with_rank(
                    f"Saved Huggingface config and tokenizer to {hf_config_tokenizer_path}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )

        if self.should_save_extra:
            if self.rank == 0:
                # Save transformer config
                print(self.transformer_config)
                bypass_keys = [
                    "finalize_model_grads_func",
                    "grad_scale_func",
                    "no_sync_func",
                    "grad_sync_func",
                    "param_sync_func",
                    "generation_config",
                    "_pg_collection",
                ]
                backup = {}
                for k in bypass_keys:
                    if hasattr(self.transformer_config, k):
                        backup[k] = getattr(self.transformer_config, k, None)
                        delattr(self.transformer_config, k)
                transformer_config_dict = asdict(self.transformer_config)
                for k in backup:
                    setattr(self.transformer_config, k, backup[k])
                to_convert_types = {torch.dtype: str, AttnBackend: str}
                ignore_types = [Callable]
                pop_keys = []
                for key, value in transformer_config_dict.items():
                    if type(value) in to_convert_types:
                        transformer_config_dict[key] = to_convert_types[type(value)](value)
                    if type(value) in ignore_types:
                        pop_keys.append(key)
                    if callable(value):
                        pop_keys.append(key)
                for key in pop_keys:
                    transformer_config_dict.pop(key)
                transformer_config_path = get_transformer_config_checkpoint_path(local_path)
                with open(transformer_config_path, "w") as f:
                    json.dump(transformer_config_dict, f, indent=2)

        if self.should_save_hf_model and not self.use_hf_checkpoint:
            # wait for everyone to dump to local
            if self.bridge is not None:
                hf_model_ckpt_path = get_hf_model_checkpoint_path(local_path)
                if self.vanilla_bridge:
                    self.bridge.save_weights(
                        self.model, hf_model_ckpt_path, distributed_filesystem=True, memory_efficient=True
                    )
                else:
                    self.bridge.save_hf_weights(self.model, hf_model_ckpt_path)
            else:
                state_dict = self.weight_saver(
                    self.model,
                    self.hf_config,
                    dtype=self.param_dtype,
                    is_value_model=self.is_value_model,
                    tie_word_embeddings=self.share_embeddings_and_output_weights,
                )

                torch.distributed.barrier()
                if self.rank == 0:
                    hf_model_ckpt_path = get_hf_model_checkpoint_path(local_path)
                    import warnings

                    from accelerate import init_empty_weights

                    with init_empty_weights(), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if "mistral7b-rm" in self.config.model.path:
                            from transformers import MistralForSequenceClassification

                            model = MistralForSequenceClassification.from_pretrained(
                                self.config.model.path
                            )  # use score head instead of lm_head
                            state_dict["score.weight"] = state_dict["score.weight"]
                        else:
                            from transformers import AutoModelForCausalLM

                            model = AutoModelForCausalLM.from_pretrained(self.config.model.path, torch_dtype="auto")
                    model.save_pretrained(hf_model_ckpt_path, state_dict=state_dict)
                    log_with_rank(
                        f"Saved Huggingface config and tokenizer to {hf_model_ckpt_path}",
                        rank=self.rank,
                        logger=logger,
                        log_only_rank_0=True,
                    )

                    if hdfs_path is not None:
                        log_with_rank(
                            f"Uploading checkpoint to {hdfs_path}", rank=self.rank, logger=logger, log_only_rank_0=True
                        )
                        from verl.utils import hdfs_io

                        hdfs_io.makedirs(hdfs_path, exist_ok=True)
                        hdfs_io.copy(src=hf_model_ckpt_path, dst=hdfs_path, dirs_exist_ok=True)
                        log_with_rank(
                            f"HDFS checkpoint uploaded to {hdfs_path}",
                            rank=self.rank,
                            logger=logger,
                            log_only_rank_0=True,
                        )

        def finalize_save_fn():
            # Rank 0 uploads checkpoint to HDFS if hdfs_path is provided
            log_with_rank(
                f"Dist checkpointing save completed for {dist_checkpoint_path}", rank=self.rank, logger=logger
            )
            if self.rank == 0:
                if hdfs_path is not None:
                    log_with_rank(f"Uploading checkpoint to {hdfs_path}", rank=self.rank, logger=logger)
                    from verl.utils import hdfs_io

                    hdfs_io.makedirs(hdfs_path, exist_ok=True)
                    hdfs_io.copy(src=dist_checkpoint_path, dst=hdfs_path, dirs_exist_ok=True)
                    hdfs_io.copy(src=hf_config_tokenizer_path, dst=hdfs_path, dirs_exist_ok=True)

            # update latest_checkpointed_iteration.txt when async_save is True
            if self.checkpoint_config.async_save and self.rank == 0:
                log_with_rank(
                    f"Update latest_checkpointed_iteration.txt to step {global_step}",
                    rank=self.rank,
                    logger=logger,
                )
                local_latest_checkpointed_iteration = os.path.join(
                    os.path.dirname(os.path.dirname(local_path)), "latest_checkpointed_iteration.txt"
                )
                with open(local_latest_checkpointed_iteration, "w") as f:
                    f.write(str(global_step))

            self.register_checkpoint(local_path, max_ckpt_to_keep)

        if self.checkpoint_config.async_save:
            assert async_save_request is not None, "Async save request should not be None when using async save."
            async_save_request.add_finalize_fn(finalize_save_fn)
            from megatron.core.dist_checkpointing.strategies.base import async_calls

            async_calls.schedule_async_request(async_save_request)
        else:
            finalize_save_fn()
