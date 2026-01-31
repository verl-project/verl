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
import shutil
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.distributed
import torch.distributed.checkpoint as dcp
from accelerate import init_empty_weights
from omegaconf import DictConfig
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin
from transformers.dynamic_module_utils import custom_object_save

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, is_non_local, local_mkdir_safe
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_full_state_dict, get_fsdp_state_ctx
from verl.utils.logger import log_with_rank

from .checkpoint_manager import BaseCheckpointManager

# Setup logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@dataclass
class FSDPConfig:
    """Configuration for FSDP checkpointing.

    Args:
        FSDP_version (int): Version of FSDP being used.
        world_size (int): Number of processes in the distributed training setup.
    """

    FSDP_version: int
    world_size: int


APP_STATE_KEY = "fsdp_app_state"
EXTRA_STATE_KEY = "extra_state"


class _FSDPAppState(Stateful):
    """Wraps model/optimizer state handling for DCP saves/loads."""

    def __init__(
        self,
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer],
        include_model: bool,
        include_optimizer: bool,
        state_dict_cfg: ShardedStateDictConfig | None,
        optim_cfg: ShardedOptimStateDictConfig | None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.include_model = include_model
        self.include_optimizer = include_optimizer
        self.state_dict_cfg = state_dict_cfg
        self.optim_cfg = optim_cfg

    def _fsdp_state_ctx(self):
        return get_fsdp_state_ctx(
            self.model,
            StateDictType.SHARDED_STATE_DICT,
            self.state_dict_cfg,
            self.optim_cfg,
        )

    def state_dict(self):
        if not (self.include_model or self.include_optimizer):
            return {}
        with self._fsdp_state_ctx():
            model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        state = {}
        if self.include_model:
            state["model"] = model_state_dict
        if self.include_optimizer and optimizer_state_dict is not None:
            state["optimizer"] = optimizer_state_dict
        return state

    def load_state_dict(self, state_dict):
        if not state_dict:
            return
        load_kwargs = {}
        if self.include_model and "model" in state_dict:
            load_kwargs["model_state_dict"] = state_dict["model"]
        if self.include_optimizer and "optimizer" in state_dict:
            load_kwargs["optim_state_dict"] = state_dict["optimizer"]
        if not load_kwargs:
            return
        with self._fsdp_state_ctx():
            set_state_dict(
                self.model,
                self.optimizer,
                **load_kwargs,
            )


class _ExtraState(Stateful):
    """Handles lr_scheduler and RNG state via DCP."""

    def __init__(self, manager: "FSDPCheckpointManager"):
        self.manager = manager

    def state_dict(self):
        lr_scheduler_state = None
        if self.manager.lr_scheduler is not None:
            lr_scheduler_state = self.manager.lr_scheduler.state_dict()
        return {
            "lr_scheduler": lr_scheduler_state,
            "rng": self.manager.get_rng_state(),
        }

    def load_state_dict(self, state_dict):
        if not state_dict:
            return
        lr_scheduler_state = state_dict.get("lr_scheduler")
        if lr_scheduler_state is not None and self.manager.lr_scheduler is not None:
            self.manager.lr_scheduler.load_state_dict(lr_scheduler_state)
            log_with_rank(
                "Loaded lr_scheduler via DCP extras",
                rank=self.manager.rank,
                logger=logger,
            )
        rng_state = state_dict.get("rng")
        if rng_state is not None:
            self.manager.load_rng_state(rng_state)
            log_with_rank(
                "Loaded RNG state via DCP extras",
                rank=self.manager.rank,
                logger=logger,
            )


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    Manage FSDP checkpointing in SPMD training.

    - Saves/loads per-rank sharded model & optimizer states
    - Persists full lr_scheduler and RNG state
    - Stores HF tokenizer/processor and model/config for unified restore

    Args:
        model (FSDP): Wrapped model instance.
        optimizer (Optimizer): Training optimizer.
        lr_scheduler (LRScheduler): Learning-rate scheduler.
        processing_class (PreTrainedTokenizer or ProcessorMixin, optional):
            Pre-/post-processing artifact handler.
        checkpoint_contents DictConfig: Configuration for checkpoint contents.
            - 'load': Components to load; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
            - 'save': Components to save; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig = None,
        **kwargs,
    ):
        if processing_class is None and "tokenizer" in kwargs:
            warnings.warn(
                "`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2
            )
            processing_class = kwargs.pop("tokenizer")

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_config=checkpoint_config,
        )
        self._async_checkpoint_future = None
        self._async_checkpoint_start_time = None

    def _wait_for_pending_async_save(self):
        """Block until the outstanding async checkpoint completes."""
        if self._async_checkpoint_future is None:
            return
        self._async_checkpoint_future.result()
        self._async_checkpoint_future = None
        self._async_checkpoint_start_time = None
        if hasattr(self, "_async_checkpoint_path"):
            delattr(self, "_async_checkpoint_path")

    def _spawn_async_completion_logger(self, future, checkpoint_path, start_time):
        """Spawn a daemon thread that logs when async save finishes."""

        def _worker():
            try:
                future.result()
                duration = time.time() - start_time
                log_with_rank(
                    f"Async DCP save finished in {duration:.2f}s at {checkpoint_path}",
                    rank=self.rank,
                    logger=logger,
                )
            except Exception as e:
                log_with_rank(
                    f"Async DCP save failed for {checkpoint_path}: {e}",
                    rank=self.rank,
                    logger=logger,
                    level=logging.ERROR,
                )

        threading.Thread(target=_worker, daemon=True).start()

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        """
        Load an FSDP checkpoint for this rank using torch.distributed.checkpoint (DCP).

        Args:
            local_path: Directory with checkpoint files (local or remote via copy_to_local).
            hdfs_path: Unused (for API compatibility).
            del_local_after_load: Remove local files after loading.
        """
        if local_path is None:
            return

        # check if the checkpoint_load_contents is valid
        if self.should_load_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.load includes ['model']"
        if self.should_load_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.load includes ['optimizer']"
            )

        checkpoint_path = local_path.rstrip("/")
        copied_local_path = None
        if is_non_local(checkpoint_path):
            checkpoint_path = copy_to_local(checkpoint_path)
            copied_local_path = checkpoint_path

        state_dict_cfg = (
            ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_model
            else None
        )
        optim_cfg = (
            ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_optimizer
            else None
        )

        state_dict_entries = {}
        if self.should_load_model or self.should_load_optimizer:
            state_dict_entries[APP_STATE_KEY] = _FSDPAppState(
                model=self.model,
                optimizer=self.optimizer,
                include_model=self.should_load_model,
                include_optimizer=self.should_load_optimizer,
                state_dict_cfg=state_dict_cfg,
                optim_cfg=optim_cfg,
            )
        if self.should_load_extra:
            state_dict_entries[EXTRA_STATE_KEY] = _ExtraState(self)

        if state_dict_entries:
            dcp.load(state_dict=state_dict_entries, checkpoint_id=checkpoint_path)
            log_with_rank(
                f"Loaded checkpoint via DCP from {checkpoint_path}",
                rank=self.rank,
                logger=logger,
            )

        if self.rank == 0 and del_local_after_load and copied_local_path:
            try:
                if os.path.isdir(copied_local_path):
                    shutil.rmtree(copied_local_path, ignore_errors=True)
                else:
                    os.remove(copied_local_path)
            except Exception as e:
                log_with_rank(
                    f"remove local resume ckpt directory after loading failed, exception {e} will be ignored",
                    rank=self.rank,
                    logger=logger,
                )

        # wait for everyone to load checkpoints
        torch.distributed.barrier()

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        """
        Save an FSDP checkpoint for this rank.

        Writes:
          - model & optimizer shard files
          - extra state dict (scheduler + RNG)
          - HF tokenizer/processor and model/config on rank 0
          - optional full HF model under 'huggingface/' if requested

        Rotates old checkpoints, keeping at most `max_ckpt_to_keep`.

        Args:
            local_path: Target directory for checkpoint files.
            hdfs_path: Unused (for API compatibility).
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent checkpoints to retain.
        """
        if local_path is None:
            return

        # record the previous global step
        self.previous_global_step = global_step

        if self.rank == 0:
            self.ensure_checkpoint_capacity(max_ckpt_to_keep)

        local_path = local_mkdir_safe(local_path)
        torch.distributed.barrier()

        # check if the checkpoint_save_contents is valid
        if self.should_save_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.save includes ['model']"
        if self.should_save_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.save includes ['optimizer']"
            )

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)

        use_async_save = self.checkpoint_config.get("async_save", False) if self.checkpoint_config else False

        if use_async_save:
            self._wait_for_pending_async_save()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state_dict_entries = {}
            if self.should_save_model or self.should_save_optimizer:
                state_dict_entries[APP_STATE_KEY] = _FSDPAppState(
                    model=self.model,
                    optimizer=self.optimizer,
                    include_model=self.should_save_model,
                    include_optimizer=self.should_save_optimizer,
                    state_dict_cfg=state_dict_cfg,
                    optim_cfg=optim_cfg,
                )
            if self.should_save_extra:
                state_dict_entries[EXTRA_STATE_KEY] = _ExtraState(self)

            if state_dict_entries:
                if use_async_save:
                    self._async_checkpoint_start_time = time.time()
                    self._async_checkpoint_path = os.path.abspath(local_path)
                    storage_writer = FileSystemWriter(path=local_path)
                    self._async_checkpoint_future = dcp.async_save(
                        state_dict=state_dict_entries,
                        storage_writer=storage_writer,
                        checkpoint_id=local_path,
                    )
                    log_with_rank(
                        f"Started async DCP save to {os.path.abspath(local_path)}",
                        rank=self.rank,
                        logger=logger,
                    )
                    self._spawn_async_completion_logger(
                        self._async_checkpoint_future,
                        self._async_checkpoint_path,
                        self._async_checkpoint_start_time,
                    )
                else:
                    start_time = time.time()
                    dcp.save(state_dict=state_dict_entries, checkpoint_id=local_path)
                    duration = time.time() - start_time
                    log_with_rank(
                        f"Saved checkpoint via DCP to {os.path.abspath(local_path)} in {duration:.2f}s",
                        rank=self.rank,
                        logger=logger,
                    )

        if self.rank == 0:
            # Save HF tokenizer/processor and model config on rank 0 to huggingface/ directory, no matter whether
            # huggingface model is requested to be saved or not.

            if fsdp_version(self.model) == 1:
                unwrap_model = self.model._fsdp_wrapped_module
            else:
                unwrap_model = self.model

            hf_config_tokenizer_path = os.path.join(local_path, "huggingface")
            local_mkdir_safe(hf_config_tokenizer_path)
            model_config = unwrap_model.config
            generation_config = None
            if unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:
                try:
                    # Some model's name_or_path is empty if not initialized from pretrained,
                    # in this cases, we don't save generation config.
                    generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                    generation_config.save_pretrained(hf_config_tokenizer_path)
                except Exception:
                    # if the generation config isn't available, we don't save it
                    pass

            if hasattr(model_config, "auto_map") and None in model_config.auto_map:
                model_config.auto_map = {k: v for k, v in model_config.auto_map.items() if k is not None}

            model_config.save_pretrained(hf_config_tokenizer_path)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(hf_config_tokenizer_path)
            log_with_rank(
                f"Saved model config and tokenizer class to {os.path.abspath(hf_config_tokenizer_path)}",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )

            # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
            # loaded from the Hub.
            if hasattr(model_config, "auto_map"):
                custom_object_save(unwrap_model, hf_config_tokenizer_path, config=model_config)

            # Also save runtime FSDP config
            fsdp_config_path = os.path.join(local_path, "fsdp_config.json")
            fsdp_config = FSDPConfig(
                FSDP_version=fsdp_version(self.model),
                world_size=self.world_size,
            )
            with open(fsdp_config_path, "w") as f:
                json.dump(asdict(fsdp_config), f, indent=4)

        # wait for everyone to dump to local (only when not using async save)
        if not use_async_save:
            torch.distributed.barrier()

        if self.should_save_hf_model:
            # Only rank 0 will save hf model and,
            # offload to cpu to save LLMs which may be too large to fit in one GPU
            state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)

            if self.rank == 0:
                hf_local_path = os.path.join(local_path, "huggingface")
                os.makedirs(hf_local_path, exist_ok=True)

                if "ForTokenClassification" in model_config.architectures[0]:
                    from transformers import AutoModelForTokenClassification

                    auto_model_cls = AutoModelForTokenClassification
                elif "ForCausalLM" in model_config.architectures[0]:
                    from transformers import AutoModelForCausalLM

                    auto_model_cls = AutoModelForCausalLM
                elif "ForConditionalGeneration" in model_config.architectures[0]:
                    # Handle different transformers versions for Vision2Seq models
                    import transformers
                    from packaging import version

                    if version.parse(transformers.__version__) >= version.parse("4.54.0"):
                        # transformers >= 4.54.0 uses AutoModelForImageTextToText
                        from transformers import AutoModelForImageTextToText

                        auto_model_cls = AutoModelForImageTextToText
                    else:
                        # transformers < 4.54.0 uses AutoModelForVision2Seq
                        from transformers import AutoModelForVision2Seq

                        auto_model_cls = AutoModelForVision2Seq
                else:
                    raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

                with init_empty_weights():
                    save_model = auto_model_cls.from_config(model_config, torch_dtype=torch.bfloat16)
                save_model.to_empty(device="cpu")

                if save_model.can_generate():
                    if generation_config is not None:
                        save_model.generation_config = generation_config
                    else:
                        print(
                            f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found "
                            f"in, using a generation config created from the model config when saving hf_model."
                        )

                save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                log_with_rank(
                    f"Saved hf_model to {os.path.abspath(hf_local_path)}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
                del state_dict
                del save_model

            # wait for rank0 to dump hf_model to local
            torch.distributed.barrier()

        if self.rank == 0:
            self.register_checkpoint(local_path, max_ckpt_to_keep)
