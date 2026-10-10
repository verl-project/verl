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

import copy
import json
import logging
import os
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, Generator, Optional

import torch
import torch.distributed
from omegaconf import DictConfig
from safetensors.torch import save_file
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp._common_utils import clean_tensor_name
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin
from transformers.dynamic_module_utils import custom_object_save

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, is_non_local, local_mkdir_safe
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_state_ctx, normalize_peft_param_name
from verl.utils.logger import log_with_rank
from verl.utils.model import convert_weight_keys

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
        trust_remote_code: Whether to trust_remote_code when loading the model configuration
        per_tensor_param_fn: Callable with the engine ``get_per_tensor_param`` contract yielding the full
            HF-format weights streamed into the 'hf_model' export; required when 'hf_model' is saved. It is called
            collectively on every rank.
        hf_export_dtype: Optional dtype the forward computes parameters in (the mixed-precision param_dtype).
            fp32 parameters are exported in this dtype; buffers and parameters of modules excluded from mixed
            precision keep their dtype, as the forward uses them unconverted.
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig = None,
        trust_remote_code: bool = False,
        per_tensor_param_fn: Optional[
            Callable[[], tuple[Generator[tuple[str, torch.Tensor], None, None], Optional[dict]]]
        ] = None,
        hf_export_dtype: Optional[torch.dtype] = None,
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
        self.trust_remote_code = trust_remote_code
        if self.should_save_hf_model and per_tensor_param_fn is None:
            raise ValueError("Saving 'hf_model' requires `per_tensor_param_fn` to stream the full HF weights.")
        self.per_tensor_param_fn = per_tensor_param_fn
        self.hf_export_dtype = hf_export_dtype

    def _to_export_names(self, unwrap_model: torch.nn.Module, names) -> set[str]:
        """Map module FQNs to the HF names ``per_tensor_param_fn`` yields."""
        names = normalize_peft_param_name(dict.fromkeys(clean_tensor_name(name) for name in names))
        return set(convert_weight_keys(names, unwrap_model))

    def _get_dtype_preserved_names(self, unwrap_model: torch.nn.Module) -> set[str]:
        """Tensors the forward uses in their stored dtype: buffers and params of mixed-precision-ignored modules."""
        names = [name for name, _ in unwrap_model.named_buffers()]
        get_ignored_modules = getattr(unwrap_model, "get_ignore_modules_in_mixed_precision", None)
        ignored_classes = get_ignored_modules() if get_ignored_modules is not None else None
        if ignored_classes:
            for module_name, module in unwrap_model.named_modules():
                if isinstance(module, ignored_classes):
                    names.extend(f"{module_name}.{name}" for name, _ in module.named_parameters())
        return self._to_export_names(unwrap_model, names)

    def _get_tied_alias_names(self, unwrap_model: torch.nn.Module) -> set[str]:
        """Names of tied aliases (e.g. ``lm_head.weight``), which HF re-ties on load and must not be saved."""
        return self._to_export_names(unwrap_model, unwrap_model.get_expanded_tied_weights_keys(all_submodels=True))

    def _save_hf_weights_from_per_tensor_param(self, hf_local_path: str, max_shard_bytes: int = 5 * 1024**3):
        """Stream HF-format weights from ``per_tensor_param_fn`` into safetensors shards on rank 0.

        Host memory peaks at about one shard instead of the whole model, so models larger than host
        memory can be exported. fp32 parameters are narrowed to ``hf_export_dtype``.
        """
        per_tensor_param, _ = self.per_tensor_param_fn()
        unwrap_model = getattr(self.model, "_fsdp_wrapped_module", self.model)
        tied_aliases = self._get_tied_alias_names(unwrap_model) if self.rank == 0 else set()
        preserved_names = self._get_dtype_preserved_names(unwrap_model) if self.rank == 0 else set()

        shard, shard_bytes, total_size = {}, 0, 0
        tmp_shards: list[tuple[str, list[str]]] = []

        def flush_shard():
            nonlocal shard, shard_bytes
            tmp_path = os.path.join(hf_local_path, f"model-{len(tmp_shards) + 1:05d}.safetensors.tmp")
            save_file(shard, tmp_path, metadata={"format": "pt"})
            tmp_shards.append((tmp_path, list(shard)))
            shard, shard_bytes = {}, 0

        # Every rank must drain the generator since it issues collectives; only rank 0 writes.
        for name, tensor in per_tensor_param:
            if self.rank != 0 or name in tied_aliases:
                continue
            if self.hf_export_dtype is not None and tensor.dtype == torch.float32 and name not in preserved_names:
                tensor = tensor.to(self.hf_export_dtype)
            nbytes = tensor.numel() * tensor.element_size()
            if shard and shard_bytes + nbytes > max_shard_bytes:
                flush_shard()
            # Yielded tensors may be views of a buffer the generator overwrites later (EP broadcast),
            # so copy them out before advancing.
            shard[name] = tensor.detach().contiguous().to("cpu", copy=True)
            shard_bytes += nbytes
            total_size += nbytes

        if self.rank != 0:
            return
        if shard:
            flush_shard()

        weight_map = {}
        for i, (tmp_path, names) in enumerate(tmp_shards, start=1):
            file_name = f"model-{i:05d}-of-{len(tmp_shards):05d}.safetensors"
            os.replace(tmp_path, os.path.join(hf_local_path, file_name))
            weight_map.update(dict.fromkeys(names, file_name))
        with open(os.path.join(hf_local_path, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)
        log_with_rank(
            f"Saved hf_model ({len(weight_map)} tensors in {len(tmp_shards)} shards) "
            f"to {os.path.abspath(hf_local_path)}",
            rank=self.rank,
            logger=logger,
            log_only_rank_0=True,
        )

    def _get_lora_train_meta(self, unwrap_model):
        peft_config = getattr(unwrap_model, "peft_config", None)
        if not peft_config:
            return None
        if isinstance(peft_config, dict):
            peft_config = peft_config.get("default") or next(iter(peft_config.values()), None)
        if peft_config is None:
            return None

        lora_rank = int(getattr(peft_config, "r", 0) or 0)
        if lora_rank <= 0:
            return None

        lora_alpha = int(getattr(peft_config, "lora_alpha", lora_rank) or 0)
        task_type = getattr(peft_config, "task_type", None) or "CAUSAL_LM"
        if hasattr(task_type, "value"):
            task_type = task_type.value

        return {"r": lora_rank, "lora_alpha": lora_alpha, "task_type": str(task_type)}

    def _save_lora_train_meta(self, local_path: str, unwrap_model):
        lora_train_meta = self._get_lora_train_meta(unwrap_model)
        if lora_train_meta is None:
            return None

        lora_meta_path = os.path.join(local_path, "lora_train_meta.json")
        with open(lora_meta_path, "w", encoding="utf-8") as f:
            json.dump(lora_train_meta, f, ensure_ascii=False, indent=4)
        log_with_rank(
            f"Saved LoRA rank/alpha metadata to {os.path.abspath(lora_meta_path)}",
            rank=self.rank,
            logger=logger,
            log_only_rank_0=True,
        )
        return lora_meta_path

    def _has_lora(self) -> bool:
        unwrap = getattr(self.model, "_fsdp_wrapped_module", self.model)
        return hasattr(unwrap, "peft_config")

    def _backfill_optimizer_state(self, state_dict: dict) -> dict:
        """Fill in entries that a flattened DSD optimizer checkpoint does not contain.

        Engines that save through ``torch.distributed.checkpoint.state_dict`` with
        ``flatten_optimizer_state_dict=True`` (veomni's ``MultiOptimizer``) only persist
        params that already own optimizer state, i.e. params that received a gradient at
        least once. Params that never do -- the DeepSeek-V4 sparse-attention indexer for
        instance, whose forward only returns top-k indices -- are absent from the file,
        and torch's unflatten path turns each of their missing entries into an empty dict
        rather than skipping it, which then fails in ``Adam.__setstate__``. Backfilling
        from the freshly initialized optimizer mirrors what DCP's ``allow_partial_load``
        does: those params resume with default state.
        """
        if "state" in state_dict or "param_groups" in state_dict:
            # Plain optimizer state dict; torch already tolerates params without state.
            return state_dict

        current_state_dict = self.optimizer.state_dict()
        if "state" in current_state_dict or "param_groups" in current_state_dict:
            return state_dict

        missing_keys = [key for key in current_state_dict if key not in state_dict]
        if not missing_keys:
            return state_dict

        log_with_rank(
            f"{len(missing_keys)} optimizer state entries are absent from the checkpoint and keep their "
            f"initial value, e.g. {missing_keys[:5]}",
            rank=self.rank,
            logger=logger,
            log_only_rank_0=True,
        )
        return {**current_state_dict, **state_dict}

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        """
        Load an FSDP checkpoint for this rank.

        Downloads and loads:
          - model and optimizer shards
          - extra state dict (scheduler + RNG)

        Args:
            local_path: Directory with per-rank checkpoint files.
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

        # every rank download its own checkpoint
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
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            if self.should_load_model:
                remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_model_path = copy_to_local(remote_model_path)
                model_state_dict = torch.load(local_model_path, weights_only=False)
                if self.is_lora_only_state_dict(model_state_dict):
                    result = self.model.load_state_dict(model_state_dict, strict=False)
                    if result is not None and result.unexpected_keys:
                        raise ValueError(
                            f"Failed to load LoRA-only checkpoint: unexpected keys {result.unexpected_keys}. "
                            f"Ensure the model has the correct LoRA adapters configured."
                        )
                    log_with_rank(
                        f"Loaded LoRA-only checkpoint ({len(model_state_dict)} keys) from {remote_model_path}",
                        rank=self.rank,
                        logger=logger,
                    )
                else:
                    self.model.load_state_dict(model_state_dict)
                    log_with_rank(f"Loaded model from {remote_model_path}", rank=self.rank, logger=logger)

            if self.should_load_optimizer:
                remote_optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_optim_path = copy_to_local(remote_optim_path)
                optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
                optimizer_state_dict = self._backfill_optimizer_state(optimizer_state_dict)
                self.optimizer.load_state_dict(optimizer_state_dict)
                log_with_rank(f"Loaded optimizer from {remote_optim_path}", rank=self.rank, logger=logger)

        if self.should_load_extra:
            remote_extra_state_path = os.path.join(
                local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt"
            )
            local_extra_state_path = copy_to_local(remote_extra_state_path)
            extra_state_dict = torch.load(local_extra_state_path, weights_only=False)
            # recover random state
            if "rng" in extra_state_dict:
                # 'rng' may not exist for backward compatibility
                self.load_rng_state(extra_state_dict["rng"])
                log_with_rank(f"Loaded rng from {remote_extra_state_path}", rank=self.rank, logger=logger)

            lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]
            if lr_scheduler_state_dict is not None and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                log_with_rank(f"Loaded lr_scheduler from {remote_extra_state_path}", rank=self.rank, logger=logger)

        if self.rank == 0 and del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            except Exception as e:
                log_with_rank(
                    f"remove local resume ckpt file after loading failed, exception {e} will be ignored",
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

        # every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                if self.should_save_model:
                    model_state_dict = self.model.state_dict()
                    if self.should_save_lora_only and self._has_lora():
                        n_total = len(model_state_dict)
                        model_state_dict = {
                            k: v for k, v in model_state_dict.items() if "lora_" in k or ".adapter_" in k
                        }
                        if not model_state_dict:
                            raise ValueError(
                                f"save_lora_only is True and the model has a peft_config, "
                                f"but no LoRA/adapter parameters were found in the state dict. "
                                f"Total params checked: {n_total}."
                            )
                        lora_bytes = 0
                        for v in model_state_dict.values():
                            if hasattr(v, "numel") and hasattr(v, "element_size"):
                                lora_bytes += v.numel() * v.element_size()
                            elif hasattr(v, "local_shards"):
                                for shard in v.local_shards():
                                    lora_bytes += shard.tensor.numel() * shard.tensor.element_size()
                        lora_mib = lora_bytes / 1024**2
                        log_with_rank(
                            f"LoRA-only save: {len(model_state_dict)}/{n_total} params ({lora_mib:.1f} MiB)",
                            rank=self.rank,
                            logger=logger,
                            log_only_rank_0=True,
                        )
                    torch.save(model_state_dict, model_path)
                    log_with_rank(f"Saved model to {os.path.abspath(model_path)}", rank=self.rank, logger=logger)

                if self.should_save_optimizer:
                    optimizer_state_dict = self.optimizer.state_dict()
                    torch.save(optimizer_state_dict, optim_path)
                    log_with_rank(f"Saved optim to {os.path.abspath(optim_path)}", rank=self.rank, logger=logger)

                if self.should_save_extra:
                    lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                    extra_state_dict = {
                        "lr_scheduler": lr_scheduler_state_dict,
                        "rng": self.get_rng_state(),
                    }
                    torch.save(extra_state_dict, extra_path)
                    log_with_rank(f"Saved extra_state to {os.path.abspath(extra_path)}", rank=self.rank, logger=logger)

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

            if self.should_save_hf_model and self.hf_export_dtype is not None:
                # Record the exported weight dtype instead of the training dtype the model was built with.
                model_config = copy.deepcopy(model_config)
                for cfg in (model_config, *(getattr(model_config, k, None) for k in model_config.sub_configs)):
                    if cfg is not None:
                        cfg.dtype = self.hf_export_dtype

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
                # custom_object_save copies the source of type(obj).__module__, so it needs the base
                # model's module. For a PEFT model unwrap_model is the PeftModel wrapper, so unwrap it
                # first; get_base_model() is peft-only (a plain model lacks it and is passed through).
                save_obj = unwrap_model.get_base_model() if hasattr(unwrap_model, "get_base_model") else unwrap_model
                custom_object_save(save_obj, hf_config_tokenizer_path, config=model_config)

            # Also save runtime FSDP config
            fsdp_config_path = os.path.join(local_path, "fsdp_config.json")
            fsdp_config = FSDPConfig(
                FSDP_version=fsdp_version(self.model),
                world_size=self.world_size,
            )
            with open(fsdp_config_path, "w") as f:
                json.dump(asdict(fsdp_config), f, indent=4)
            self._save_lora_train_meta(local_path, unwrap_model)

        # wait for everyone to dump to local
        torch.distributed.barrier()

        if self.should_save_hf_model:
            self._save_hf_weights_from_per_tensor_param(os.path.join(local_path, "huggingface"))
            # wait for rank0 to dump hf_model to local
            torch.distributed.barrier()

        if self.rank == 0:
            self.register_checkpoint(local_path, max_ckpt_to_keep)
