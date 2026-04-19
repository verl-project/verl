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

import inspect
import json
import logging
import os
import random
from collections.abc import Callable
from dataclasses import asdict

import megatron.core
import numpy as np
import torch
import torch.distributed
from megatron.core import dist_checkpointing, mpu, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.transformer.enums import AttnBackend
from packaging import version
from transformers import GenerationConfig

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
mcore_ge_014 = version.parse(megatron.core.__version__) >= version.parse("0.14.0")
if not mcore_ge_014:
    logger.warning(
        "Detected megatron.core %s, recommend upgrading to >= 0.14.0 for better checkpoint compatibility",
        megatron.core.__version__,
    )


class MegatronCheckpointManager(BaseCheckpointManager):
    """Checkpoint manager for Megatron-LM distributed training.

    Two model-checkpoint backends are supported, controlled by ``use_mbridge``:

    * **mbridge (default)** -- saves / loads model weights in HuggingFace format
      via megatron-bridge.  Both ``"model"`` and ``"hf_model"`` in
      ``save_contents`` produce the same HF checkpoint (saved only once).

    * **dist_checkpoint** -- saves model weights using Megatron's native
      ``dist_checkpointing`` (sharded across ranks).  ``"hf_model"`` in
      ``save_contents`` is **not** supported with this backend (raises error).

    Optimizer, LR-scheduler, and RNG states always go through
    ``dist_checkpointing`` regardless of the model backend.

    Args:
        model: The Megatron model instance to checkpoint.
        optimizer: The optimizer instance.
        lr_scheduler: The learning rate scheduler instance.
        use_mbridge: If ``True`` (default), model weights are saved/loaded via
            megatron-bridge in HuggingFace format.  Requires ``bridge`` to be
            provided.  If ``False``, ``dist_checkpointing`` is used for model
            weights.
        use_dist_checkpointing: Legacy alias -- when provided, it sets
            ``use_mbridge = not use_dist_checkpointing``.
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
        use_dist_checkpointing: bool | None = None,
        use_mbridge: bool = True,
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
        self.is_value_model = self.role in ("reward", "critic")
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

        # --- Resolve backend ---------------------------------------------------
        # Legacy callers may still pass ``use_dist_checkpointing``; translate it.
        if use_dist_checkpointing is not None:
            use_mbridge = not use_dist_checkpointing

        if use_mbridge and self.bridge is None:
            raise ValueError(
                "MegatronCheckpointManager: use_mbridge=True requires a bridge "
                "instance. Either pass `bridge=...` or set `use_mbridge=False` "
                "to use dist_checkpointing for model weights."
            )

        self.use_mbridge = use_mbridge
        # Aliases kept so the rest of the codebase can read them without change.
        self.use_dist_checkpointing = not self.use_mbridge
        self.use_hf_checkpoint = self.use_mbridge

        # --- Validate & resolve save_contents vs backend ----------------------
        # Per the RFC:
        #   mbridge   + model    → save HF model via bridge
        #   mbridge   + hf_model → same as model (deduplicated, save once)
        #   dist_ckpt + model    → save via mcore dist_checkpointing
        #   dist_ckpt + hf_model → ERROR (not supported)
        #   both backends False  → ERROR
        if not self.use_mbridge and not self.use_dist_checkpointing:
            raise ValueError(
                "MegatronCheckpointManager: at least one model-weight backend "
                "must be enabled (use_mbridge or dist_checkpointing)."
            )

        if self.should_save_hf_model and not self.use_mbridge:
            raise ValueError(
                "save_contents contains 'hf_model' but mbridge is disabled. "
                "'hf_model' is only supported with mbridge. Either enable "
                "mbridge or remove 'hf_model' from save_contents."
            )

        # Whether we actually need to write model weights via each backend.
        # When mbridge is active, both 'model' and 'hf_model' resolve to a
        # single HF save (deduplicated).
        self._save_model_via_mbridge = self.use_mbridge and (self.should_save_model or self.should_save_hf_model)
        self._save_model_via_dist_ckpt = self.use_dist_checkpointing and self.should_save_model

        # PEFT adapter shards always live in dist_checkpoint even with mbridge,
        # because bridge handles only the base model.
        self._save_peft_adapters_in_dist_ckpt = self._save_model_via_mbridge and self.peft_cls is not None

        self._load_model_via_mbridge = self.use_mbridge and self.should_load_model and self.peft_cls is None
        # PEFT adapters always live in the dist_checkpoint directory.
        self._load_model_via_dist_ckpt = self.should_load_model and (not self.use_mbridge or self.peft_cls is not None)

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

    # -- Sharded state dict builders -------------------------------------------
    # Each builder produces an independent piece of the dist_checkpoint state
    # dict.  Callers compose exactly the pieces they need rather than building
    # one monolithic dict every time.

    def _build_model_sharded_state_dict(self, metadata: dict) -> dict:
        """Build the model's sharded state dict for all VPP ranks.

        This is used both for persisting model weights (dist_ckpt backend) and
        as metadata input for the optimizer's ``sharded_state_dict()`` method
        (every Megatron optimizer takes ``model_sharded_state_dict`` as its
        first argument but only reads it — the model tensors are not included
        in the optimizer's output).
        """
        model_sharded_state_dict = {}
        model_metadata = dict(metadata)
        model_metadata["dp_cp_group"] = mpu.get_data_parallel_group(with_context_parallel=True)
        for vpp_rank, model in enumerate(self.model):
            if len(self.model) > 1:
                mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
                key = f"model{vpp_rank}"
            else:
                key = "model"
            if hasattr(model, "module"):
                model = model.module
            model_sharded_state_dict[key] = model.sharded_state_dict(metadata=model_metadata)
        return model_sharded_state_dict

    def _build_optimizer_state_dict(
        self,
        model_sharded_state_dict: dict,
        metadata: dict,
        is_loading: bool = False,
    ) -> dict:
        """Build the optimizer (+ LR scheduler) sharded state dict.

        ``model_sharded_state_dict`` is required because Megatron's optimizer
        ``sharded_state_dict()`` uses the model's sharding layout to map
        optimizer states to parameter shards.  It is consumed read-only and
        its entries do **not** appear in the returned dict.
        """
        torch.distributed.barrier()
        sharded_state_dict_kwargs = {"is_loading": is_loading}
        if metadata is not None and mcore_ge_014:
            sharded_state_dict_kwargs["metadata"] = metadata
        state_dict = {}
        state_dict["optimizer"] = self.optimizer.sharded_state_dict(
            model_sharded_state_dict, **sharded_state_dict_kwargs
        )
        if self.lr_scheduler is not None:
            state_dict["lr_scheduler"] = self.lr_scheduler.state_dict()
        return state_dict

    def _build_extra_state_dict(self) -> dict:
        """Build the extra state dict (RNG states)."""
        torch.distributed.barrier()
        return {"rng_state": self.get_rng_state()}

    def _build_sharded_state_dict_metadata(self) -> dict:
        """Builds metadata used for sharded_state_dict versioning.


        The whole content metadata is passed to ``sharded_state_dict`` model and optimizer methods
        and therefore affects only the logic behind sharded_state_dict creation.
        The content metadata should be minimalistic, ideally flat (or with a single nesting level)
        and with semantically meaningful flag names (e.g. `distrib_optim_sharding_type`).
        In particular, a simple integer (or SemVer) versioning flag (e.g. `metadata['version'] = 3.4`)
        is discouraged, because the metadata serves for all models and optimizers and it's practically
        impossible to enforce a linearly increasing versioning for this whole space.
        """
        metadata: dict = {}

        if not mcore_ge_014:
            # For backward compatibility with Megatron core < v0.14.0
            if self.use_distributed_optimizer:
                metadata["distrib_optim_sharding_type"] = "fully_sharded_model_space"
            return metadata

        if self.use_distributed_optimizer:
            megatron_config = getattr(self.config, self.role, self.config).megatron
            dist_ckpt_optim_fully_reshardable = megatron_config.dist_ckpt_optim_fully_reshardable
            distrib_optim_fully_reshardable_mem_efficient = (
                megatron_config.distrib_optim_fully_reshardable_mem_efficient
            )
            if dist_ckpt_optim_fully_reshardable:
                metadata["distrib_optim_sharding_type"] = "fully_reshardable"
                metadata["distrib_optim_fully_reshardable_mem_efficient"] = (
                    distrib_optim_fully_reshardable_mem_efficient
                )
            else:
                metadata["distrib_optim_sharding_type"] = "dp_reshardable"

        metadata["singleton_local_shards"] = False
        metadata["chained_optim_avoid_prefix"] = True
        return metadata

    @staticmethod
    def _has_checkpoint_files(path: str) -> bool:
        return os.path.isdir(path) and any(os.scandir(path))

    def _raise_for_unsupported_peft_checkpoint_layout(self, local_path: str, dist_checkpoint_path: str):
        if self.peft_cls is None or not self.should_load_model or self._has_checkpoint_files(dist_checkpoint_path):
            return

        legacy_adapter_ckpt_path = os.path.join(local_path, "adapter_checkpoint")
        hf_adapter_ckpt_path = os.path.join(local_path, "huggingface", "adapter")

        if os.path.isdir(legacy_adapter_ckpt_path):
            raise RuntimeError(
                f"Found legacy PEFT checkpoint at {legacy_adapter_ckpt_path}, but checkpoint resume now expects "
                f"adapter weights in {dist_checkpoint_path}. Resave/convert the checkpoint or load the adapter via "
                "`lora.adapter_path`."
            )

        if os.path.isfile(os.path.join(hf_adapter_ckpt_path, "adapter_config.json")):
            raise RuntimeError(
                f"Found exported HF PEFT adapter at {hf_adapter_ckpt_path}, but `load_checkpoint()` resumes from "
                f"{dist_checkpoint_path}. HF adapter exports are not used for trainer resume; keep the distributed "
                "checkpoint or load the adapter separately via `lora.adapter_path`."
            )

    def _maybe_filter_peft_state_dict(self, state_dict: dict):
        if self.peft_cls is None:
            return state_dict

        from megatron.bridge.training.checkpointing import apply_peft_adapter_filter_to_state_dict

        return apply_peft_adapter_filter_to_state_dict(state_dict, self.peft_cls)

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

        try:
            import transformer_engine

            torch.serialization.add_safe_globals([torch.optim.AdamW])
            torch.serialization.add_safe_globals([transformer_engine.pytorch.optimizers.fused_adam.FusedAdam])
        except Exception:
            pass

        dist_checkpoint_path = get_dist_checkpoint_path(local_path)
        self._raise_for_unsupported_peft_checkpoint_layout(local_path, dist_checkpoint_path)

        load_content_metadata = getattr(dist_checkpointing, "load_content_metadata", None)
        if load_content_metadata is None:
            metadata = None
        else:
            metadata = load_content_metadata(checkpoint_dir=dist_checkpoint_path)
        if metadata is None:
            if self.use_distributed_optimizer:
                metadata = {"distrib_optim_sharding_type": "fully_sharded_model_space"}
            else:
                metadata = self._build_sharded_state_dict_metadata()

        # ── Compose the sharded state dict from individual pieces ─────────────
        # Same principle as save: optimizer and extra always go through
        # dist_checkpointing.  Model sharded state dict is only built when
        # needed (optimizer metadata input or loading model weights via dist_ckpt).
        sharded_state_dict = {}

        model_sharded_state_dict = None
        if self.should_load_optimizer or self._load_model_via_dist_ckpt:
            model_sharded_state_dict = self._build_model_sharded_state_dict(metadata)

        if self.should_load_optimizer:
            sharded_state_dict.update(
                self._build_optimizer_state_dict(model_sharded_state_dict, metadata, is_loading=True)
            )

        if self.should_load_extra:
            sharded_state_dict.update(self._build_extra_state_dict())

        if self._load_model_via_dist_ckpt:
            sharded_state_dict.update(model_sharded_state_dict)

        sharded_state_dict = self._maybe_filter_peft_state_dict(sharded_state_dict)
        log_with_rank(f"Generated state dict for loading: {sharded_state_dict.keys()}", rank=self.rank, logger=logger)

        state_dict = load_dist_checkpointing(
            sharded_state_dict=sharded_state_dict,
            ckpt_dir=dist_checkpoint_path,
        )

        # ── Load model weights ────────────────────────────────────────────────
        if self._load_model_via_dist_ckpt:
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
                self.model[vpp_rank].load_state_dict(model_state_dict, strict=self.peft_cls is None)
            if self.peft_cls is not None:
                log_with_rank(
                    f"Loaded PEFT adapter checkpoint from {dist_checkpoint_path}", rank=self.rank, logger=logger
                )
            else:
                log_with_rank(f"Loaded sharded model checkpoint from {local_path}", rank=self.rank, logger=logger)

        elif self._load_model_via_mbridge:
            hf_model_path = get_hf_model_checkpoint_path(local_path)
            self._load_model_via_bridge(hf_model_path)
            log_with_rank(f"Loaded HF model checkpoint from {hf_model_path} with bridge", rank=self.rank, logger=logger)

        # ── Load optimizer / LR scheduler ─────────────────────────────────────
        if self.should_load_optimizer:
            assert "optimizer" in state_dict, (
                f"Optimizer state dict not found in {state_dict.keys()}. Please check the checkpoint file {local_path}."
            )
            self.optimizer.load_state_dict(state_dict["optimizer"])
            log_with_rank(f"Loaded optimizer checkpoint from {local_path}", rank=self.rank, logger=logger)
            if self.use_checkpoint_opt_param_scheduler:
                assert "lr_scheduler" in state_dict, (
                    f"LR scheduler state dict not found in {state_dict.keys()}. Please check the checkpoint file "
                    f"{local_path}."
                )
                if self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
                    log_with_rank(f"Loaded LR scheduler checkpoint from {local_path}", rank=self.rank, logger=logger)

        # ── Load RNG states ───────────────────────────────────────────────────
        if self.should_load_extra:
            assert "rng_state" in state_dict, (
                f"RNG state dict not found in {state_dict.keys()}. Please check the checkpoint file {local_path}."
            )
            self.load_rng_states(state_dict["rng_state"])
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

    # -- Bridge helpers --------------------------------------------------------

    def _get_bridge_extended_args(self):
        """Build extra kwargs for ``bridge.save_weights`` from checkpoint config."""
        extended_args = {}
        mbridge_config = getattr(self.checkpoint_config, "mbridge_config", None) or {}
        for sig in inspect.signature(self.bridge.save_weights).parameters:
            if sig in ("weights_path", "models"):
                continue
            if sig in mbridge_config:
                extended_args[sig] = mbridge_config[sig]
        return extended_args

    def _save_model_via_bridge(self, hf_ckpt_path: str):
        """Save model weights through megatron-bridge."""
        if self.vanilla_bridge:
            self.bridge.save_weights(self.model, hf_ckpt_path, **self._get_bridge_extended_args())
        else:
            if self.peft_cls is not None:
                hf_adapter_ckpt_path = os.path.join(hf_ckpt_path, "adapter")
                self.bridge.save_hf_adapter(self.model, hf_adapter_ckpt_path, self.peft_cls)
                log_with_rank(
                    f"Saved HF PEFT adapter checkpoint to {hf_adapter_ckpt_path}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
            else:
                self.bridge.save_hf_weights(self.model, hf_ckpt_path)

    def _load_model_via_bridge(self, hf_model_path: str):
        """Load model weights through megatron-bridge."""
        if self.vanilla_bridge:
            self.bridge.load_weights(self.model, hf_model_path)
        else:
            self.bridge.load_hf_weights(self.model, hf_model_path)

    def _save_hf_config_and_tokenizer(self, local_path: str):
        """Rank-0 saves HF config, tokenizer, and generation config."""
        if self.rank != 0:
            return
        hf_config_tokenizer_path = get_hf_model_checkpoint_path(local_path)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(hf_config_tokenizer_path)
        self.hf_config.save_pretrained(hf_config_tokenizer_path)
        if hasattr(self.hf_config, "name_or_path") and self.hf_config.name_or_path:
            try:
                generation_config = GenerationConfig.from_pretrained(self.hf_config.name_or_path)
                generation_config.save_pretrained(hf_config_tokenizer_path)
            except Exception:
                pass
        log_with_rank(
            f"Saved Huggingface config and tokenizer to {hf_config_tokenizer_path}",
            rank=self.rank,
            logger=logger,
            log_only_rank_0=True,
        )

    def _save_transformer_config(self, local_path: str):
        """Rank-0 serialises the Megatron TransformerConfig to JSON."""
        if self.rank != 0:
            return
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
        try:
            for k in bypass_keys:
                if hasattr(self.transformer_config, k):
                    backup[k] = getattr(self.transformer_config, k, None)
                    delattr(self.transformer_config, k)
            transformer_config_dict = asdict(self.transformer_config)
        finally:
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

    # -- Save / Load ----------------------------------------------------------

    def _save_dist_checkpoint(self, dist_checkpoint_path: str, state_dict: dict):
        """Persist ``state_dict`` via Megatron dist_checkpointing.

        The caller is responsible for composing exactly the pieces that belong
        in the dist_checkpoint directory (optimizer, extra, and optionally
        model / PEFT adapter shards).

        Returns the async save request when ``async_save`` is enabled, or
        ``None`` for synchronous saves.
        """
        sharded_sd_metadata = self._build_sharded_state_dict_metadata()

        async_save_request = save_dist_checkpointing(
            sharded_state_dict=state_dict,
            ckpt_path=dist_checkpoint_path,
            async_save=self.checkpoint_config.async_save,
            content_metadata=sharded_sd_metadata,
        )

        if not self.checkpoint_config.async_save:
            assert async_save_request is None, "Async save request should be None when not using async save."
            torch.distributed.barrier()

        return async_save_request

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        self.previous_global_step = global_step

        if not self.checkpoint_config.async_save:
            self.ensure_checkpoint_capacity(max_ckpt_to_keep)

        local_path = local_mkdir_safe(local_path)
        dist_checkpoint_path = get_dist_checkpoint_path(local_path)
        metadata = self._build_sharded_state_dict_metadata()

        # ── Compose the dist_checkpoint state dict ──────────────────────────────
        # dist_checkpointing is the universal backend for optimizer and extra
        # states, regardless of which model backend (mbridge / dist_ckpt) is
        # used.  Model weights only go into dist_ckpt when mbridge is disabled
        # (or for PEFT adapters which bridge doesn't handle).
        #
        # The model sharded state dict is built as a transient dependency when
        # the optimizer needs it — Megatron's optimizer.sharded_state_dict()
        # requires the model's sharding layout as read-only input.
        dist_ckpt_state_dict = {}

        # Optimizer needs model sharded state dict as metadata input.
        model_sharded_state_dict = None
        if self.should_save_optimizer or self._save_model_via_dist_ckpt or self._save_peft_adapters_in_dist_ckpt:
            model_sharded_state_dict = self._build_model_sharded_state_dict(metadata)

        if self.should_save_optimizer:
            dist_ckpt_state_dict.update(self._build_optimizer_state_dict(model_sharded_state_dict, metadata))

        if self.should_save_extra:
            dist_ckpt_state_dict.update(self._build_extra_state_dict())

        # Model shards go into dist_ckpt only when mbridge is disabled.
        if self._save_model_via_dist_ckpt:
            dist_ckpt_state_dict.update(model_sharded_state_dict)
            log_with_rank(
                f"dist_ckpt will save model shards: {model_sharded_state_dict.keys()}",
                rank=self.rank,
                logger=logger,
            )

        # PEFT adapter shards always live in dist_ckpt (bridge handles base model only).
        if self._save_peft_adapters_in_dist_ckpt:
            peft_state = self._maybe_filter_peft_state_dict(dict(model_sharded_state_dict))
            dist_ckpt_state_dict.update(peft_state)

        # ── 1. Save composed state dict via dist_checkpointing ─────────────────
        async_save_request = None
        if dist_ckpt_state_dict:
            async_save_request = self._save_dist_checkpoint(dist_checkpoint_path, dist_ckpt_state_dict)

        # ── 2. Save model weights via mbridge (HF format) ─────────────────────
        if self._save_model_via_mbridge:
            hf_ckpt_path = get_hf_model_checkpoint_path(local_path)
            log_with_rank(f"Saving HF model checkpoint to {hf_ckpt_path} with bridge", rank=self.rank, logger=logger)
            self._save_model_via_bridge(hf_ckpt_path)
            log_with_rank(f"Saved bridge checkpoint to {hf_ckpt_path}", rank=self.rank, logger=logger)

        # ── 3. HF config / tokenizer (rank 0) ────────────────────────────────
        if self._save_model_via_mbridge:
            self._save_hf_config_and_tokenizer(local_path)

        # ── 4. Transformer config (rank 0) ────────────────────────────────────
        if self.should_save_extra:
            self._save_transformer_config(local_path)

        # ── 5. Finalization (HDFS upload, tracker, retention) ─────────────────
        hf_config_tokenizer_path = get_hf_model_checkpoint_path(local_path)
        saved_dist_ckpt = bool(dist_ckpt_state_dict)

        def finalize_save_fn():
            log_with_rank(f"Checkpoint save completed for {local_path}", rank=self.rank, logger=logger)
            if self.rank == 0 and hdfs_path is not None:
                log_with_rank(f"Uploading checkpoint to {hdfs_path}", rank=self.rank, logger=logger)
                from verl.utils import hdfs_io

                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                if saved_dist_ckpt:
                    hdfs_io.copy(src=dist_checkpoint_path, dst=hdfs_path, dirs_exist_ok=True)
                if self._save_model_via_mbridge:
                    hdfs_io.copy(src=hf_config_tokenizer_path, dst=hdfs_path, dirs_exist_ok=True)

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

        if self.checkpoint_config.async_save and async_save_request is not None:
            async_save_request.add_finalize_fn(finalize_save_fn)
            from megatron.core.dist_checkpointing.strategies.base import async_calls

            async_calls.schedule_async_request(async_save_request)
        else:
            finalize_save_fn()
