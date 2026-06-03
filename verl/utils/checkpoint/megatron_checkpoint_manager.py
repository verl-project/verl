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
from dataclasses import fields, is_dataclass
from enum import Enum

import megatron.core
import numpy as np
import torch
import torch.distributed
from megatron.core import dist_checkpointing, mpu, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedObject
from packaging import version
from transformers import GenerationConfig

from verl.utils.device import get_device_name, get_torch_device
from verl.utils.fs import is_non_local, local_mkdir_safe
from verl.utils.logger import log_with_rank
from verl.utils.megatron.dist_checkpointing import load_dist_checkpointing, save_dist_checkpointing
from verl.utils.megatron_utils import (
    get_checkpoint_contents_manifest_path,
    get_extra_dist_checkpoint_path,
    get_hf_model_checkpoint_path,
    get_legacy_dist_checkpoint_path,
    get_legacy_hf_model_checkpoint_path,
    get_model_dist_checkpoint_path,
    get_optimizer_dist_checkpoint_path,
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


_SKIP_CONFIG_VALUE = object()


def _to_json_safe_config_value(value, seen):
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if type(value) is torch.dtype or isinstance(value, Enum):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return _SKIP_CONFIG_VALUE
    if isinstance(value, list | tuple):
        value_id = id(value)
        if value_id in seen:
            return _SKIP_CONFIG_VALUE
        seen.add(value_id)
        converted = []
        for item in value:
            converted_item = _to_json_safe_config_value(item, seen)
            if converted_item is not _SKIP_CONFIG_VALUE:
                converted.append(converted_item)
        seen.remove(value_id)
        return converted
    if isinstance(value, dict):
        value_id = id(value)
        if value_id in seen:
            return _SKIP_CONFIG_VALUE
        seen.add(value_id)
        converted = {}
        for key, item in value.items():
            converted_key = _to_json_safe_config_value(key, seen)
            converted_item = _to_json_safe_config_value(item, seen)
            if converted_key is not _SKIP_CONFIG_VALUE and converted_item is not _SKIP_CONFIG_VALUE:
                converted[str(converted_key)] = converted_item
        seen.remove(value_id)
        return converted
    return _SKIP_CONFIG_VALUE


def _to_json_safe_config_dict(config_dict):
    json_safe_config = {}
    for key, value in config_dict.items():
        converted = _to_json_safe_config_value(value, set())
        if converted is not _SKIP_CONFIG_VALUE:
            json_safe_config[key] = converted
    return json_safe_config


def _config_to_shallow_dict(config):
    if is_dataclass(config):
        return {field.name: getattr(config, field.name) for field in fields(config) if hasattr(config, field.name)}
    return vars(config)


class MegatronCheckpointManager(BaseCheckpointManager):
    """Checkpoint manager for Megatron-LM distributed training.

    * ``use_dist_checkpointing`` -- when ``True``, Megatron sharded weights for
      the ``model`` slot are saved/loaded via ``dist_checkpointing`` under
      ``model/dist_ckpt/`` (mirrors ``*.megatron.use_dist_checkpointing``).

    * **HuggingFace / mbridge** -- any HF-format model weights (``hf_model`` in
      contents, or ``model`` when not using dist shards for that slot) require a
      non-``None`` ``bridge`` instance. The engine constructs the bridge when
      mbridge is enabled.

    These combine: e.g. ``save_contents=['model', 'hf_model', ...]`` with
    ``use_dist_checkpointing=True`` and a ``bridge`` writes both
    ``model/dist_ckpt/`` and ``model/huggingface/``.

    When ``use_dist_checkpointing=False``, the ``model`` entry maps to the HF
    tree only (no Megatron model shards unless PEFT adapter shards are needed).

    Optimizer, LR-scheduler, and RNG states always go through
    ``dist_checkpointing`` regardless of the model format.

    Interaction with ``save_contents`` / ``load_contents``:

    * ``model`` -- HF via ``bridge`` if not ``use_dist_checkpointing``; Megatron
      shards if ``use_dist_checkpointing``; both trees when ``hf_model`` is also
      listed and ``bridge`` is provided.
    * ``hf_model`` -- HF weights via ``bridge``; requires ``bridge`` is not
      ``None``. With only the HF ``model`` path, ``model`` and ``hf_model`` are
      deduplicated (saved once).

    Args:
        model: The Megatron model instance to checkpoint.
        optimizer: The optimizer instance.
        lr_scheduler: The learning rate scheduler instance.
        use_dist_checkpointing: If ``True``, include Megatron ``dist_checkpointing``
            shards for the ``model`` slot. Mirrors ``*.megatron.use_dist_checkpointing``.
        bridge: mbridge / Megatron-Bridge instance for HF save/load; required whenever
            checkpoint contents request HF-format model weights.
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
        use_dist_checkpointing: bool = False,
        use_megatron_fsdp: bool = False,
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
        self.use_megatron_fsdp = use_megatron_fsdp
        self.rank = torch.distributed.get_rank()

        # ``use_dist_checkpointing`` selects Megatron shards for the ``model``
        # slot; HF weights always go through ``bridge`` when requested by
        # ``save_contents`` / ``load_contents``.
        self.use_dist_checkpointing = use_dist_checkpointing

        if "hf_model" in self.checkpoint_save_contents and self.bridge is None:
            raise ValueError(
                "`save_contents` contains 'hf_model' but `bridge` is None. "
                "HuggingFace-format model weights require mbridge (pass `bridge=...`) "
                "or remove 'hf_model' from `save_contents`."
            )
        if "hf_model" in self.checkpoint_load_contents and self.bridge is None:
            raise ValueError(
                "`load_contents` contains 'hf_model' but `bridge` is None. "
                "Pass `bridge=...` or remove 'hf_model' from `load_contents`."
            )

        if self.should_save_hf_model and self.bridge is None:
            raise ValueError(
                "MegatronCheckpointManager: HF-format model weights require "
                "a bridge instance. Either pass `bridge=...` or set "
                "`use_dist_checkpointing=True` to use dist_checkpointing for the "
                "`model` slot only."
            )

        if self.should_load_hf_model and self.bridge is None:
            raise ValueError(
                "MegatronCheckpointManager: loading HF-format model weights requires "
                "a bridge instance. Pass `bridge=...` or set "
                "`use_dist_checkpointing=True` for the `model` slot."
            )

    # ------------------------------------------------------------------
    # Backend-aware specialisations of ``should_{save,load}_{hf,dist_ckpt}_model``.
    # Each property is the *single expression* guarding one dispatch
    # branch, so save/load code never re-derives the same boolean.
    # ------------------------------------------------------------------

    @property
    def should_save_hf_model(self) -> bool:
        """True when this save must emit HF-format model weights via bridge."""
        return "hf_model" in self.checkpoint_save_contents or (
            "model" in self.checkpoint_save_contents and not self.use_dist_checkpointing
        )

    @property
    def should_save_dist_ckpt_model(self) -> bool:
        """True when this save must emit Megatron sharded model weights."""
        return "model" in self.checkpoint_save_contents and self.use_dist_checkpointing

    @property
    def should_load_hf_model(self) -> bool:
        """True when model weights must be loaded from an HF-format checkpoint via bridge."""
        return "hf_model" in self.checkpoint_load_contents or (
            "model" in self.checkpoint_load_contents and not self.use_dist_checkpointing
        )

    @property
    def should_load_dist_ckpt_model(self) -> bool:
        """True when model weights must be loaded from a Megatron sharded checkpoint."""
        return "model" in self.checkpoint_load_contents and self.use_dist_checkpointing

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

        if self.use_megatron_fsdp:
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            return {f"({pp_rank}, {tp_rank})": rng_state_list}

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
            if self.use_megatron_fsdp:
                # Megatron-FSDP wraps the model and exposes a DTensor-aware
                # state-dict; the standard ``sharded_state_dict`` path doesn't
                # apply.
                model_sharded_state_dict[key] = model.state_dict_for_save_checkpoint()
                continue
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

        if self.use_megatron_fsdp:
            metadata["distrib_optim_sharding_type"] = "fsdp_dtensor"
        elif self.use_distributed_optimizer:
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

    # -- Contents manifest -----------------------------------------------------
    # A small human-readable JSON file (``ckpt_contents.json``) written at the
    # root of the checkpoint directory, describing which logical contents were
    # saved and where to find them.  Intended for users who need to locate a
    # particular artifact (HF weights, optimizer shards, transformer config,
    # …) without re-deriving the layout from code.

    # Bump when the schema changes in a backwards-incompatible way.
    #
    #  v1: single ``dist_ckpt/`` and ``huggingface/`` directly under the
    #      checkpoint root (legacy, no longer produced).
    #  v2: split into ``model/``, ``optimizer/``, ``extra/`` subdirectories;
    #      HF tree is nested at ``model/huggingface/``.
    MANIFEST_SCHEMA_VERSION = 2

    def _build_checkpoint_manifest(
        self,
        local_path: str,
        global_step: int,
        saved_any_dist_ckpt: bool,
    ) -> dict:
        """Compose the ``ckpt_contents.json`` payload for this save.

        All paths in the returned mapping are relative to ``local_path`` so
        the manifest stays valid if the checkpoint directory is moved or
        uploaded to remote storage.  The ``contents`` section maps each
        logical piece that was saved to its on-disk location; the
        ``directories`` section lists each top-level subdirectory with a
        short description of what it contains.
        """

        def _rel(abs_path: str) -> str:
            return os.path.relpath(abs_path, local_path)

        model_dist_rel = _rel(get_model_dist_checkpoint_path(local_path))
        optim_dist_rel = _rel(get_optimizer_dist_checkpoint_path(local_path))
        extra_dist_rel = _rel(get_extra_dist_checkpoint_path(local_path))
        hf_rel = _rel(get_hf_model_checkpoint_path(local_path))
        transformer_config_rel = _rel(get_transformer_config_checkpoint_path(local_path))

        contents: dict[str, dict] = {}

        if self.should_save_hf_model:
            model_entry: dict = {
                "path": hf_rel,
                "format": "huggingface",
                "backend": "mbridge",
            }
            if self.peft_cls is not None:
                model_entry["peft_adapter_path"] = os.path.join(hf_rel, "adapter")
            hybrid_weights = self.should_save_dist_ckpt_model
            if not hybrid_weights:
                contents["model"] = model_entry
            if "hf_model" in self.checkpoint_save_contents:
                contents["hf_model"] = {
                    "path": hf_rel,
                    "format": "huggingface",
                    "note": (
                        "deduplicated with 'model' on the HF path"
                        if not hybrid_weights
                        else "HF export alongside Megatron model shards under model/dist_ckpt/",
                    ),
                }

        if self.should_save_dist_ckpt_model:
            contents["model"] = {
                "path": model_dist_rel,
                "format": "megatron_dist_checkpoint",
                "backend": "dist_checkpointing",
            }

        if self.should_save_hf_model and self.peft_cls is not None:
            contents["peft_adapters"] = {
                "path": model_dist_rel,
                "format": "megatron_dist_checkpoint",
                "note": "PEFT adapter shards sit next to base-model HF weights, under model/dist_ckpt/.",
            }

        if self.should_save_optimizer:
            contents["optimizer"] = {
                "path": optim_dist_rel,
                "format": "megatron_dist_checkpoint",
            }
            if self.lr_scheduler is not None:
                contents["lr_scheduler"] = {
                    "path": optim_dist_rel,
                    "format": "megatron_dist_checkpoint",
                    "key": "lr_scheduler",
                }

        if self.should_save_extra:
            contents["rng_state"] = {
                "path": extra_dist_rel,
                "format": "megatron_dist_checkpoint",
                "key": "rng_state",
            }
            contents["transformer_config"] = {
                "path": transformer_config_rel,
                "format": "json",
            }

        if self.should_save_hf_model:
            contents["hf_config"] = {
                "path": hf_rel,
                "format": "huggingface",
                "note": "config.json, tokenizer, and optional generation_config.json",
            }
            if self.processing_class is not None:
                contents["tokenizer"] = {
                    "path": hf_rel,
                    "format": "huggingface",
                }

        directories: dict[str, str] = {}
        if self.should_save_hf_model:
            directories[hf_rel] = (
                "HuggingFace-format artifacts written via mbridge: model weights, "
                "config.json, tokenizer files, and optional generation_config.json."
            )
        if self.should_save_dist_ckpt_model or (self.should_save_hf_model and self.peft_cls is not None):
            directories[model_dist_rel] = (
                "Megatron dist_checkpointing shards for model weights (or PEFT adapter shards when mbridge is enabled)."
            )
        if self.should_save_optimizer:
            directories[optim_dist_rel] = (
                "Megatron dist_checkpointing shards for the optimizer state (includes lr_scheduler when present)."
            )
        if self.should_save_extra:
            directories[extra_dist_rel] = "Megatron dist_checkpointing shards for extra state (rng_state)."

        manifest = {
            "schema_version": self.MANIFEST_SCHEMA_VERSION,
            "framework": "megatron",
            "role": self.role,
            "arch": self.arch,
            "global_step": int(global_step),
            "world_size": self.world_size,
            "backend": {
                "has_bridge": self.bridge is not None,
                "use_dist_checkpointing": self.use_dist_checkpointing,
                "peft": self.peft_cls is not None,
            },
            "save_contents": list(self.checkpoint_save_contents),
            "contents": contents,
            "directories": directories,
            "saved_any_dist_ckpt": bool(saved_any_dist_ckpt),
        }
        return manifest

    def _write_checkpoint_manifest(
        self,
        local_path: str,
        global_step: int,
        saved_any_dist_ckpt: bool,
    ):
        """Rank-0 writes the ``ckpt_contents.json`` manifest.

        Written last so its presence signals a fully-completed save (useful
        when scanning checkpoint directories to find valid snapshots).
        """
        if self.rank != 0:
            return
        manifest = self._build_checkpoint_manifest(local_path, global_step, saved_any_dist_ckpt)
        manifest_path = get_checkpoint_contents_manifest_path(local_path)
        tmp_path = manifest_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, manifest_path)
        log_with_rank(
            f"Wrote checkpoint contents manifest to {manifest_path}",
            rank=self.rank,
            logger=logger,
            log_only_rank_0=True,
        )

    # -- Load ------------------------------------------------------------------

    def _load_model_as_hf_via_bridge(self, hf_model_path: str):
        """Load model weights through megatron-bridge."""
        if self.vanilla_bridge:
            self.bridge.load_weights(self.model, hf_model_path)
        else:
            self.bridge.load_hf_weights(self.model, hf_model_path)

    @staticmethod
    def _has_checkpoint_files(path: str) -> bool:
        return os.path.isdir(path) and any(os.scandir(path))

    @staticmethod
    def _raise_for_old_layout(local_path: str) -> None:
        """Fail fast with a pointer to the migration script for pre-v2 layouts.

        The v2 layout nests dist_ckpt under ``model/``, ``optimizer/``,
        ``extra/``; the legacy (v1) layout put ``dist_ckpt/`` and
        ``huggingface/`` directly at the checkpoint root.  We only flag a
        directory as legacy when it has no v2 subdirectories **and** still
        has a non-empty root-level ``dist_ckpt/`` or ``huggingface/``.
        """
        if not os.path.isdir(local_path):
            return
        has_v2 = any(os.path.isdir(os.path.join(local_path, sub)) for sub in ("model", "optimizer", "extra"))
        if has_v2:
            return

        legacy_dist = get_legacy_dist_checkpoint_path(local_path)
        legacy_hf = get_legacy_hf_model_checkpoint_path(local_path)
        if os.path.isdir(legacy_dist) and any(os.scandir(legacy_dist)):
            detected = f"legacy dist_ckpt/ directly under {local_path}"
        elif os.path.isdir(legacy_hf) and any(os.scandir(legacy_hf)):
            detected = f"legacy huggingface/ directly under {local_path}"
        else:
            return

        raise RuntimeError(
            f"Detected deprecated Megatron checkpoint layout ({detected}). The current "
            "loader expects the split layout (model/, optimizer/, extra/ subdirectories). "
            "Migrate the checkpoint in place with:\n"
            "    python scripts/migrate_megatron_checkpoint_layout.py --checkpoint "
            f"{local_path}\n"
            "or use a freshly-saved checkpoint."
        )

    def _raise_for_unsupported_peft_checkpoint_layout(self, local_path: str, model_dist_path: str):
        if self.peft_cls is None or not self.should_load_model or self._has_checkpoint_files(model_dist_path):
            return

        legacy_adapter_ckpt_path = os.path.join(local_path, "adapter_checkpoint")
        hf_adapter_ckpt_path = os.path.join(local_path, "model", "huggingface", "adapter")

        if os.path.isdir(legacy_adapter_ckpt_path):
            raise RuntimeError(
                f"Found legacy PEFT checkpoint at {legacy_adapter_ckpt_path}, but checkpoint resume now expects "
                f"adapter weights in {model_dist_path}. Resave/convert the checkpoint or load the adapter via "
                "`lora.adapter_path`."
            )

        if os.path.isfile(os.path.join(hf_adapter_ckpt_path, "adapter_config.json")):
            raise RuntimeError(
                f"Found exported HF PEFT adapter at {hf_adapter_ckpt_path}, but `load_checkpoint()` resumes from "
                f"{model_dist_path}. HF adapter exports are not used for trainer resume; keep the distributed "
                "checkpoint or load the adapter separately via `lora.adapter_path`."
            )

    def _maybe_filter_peft_state_dict(self, state_dict: dict):
        if self.peft_cls is None:
            return state_dict

        from megatron.bridge.training.checkpointing import apply_peft_adapter_filter_to_state_dict

        return apply_peft_adapter_filter_to_state_dict(state_dict, self.peft_cls)

    def _load_megatron_fsdp_checkpoint(self, local_path: str, del_local_after_load=False):
        # Megatron-FSDP writes a single combined DTensor checkpoint at the
        # model dist path (see ``_save_megatron_fsdp_checkpoint``); load it
        # back from the same location under the v2 layout.
        dist_checkpoint_path = get_model_dist_checkpoint_path(local_path)
        if not os.path.isfile(os.path.join(dist_checkpoint_path, ".metadata")):
            raise FileNotFoundError(f"Megatron-FSDP checkpoint metadata not found at {dist_checkpoint_path}/.metadata.")

        # Build a full template (model + optimizer + extra) regardless of the
        # ``should_load_*`` flags so the DTensor loader can match every entry
        # written at save time.  Per-section gating happens below when the
        # loaded tensors are applied back to the live objects.
        metadata = self._build_sharded_state_dict_metadata()
        sharded_state_dict = {}
        model_sharded_state_dict = self._build_model_sharded_state_dict(metadata)
        sharded_state_dict.update(model_sharded_state_dict)
        sharded_state_dict.update(self._build_optimizer_state_dict(model_sharded_state_dict, metadata, is_loading=True))
        sharded_state_dict.update(self._build_extra_state_dict())

        from megatron.bridge.training.checkpointing import load_fsdp_dtensor_checkpoint

        checkpoint_model = getattr(self.model[0], "module", self.model[0])
        sharded_state_dict["_model"] = [checkpoint_model]
        state_dict, _, _, _ = load_fsdp_dtensor_checkpoint(
            load_dir=dist_checkpoint_path,
            ckpt_cfg=self.checkpoint_config,
            rank0=False,
            sharded_state_dict=sharded_state_dict,
            iteration=None,
            release=False,
            checkpoint_path_override=dist_checkpoint_path,
            cfg=self.transformer_config,
        )

        if self.should_load_model:
            self.model[0].load_state_dict(state_dict["model"], strict=True)
            log_with_rank(f"Loaded sharded model checkpoint from {local_path}", rank=self.rank, logger=logger)
        if self.should_load_optimizer:
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
        if self.should_load_extra:
            self.load_rng_states(state_dict["rng_state"])
            log_with_rank(f"Loaded RNG states from {local_path}", rank=self.rank, logger=logger)
        log_with_rank(f"Loaded Megatron-FSDP checkpoint from {local_path}", rank=self.rank, logger=logger)

        if del_local_after_load:
            try:
                os.remove(local_path) if is_non_local(local_path) else None
            except Exception as e:
                log_with_rank(
                    f"remove local resume ckpt file after loading failed, exception {e} will be ignored",
                    rank=self.rank,
                    logger=logger,
                )

    def _save_megatron_fsdp_checkpoint(self, dist_checkpoint_path: str):
        """Save a Megatron-FSDP (DTensor) checkpoint.

        Composes the state dict from the same builders used by the
        dist_checkpointing path, then preprocesses it for the DTensor
        ``torch.distributed.checkpoint`` writer.  Megatron-FSDP saves
        synchronously and returns no async request.
        """
        metadata = self._build_sharded_state_dict_metadata()
        state_dict = {}

        # The optimizer's ``sharded_state_dict()`` needs the model layout as
        # input; build the model state dict whenever we save model weights or
        # optimizer states.
        model_sharded_state_dict = None
        if self.should_save_model or self.should_save_optimizer:
            model_sharded_state_dict = self._build_model_sharded_state_dict(metadata)
            if self.should_save_model:
                state_dict.update(model_sharded_state_dict)

        if self.should_save_optimizer:
            state_dict.update(self._build_optimizer_state_dict(model_sharded_state_dict, metadata))

        if self.should_save_extra:
            state_dict.update(self._build_extra_state_dict())

        from megatron.bridge.training.checkpointing import save_fsdp_dtensor_checkpoint

        checkpoint_model = getattr(self.model[0], "module", self.model[0])
        save_fsdp_dtensor_checkpoint(
            dist_checkpoint_path,
            state_dict,
            cfg=self.transformer_config,
            model=checkpoint_model,
        )
        return None

    def load_rng_states(self, rng_states, data_parallel_random_init=False, use_dist_ckpt=True):
        if self.use_megatron_fsdp:
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            key = f"({pp_rank}, {tp_rank})"
            if key in rng_states:
                rng_states = rng_states[key]
            else:
                log_with_rank(
                    f"RNG state for PP/TP key {key} not found; falling back to the first saved RNG state.",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
                rng_states = next(iter(rng_states.values()))

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

    def _load_content_metadata(self, ckpt_dir: str) -> dict:
        """Fetch MCore's content_metadata for a dist_ckpt dir with a fallback."""
        load_content_metadata = getattr(dist_checkpointing, "load_content_metadata", None)
        metadata = None
        if load_content_metadata is not None and self._has_checkpoint_files(ckpt_dir):
            metadata = load_content_metadata(checkpoint_dir=ckpt_dir)
        if metadata is None:
            if self.use_distributed_optimizer:
                metadata = {"distrib_optim_sharding_type": "fully_sharded_model_space"}
            else:
                metadata = self._build_sharded_state_dict_metadata()
        return metadata

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        """Load a Megatron checkpoint in the v2 split layout.

        Each dist_checkpointing subtree (``model/dist_ckpt/``,
        ``optimizer/dist_ckpt/``, ``extra/dist_ckpt/``) is loaded
        independently, and any pre-v2 layout is rejected upfront with a
        pointer at the migration script.
        """
        if local_path is not None:
            assert os.path.exists(local_path), f"Checkpoint path {local_path} does not exist."

        self._raise_for_old_layout(local_path)

        try:
            import transformer_engine

            torch.serialization.add_safe_globals([torch.optim.AdamW])
            torch.serialization.add_safe_globals([transformer_engine.pytorch.optimizers.fused_adam.FusedAdam])
        except Exception:
            pass

        # Megatron-FSDP keeps a single combined DTensor checkpoint (model +
        # optimizer + extra in one directory) rather than the v2 three-tree
        # split, so it has its own dedicated loader and short-circuits here.
        if self.use_megatron_fsdp:
            self._load_megatron_fsdp_checkpoint(local_path, del_local_after_load=del_local_after_load)
            return

        model_dist_path = get_model_dist_checkpoint_path(local_path)
        optim_dist_path = get_optimizer_dist_checkpoint_path(local_path)
        extra_dist_path = get_extra_dist_checkpoint_path(local_path)

        self._raise_for_unsupported_peft_checkpoint_layout(local_path, model_dist_path)

        # Pick metadata from the first subtree that actually exists; all
        # three trees are written with the same content_metadata and we
        # need a single consistent value for composing sharded state dicts.
        metadata_source = next(
            (p for p in (model_dist_path, optim_dist_path, extra_dist_path) if self._has_checkpoint_files(p)),
            model_dist_path,
        )
        metadata = self._load_content_metadata(metadata_source)

        # Model sharded state dict is shared across the model-load and
        # optimizer-load metadata inputs, so build it once when either path
        # needs it. The dist_ckpt model load and the HF+PEFT load both read
        # Megatron shards, so either branch requires the sharded SD.
        model_sharded_state_dict = None
        if (
            self.should_load_optimizer
            or self.should_load_dist_ckpt_model
            or (self.should_load_hf_model and self.peft_cls is not None)
        ):
            model_sharded_state_dict = self._build_model_sharded_state_dict(metadata)

        # ── Load model weights ──────────────────────────────────────────────
        if self.should_load_dist_ckpt_model or (self.should_load_hf_model and self.peft_cls is not None):
            model_sd = self._maybe_filter_peft_state_dict(dict(model_sharded_state_dict))
            loaded_model = load_dist_checkpointing(
                sharded_state_dict=model_sd,
                ckpt_dir=model_dist_path,
            )
            assert "model" in loaded_model or any(
                f"model{vpp_rank}" in loaded_model for vpp_rank in range(len(self.model))
            ), f"Model state dict not found in {loaded_model.keys()}. Please check the checkpoint file {local_path}."
            for vpp_rank in range(len(self.model)):
                if len(self.model) == 1:
                    model_state_dict = loaded_model["model"]
                else:
                    assert f"model{vpp_rank}" in loaded_model, f"model{vpp_rank} not found in state_dict"
                    model_state_dict = loaded_model[f"model{vpp_rank}"]
                mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
                self.model[vpp_rank].load_state_dict(model_state_dict, strict=self.peft_cls is None)
            if self.peft_cls is not None:
                log_with_rank(f"Loaded PEFT adapter checkpoint from {model_dist_path}", rank=self.rank, logger=logger)
            else:
                log_with_rank(f"Loaded sharded model checkpoint from {model_dist_path}", rank=self.rank, logger=logger)

        elif self.should_load_hf_model and self.peft_cls is None:
            hf_model_path = get_hf_model_checkpoint_path(local_path)
            self._load_model_as_hf_via_bridge(hf_model_path)
            log_with_rank(f"Loaded HF model checkpoint from {hf_model_path} with bridge", rank=self.rank, logger=logger)

        # ── Load optimizer / LR scheduler ───────────────────────────────────
        if self.should_load_optimizer:
            optim_sd = self._build_optimizer_state_dict(model_sharded_state_dict, metadata, is_loading=True)
            loaded_optim = load_dist_checkpointing(
                sharded_state_dict=optim_sd,
                ckpt_dir=optim_dist_path,
            )
            assert "optimizer" in loaded_optim, (
                f"Optimizer state dict not found in {loaded_optim.keys()}. "
                f"Please check the checkpoint file {optim_dist_path}."
            )
            self.optimizer.load_state_dict(loaded_optim["optimizer"])
            log_with_rank(f"Loaded optimizer checkpoint from {optim_dist_path}", rank=self.rank, logger=logger)
            if self.use_checkpoint_opt_param_scheduler:
                assert "lr_scheduler" in loaded_optim, (
                    f"LR scheduler state dict not found in {loaded_optim.keys()}. "
                    f"Please check the checkpoint file {optim_dist_path}."
                )
                if self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(loaded_optim["lr_scheduler"])
                    log_with_rank(
                        f"Loaded LR scheduler checkpoint from {optim_dist_path}", rank=self.rank, logger=logger
                    )

        # ── Load RNG states ─────────────────────────────────────────────────
        if self.should_load_extra:
            extra_sd = self._build_extra_state_dict()
            loaded_extra = load_dist_checkpointing(
                sharded_state_dict=extra_sd,
                ckpt_dir=extra_dist_path,
            )
            assert "rng_state" in loaded_extra, (
                f"RNG state dict not found in {loaded_extra.keys()}. "
                f"Please check the checkpoint file {extra_dist_path}."
            )
            self.load_rng_states(loaded_extra["rng_state"])
            log_with_rank(f"Loaded RNG states from {extra_dist_path}", rank=self.rank, logger=logger)

        if del_local_after_load:
            try:
                os.remove(local_path) if is_non_local(local_path) else None
            except Exception as e:
                log_with_rank(
                    f"remove local resume ckpt file after loading failed, exception {e} will be ignored",
                    rank=self.rank,
                    logger=logger,
                )

    # -- Save ------------------------------------------------------------------

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

    def _save_model_as_hf_via_bridge(self, hf_ckpt_path: str):
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
        transformer_config_dict = _to_json_safe_config_dict(_config_to_shallow_dict(self.transformer_config))
        transformer_config_path = get_transformer_config_checkpoint_path(local_path)
        # NOTE: With Megatron-Bridge backend, a circular import issue occurs when transformers version >= 5.4.0.
        with open(transformer_config_path, "w") as f:
            json.dump(
                transformer_config_dict,
                f,
                indent=2,
                default=lambda o: o.to_dict() if hasattr(o, "to_dict") else o,
            )

    def _save_dist_checkpoint(self, dist_checkpoint_path: str, state_dict: dict):
        """Persist ``state_dict`` via Megatron dist_checkpointing.

        Writes one self-contained ``torch_dist`` directory under
        ``dist_checkpoint_path``.  Callers compose the exact state dict to go
        into this directory; in the v2 layout there are up to three such
        directories per checkpoint (``model/dist_ckpt/``,
        ``optimizer/dist_ckpt/``, ``extra/dist_ckpt/``).

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

    def _schedule_dist_checkpoint_saves(self, items: list[tuple[str, dict]]) -> list:
        """Schedule one dist_checkpointing save per non-empty state dict.

        ``items`` is an ordered list of ``(path, state_dict)`` pairs.  Empty
        state dicts are skipped so a disabled content (e.g. no optimizer)
        does not create an empty on-disk directory.  Returns the async save
        requests in the same order they were scheduled; for synchronous
        saves the list is empty.
        """
        async_requests: list = []
        for path, state_dict in items:
            if not state_dict:
                continue
            req = self._save_dist_checkpoint(path, state_dict)
            if req is not None:
                async_requests.append(req)
        return async_requests

    def _finalize_save(
        self,
        *,
        local_path: str,
        hdfs_path: str | None,
        global_step: int,
        max_ckpt_to_keep,
        saved_any_dist_ckpt: bool,
    ) -> None:
        """Run post-write bookkeeping: manifest, HDFS upload, tracker, retention.

        This is invoked either immediately after synchronous saves, or via
        Megatron's ``AsyncCallsQueue`` finalize hook attached to the last
        scheduled async request — see ``_dispatch_finalize``.  Either way it
        runs exactly once per save, after every per-content write has
        completed on every rank.
        """
        log_with_rank(f"Checkpoint save completed for {local_path}", rank=self.rank, logger=logger)

        # Write the contents manifest last so its presence indicates a
        # fully-complete save (including async dist_checkpointing writes).
        self._write_checkpoint_manifest(local_path, global_step, saved_any_dist_ckpt)

        if self.rank == 0 and hdfs_path is not None:
            log_with_rank(f"Uploading checkpoint to {hdfs_path}", rank=self.rank, logger=logger)
            from verl.utils import hdfs_io

            hdfs_io.makedirs(hdfs_path, exist_ok=True)
            # Upload the entire checkpoint directory as a single recursive
            # copy.  This keeps HDFS and local layouts identical (model/,
            # optimizer/, extra/, manifest, transformer_config.json).
            hdfs_io.copy(src=local_path, dst=hdfs_path, dirs_exist_ok=True)

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

    def _dispatch_finalize(self, async_requests: list, finalize_save_fn) -> None:
        """Run ``finalize_save_fn`` now, or after all async writes complete.

        When ``async_save`` is enabled and at least one dist_checkpointing
        request was scheduled, attach the finalize callback to the **last**
        request and enqueue every request on Megatron's ``AsyncCallsQueue``.
        The queue processes requests in FIFO order within a rank and
        all-reduces between each to keep ranks in sync, so attaching to the
        last request guarantees ``finalize_save_fn`` runs only after every
        per-content write has completed on every rank.

        In all other cases (sync save, or async save with nothing to write)
        the callback fires immediately on the calling thread.
        """
        if self.checkpoint_config.async_save and async_requests:
            async_requests[-1].add_finalize_fn(finalize_save_fn)
            from megatron.core.dist_checkpointing.strategies.base import async_calls

            for req in async_requests:
                async_calls.schedule_async_request(req)
        else:
            finalize_save_fn()

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        """Save a Megatron checkpoint under ``local_path`` (layout schema v2).

        Contents are split across three sibling directories so each piece can
        be inspected or managed independently::

            local_path/
            ├── ckpt_contents.json       # manifest (read first to locate any piece)
            ├── transformer_config.json  # when 'extra' is in save_contents
            ├── model/
            │   ├── huggingface/         # HF weights when ``bridge`` is set (+ ``hf_model`` / HF ``model`` path)
            │   └── dist_ckpt/           # Megatron model shards when ``use_dist_checkpointing``;
            │                            # also PEFT adapters with mbridge
            ├── optimizer/dist_ckpt/     # optimizer + lr_scheduler shards
            └── extra/dist_ckpt/         # rng_state shards

        With a non-``None`` ``bridge`` and ``use_dist_checkpointing=True``, both
        ``model/huggingface/`` and ``model/dist_ckpt/`` may be populated in one save.

        Rank 0 writes ``ckpt_contents.json`` last, so its presence signals a
        fully-complete save (including async dist_checkpointing writes).
        See ``docs/advance/checkpoint.rst`` ("Locating saved contents") for
        the manifest schema.
        """
        self.previous_global_step = global_step

        if not self.checkpoint_config.async_save:
            self.ensure_checkpoint_capacity(max_ckpt_to_keep)

        local_path = local_mkdir_safe(local_path)
        # ── 1. Save dist_checkpoint payload ───────────────────────────────────
        # Two save paths exist:
        #
        # * Megatron-FSDP uses its own DTensor-based writer that bundles
        #   model + optimizer + extra into a single combined directory.  It
        #   replaces the per-content dist_checkpointing composition entirely
        #   and saves synchronously (no async request is produced).  We
        #   route the combined checkpoint to the model dist path so the rest
        #   of the v2 layout (HF model, transformer config, manifest) still
        #   lines up at the sibling locations.
        #
        # * Otherwise we use Megatron dist_checkpointing per-content (model /
        #   optimizer / extra) so each piece can be independently located,
        #   GC'd, or uploaded.  Model weights only go into dist_ckpt when
        #   ``use_dist_checkpointing=True`` (or for PEFT adapters which
        #   bridge doesn't handle).  The model sharded state dict is built
        #   once as a shared dependency: Megatron's optimizer.sharded_state_dict()
        #   reads it as metadata, and we may additionally persist it (or a
        #   PEFT-filtered subset) as the model content itself.
        model_state_dict: dict = {}
        optimizer_state_dict: dict = {}
        extra_state_dict: dict = {}
        fsdp_saved = False

        if self.use_megatron_fsdp:
            self._save_megatron_fsdp_checkpoint(get_model_dist_checkpoint_path(local_path))
            fsdp_saved = True
        else:
            metadata = self._build_sharded_state_dict_metadata()

            model_sharded_state_dict = None
            if (
                self.should_save_optimizer
                or self.should_save_dist_ckpt_model
                or (self.should_save_hf_model and self.peft_cls is not None)
            ):
                model_sharded_state_dict = self._build_model_sharded_state_dict(metadata)

            if self.should_save_dist_ckpt_model:
                model_state_dict.update(model_sharded_state_dict)
                log_with_rank(
                    f"model/dist_ckpt will save model shards: {model_sharded_state_dict.keys()}",
                    rank=self.rank,
                    logger=logger,
                )
            if self.should_save_hf_model and self.peft_cls is not None:
                peft_state = self._maybe_filter_peft_state_dict(dict(model_sharded_state_dict))
                model_state_dict.update(peft_state)

            if self.should_save_optimizer:
                optimizer_state_dict.update(self._build_optimizer_state_dict(model_sharded_state_dict, metadata))

        # FSDP already wrote model + optimizer + extra into one combined
        # checkpoint above, so the per-content state dicts stay empty and
        # ``_schedule_dist_checkpoint_saves`` becomes a no-op for that path.
        if self.should_save_extra and not fsdp_saved:
            extra_state_dict.update(self._build_extra_state_dict())

        # ── 1b. Save each dist_checkpointing tree ───────────────────────────
        # Preserve a stable schedule order (model → optimizer → extra) so the
        # async finalize callback (attached to the last request by
        # ``_dispatch_finalize``) only fires after all three writes complete.
        async_requests = self._schedule_dist_checkpoint_saves(
            [
                (get_model_dist_checkpoint_path(local_path), model_state_dict),
                (get_optimizer_dist_checkpoint_path(local_path), optimizer_state_dict),
                (get_extra_dist_checkpoint_path(local_path), extra_state_dict),
            ]
        )

        # ── 2. Save model weights in HF format via bridge ───────────────────
        if self.should_save_hf_model:
            hf_ckpt_path = get_hf_model_checkpoint_path(local_path)
            log_with_rank(f"Saving HF model checkpoint to {hf_ckpt_path} with bridge", rank=self.rank, logger=logger)
            self._save_model_as_hf_via_bridge(hf_ckpt_path)
            log_with_rank(f"Saved bridge checkpoint to {hf_ckpt_path}", rank=self.rank, logger=logger)

        # ── 3. HF config / tokenizer (rank 0) ───────────────────────────────
        if self.should_save_hf_model:
            self._save_hf_config_and_tokenizer(local_path)

        # ── 4. Transformer config (rank 0, at checkpoint root) ──────────────
        if self.should_save_extra:
            self._save_transformer_config(local_path)

        # ── 5. Finalization (manifest, HDFS upload, tracker, retention) ─────
        # FSDP saves synchronously and produces no async request, so the
        # finalize callback fires immediately for that path even though we
        # still need to record that a dist_ckpt was written.
        saved_any_dist_ckpt = fsdp_saved or bool(model_state_dict or optimizer_state_dict or extra_state_dict)
        self._dispatch_finalize(
            async_requests,
            lambda: self._finalize_save(
                local_path=local_path,
                hdfs_path=hdfs_path,
                global_step=global_step,
                max_ckpt_to_keep=max_ckpt_to_keep,
                saved_any_dist_ckpt=saved_any_dist_ckpt,
            ),
        )
