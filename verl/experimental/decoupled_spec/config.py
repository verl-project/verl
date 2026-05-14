from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from verl.base_config import BaseConfig


@dataclass
class DraftConfig(BaseConfig):
    enable: bool = False
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    load_format: str = "auto"
    nnodes: int = 1
    ngpus: int = 0
    tp_size: int = 1
    speculative_num_steps: int = 3
    trace_dir: Optional[str] = None

    def __post_init__(self):
        if not self.enable:
            return
        if not self.model_path:
            raise ValueError("draft.model_path must be set when draft.enable=True")
        if self.nnodes <= 0:
            raise ValueError("draft.nnodes must be > 0")
        if self.ngpus <= 0:
            raise ValueError("draft.ngpus must be > 0")
        if self.tp_size <= 0:
            raise ValueError("draft.tp_size must be > 0")
        if self.speculative_num_steps <= 0:
            raise ValueError("draft.speculative_num_steps must be > 0")
        if self.ngpus % self.tp_size != 0:
            raise ValueError("draft.ngpus must be divisible by draft.tp_size")
        if self.ngpus % self.nnodes != 0:
            raise ValueError("draft.ngpus must be divisible by draft.nnodes")
        if (self.ngpus // self.nnodes) % self.tp_size != 0:
            raise ValueError("draft.ngpus / draft.nnodes must be divisible by draft.tp_size")

    @property
    def num_drafters(self) -> int:
        if self.tp_size <= 0:
            return 0
        return self.ngpus // self.tp_size

    @property
    def ngpus_per_node(self) -> int:
        if self.nnodes <= 0:
            return 0
        return self.ngpus // self.nnodes


def get_draft_config(config: DictConfig | dict[str, Any] | None) -> DraftConfig:
    if config is None:
        return DraftConfig()

    draft_cfg = config.get("draft") if isinstance(config, (DictConfig, dict)) else None
    if draft_cfg is None:
        return DraftConfig()

    structured = OmegaConf.structured(DraftConfig)
    merged = OmegaConf.merge(structured, draft_cfg)
    return OmegaConf.to_object(merged)


def _copy_config(config: DictConfig | dict[str, Any]) -> DictConfig:
    if isinstance(config, DictConfig):
        return OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    return OmegaConf.create(copy.deepcopy(config))


def build_draft_model_config(
    model_config: DictConfig | dict[str, Any],
    draft_config: DraftConfig,
) -> DictConfig:
    cfg = _copy_config(model_config)
    cfg.path = draft_config.model_path
    cfg.local_path = None
    cfg.hf_config_path = draft_config.model_path
    cfg.local_hf_config_path = None
    cfg.tokenizer_path = draft_config.tokenizer_path or draft_config.model_path
    cfg.local_tokenizer_path = None
    cfg.hf_config = None
    cfg.generation_config = None
    cfg.tokenizer = None
    cfg.processor = None
    cfg.architectures = None
    return cfg


def build_draft_rollout_config(
    rollout_config: DictConfig | dict[str, Any],
    draft_config: DraftConfig,
) -> DictConfig:
    cfg = _copy_config(rollout_config)
    cfg.name = "sglang"
    cfg.nnodes = draft_config.nnodes
    cfg.n_gpus_per_node = draft_config.ngpus_per_node
    cfg.tensor_model_parallel_size = draft_config.tp_size
    cfg.data_parallel_size = 1
    cfg.pipeline_model_parallel_size = 1
    cfg.expert_parallel_size = 1
    cfg.load_format = draft_config.load_format
    cfg.enable_decoupled_spec = True
    return cfg


def validate_decoupled_spec_config(rollout_config: Any, draft_config: DraftConfig):
    if rollout_config.name != "sglang":
        raise ValueError("rollout.enable_decoupled_spec=True only supports rollout.name=sglang")
    if not draft_config.enable:
        raise ValueError("draft.enable must be True when rollout.enable_decoupled_spec=True")
    if rollout_config.data_parallel_size != 1:
        raise ValueError("decoupled speculative decoding currently requires verifier data_parallel_size == 1")
    if rollout_config.pipeline_model_parallel_size != 1:
        raise ValueError("decoupled speculative decoding currently requires verifier pipeline_model_parallel_size == 1")
    mtp_config = rollout_config.get("mtp", None)
    if mtp_config is not None and mtp_config.get("enable", False) and mtp_config.get("enable_rollout", False):
        raise ValueError("decoupled speculative decoding cannot be enabled together with MTP rollout")
    if draft_config.tp_size > draft_config.ngpus_per_node:
        raise ValueError(
            f"draft.tp_size ({draft_config.tp_size}) must be <= draft.ngpus / draft.nnodes "
            f"({draft_config.ngpus_per_node})"
        )
