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
        if self.ngpus % self.tp_size != 0:
            raise ValueError("draft.ngpus must be divisible by draft.tp_size")

    @property
    def num_drafters(self) -> int:
        if self.tp_size <= 0:
            return 0
        return self.ngpus // self.tp_size


def get_draft_config(config: DictConfig | dict[str, Any] | None) -> DraftConfig:
    if config is None:
        return DraftConfig()

    draft_cfg = config.get("draft") if isinstance(config, (DictConfig, dict)) else None
    if draft_cfg is None:
        return DraftConfig()

    structured = OmegaConf.structured(DraftConfig)
    merged = OmegaConf.merge(structured, draft_cfg)
    return OmegaConf.to_object(merged)


def build_draft_model_config(
    model_config: DictConfig | dict[str, Any],
    draft_config: DraftConfig,
) -> DictConfig:
    if isinstance(model_config, DictConfig):
        cfg = OmegaConf.create(OmegaConf.to_container(model_config, resolve=False))
    else:
        cfg = OmegaConf.create(copy.deepcopy(model_config))

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
