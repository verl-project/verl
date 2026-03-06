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

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = ["CheckpointConfig", "ProfileConfig", "BaseModelConfig"]


@dataclass
class CheckpointConfig(BaseConfig):
    """Configuration for model checkpointing.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        save_contents (list[str]): What to include in saved checkpoints.
            Options: 'model', 'optimizer', 'extra', 'hf_model'.
        load_contents (list[str]): Contents to load from checkpoint. Defaults to same as save_contents.
        async_save (bool): Whether to save checkpoints asynchronously. Only implemented for Megatron as of now.
    """

    save_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    load_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    async_save: bool = False
    mbridge_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileConfig(BaseConfig):
    """Configuration for profiling.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        profile_ranks (Optional[list[int]]): List of ranks to profile. None means all ranks.
        step_start (int): Starting step for profiling.
        step_end (int): Ending step for profiling.
        save_path (Optional[str]): Path to save profiling results.
    """

    profile_ranks: Optional[list[int]] = None
    step_start: int = -1
    step_end: int = -1
    save_path: Optional[str] = None


@dataclass
class BaseModelConfig(BaseConfig):
    """Base configuration for a model.
    Contains core settings for loading and initializing a pretrained model checkpoint.

    Args:
        path (str): Path to pretrained model weights.
        tokenizer_path (Optional[str]): Tokenizer path (defaults to actor's model path if not set).
        override_config (dict): Hugging Face config override.
        external_lib (Optional[str]): External model implementation (optional).
        trust_remote_code (bool): Whether to trust remote code from Hugging Face models.
        lora (dict[str, Any]): LoRA configuration dictionary.
    """

    path: str = "~/models/deepseek-llm-7b-chat"
    tokenizer_path: Optional[str] = None
    override_config: dict[str, Any] = field(default_factory=dict)
    external_lib: Optional[str] = None
    trust_remote_code: bool = False
    lora: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleConfig(BaseConfig):
    """Configuration for external Python module, which can be loaded, executed (and optionally, ``import``ed).

    Args:
        path (str, optional): Path to the module file to load and execute.
        name (str, optional): Name of the module to ``import``. Format: ``"import.path.to.module"``.
            If ``None``, the module will be loaded with a hased name and
                will not be added to ``sys.modules``, thus can not be ``import``ed as ``name``.
    """

    path: Optional[str] = None
    name: Optional[str] = None
