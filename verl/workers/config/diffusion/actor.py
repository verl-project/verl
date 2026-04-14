# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig

from ..engine import FSDPEngineConfig
from ..optimizer import OptimizerConfig
from .model import DiffusionModelConfig

__all__ = [
    "DiffusionLossConfig",
    "DiffusionActorConfig",
    "FSDPDiffusionActorConfig",
]


@dataclass
class DiffusionLossConfig(BaseConfig):
    """Configuration for diffusion policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'flow_grpo'.
        clip_ratio (float): PPO clipping ratio for FlowGRPO policy loss.
        adv_clip_max (float): Maximum value to clamp advantages before computing policy loss.
    """

    loss_mode: str = "flow_grpo"
    clip_ratio: float = 0.0001
    adv_clip_max: float = 5.0

    def __post_init__(self):
        """Validate diffusion loss configuration."""
        valid_modes = ["flow_grpo"]
        if self.loss_mode not in valid_modes:
            raise ValueError(f"Invalid diffusion loss_mode: {self.loss_mode}. Must be one of {valid_modes}")


@dataclass
class DiffusionActorConfig(BaseConfig):
    """Configuration for diffusion actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for gradient accumulation.
        diffusion_loss (DiffusionLossConfig): Diffusion loss config (loss mode, clip ratio, adv clamp).
        use_kl_loss (bool): Whether to use KL loss.
        kl_loss_coef (float): KL loss coefficient.
        ppo_epochs (int): Number of PPO epochs per batch.
        shuffle (bool): Shuffle training data across PPO epochs.
        data_loader_seed (int): Seed for data loader.
        checkpoint (CheckpointConfig): Checkpoint config.
        optim (OptimizerConfig): Optimizer config.
        log_prob_micro_batch_size_per_gpu (Optional[int]): Log-prob micro batch size per GPU for ref.
        engine (BaseConfig): Engine config (set in subclass __post_init__).
        rollout_n (int): Number of rollouts per update.
        model_config (DiffusionModelConfig): Model config (set at runtime by engine_workers.py).
        global_batch_info (dict): Global batch info for loss aggregation (set at runtime).
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    diffusion_loss: DiffusionLossConfig = field(default_factory=DiffusionLossConfig)
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 42
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING  # must be override by sampling config
    model_config: DiffusionModelConfig = field(default_factory=DiffusionModelConfig)
    log_prob_micro_batch_size_per_gpu: Optional[int] = None

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate diffusion actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING


@dataclass
class FSDPDiffusionActorConfig(DiffusionActorConfig):
    """Configuration for FSDP diffusion actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy: fsdp or fsdp2.
        grad_clip (float): Gradient clipping threshold for actor updates.
        fsdp_config (FSDPEngineConfig): FSDP engine config.
    """

    # Training strategy: fsdp or fsdp2
    strategy: str = "fsdp"
    grad_clip: float = 1.0
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)

    def __post_init__(self):
        """Validate diffusion FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.fsdp_config
        # Sync strategy to engine config so engine_workers can pick the right FSDP version.
        # EngineConfig.strategy defaults to None, so without this, engine_workers.py always
        # falls back to FSDP1 even when actor.strategy="fsdp2".
        object.__setattr__(self.engine, "strategy", self.strategy)
