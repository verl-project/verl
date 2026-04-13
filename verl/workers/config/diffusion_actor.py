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

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig

from .actor import PolicyLossConfig
from .engine import FSDPEngineConfig
from .optimizer import OptimizerConfig

__all__ = [
    "DiffusionActorConfig",
    "FSDPDiffusersActorConfig",
]


@dataclass
class DiffusionActorConfig(BaseConfig):
    """Base actor config for diffusion training.

    Does NOT inherit from ActorConfig. Only contains fields actually used by
    the diffusion training pipeline (diffusion_algos.py, diffusion_loss,
    engine_workers.py, ray_diffusion_trainer.py).
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    # Training strategy
    strategy: str = MISSING

    # Number of rollouts per update
    rollout_n: int = MISSING

    # Mini-batch size for PPO training
    ppo_mini_batch_size: int = 256

    # Micro-batch size per GPU for gradient accumulation
    ppo_micro_batch_size_per_gpu: Optional[int] = None

    # PPO clip ratio (FlowGRPO-style; tighter than LLM default 0.2)
    clip_ratio: float = 0.0001

    # Maximum value to clamp advantages before computing policy loss
    adv_clip_max: float = 5.0

    # Policy loss config
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)

    # Scale factor for 'seq-mean-token-sum-norm' loss aggregation mode
    loss_scale_factor: Optional[int] = None

    # Whether to use KL loss
    use_kl_loss: bool = False

    # KL loss coefficient
    kl_loss_coef: float = 0.001

    # Number of PPO epochs per batch
    ppo_epochs: int = 1

    # Shuffle training data across PPO epochs
    shuffle: bool = False

    # Seed for data loader
    data_loader_seed: int = 42

    # Gradient clipping threshold
    grad_clip: float = 1.0

    # Optimizer config
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Checkpoint config
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    # Engine config (set in subclass __post_init__)
    engine: BaseConfig = field(default_factory=BaseConfig)

    # Model config (set at runtime by engine_workers.py)
    model_config: Any = field(default_factory=BaseConfig)

    # Global batch info for loss aggregation (set at runtime)
    global_batch_info: dict = field(default_factory=dict)

    # Profiler config
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

    def __post_init__(self):
        """Validate diffusion actor configuration."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING


@dataclass
class FSDPDiffusersActorConfig(DiffusionActorConfig):
    """FSDP actor config for diffusion training.

    Extends DiffusionActorConfig with FSDP-specific fields (fsdp_config,
    ulysses_sequence_parallel_size).
    """

    # Training strategy: fsdp or fsdp2
    strategy: str = "fsdp"

    # Ulysses sequence parallel size (for future use)
    ulysses_sequence_parallel_size: int = 1

    # FSDP engine config
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)

    def __post_init__(self):
        """Validate diffusion FSDP actor configuration."""
        super().__post_init__()
        self.engine = self.fsdp_config
        object.__setattr__(self.engine, "strategy", self.strategy)

        if self.ulysses_sequence_parallel_size > 1:
            self.fsdp_config.ulysses_sequence_parallel_size = self.ulysses_sequence_parallel_size
