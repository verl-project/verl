# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from verl.base_config import BaseConfig

__all__ = ["TeacherConfig", "MOPDConfig"]


@dataclass
class TeacherConfig(BaseConfig):
    """Configuration for a single teacher model in MOPD.

    Args:
        name: Unique teacher identifier (e.g., "math", "code")
        model_path: HuggingFace model path or local checkpoint
        weight: Teacher weight for weighted composition (unused in current impl)
        resource_pool: Ray resource pool name (default: "global_pool")
        log_prob_micro_batch_size: Micro-batch size for teacher forward pass
        base_model_path: Optional base model for ExOPD normalization
    """

    name: str = ""
    model_path: str = ""
    weight: float = 1.0
    resource_pool: str = "global_pool"
    log_prob_micro_batch_size: int = 8
    base_model_path: Optional[str] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("TeacherConfig.name must be non-empty")
        if not self.model_path:
            raise ValueError("TeacherConfig.model_path must be non-empty")
        if self.weight <= 0:
            raise ValueError(f"TeacherConfig.weight must be positive: {self.weight}")


@dataclass
class MOPDConfig(BaseConfig):
    """Configuration for Multi-Teacher On-Policy Distillation.

    Implements MiMo paper (arXiv:2601.02780) Eq. 7-9 + G-OPD ExOPD mode.

    Args:
        enabled: Enable MOPD training (default: False for backward compat)
        teachers: List of teacher configurations
        lambda_val: G-OPD scaling coefficient (1.0=standard MOPD, >1.0=extrapolation)
        orm_weight: Weight for outcome reward mixing (α in A_final = A_mopd + α·A_orm)
        is_correction: Enable importance sampling correction for train/inference mismatch
        is_epsilon_low: Lower bound for IS ratio acceptance
        is_epsilon_high: Upper bound for IS ratio acceptance
        use_base_normalization: Enable ExOPD base model normalization
        base_model_path: Path to base model for ExOPD (shared across teachers)
    """

    enabled: bool = False
    teachers: list[TeacherConfig] = field(default_factory=list)
    lambda_val: float = 1.0
    orm_weight: float = 0.0
    is_correction: bool = True
    is_epsilon_low: float = 0.1
    is_epsilon_high: float = 10.0
    use_base_normalization: bool = False
    base_model_path: Optional[str] = None

    def __post_init__(self):
        # Validate non-empty teachers when enabled
        if self.enabled and len(self.teachers) == 0:
            raise ValueError(
                "MOPD enabled=True requires at least one teacher. " "Add teachers to algorithm.mopd.teachers."
            )

        # Validate unique teacher names
        names = [t.name for t in self.teachers]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate teacher names: {set(duplicates)}")

        # Validate lambda_val
        if self.lambda_val <= 0:
            raise ValueError(f"lambda_val must be positive: {self.lambda_val}")

        # Validate IS epsilon bounds
        if self.is_epsilon_low >= self.is_epsilon_high:
            raise ValueError(
                f"is_epsilon_low ({self.is_epsilon_low}) must be < " f"is_epsilon_high ({self.is_epsilon_high})"
            )

        # Validate base normalization config
        if self.use_base_normalization and not self.base_model_path:
            raise ValueError("use_base_normalization=True requires base_model_path to be set")
