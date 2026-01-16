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

from verl.base_config import BaseConfig
from dataclasses import dataclass
from typing import Optional

__all__ = ["DistillationConfig", "FSDPDistillationConfig"]

@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for distillation training.
    TODO
    """
    enabled: bool = False
    loss_mode: str = "k3"
    topk: Optional[int] = 128
    use_policy_loss: bool = False
    distillation_loss_coef: float = 1.0


@dataclass
class FSDPDistillationConfig(DistillationConfig):
    """Configuration for distillation training with FSDP."""
    pass