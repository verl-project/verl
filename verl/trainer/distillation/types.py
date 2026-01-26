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

from dataclasses import dataclass
from typing import Optional

import torch

from verl.base_config import BaseConfig


@dataclass
class DistillationLossInputs(BaseConfig):
    """Storage class for distillation loss inputs."""

    student_log_probs: Optional[torch.Tensor] = None
    teacher_log_probs: Optional[torch.Tensor] = None
    student_logits: Optional[torch.Tensor] = None
    teacher_logits: Optional[torch.Tensor] = None
    teacher_topk_indices: Optional[torch.Tensor] = None
