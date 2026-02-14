# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
from typing import Any, Callable

import torch


@dataclass
class DiffusionOutput:
    """
    Final output (after pipeline completion)
    """

    output: torch.Tensor | None = None
    all_latents: torch.Tensor | None = None
    all_log_probs: torch.Tensor | None = None
    all_timesteps: torch.Tensor | None = None
    prompt_embeds: torch.Tensor | None = None
    prompt_embeds_mask: torch.Tensor | None = None
    negative_prompt_embeds: torch.Tensor | None = None
    negative_prompt_embeds_mask: torch.Tensor | None = None

    # default variables
    trajectory_timesteps: list[torch.Tensor] | None = None
    trajectory_latents: torch.Tensor | None = None
    trajectory_decoded: list[torch.Tensor] | None = None
    error: str | None = None
    post_process_func: Callable[..., Any] | None = None
