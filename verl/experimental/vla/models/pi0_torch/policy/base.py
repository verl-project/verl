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

from abc import ABC, abstractmethod

import torch


class Pi0Input(ABC):
    def __init__(self):
        # three images for pi0 input with keys:
        # [
        #     'observation.images.cam_high',
        #     'observation.images.cam_left_wrist',
        #     'observation.images.cam_right_wrist',
        # ],
        # each with shape (B, C, H, W)
        self.images: dict[str, torch.Tensor] = {}

        # image masks corresponding to the images, each with shape (B,)
        self.img_masks: list[torch.Tensor] = []

        # task description as a list of strings
        self.task: list[str] = []

        # robot state with shape (B, state_dim)
        self.state: torch.Tensor = None

    @classmethod
    @abstractmethod
    def from_env_obs(cls, env_obs) -> "Pi0Input": ...


class Pi0Output:
    def __init__(self):
        self.action: torch.Tensor = None

    @classmethod
    @abstractmethod
    def from_model_output(cls, model_output) -> "Pi0Output": ...
