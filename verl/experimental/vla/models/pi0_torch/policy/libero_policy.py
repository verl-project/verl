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

import torch
from typing_extensions import override

from verl.protocol import DataProto

from .base import Pi0Input, Pi0Output

PI0_MAX_STATE_DIM = 32
PI0_ACTION_CHUNK_SIZE = 10
LIBERO_ACTION_DIM = 7


class LiberoPi0Input(Pi0Input):
    @override
    @classmethod
    def from_env_obs(cls, env_obs: DataProto) -> "LiberoPi0Input":
        input = cls()

        # Process images
        images = env_obs.batch["full_image"]
        wrist_images = env_obs.batch["wrist_image"]

        batch_size = images.shape[0]
        cam_high = images.permute(0, 3, 1, 2)
        left_wrist = wrist_images.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        empty_images = torch.zeros(
            (batch_size, 3, cam_high.shape[2], cam_high.shape[3]),
            device=env_obs.batch.device,
            dtype=torch.bfloat16,
        )

        input.images = {
            "observation.images.cam_high": cam_high.to(torch.bfloat16),
            "observation.images.cam_left_wrist": left_wrist.to(torch.bfloat16),
            "observation.images.cam_right_wrist": empty_images,
        }
        input.img_masks = [
            torch.ones((batch_size,), device=env_obs.batch.device, dtype=torch.bool),
            torch.ones((batch_size,), device=env_obs.batch.device, dtype=torch.bool),
            torch.zeros((batch_size,), device=env_obs.batch.device, dtype=torch.bool),
        ]

        # Process other data
        input.task = list(env_obs.non_tensor_batch["task_descriptions"])

        state = env_obs.batch["state"]
        input.state = torch.nn.functional.pad(
            state, (0, max(0, PI0_MAX_STATE_DIM - state.shape[-1])), "constant", 0
        ).to(env_obs.batch.device, dtype=torch.float32)

        return input


class LiberoPi0Output(Pi0Output):
    @override
    @classmethod
    def from_model_output(cls, model_output: dict) -> "LiberoPi0Output":
        output = cls()
        output.action = model_output["full_action"][:, :PI0_ACTION_CHUNK_SIZE, :LIBERO_ACTION_DIM]
        return output
