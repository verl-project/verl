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

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline

__all__ = ["prepare_pipeline", "load_to_device"]


def prepare_pipeline(pipeline: "DiffusionPipeline", dtype: torch.dtype) -> None:
    """Prepare the diffusion pipeline by casting to the specified dtype.

    Args:
        pipeline (DiffusionPipeline): The diffusion pipeline.
        dtype (torch.dtype): The target data type.
    """
    from diffusers import QwenImagePipeline

    if isinstance(pipeline, QwenImagePipeline):
        # Move vae and text_encoder to device and cast to inference_dtype
        pipeline.text_encoder.to(dtype=dtype)
        pipeline.transformer.to(dtype=dtype)
        # set eval mode
        pipeline.vae.eval()
        pipeline.text_encoder.eval()
        pipeline.transformer.eval()
    else:
        raise NotImplementedError()


def load_to_device(pipeline: "DiffusionPipeline", device: str) -> None:
    """Load the diffusion pipeline to the specified device.

    Args:
        pipeline (DiffusionPipeline): The diffusion pipeline.
        device (str): The target device.
    """
    from diffusers import QwenImagePipeline

    if isinstance(pipeline, QwenImagePipeline):
        # Move vae and text_encoder to device and cast to inference_dtype
        pipeline.vae.to(device)
        pipeline.text_encoder.to(device)
        pipeline.transformer.to(device)
    else:
        raise NotImplementedError()
