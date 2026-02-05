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


"""Utility helpers to register custom VLA models with Hugging Face Auto classes."""

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from .openvla_oft.configuration_prismatic import OpenVLAConfig
from .openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
from .openvla_oft.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from .pi0_torch import PI0ForActionPrediction, PI0TorchConfig

_REGISTERED_MODELS = {
    "openvla_oft": False,
    "pi0_torch": False,
}


def register_openvla_oft() -> None:
    """Register the OpenVLA OFT model and processors."""
    if _REGISTERED_MODELS["openvla_oft"]:
        return

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    _REGISTERED_MODELS["openvla_oft"] = True


def register_pi0_torch_model() -> None:
    """Register the PI0 wrapper with the HF auto classes."""
    if _REGISTERED_MODELS["pi0_torch"]:
        return

    AutoConfig.register("pi0_torch", PI0TorchConfig)
    AutoModelForVision2Seq.register(PI0TorchConfig, PI0ForActionPrediction)

    _REGISTERED_MODELS["pi0_torch"] = True


def register_vla_models() -> None:
    """Register all custom VLA models with Hugging Face."""
    register_openvla_oft()
    register_pi0_torch_model()
