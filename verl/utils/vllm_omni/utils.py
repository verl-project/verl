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


from msgspec import field

try:
    from vllm.lora.lora_model import LoRAModel
except ImportError:
    from vllm.lora.models import LoRAModel

from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import get_adapter_absolute_path
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager, logger
from vllm_omni.lora.request import LoRARequest as OmniLoRARequest


class OmniTensorLoRARequest(OmniLoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


class VLLMOmniHijack:
    @staticmethod
    def hijack():
        def hijack__load_adapter(self, lora_request: OmniTensorLoRARequest) -> tuple[LoRAModel, PEFTHelper]:
            """
            based on vllm_omni.diffusion.lora.manager.DiffusionLoRAManager._load_adapter,
            support load adapter with lora tensors

            Reason:
            VLLM-Omni does not support adding LoRA from tensors directly. It only supports adding LoRA via file paths.
            To synchronize the LoRA tensors of the actor model, we need to find a workaround to enable VLLM to
            load memory-based LoRA tensors.
            """
            if not self._expected_lora_modules:
                raise ValueError("No supported LoRA modules found in the diffusion pipeline.")

            logger.debug("Supported LoRA modules: %s", self._expected_lora_modules)

            lora_tensors = None

            if isinstance(lora_request, OmniTensorLoRARequest):
                peft_config = lora_request.peft_config
                lora_tensors = lora_request.lora_tensors
                peft_helper = PEFTHelper.from_dict(peft_config)
            else:
                lora_path = get_adapter_absolute_path(lora_request.lora_path)
                logger.debug("Resolved LoRA path: %s", lora_path)

                peft_helper = PEFTHelper.from_local_dir(
                    lora_path,
                    max_position_embeddings=None,  # no need in diffusion
                    tensorizer_config_dict=lora_request.tensorizer_config_dict,
                )

            logger.info(
                "Loaded PEFT config: r=%d, lora_alpha=%d, target_modules=%s",
                peft_helper.r,
                peft_helper.lora_alpha,
                peft_helper.target_modules,
            )

            if isinstance(lora_request, OmniTensorLoRARequest):
                lora_model = LoRAModel.from_lora_tensors(
                    tensors=lora_tensors,
                    peft_helper=peft_helper,
                    lora_model_id=lora_request.lora_int_id,
                    device="cpu",  # consistent w/ vllm's behavior
                    dtype=self.dtype,
                    model_vocab_size=None,
                    weights_mapper=None,
                )
            else:
                lora_model = LoRAModel.from_local_checkpoint(
                    lora_path,
                    expected_lora_modules=self._expected_lora_modules,
                    peft_helper=peft_helper,
                    lora_model_id=lora_request.lora_int_id,
                    device="cpu",  # consistent w/ vllm's behavior
                    dtype=self.dtype,
                    model_vocab_size=None,
                    tensorizer_config_dict=lora_request.tensorizer_config_dict,
                    weights_mapper=None,
                )

            logger.info(
                "Loaded LoRA model: id=%d, num_modules=%d, modules=%s",
                lora_model.id,
                len(lora_model.loras),
                list(lora_model.loras.keys()),
            )

            for lora in lora_model.loras.values():
                lora.optimize()  # ref: _create_merged_loras_inplace, internal scaling

            return lora_model, peft_helper

        def do_hijack(target_cls, target_method_name, hooking_method):
            setattr(target_cls, target_method_name, hooking_method)

        do_hijack(DiffusionLoRAManager, "_load_adapter", hijack__load_adapter)
