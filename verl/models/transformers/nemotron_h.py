# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright Amazon.com
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
"""
Patch for NemotronH models to enable flash_attention_2 support.

The HuggingFace NemotronH model doesn't declare _supports_flash_attn_2 = True,
but the model architecture does support it. This patch enables flash attention 2
support by patching the model class before instantiation.

Reference: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/discussions
"""

import sys


def patch_nemotron_h_flash_attention_support(model_config):
    """
    Patch NemotronH model to support flash_attention_2.

    This function patches the NemotronHPreTrainedModel class to declare
    flash attention 2 support. Must be called AFTER loading the config
    (which imports the model module) but BEFORE calling from_pretrained().

    Args:
        model_config: The model config object from AutoConfig.from_pretrained()
    """
    try:
        # Force-load the modeling module using transformers' dynamic module utilities
        # At this point, only the config module is loaded, we need to load the modeling module
        if hasattr(model_config, "auto_map") and "AutoModelForCausalLM" in model_config.auto_map:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            module_path = model_config.auto_map["AutoModelForCausalLM"]

            # Force import the modeling module by getting the class
            # This will load modeling_nemotron_h into sys.modules
            try:
                # We don't actually need the class, just need to trigger the import
                _ = get_class_from_dynamic_module(
                    class_reference=module_path,
                    pretrained_model_name_or_path=model_config.name_or_path,
                )
            except Exception as e:
                print(f"Error loading modeling module: {e}")

        # Now search for the modeling module which should be loaded
        nemotron_module = None
        for module_name, module in sys.modules.items():
            if (
                "transformers_modules" in module_name
                and "nemotron" in module_name.lower()
                and "modeling" in module_name
            ):
                if hasattr(module, "NemotronHPreTrainedModel"):
                    nemotron_module = module
                    break

        if nemotron_module is not None:
            # Patch the base class to support flash attention 2
            if hasattr(nemotron_module, "NemotronHPreTrainedModel"):
                nemotron_module.NemotronHPreTrainedModel._supports_flash_attn_2 = True
            else:
                print("[NemotronH Patch] Warning: Could not find NemotronHPreTrainedModel class to patch")
        else:
            print("[NemotronH Patch] Warning: Could not find NemotronH modeling module to patch")

    except Exception as e:
        print(f"[NemotronH Patch] Warning: Failed to patch NemotronH for flash attention support: {e}")
        # Don't raise - let the model loading continue and fail naturally if flash attention is truly unsupported
