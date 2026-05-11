# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""ModelOpt integration for NVFP4 quantization with Megatron QAT training and vLLM inference."""

from importlib import import_module

_EXPORTS = {
    "megatron_qat_patch": ("apply_qat_patch", "revert_qat_patch"),
    "qat_utils": ("patch_provider_for_qat", "apply_qat_to_modules", "export_qat_weights"),
    "qat_weight_exporter": ("QATWeightExporter",),
    "quantize": ("build_quantize_config", "apply_qat"),
    "vllm_modelopt_patch": (
        "apply_modelopt_nvfp4_patches",
        "prepare_modelopt_for_weight_reload",
        "modelopt_process_weights_after_loading",
    ),
}
_MODULE_BY_ATTR = {attr: f"verl.utils.modelopt.{module}" for module, attrs in _EXPORTS.items() for attr in attrs}

__all__ = list(_MODULE_BY_ATTR)


def __getattr__(name):
    module_name = _MODULE_BY_ATTR.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    attr = getattr(import_module(module_name), name)
    globals()[name] = attr
    return attr
