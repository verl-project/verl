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


def patch_provider_for_qat(provider):
    """Patch the Megatron-Bridge provider to support QAT quantized layers."""
    from megatron.bridge.models.conversion.param_mapping import AutoMapping
    from megatron.bridge.models.gpt_provider import quantization_layer_spec

    from verl.utils.modelopt.megatron_qat_patch import apply_qat_patch

    provider.transformer_layer_spec = quantization_layer_spec
    apply_qat_patch()
    AutoMapping.register_module_type("QuantColumnParallelLinear", "column")
    AutoMapping.register_module_type("QuantRowParallelLinear", "row")


def apply_qat_to_modules(modules, qat_mode, ignore_patterns=None):
    """Apply ModelOpt fake quantization to a list of Megatron module chunks."""
    from verl.utils.modelopt.quantize import apply_qat

    for i in range(len(modules)):
        modules[i] = apply_qat(modules[i], qat_mode, ignore_patterns=ignore_patterns)
    return modules


def export_qat_weights(per_tensor_param, modules, qat_mode, bridge):
    """Process exported weights through QATWeightExporter for quantized weight sync."""
    from verl.utils.modelopt.qat_weight_exporter import QATWeightExporter

    qat_weight_exporter = QATWeightExporter(modules, qat_mode, bridge=bridge)
    return qat_weight_exporter.process_weights_iterator(per_tensor_param)
