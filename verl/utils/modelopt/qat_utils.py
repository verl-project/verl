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

"""High-level QAT workflow helpers for Megatron backend."""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))  # Ray workers keep root at WARNING


def _check_mxfp4_experts(modules):
    """Fail fast if the experts-only MXFP4 config did not land where it should.

    (a) at least one routed-expert weight_quantizer enabled, (b) nothing else enabled (a mis-ordered
    cfg leaves every other linear on modelopt's default INT8 fake-quant), (c) fake-quant is the
    identity on the freshly loaded expert weights, which are already on the MXFP4 grid.
    """
    import torch
    from modelopt.torch.quantization.nn import TensorQuantizer

    n_expert_w, others, identity = 0, [], None
    for m in modules:
        for name, q in m.named_modules():
            if not isinstance(q, TensorQuantizer) or not getattr(q, "is_enabled", True):
                continue
            if name.endswith("weight_quantizer") and ".mlp.experts." in name:
                n_expert_w += 1
            else:
                others.append(name)
        if identity is None:
            for name, sub in m.named_modules():
                w0, wq = getattr(sub, "weight0", None), getattr(sub, "weight_quantizer", None)
                if w0 is not None and wq is not None and ".mlp.experts." in name:
                    try:
                        with torch.no_grad():
                            fq = wq(w0)
                            identity = (name, bool(torch.equal(fq, w0)), (fq.float() - w0.float()).abs().max().item())
                    except Exception as e:  # diagnostic only (e.g. weights not materialized yet)
                        identity = (name, "skipped", repr(e)[:120])
                    break
    logger.info(
        "[QAT mxfp4_experts] enabled expert weight_quantizers=%d, other enabled quantizers=%d, "
        "identity_check(name, equal, max_abs_diff)=%s",
        n_expert_w,
        len(others),
        identity,
    )
    has_experts = any(".mlp.experts." in n for m in modules for n, _ in m.named_modules())
    if has_experts and n_expert_w == 0:
        raise RuntimeError("QAT mxfp4_experts: no routed-expert weight_quantizer enabled -- pattern did not match")
    if others:
        raise RuntimeError(f"QAT mxfp4_experts: {len(others)} unexpected quantizers enabled, e.g. {others[:3]}")


def _get_qat_field(qat_config, key, default=None):
    """Extract a field from qat_config, supporting both dict and object-style access."""
    if isinstance(qat_config, dict):
        return qat_config.get(key, default)
    return getattr(qat_config, key, default)


def apply_qat_to_modules(modules, qat_config):
    """Apply ModelOpt fake quantization to a list of Megatron module chunks."""
    from verl.utils.modelopt.quantize import apply_qat

    qat_mode = _get_qat_field(qat_config, "mode", "w4a16")
    ignore_patterns = _get_qat_field(qat_config, "ignore_patterns", None)
    if ignore_patterns is not None:
        ignore_patterns = list(ignore_patterns)

    for i in range(len(modules)):
        modules[i] = apply_qat(modules[i], qat_mode, ignore_patterns=ignore_patterns)
    if qat_mode == "mxfp4_experts":
        _check_mxfp4_experts(modules)
    return modules


def export_qat_weights(per_tensor_param, modules, qat_mode, bridge):
    """Process exported weights through QATWeightExporter for quantized weight sync."""
    from verl.utils.modelopt.qat_weight_exporter import QATWeightExporter

    qat_weight_exporter = QATWeightExporter(modules, bridge, qat_mode)
    return qat_weight_exporter.process_weights_iterator(per_tensor_param)
