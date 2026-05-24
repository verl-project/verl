# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Internal helpers shared by the verl-side TRT-LLM rollout adapter and the
TRT-LLM rollout worker extension for W4A4 (NVFP4 QAT) weight sync over IPC."""

import json
import torch


def _register_float8_storage_stubs() -> None:
    """Register float8_e4m3fn / float8_e5m2 in torch's legacy storage pickle
    maps and create UntypedStorage stub classes so the legacy save/load path
    round-trips W4A4 weight_scale tensors via IPC.

    Must run on both the sender (verl rollout worker) and receiver (TRT-LLM
    rollout worker). Safe to call multiple times.
    """
    try:
        from torch import storage as torch_storage
        fwd = torch_storage._dtype_to_storage_type_map
        bwd = torch_storage._storage_type_to_dtype_map
        if callable(fwd):
            fwd = fwd()
        if callable(bwd):
            bwd = bwd()
        for dt, name in (
            (torch.float8_e4m3fn, "Float8_e4m3fnStorage"),
            (torch.float8_e5m2, "Float8_e5m2Storage"),
        ):
            if dt not in fwd:
                fwd[dt] = name
            if name not in bwd:
                bwd[name] = dt
            if not hasattr(torch, name):
                stub = type(name, (torch.UntypedStorage,), {"dtype": dt, "__module__": "torch"})
                setattr(torch, name, stub)
    except (AttributeError, ImportError):
        pass


def build_nvfp4_quantization_config(qat_cfg) -> dict:
    """Build the HF/TRT-LLM quantization_config dict from a verl QAT config.

    `qat_cfg` is the rollout-side QAT sub-config (OmegaConf-style with
    `.get(...)`). Returns the dict that goes into either
    `engine_kwargs["model_kwargs"]["quantization_config"]` (TRT-LLM side) or
    `model_config.hf_config.quantization_config` (verl ServerAdapter side).
    """
    quant_json_path = qat_cfg.get("quantization_config_path")
    with open(quant_json_path) as f:
        quant_json = json.load(f)
    group_size = (
        (quant_json.get("config_groups", {}).get("group_0", {}).get("weights", {}) or {})
        .get("group_size", 16)
    )
    return {
        "quant_method": "nvfp4",
        "group_size": group_size,
        "modules_to_not_convert": quant_json.get("ignore", []) or ["lm_head"],
    }


_register_float8_storage_stubs()
