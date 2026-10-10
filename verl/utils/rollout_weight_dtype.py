# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Receiver-verified export dtype contract; constructing it never changes weights.

This initial contract supports unquantized Qwen3 MoE receivers only. Metadata
must come from every actual TP worker, rather than from model configuration.
The caller must finish preflight on every rollout replica before collectives.
"""

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

_DTYPES = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}


def worker_weight_dtype_metadata(worker) -> dict:
    """Read actual parameter and buffer metadata without accessing tensor values."""
    runner = worker.model_runner
    config = runner.vllm_config
    model = runner.model
    if config.quant_config is not None:
        raise ValueError("Quantized receivers cannot use export_receiver_dtype")
    if getattr(config, "lora_config", None) is not None:
        raise ValueError("LoRA receivers cannot use export_receiver_dtype")

    def entries(tensors):
        return [{"name": name, "dtype": str(t.dtype), "shape": list(t.shape)} for name, t in tensors]

    return {
        "rank": worker.rank,
        "model_type": config.model_config.hf_config.model_type,
        "model_class": type(model).__name__,
        "model_dtype": str(config.model_config.dtype),
        "quantization": None,
        "parameters": entries(model.named_parameters()),
        "buffers": entries(model.named_buffers()),
    }


def stage_export_tensor(tensor, device, dtype):
    """Cast a temporary shard before gathering; preserve the source tensor."""
    if dtype is None:
        return tensor.to(device, non_blocking=True)
    return tensor.to(device=device, dtype=dtype, non_blocking=True)


def _target_name(name: str) -> str:
    name = re.sub(r"\.self_attn\.[qkv]_proj\.weight$", ".self_attn.qkv_proj.weight", name)
    name = re.sub(
        r"\.mlp\.experts\.\d+\.(gate_proj|up_proj)\.weight$",
        ".mlp.experts.routed_experts.w13_weight",
        name,
    )
    name = re.sub(
        r"\.mlp\.experts\.\d+\.down_proj\.weight$",
        ".mlp.experts.routed_experts.w2_weight",
        name,
    )
    # VeOmni stacked experts are split into the HF aliases above after gathering.
    name = re.sub(r"\.mlp\.experts\.gate_up_proj$", ".mlp.experts.routed_experts.w13_weight", name)
    return re.sub(r"\.mlp\.experts\.down_proj$", ".mlp.experts.routed_experts.w2_weight", name)


def _entries(rows: list[dict]) -> dict[str, str]:
    result = {}
    if not rows:
        raise ValueError("Receiver tensor metadata is empty")
    for row in rows:
        name, dtype, shape = row["name"], row["dtype"], row["shape"]
        if not isinstance(name, str) or not name or name in result:
            raise ValueError("Receiver tensor names must be nonempty and unique")
        if dtype not in _DTYPES:
            raise ValueError(f"Unsupported receiver dtype for {name}: {dtype}")
        if not isinstance(shape, list) or any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ValueError(f"Invalid receiver shape for {name}")
        result[name] = dtype
    return result


@dataclass(frozen=True)
class RolloutWeightDtypeContract:
    """Locally built immutable contract; serialize raw metadata over RPC instead."""

    parameters: Mapping[str, str]
    buffers: Mapping[str, str]
    expert_counts: Mapping[str, int]

    @classmethod
    def from_workers(cls, workers: list[dict], *, tensor_parallel_size: int):
        if type(tensor_parallel_size) is not int or tensor_parallel_size <= 0:
            raise ValueError("Invalid tensor parallel size")
        if len(workers) != tensor_parallel_size:
            raise ValueError("Metadata must include every TP worker")
        ranks = [worker["rank"] for worker in workers]
        if any(type(rank) is not int for rank in ranks) or set(ranks) != set(range(tensor_parallel_size)):
            raise ValueError("Receiver TP ranks are missing or duplicated")
        reference = None
        for worker in workers:
            if (
                worker["model_type"] != "qwen3_moe"
                or worker["model_class"] != "Qwen3MoeForCausalLM"
                or worker["model_dtype"] != "torch.bfloat16"
                or worker["quantization"] is not None
            ):
                raise ValueError("Unsupported receiver model or quantization")
            params, buffers = _entries(worker["parameters"]), _entries(worker["buffers"])
            experts = {
                row["name"]: row["shape"][0]
                for row in worker["parameters"]
                if row["name"].endswith((".routed_experts.w13_weight", ".routed_experts.w2_weight"))
                and len(row["shape"]) == 3
            }
            expert_names = {name for name in params if ".routed_experts." in name}
            if set(experts) != expert_names:
                raise ValueError("Receiver expert weights must have an expert dimension")
            if params.keys() & buffers.keys():
                raise ValueError("Receiver parameter and buffer names overlap")
            if reference is None:
                reference = (params, buffers, experts)
            elif reference != (params, buffers, experts):
                raise ValueError("Receiver TP workers disagree about names or dtypes")
        return cls(*(MappingProxyType(mapping) for mapping in reference))

    def dtype_for_export(self, name: str, *, is_parameter: bool) -> torch.dtype | None:
        """Return a verified parameter dtype; known buffers keep their original dtype."""
        if not is_parameter:
            if name not in self.buffers:
                raise ValueError(f"Unknown exported buffer: {name}")
            return None
        target = _target_name(name)
        if target not in self.parameters:
            raise ValueError(f"Unknown exported parameter: {name} (receiver {target})")
        expert = re.search(r"\.mlp\.experts\.(\d+)\.", name)
        if expert is not None and int(expert.group(1)) >= self.expert_counts[target]:
            raise ValueError(f"Exported expert is outside the receiver range: {name}")
        return _DTYPES[self.parameters[target]]

    def validate_export_parameters(self, names: list[str]) -> None:
        """Require the export name set to cover every measured receiver parameter."""
        if len(names) != len(set(names)):
            raise ValueError("Exported parameter names are duplicated")
        for name in names:
            self.dtype_for_export(name, is_parameter=True)
        name_set = set(names)
        for name in names:
            qkv = re.match(r"(.*\.self_attn\.)[qkv]_proj\.weight$", name)
            if qkv and not {qkv[1] + part + "_proj.weight" for part in "qkv"} <= name_set:
                raise ValueError("Export is missing a Q/K/V projection")
        for target, count in self.expert_counts.items():
            prefix, kind = target.split(".routed_experts.")
            individual = [
                name for name in names if name.startswith(prefix + ".") and re.search(r"\.\d+\.", name[len(prefix) :])
            ]
            if not individual:
                continue
            parts = ("gate_proj", "up_proj") if kind == "w13_weight" else ("down_proj",)
            required = {f"{prefix}.{expert}.{part}.weight" for expert in range(count) for part in parts}
            if not required <= name_set:
                raise ValueError("Export is missing an expert projection")
        missing = self.parameters.keys() - {_target_name(name) for name in names}
        if missing:
            raise ValueError(f"Export is missing receiver parameters: {sorted(missing)}")
