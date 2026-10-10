# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
import torch

from verl.utils.rollout_weight_dtype import RolloutWeightDtypeContract


def metadata():
    names = [
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.mlp.experts.routed_experts.w13_weight",
        "model.layers.0.mlp.experts.routed_experts.w2_weight",
        "model.layers.0.mlp.gate.weight",
    ]
    worker = {
        "rank": 0,
        "model_type": "qwen3_moe",
        "model_class": "Qwen3MoeForCausalLM",
        "model_dtype": "torch.bfloat16",
        "quantization": None,
        "parameters": [{"name": name, "dtype": "torch.bfloat16", "shape": [8, 4]} for name in names],
        "buffers": [{"name": "model.layers.0.self_attn._k_scale", "dtype": "torch.float32", "shape": []}],
    }
    workers = [deepcopy(worker) for _ in range(4)]
    for rank, row in enumerate(workers):
        row["rank"] = rank
        for param in row["parameters"]:
            if ".routed_experts." in param["name"]:
                param["shape"] = [128, 8, 4]
    return workers


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.experts.127.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.32.down_proj.weight",
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    ],
)
def test_hf_and_stacked_aliases(name):
    contract = RolloutWeightDtypeContract.from_workers(metadata(), tensor_parallel_size=4)
    assert contract.dtype_for_export(name, is_parameter=True) == torch.bfloat16


def test_preserve_actual_fp32_router_and_buffers():
    workers = metadata()
    for worker in workers:
        worker["parameters"][-1]["dtype"] = "torch.float32"
    contract = RolloutWeightDtypeContract.from_workers(workers, tensor_parallel_size=4)
    assert contract.dtype_for_export("model.layers.0.mlp.gate.weight", is_parameter=True) == torch.float32
    assert contract.dtype_for_export("model.layers.0.self_attn._k_scale", is_parameter=False) is None
    with pytest.raises(TypeError):
        contract.parameters["extra"] = "torch.bfloat16"


@pytest.mark.parametrize(
    "fault", ["missing_rank", "duplicate_rank", "dtype", "name", "quant", "model", "duplicate_name", "shape", "overlap"]
)
def test_reject_incomplete_or_inconsistent_metadata(fault):
    workers = metadata()
    if fault == "missing_rank":
        workers.pop()
    elif fault == "duplicate_rank":
        workers[1]["rank"] = 0
    elif fault == "dtype":
        workers[1]["parameters"][0]["dtype"] = "torch.float32"
    elif fault == "name":
        workers[1]["parameters"].pop()
    elif fault == "quant":
        workers[1]["quantization"] = "fp8"
    elif fault == "model":
        workers[1]["model_type"] = "unknown"
    elif fault == "duplicate_name":
        workers[1]["parameters"].append(workers[1]["parameters"][0])
    elif fault == "shape":
        workers[1]["parameters"][0]["shape"] = [-1]
    else:
        workers[1]["buffers"].append(workers[1]["parameters"][0])
    with pytest.raises(ValueError):
        RolloutWeightDtypeContract.from_workers(workers, tensor_parallel_size=4)


@pytest.mark.parametrize("parameter", [False, True])
def test_unknown_exports_rejected(parameter):
    contract = RolloutWeightDtypeContract.from_workers(metadata(), tensor_parallel_size=4)
    with pytest.raises(ValueError):
        contract.dtype_for_export("unknown.weight", is_parameter=parameter)


def test_cast_before_expert_split_has_identical_cpu_bytes():
    # Includes overflow, signed zero and subnormal rounding; no GPU equivalence claim.
    tensor = torch.tensor([0.0, -0.0, 1e-40, 3.4e38, -1.003, 1.003, float("inf"), float("nan")])
    before = tensor.to(torch.bfloat16).chunk(2)
    after = [part.to(torch.bfloat16) for part in tensor.chunk(2)]
    assert all(
        torch.equal(left.view(torch.uint8), right.view(torch.uint8)) for left, right in zip(before, after, strict=True)
    )


def test_export_coverage_and_expert_range():
    workers = metadata()
    contract = RolloutWeightDtypeContract.from_workers(workers, tensor_parallel_size=4)
    names = [row["name"] for row in workers[0]["parameters"]]
    contract.validate_export_parameters(names)
    with pytest.raises(ValueError, match="missing receiver"):
        contract.validate_export_parameters(names[:-1])
    with pytest.raises(ValueError, match="duplicated"):
        contract.validate_export_parameters(names + names[:1])
    with pytest.raises(ValueError, match="outside the receiver"):
        contract.dtype_for_export("model.layers.0.mlp.experts.128.gate_proj.weight", is_parameter=True)


def test_fused_target_coverage_requires_every_split_projection():
    workers = metadata()
    contract = RolloutWeightDtypeContract.from_workers(workers, tensor_parallel_size=4)
    prefix = "model.layers.0."
    names = [prefix + "self_attn." + part + "_proj.weight" for part in "qkv"]
    names += [prefix + "mlp.gate.weight"]
    names += [prefix + f"mlp.experts.{i}.{part}_proj.weight" for i in range(128) for part in ("gate", "up", "down")]
    contract.validate_export_parameters(names)
    for missing in (
        prefix + "self_attn.k_proj.weight",
        prefix + "mlp.experts.72.up_proj.weight",
        prefix + "mlp.experts.2.down_proj.weight",
    ):
        with pytest.raises(ValueError, match="missing"):
            contract.validate_export_parameters([name for name in names if name != missing])
