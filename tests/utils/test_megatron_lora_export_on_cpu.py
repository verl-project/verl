# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest
import torch

from verl.utils import megatron_peft_utils as peft_utils

pytestmark = pytest.mark.cpu


def test_build_peft_config_for_vllm_expands_megatron_target_modules():
    peft_config = peft_utils.build_peft_config_for_vllm(
        {
            "rank": 16,
            "alpha": 32,
            "target_modules": ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        }
    )

    assert peft_config["r"] == 16
    assert peft_config["lora_alpha"] == 32
    assert peft_config["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_gather_ep_lora_adapter_weights_passthrough_without_ep():
    params = [
        ("thinker.model.layers.0.self_attn.q_proj.lora_A.weight", torch.tensor([1.0])),
        ("thinker.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight", torch.tensor([2.0])),
    ]

    result = list(peft_utils.gather_ep_lora_adapter_weights_for_vllm(iter(params)))

    assert [(name, tensor.item()) for name, tensor in result] == [
        ("thinker.model.layers.0.self_attn.q_proj.lora_A.weight", 1.0),
        ("thinker.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight", 2.0),
    ]


def test_gather_ep_lora_adapter_weights_renames_local_to_global_experts(monkeypatch):
    ep_group = object()
    monkeypatch.setattr(peft_utils, "_get_expert_parallel_info", lambda: (2, 0, ep_group))

    def fake_all_gather(tensor, ep_size, group):
        assert ep_size == 2
        assert group is ep_group
        return [tensor, tensor + 100]

    monkeypatch.setattr(peft_utils, "_all_gather_tensor_for_ep", fake_all_gather)
    params = [
        ("thinker.model.layers.0.self_attn.q_proj.lora_A.weight", torch.tensor([1.0])),
        ("thinker.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight", torch.tensor([2.0])),
        ("thinker.model.layers.0.mlp.experts.1.gate_proj.lora_A.weight", torch.tensor([3.0])),
    ]

    result = list(peft_utils.gather_ep_lora_adapter_weights_for_vllm(iter(params)))

    assert [(name, tensor.item()) for name, tensor in result] == [
        ("thinker.model.layers.0.self_attn.q_proj.lora_A.weight", 1.0),
        ("thinker.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight", 2.0),
        ("thinker.model.layers.0.mlp.experts.2.gate_proj.lora_A.weight", 102.0),
        ("thinker.model.layers.0.mlp.experts.1.gate_proj.lora_A.weight", 3.0),
        ("thinker.model.layers.0.mlp.experts.3.gate_proj.lora_A.weight", 103.0),
    ]


def test_pack_3d_moe_lora_adapter_weights_for_vllm_packs_complete_expert_groups():
    rank = 2
    hidden = 3
    intermediate = 4
    params = [("thinker.model.layers.0.self_attn.q_proj.lora_A.weight", torch.full((rank, hidden), -1.0))]
    for expert_id in range(2):
        shared_a = torch.full((rank, hidden), float(expert_id + 1))
        params.extend(
            [
                (f"thinker.model.layers.0.mlp.experts.{expert_id}.gate_proj.lora_A.weight", shared_a),
                (
                    f"thinker.model.layers.0.mlp.experts.{expert_id}.gate_proj.lora_B.weight",
                    torch.full((intermediate, rank), 10.0 + expert_id),
                ),
                (f"thinker.model.layers.0.mlp.experts.{expert_id}.up_proj.lora_A.weight", shared_a.clone()),
                (
                    f"thinker.model.layers.0.mlp.experts.{expert_id}.up_proj.lora_B.weight",
                    torch.full((intermediate, rank), 20.0 + expert_id),
                ),
                (
                    f"thinker.model.layers.0.mlp.experts.{expert_id}.down_proj.lora_A.weight",
                    torch.full((rank, intermediate), 30.0 + expert_id),
                ),
                (
                    f"thinker.model.layers.0.mlp.experts.{expert_id}.down_proj.lora_B.weight",
                    torch.full((hidden, rank), 40.0 + expert_id),
                ),
            ]
        )

    result = list(peft_utils.pack_3d_moe_lora_adapter_weights_for_vllm(iter(params), model_type="qwen3_omni_moe"))
    tensors = dict(result)

    assert [name for name, _tensor in result] == [
        "thinker.model.layers.0.self_attn.q_proj.lora_A.weight",
        "thinker.model.layers.0.mlp.experts.base_layer.lora_A.weight",
        "thinker.model.layers.0.mlp.experts.base_layer.lora_B.weight",
        "thinker.model.layers.0.mlp.experts.lora_A.weight",
        "thinker.model.layers.0.mlp.experts.lora_B.weight",
    ]
    assert tensors["thinker.model.layers.0.mlp.experts.base_layer.lora_A.weight"].shape == (2 * rank, hidden)
    assert tensors["thinker.model.layers.0.mlp.experts.base_layer.lora_B.weight"].shape == (
        2 * intermediate,
        2 * rank,
    )
    assert tensors["thinker.model.layers.0.mlp.experts.lora_A.weight"].shape == (2 * rank, intermediate)
    assert tensors["thinker.model.layers.0.mlp.experts.lora_B.weight"].shape == (hidden, 2 * rank)


def test_pack_3d_moe_lora_adapter_weights_for_vllm_rejects_split_gate_up_lora_a():
    params = [
        ("thinker.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight", torch.ones(2, 3)),
        ("thinker.model.layers.0.mlp.experts.0.gate_proj.lora_B.weight", torch.ones(4, 2)),
        ("thinker.model.layers.0.mlp.experts.0.up_proj.lora_A.weight", torch.zeros(2, 3)),
        ("thinker.model.layers.0.mlp.experts.0.up_proj.lora_B.weight", torch.ones(4, 2)),
        ("thinker.model.layers.0.mlp.experts.0.down_proj.lora_A.weight", torch.ones(2, 4)),
        ("thinker.model.layers.0.mlp.experts.0.down_proj.lora_B.weight", torch.ones(3, 2)),
    ]

    with pytest.raises(ValueError, match="different gate_proj and up_proj lora_A"):
        list(peft_utils.pack_3d_moe_lora_adapter_weights_for_vllm(iter(params), model_type="qwen3_omni_moe"))
