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


import torch

from verl.workers.engine.fsdp.utils import unfuse_moe_params


def _collect(weights, model_type, hidden_size=None, **kwargs):
    return [item for item in unfuse_moe_params(weights, model_type, hidden_size, **kwargs)]


def test_qwen_moe_packed_weights_are_expanded_per_expert():
    gate_up = torch.randn(2, 6, 8)
    down = torch.randn(2, 8, 3)

    converted = _collect(
        [
            ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
            ("model.layers.0.mlp.experts.down_proj", down),
        ],
        "qwen3_moe",
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.experts.1.down_proj.weight",
    ]
    assert [tensor.shape for _, tensor in converted] == [
        (3, 8),
        (3, 8),
        (3, 8),
        (3, 8),
        (8, 3),
        (8, 3),
    ]


def test_qwen_moe_npu_gmm_layout_is_normalized_before_expansion():
    # NPU patches (verl/models/transformers/npu_patch.py) keep experts packed as
    # [num_experts, hidden_size, 2 * intermediate_size] and
    # [num_experts, intermediate_size, hidden_size] for npu_grouped_matmul, i.e.
    # transposed w.r.t. the HF checkpoint layout that vLLM's loader consumes.
    hidden_size, intermediate = 8, 3
    gate_up = torch.randn(2, hidden_size, 2 * intermediate)
    down = torch.randn(2, intermediate, hidden_size)

    converted = _collect(
        [
            ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
            ("model.layers.0.mlp.experts.down_proj", down),
        ],
        "qwen3_vl_moe",
        hidden_size,
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.experts.1.down_proj.weight",
    ]
    assert [tensor.shape for _, tensor in converted] == [
        (intermediate, hidden_size),
        (intermediate, hidden_size),
        (intermediate, hidden_size),
        (intermediate, hidden_size),
        (hidden_size, intermediate),
        (hidden_size, intermediate),
    ]
    assert torch.equal(converted[0][1], gate_up[0][:, :intermediate].T.contiguous())
    assert torch.equal(converted[1][1], gate_up[0][:, intermediate:].T.contiguous())
    assert torch.equal(converted[4][1], down[0].T.contiguous())


def test_qwen_moe_ambiguous_shape_falls_back_to_standard_layout():
    # 2 * intermediate == hidden makes the two layouts shape-identical, so the
    # converter must keep the previous behaviour instead of guessing.
    hidden_size, intermediate = 8, 4
    gate_up = torch.randn(2, 2 * intermediate, hidden_size)
    down = torch.randn(2, hidden_size, intermediate)
    weights = [
        ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
        ("model.layers.0.mlp.experts.down_proj", down),
    ]

    with_hidden_size = _collect(weights, "qwen3_vl_moe", hidden_size)
    without_hidden_size = _collect(weights, "qwen3_vl_moe")

    assert [name for name, _ in with_hidden_size] == [name for name, _ in without_hidden_size]
    for (_, lhs), (_, rhs) in zip(with_hidden_size, without_hidden_size, strict=True):
        assert lhs.shape == rhs.shape
        assert torch.equal(lhs, rhs)


def test_qwen_moe_square_weights_use_provenance_not_shape():
    # 2 * intermediate == hidden makes gate_up_proj square, and intermediate == hidden
    # makes down_proj square, so the two layouts are shape-identical and only
    # npu_gmm_layout can select the NPU grouped-GEMM orientation.
    hidden_size, num_experts = 4, 2
    gate_up = torch.arange(num_experts * hidden_size * hidden_size, dtype=torch.float32).reshape(
        num_experts, hidden_size, hidden_size
    )
    down = gate_up.flip(1).contiguous()

    converted = dict(
        _collect(
            [
                ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
                ("model.layers.0.mlp.experts.down_proj", down),
            ],
            "qwen3_vl_moe",
            hidden_size,
            npu_gmm_layout=True,
        )
    )

    assert torch.equal(
        converted["model.layers.0.mlp.experts.0.gate_proj.weight"], gate_up[0, :, :2].T.contiguous()
    )
    assert torch.equal(
        converted["model.layers.0.mlp.experts.0.up_proj.weight"], gate_up[0, :, 2:].T.contiguous()
    )
    assert torch.equal(converted["model.layers.0.mlp.experts.0.down_proj.weight"], down[0].T.contiguous())

    # Without provenance the same square tensor keeps the previous (standard) split,
    # which proves the flag, not the shape, decided the layout above.
    legacy = dict(
        _collect([("model.layers.0.mlp.experts.gate_up_proj", gate_up)], "qwen3_vl_moe", hidden_size)
    )
    assert torch.equal(legacy["model.layers.0.mlp.experts.0.gate_proj.weight"], gate_up[0, :2].contiguous())


def test_gpt_oss_packed_weights_are_not_expanded():
    gate_up = torch.randn(2, 8, 6)
    down = torch.randn(2, 3, 8)
    weights = [
        ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
        ("model.layers.0.mlp.experts.down_proj", down),
    ]

    converted = _collect(weights, "gpt_oss")

    assert [name for name, _ in converted] == [name for name, _ in weights]
    assert converted[0][1] is gate_up
    assert converted[1][1] is down
    assert converted[0][1].shape == (2, 8, 6)
    assert converted[1][1].shape == (2, 3, 8)
