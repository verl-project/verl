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

from verl.utils.megatron_peft_utils import convert_megatron_to_hf_target_modules, resolve_base_layer_name


def test_convert_megatron_to_hf_target_modules_expands_gdn_in_proj():
    converted = convert_megatron_to_hf_target_modules(["in_proj", "out_proj"])

    assert converted == [
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    ]


def test_resolve_base_layer_name_adds_suffix_when_target_requires_it():
    resolved_name = resolve_base_layer_name(
        "model.layers.0.self_attn.q_proj.weight",
        exists=lambda candidate: candidate == "model.layers.0.self_attn.q_proj.base_layer.weight",
    )

    assert resolved_name == "model.layers.0.self_attn.q_proj.base_layer.weight"


def test_resolve_base_layer_name_removes_suffix_when_target_does_not_use_it():
    resolved_name = resolve_base_layer_name(
        "model.visual.merger.linear_fc1.base_layer.weight",
        exists=lambda candidate: candidate == "model.visual.merger.linear_fc1.weight",
    )

    assert resolved_name == "model.visual.merger.linear_fc1.weight"


def test_resolve_base_layer_name_keeps_existing_name():
    resolved_name = resolve_base_layer_name(
        "model.visual.merger.linear_fc1.weight",
        exists=lambda candidate: candidate == "model.visual.merger.linear_fc1.weight",
    )

    assert resolved_name == "model.visual.merger.linear_fc1.weight"
