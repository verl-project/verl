# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace

from verl.models.transformers.qwen3_omni_moe import (
    _DEFAULT_EXPERTS_IMPLEMENTATION,
    _force_experts_implementation,
    patch_qwen3_omni_moe_sparse_moe_block_forward,
)


def test_force_experts_implementation_sets_when_unset():
    config = SimpleNamespace(_experts_implementation=None)

    result = _force_experts_implementation(config, "batched_mm")

    assert result == "batched_mm"
    assert config._experts_implementation == "batched_mm"


def test_force_experts_implementation_preserves_existing_value():
    config = SimpleNamespace(_experts_implementation="eager")

    result = _force_experts_implementation(config, "batched_mm")

    assert result == "eager"
    assert config._experts_implementation == "eager"


def test_patch_configures_model_instance_when_unset():
    model = SimpleNamespace(config=SimpleNamespace(_experts_implementation=None))

    patch_qwen3_omni_moe_sparse_moe_block_forward(model=model)

    assert model.config._experts_implementation == _DEFAULT_EXPERTS_IMPLEMENTATION


def test_patch_is_noop_without_model_instance():
    patch_qwen3_omni_moe_sparse_moe_block_forward()
