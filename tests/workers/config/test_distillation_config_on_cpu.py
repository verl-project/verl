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

import pytest

from verl.workers.config import DistillationConfig, DistillationTeacherModelConfig


def _config(**teachers) -> DistillationConfig:
    return DistillationConfig(
        teacher_models={key: DistillationTeacherModelConfig(**kw) for key, kw in teachers.items()}
    )


def test_routes_include_base_and_lora_teachers():
    config = _config(
        base=dict(key="base", lora_adapters={"math": "/a/math", "code": "/a/code"}),
        big=dict(key="big"),
    )
    assert config.get_teacher_routes() == {
        "base": ("base", None),
        "math": ("base", "/a/math"),
        "code": ("base", "/a/code"),
        "big": ("big", None),
    }


def test_duplicate_routing_key_raises():
    config = _config(base=dict(key="base", lora_adapters={"big": "/a/big"}), big=dict(key="big"))
    with pytest.raises(ValueError, match="Duplicate teacher routing key 'big'"):
        config.get_teacher_routes()


@pytest.mark.parametrize("key, base_route", [("base", "base"), (None, "default")])
def test_single_teacher_keeps_configured_key(key, base_route):
    teacher = DistillationTeacherModelConfig(key=key, model_path="m", lora_adapters={"math": "/a/math"})
    config = DistillationConfig(n_gpus_per_node=2, nnodes=1, teacher_models={"teacher_model": teacher})
    config.teacher_models = config._resolve_teacher_models()
    assert config.get_teacher_routes() == {base_route: (base_route, None), "math": (base_route, "/a/math")}
