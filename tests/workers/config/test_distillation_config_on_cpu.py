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

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import DistillationTeacherModelConfig, RolloutConfig


def test_sparse_teacher_inference_defaults_to_checkpoint_loading():
    config = omega_conf_to_dataclass(
        {
            "key": "general",
            "model_path": "/tmp/teacher",
            "num_replicas": 1,
            "inference": {"name": "vllm"},
        },
        dataclass_type=DistillationTeacherModelConfig,
    )

    assert config.inference.load_format == "auto"
    config.check_configured()


@pytest.mark.parametrize("load_format", ["dummy", "dummy_hf", "dummy_megatron"])
def test_teacher_rejects_dummy_load_format(load_format):
    config = DistillationTeacherModelConfig(
        key="general",
        model_path="/tmp/teacher",
        num_replicas=1,
        inference=RolloutConfig(name="vllm", load_format=load_format),
    )

    with pytest.raises(ValueError, match="must load checkpoint weights at startup"):
        config.check_configured()


def test_actor_rollout_default_load_format_remains_dummy():
    assert RolloutConfig().load_format == "dummy"
