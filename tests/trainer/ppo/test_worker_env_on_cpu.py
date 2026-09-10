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

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.utils import get_trainer_worker_env


def test_get_trainer_worker_env_resolves_values():
    config = OmegaConf.create(
        {
            "segment_size_mb": 128,
            "trainer": {
                "worker_env": {
                    "PYTORCH_CUDA_ALLOC_CONF": "large_segment_size_mb:${segment_size_mb}",
                }
            },
        }
    )

    assert get_trainer_worker_env(config) == {
        "PYTORCH_CUDA_ALLOC_CONF": "large_segment_size_mb:128",
    }


def test_get_trainer_worker_env_defaults_to_empty():
    assert get_trainer_worker_env(OmegaConf.create({"trainer": {}})) == {}


def test_get_trainer_worker_env_requires_string_values():
    config = OmegaConf.create({"trainer": {"worker_env": {"INVALID_ENV": 128}}})

    with pytest.raises(TypeError, match="must map strings to strings"):
        get_trainer_worker_env(config)
