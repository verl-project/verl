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

from verl.workers.config import RolloutConfig


@pytest.mark.parametrize("enable_dp_attention", [None, False, "true"])
def test_sglang_data_parallel_requires_dp_attention(enable_dp_attention):
    sglang_kwargs = {} if enable_dp_attention is None else {"enable_dp_attention": enable_dp_attention}

    with pytest.raises(ValueError, match=r"engine_kwargs\.sglang\.enable_dp_attention=True"):
        RolloutConfig(name="sglang", data_parallel_size=2, engine_kwargs={"sglang": sglang_kwargs})


@pytest.mark.parametrize(
    "config",
    [
        {"name": "sglang", "data_parallel_size": 1},
        {
            "name": "sglang",
            "data_parallel_size": 2,
            "engine_kwargs": {"sglang": {"enable_dp_attention": True}},
        },
        {"name": "vllm", "data_parallel_size": 2},
    ],
)
def test_rollout_data_parallel_supported_configs(config):
    RolloutConfig(**config)


@pytest.mark.parametrize("parallel_arg", ["tp_size", "dp_size"])
def test_sglang_parallel_sizes_cannot_be_overridden(parallel_arg):
    with pytest.raises(ValueError, match="tensor_model_parallel_size.*data_parallel_size"):
        RolloutConfig(name="sglang", engine_kwargs={"sglang": {parallel_arg: 2}})
