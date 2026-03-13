# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from verl.workers.rollout.base import _ROLLOUT_REGISTRY
from verl.workers.rollout.replica import ImageOutput, RolloutReplicaRegistry


def test_vllm_omni_rollout_registry_entry_exists():
    key = ("vllm_omni", "async")
    assert key in _ROLLOUT_REGISTRY
    assert _ROLLOUT_REGISTRY[key] == "verl.workers.rollout.vllm_rollout.vLLMOmniServerAdapter"


def test_vllm_omni_replica_registry_entry_exists():
    assert "vllm_omni" in RolloutReplicaRegistry._registry
    assert callable(RolloutReplicaRegistry._registry["vllm_omni"])


def test_image_output_defaults():
    output = ImageOutput(image=[[[0.0]]])
    assert output.image == [[[0.0]]]
    assert output.log_probs is None
    assert output.stop_reason is None
    assert output.num_preempted is None
    assert output.extra_info == {}
