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

from verl.workers.config.rollout import RolloutConfig


def test_max_num_batched_tokens_raised_when_chunked_prefill_disabled():
    # vLLMHttpServer._validate_configs raises max_num_batched_tokens to max_model_len
    # when chunked prefill is off; that write must not hit the frozen-field guard.
    config = RolloutConfig(name="vllm", enable_chunked_prefill=False, max_num_batched_tokens=8192)
    config.max_model_len = 32768
    config.max_num_batched_tokens = config.max_model_len
    assert config.max_num_batched_tokens == 32768
