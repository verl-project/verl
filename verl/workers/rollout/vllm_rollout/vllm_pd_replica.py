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
"""vLLM PD stub. ``rollout.name=vllm_pd`` validates at config load but raises
at launch; vLLM's KV-transfer connector wiring (NIXL/Mooncake via
``kv_transfer_config``) is a tracked follow-up."""

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica


class vLLMPDReplica(vLLMReplica):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "vLLMPDReplica is not implemented. Set rollout.name=sglang_pd for "
            "the working path, or track the vLLM PD follow-up."
        )
