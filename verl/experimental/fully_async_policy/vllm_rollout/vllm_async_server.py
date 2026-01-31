# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import asyncio
import logging
from typing import Any, Optional, Sequence

import ray
from ray.actor import ActorHandle
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput

from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    _qwen2_5_vl_dedup_image_tokens,
    vLLMHttpServer,
    vLLMReplica,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class vLLMHttpServerForPartial(vLLMHttpServer):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
    ):
        super().__init__(config, model_config, rollout_mode, workers, replica_rank, node_rank, gpus_per_node, nnodes)

        # for cancel LLMServer
        self.paused = False
        self.lock = asyncio.Lock()
        self.cancel_event: dict[str, asyncio.Event] = {}
        self.req_output: dict[str, Optional[RequestOutput]] = {}

    async def _generate_step(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ):
        max_tokens = self.config.max_model_len - len(prompt_ids)
        sampling_params["logprobs"] = 1
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt_ids = _qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)
        prompt = TokensPrompt(
            prompt_token_ids=prompt_ids, multi_modal_data={"image": image_data} if image_data else None
        )
        generator = self.engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)

        # Get final response
        async for output in generator:
            self.req_output[request_id] = output
        assert self.req_output[request_id] is not None

    async def generate_for_partial(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> tuple[list[Any], list[Any], bool] | tuple[Sequence[int], list[float], Any]:
        async with self.lock:
            if self.paused:
                # After cancel, all tasks will return directly and wait for the next submission
                return [], [], True
            self.req_output[request_id]: Optional[RequestOutput] = None
            self.cancel_event[request_id] = asyncio.Event()
            cancel_handle = asyncio.create_task(self.cancel_event[request_id].wait())
            generation_handle = asyncio.create_task(
                self._generate_step(prompt_ids, sampling_params, request_id, image_data)
            )

        done, pend = await asyncio.wait([generation_handle, cancel_handle], return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            await task

        for task in pend:
            task.cancel()

        async with self.lock:
            if self.req_output[request_id] is None:
                return [], [], True
            token_ids = self.req_output[request_id].outputs[0].token_ids
            log_probs: list[float] = []
            for i, x in enumerate(self.req_output[request_id].outputs[0].logprobs):
                # In sampling_params, logprobs is set to 1, which should return 1,
                # but in practice there are multiple. Take the log_prob corresponding to token_id
                token_id = self.req_output[request_id].outputs[0].token_ids[i]
                log_probs.append(x[token_id].logprob)
            is_cancel = generation_handle not in done
            self.cancel_event.pop(request_id, None)
            self.req_output.pop(request_id, None)
        return token_ids, log_probs, is_cancel

    async def cancel(self):
        async with self.lock:
            self.paused = True
            for request_id in self.cancel_event:
                self.cancel_event[request_id].set()

    async def resume(self):
        async with self.lock:
            self.paused = False


class FullyAsyncvLLMReplica(vLLMReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(vLLMHttpServerForPartial)

    async def cancel(self):
        """Cancel each rollout server."""
        await asyncio.gather(*[server.cancel.remote() for server in self.servers])

    async def resume(self):
        """Resume each rollout server."""
        await asyncio.gather(*[server.resume.remote() for server in self.servers])
