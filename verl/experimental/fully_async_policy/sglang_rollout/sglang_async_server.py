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
from typing import Any, Optional

import ray
import torch
from ray.actor import ActorHandle

from verl.workers.config import HFModelConfig, RewardModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.sglang_rollout.async_sglang_server import (
    SGLangHttpServer,
    SGLangReplica,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class SGLangHttpServerForPartial(SGLangHttpServer):
    def __init__(
        self,
        config: RolloutConfig | RewardModelConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        nnodes: int,
        cuda_visible_devices: str,
        base_gpu_id: int,
    ):
        super().__init__(
            config=config,
            model_config=model_config,
            rollout_mode=rollout_mode,
            workers=workers,
            replica_rank=replica_rank,
            node_rank=node_rank,
            nnodes=nnodes,
            cuda_visible_devices=cuda_visible_devices,
            base_gpu_id=base_gpu_id,
        )

        # for cancel LLMServer
        self.paused = False
        self.lock = asyncio.Lock()
        self.cancel_event: dict[str, asyncio.Event] = {}
        self.req_output: dict[str, Optional[dict[str, Any]]] = {}

    async def _generate_step(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> None:
        sampling_params = dict(sampling_params)

        max_new_tokens = min(
            self.config.response_length,
            self.config.max_model_len - len(prompt_ids) - 1,
        )
        sampling_params["max_new_tokens"] = max_new_tokens

        sampling_params.setdefault(
            "repetition_penalty",
            self.config.get("repetition_penalty", 1.0),
        )

        sampling_params.pop("logprobs", None)
        return_logprob = True
        from sglang.srt.managers.io_struct import GenerateReqInput

        if video_data is not None and len(video_data) > 0:
            logger.warning(
                f"Request {request_id} received video_data but it is not used. "
                "This is to keep consistency with the implementation in "
                "verl/workers/rollout/sglang_rollout/async_sglang_server.py. "
                "Video data will be ignored."
            )

        request = GenerateReqInput(
            rid=request_id,
            input_ids=prompt_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            image_data=image_data,
            # TODO: support video input for sglang
            # video_data=video_data,
        )

        generator = self.tokenizer_manager.generate_request(request, None)
        async for output in generator:
            self.req_output[request_id] = output

        assert self.req_output[request_id] is not None

    async def generate_for_partial(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> tuple[list[int], list[float], bool]:
        async with self.lock:
            if self.paused:
                return [], [], True
            self.req_output[request_id] = None
            self.cancel_event[request_id] = asyncio.Event()
            cancel_handle = asyncio.create_task(self.cancel_event[request_id].wait())
            generation_handle = asyncio.create_task(
                self._generate_step(prompt_ids, sampling_params, request_id, image_data, video_data)
            )
        done, pending = await asyncio.wait(
            [generation_handle, cancel_handle],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            await task

        for task in pending:
            task.cancel()
        async with self.lock:
            output = self.req_output.get(request_id)
            if output is None:
                self.cancel_event.pop(request_id, None)
                self.req_output.pop(request_id, None)
                return [], [], True
            meta_info = output.get("meta_info", {})
            output_token_logprobs = meta_info.get("output_token_logprobs")

            token_ids: list[int] = []
            log_probs: list[float] = []

            if output_token_logprobs is not None:
                for log_prob, token_id, _ in output_token_logprobs:
                    token_ids.append(int(token_id))
                    log_probs.append(float(log_prob))
            else:
                token_ids = list(output["output_ids"])
                log_probs = []
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


class FullyAsyncSGLangReplica(SGLangReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig | RewardModelConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(SGLangHttpServerForPartial)

    async def cancel(self):
        """Cancel each rollout server."""
        await asyncio.gather(*[server.cancel.remote() for server in self.servers])

    async def resume(self):
        """Resume each rollout server."""
        await asyncio.gather(*[server.resume.remote() for server in self.servers])
