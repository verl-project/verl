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

import asyncio
import logging
import os

import aiohttp
import ray
import torch
from omegaconf import DictConfig

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool
from verl.trainer.distillation.losses import DistillationLossSettings
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.ray_utils import auto_await
from verl.workers.config import DistillationConfig, DistillationLossConfig

from .teacher_model import TeacherModelManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TeacherLoopWorker:
    """TeacherLoopWorker: Computes logprobs for student completion."""

    def __init__(self, config: DictConfig, teacher_router_address: str = None):
        """
        Args:
            config: DictConfig, the config for reward loop worker.
            teacher_router_address: str, the address of teacher router.
        """
        self.config = config
        # to dataclass for the post init to handle top-k and engine kwargs and get distillation_loss_settings
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.distillation_loss_settings: DistillationLossSettings = self.distillation_loss_config.loss_settings
        self.teacher_router_address = teacher_router_address

    async def compute_logprobs(self, data: DataProto) -> dict:
        assert len(data) == 1, "TeacherLoopWorker only supports single data item"
        return await self._compute_logprobs(data)

    async def _post_request(self, payload: dict, endpoint: str, max_retries: int = 16):
        url = f"http://{self.teacher_router_address}/{endpoint}"
        last_exception = None
        for attempt in range(max_retries):
            try:
                # It's safer to have a timeout instead of None, which can hang indefinitely.
                timeout = aiohttp.ClientTimeout(total=None)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except aiohttp.ClientResponseError as e:
                # Do not retry on 4xx client errors, but retry on 5xx server errors.
                if 400 <= e.status < 500:
                    logger.error(f"Request to {url} failed with client error HTTP {e.status}: {e}. Not retrying.")
                    raise
                last_exception = e
                logger.warning(
                    f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed with HTTP {e.status}: {e}. "
                    "Retrying..."
                )
            except (asyncio.TimeoutError, aiohttp.ClientConnectorError) as e:
                last_exception = e
                logger.warning(f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed: {e}. Retrying...")
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed with unexpected error: {e}. "
                    "Retrying..."
                )

            if attempt < max_retries - 1:
                # Using exponential backoff is generally better than a fixed sleep.
                backoff_seconds = 2**attempt
                await asyncio.sleep(min(backoff_seconds, 30))

        logger.error(f"Max retries ({max_retries}) reached for request to {url}.")
        if last_exception:
            raise last_exception

    async def _compute_logprobs(self, data: DataProto) -> dict:
        prompt_ids = data.batch["prompt_ids"]
        response_ids = data.batch["response_ids"]
        input_ids = torch.cat([prompt_ids, response_ids], dim=1).squeeze(0).tolist()
        engine_name = self.config.distillation.teacher_model.inference.name
        model_name = self.config.distillation.teacher_model.model_path
        match engine_name:
            case "vllm":
                if self.distillation_loss_settings.use_topk:
                    num_logprobs = topk = self.distillation_loss_config.topk
                else:
                    num_logprobs = 0  # only the sampled logprob
                payloads = {
                    "model": model_name,
                    "prompt": input_ids,
                    "max_tokens": 1,
                    "prompt_logprobs": num_logprobs,
                }
                output = await self._post_request(payloads, "v1/completions")

                # Extract logprobs from vllm output
                choices = output["choices"]
                assert len(choices) == 1, f"Expected exactly one choice from teacher model, but got {len(choices)}"
                response_logprobs = output["choices"][0]["prompt_logprobs"]
                response_length = response_ids.shape[1]
                response_logprob_dicts = response_logprobs[-response_length:]
                response_logprobs_ls, response_ids_ls = [], []
                for logprobs_dict in response_logprob_dicts:
                    if num_logprobs == 0:
                        token_id_str = list(logprobs_dict.keys())[0]
                        logprob = logprobs_dict[token_id_str]["logprob"]
                        response_logprobs_ls.append([logprob])
                        response_ids_ls.append([int(token_id_str)])
                    else:
                        response_ids = [None] * topk
                        response_logprobs = [None] * topk
                        # We get either top-k logprobs or top-k plus the sampled logprob (if sampled token
                        # is not in top-k)
                        assert len(logprobs_dict) in [topk, topk + 1], len(logprobs_dict)
                        for token_id_str, token_dict in logprobs_dict.items():
                            if token_dict["rank"] > topk:
                                continue  # the sampled token is not in the top-k
                            rank = token_dict["rank"]
                            logprob = token_dict["logprob"]
                            response_ids[rank - 1] = int(token_id_str)
                            response_logprobs[rank - 1] = logprob
                        response_logprobs_ls.append(response_logprobs)
                        response_ids_ls.append(response_ids)
                logprobs_dtype = (
                    torch.bfloat16
                    if self.distillation_config.teacher_model.inference.dtype == "bfloat16"
                    else torch.float32
                )
                response_logprobs = torch.tensor(response_logprobs_ls, dtype=logprobs_dtype).unsqueeze(0)
                response_ids = torch.tensor(response_ids_ls, dtype=torch.long).unsqueeze(0)
            case "sglang":
                raise ValueError("SGLang backend does not support distillation currently.")
            case "trtllm":
                raise ValueError("TensorRT-LLM backend does not support distillation currently.")
            case _:
                raise NotImplementedError(f"TeacherLoopWorker does not support {engine_name}")

        return {"response_logprobs": response_logprobs, "response_ids": response_ids}


class TeacherLoopManager:
    """
    TeacherLoopManager run in single controller.
    This class will create teacher loop workers and manage them.
    """

    def __init__(self, config: DictConfig, teacher_resource_pool: RayResourcePool = None):
        self.config = config
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(
            self.config.distillation
        )  # to dataclass for the post init to handle top-k and engine kwargs and get distillation_loss_settings
        self.teacher_model_manager = TeacherModelManager(self.distillation_config.teacher_model, teacher_resource_pool)
        self.teacher_router_address = self.teacher_model_manager.get_router_address()

        self.teacher_loop_workers_class = ray.remote(TeacherLoopWorker)
        self._init_teacher_loop_workers()

    def _init_teacher_loop_workers(self):
        self.teacher_loop_workers = []
        num_workers = self.distillation_config.num_workers
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]

        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.teacher_loop_workers.append(
                self.teacher_loop_workers_class.options(
                    name=f"teacher_loop_worker_{i}",
                    max_concurrency=1,
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=True,
                    ),
                ).remote(self.config, self.teacher_router_address)
            )

    @auto_await
    async def wake_up(self):
        """Wake up all rollout replica instances."""
        await self.teacher_model_manager.wake_up()

    @auto_await
    async def sleep(self):
        """Sleep all rollout replica instances."""
        await self.teacher_model_manager.sleep()
