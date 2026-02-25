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
import numpy as np
import ray
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, open_dict
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.config import omega_conf_to_dataclass

from .teacher_model import TeacherModelManager
from verl.workers.config import DistillationConfig, DistillationLossConfig

from verl.trainer.distillation.losses import get_distillation_loss_settings, DistillationLossSettings

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TeacherLoopWorker:
    """
    TeacherLoopWorker: TODO
    """

    def __init__(self, config: DictConfig, teacher_router_address: str = None):
        """
        Args:
            config: DictConfig, the config for reward loop worker.
            teacher_router_address: str, the address of teacher router.
        """
        self.config = config
        self.distillation_config: DistillationConfig = self.config.distillation
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.distillation_loss_settings: DistillationLossSettings = get_distillation_loss_settings(self.distillation_loss_config.loss_mode)
        self.teacher_router_address = teacher_router_address
        # # Serialize teacher requests per actor to reduce pressure on the teacher vLLM router/backend.
        # self._request_semaphore = asyncio.Semaphore(1)

    async def compute_logprobs_batch(self, data: DataProto) -> list[dict]:
        raise NotImplementedError("TODO:RM")
        tasks = []
        for i in range(len(data)):
            tasks.append(asyncio.create_task(self.compute_score(data[i : i + 1])))
        outputs = await asyncio.gather(*tasks)
        return outputs

    async def compute_logprobs(self, data: DataProto) -> dict:
        assert len(data) == 1, "TeacherLoopWorker only supports single data item"
        # async with self._request_semaphore:
        #     return await self._compute_logprobs(data)
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
        prompt_ids = data.batch['prompt_ids']
        response_ids = data.batch['response_ids']
        input_ids = torch.cat([prompt_ids, response_ids], dim=1).squeeze(0).tolist()
        engine_name = self.config.distillation.teacher_model.inference.name
        model_name = self.config.distillation.teacher_model.model_path
        if engine_name == "vllm":
            if self.distillation_loss_settings.use_topk:
                num_logprobs = topk = self.distillation_loss_config.topk
            else:
                num_logprobs = 0 # only the sampled logprob    
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
                    logprob = logprobs_dict[token_id_str]['logprob']
                    response_logprobs_ls.append([logprob])
                    response_ids_ls.append([int(token_id_str)])
                else:
                    response_ids = [None] * topk
                    response_logprobs = [None] * topk
                    # We get either top-k logprobs or top-k plus the sampled logprob (if sampled token is not in top-k)
                    assert len(logprobs_dict) in [topk, topk + 1], len(logprobs_dict)
                    for token_id_str, token_dict in logprobs_dict.items():
                        if token_dict['rank'] > topk:
                            continue # the sampled token is not in the top-k
                        rank = token_dict['rank']
                        logprob = token_dict['logprob']
                        response_ids[rank - 1] = int(token_id_str)
                        response_logprobs[rank - 1] = logprob
                    response_logprobs_ls.append(response_logprobs)
                    response_ids_ls.append(response_ids)
            logprobs_dtype = torch.bfloat16 if self.distillation_config.teacher_model.inference.dtype == "bfloat16" else torch.float32
            response_logprobs = torch.tensor(response_logprobs_ls, dtype=logprobs_dtype).unsqueeze(0)
            response_ids = torch.tensor(response_ids_ls, dtype=torch.long).unsqueeze(0)
            
        elif engine_name == "sglang":
            raise ValueError("SGLang backend does not support distillation currently.")
            payloads = {
                "model": model_name,
                "input": disrm_prompt,
            }
            output = await self._post_request(payloads, "v1/embeddings")
            rm_score = output["data"][-1]["embedding"][-1]
        elif engine_name == "trtllm":
            # TODO: remove this once TRT-LLM switches to TorchSampler
            raise ValueError("TensorRT-LLM backend does not support distillation currently.")

            payloads = {
                "model": model_name,
                "prompt": disrm_prompt,
                "return_context_logits": True,
            }
            output = await self._post_request(payloads, "v1/completions")
            rm_score = output["choices"][0]["context_logits"]
            assert isinstance(rm_score, list) and len(rm_score) > 0, (
                "TensorRT-LLM OpenAI server response for reward score is not in the expected format."
            )

            rm_score = float(rm_score[0][0])
            logger.debug(f"rm score: {rm_score}")
        else:
            raise NotImplementedError(f"RewardLoopManager does not support {engine_name}")

        return {"response_logprobs": response_logprobs, "response_ids": response_ids}


class TeacherLoopManager:
    """
    TeacherLoopManager run in single controller.
    This class will create teacher loop workers and manage them.
    """

    def __init__(self, config: DictConfig, teacher_resource_pool: RayResourcePool = None):
        self.config = config
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation) # to dataclass for the post init to handle top-k and engine kwargs
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

    def compute_teacher_logprobs(self, data: DataProto) -> DataProto:
        raise NotImplementedError("TODO:RM")
        if self.teacher_model_manager is not None:
            self.teacher_model_manager.wake_up()

        chunks = data.chunk(len(self.teacher_loop_workers))
        outputs = ray.get(
            [
                worker.compute_score_batch.remote(chunk)
                for worker, chunk in zip(self.teacher_loop_workers, chunks, strict=True)
            ]
        )
        outputs_flat = [item for sublist in outputs for item in sublist]

        # compute teacher logprobs
        raise NotImplementedError
        scores = [item["reward_score"] for item in outputs_flat]
        prompt_length = data.batch["prompts"].size(1)
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=1)
        rm_scores = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        rm_scores[torch.arange(rm_scores.size(0)), valid_response_length - 1] = torch.tensor(
            scores, dtype=torch.float32
        )
        batch = TensorDict({"rm_scores": rm_scores}, batch_size=len(data))

        reward_extra_infos = [output.get("reward_extra_info", {}) for output in outputs_flat]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        non_tensor_batch = {}
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        if self.reward_model_manager is not None:
            self.reward_model_manager.sleep()

        return DataProto(
            batch=batch, non_tensor_batch=non_tensor_batch, meta_info={"reward_extra_keys": reward_extra_keys}
        )

    def _run_all(self, tasks: list[asyncio.Task]):
        raise NotImplementedError("TODO:RM")
        async def run_all():
            return await asyncio.gather(*tasks)

        return asyncio.run(run_all())
