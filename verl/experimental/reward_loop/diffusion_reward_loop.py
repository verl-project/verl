# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import base64
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import aiohttp
import numpy as np
import ray
import torch
from omegaconf import DictConfig
from PIL import Image
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.ray_utils import get_event_loop

from .reward_manager import get_reward_manager_cls
from .reward_model import RewardModelManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote
class DiffusionRewardLoopWorker:
    def __init__(self, config: DictConfig, reward_router_address: str = None):
        """
        RewardLoopWorker can tackle reward computation:
        (1) rule-based reward computation
        (2) reward model-based reward computation (both disrm and genrm)
        (3) high-flexible user-customized reward function (can access rm by posting requests to reward_model_router)

        Reward Computation Logic:
        - if user-customized reward function is provided:
            -> directly use user-customized reward function
        - if user-customized reward function is not provided:
            -> rm is not enabled: use default rule-based reward function
            -> rm is disrm: compute reward score using disrm
            -> rm is genrm: raise error (user-customized reward func must be provided)

        Args:
            config: DictConfig, the config for reward loop worker.
            reward_router_address: str, the address of reward router.
        """
        self.config = config
        self.reward_router_address = reward_router_address
        self._init_reward_fn()
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _init_reward_fn(self):
        input_tokenizer_local_path = copy_to_local(self.config.actor_rollout_ref.model.path)
        self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=True)
        self.reward_model_tokenizer = None
        if self.config.reward_model.enable:
            reward_model_tokenizer_local_path = copy_to_local(self.config.reward_model.model.path)
            self.reward_model_tokenizer = hf_tokenizer(reward_model_tokenizer_local_path, trust_remote_code=True)
        self.reward_fn = get_custom_reward_fn(self.config)

        # Load reward loop manager class
        # Support both registry and importlib loading methods
        reward_loop_source = self.config.reward_model.get("reward_loop_source", "register")

        if reward_loop_source == "register":
            # Load from registry (default behavior)
            reward_manager_cls = get_reward_manager_cls(self.config.reward_model.reward_manager)
        elif reward_loop_source == "importlib":
            # Load from external module using importlib
            from verl.utils.import_utils import load_extern_object

            reward_loop_module_path = self.config.reward_model.get("reward_loop_module_path", None)
            reward_loop_class_name = self.config.reward_model.get("reward_loop_class_name", None)

            assert reward_loop_module_path is not None, (
                "reward_loop_module_path must be set when reward_loop_source='importlib'"
            )
            assert reward_loop_class_name is not None, (
                "reward_loop_class_name must be set when reward_loop_source='importlib'"
            )

            reward_manager_cls = load_extern_object(
                module_path=reward_loop_module_path, object_name=reward_loop_class_name
            )
        else:
            raise ValueError(f"Unknown reward_loop_source: {reward_loop_source}. Must be 'register' or 'importlib'")

        self.reward_loop = reward_manager_cls(
            self.config, self.input_tokenizer, self.reward_fn, self.reward_router_address, self.reward_model_tokenizer
        )

    async def compute_score_batch(self, data: DataProto) -> list[dict]:
        tasks = []
        for i in range(len(data)):
            tasks.append(asyncio.create_task(self.compute_score(data[i : i + 1])))
        outputs = await asyncio.gather(*tasks)
        return outputs

    async def compute_score(self, data: DataProto) -> dict:
        assert len(data) == 1, "RewardLoopWorker only support single data item"
        if self.config.custom_reward_function.path is not None:
            # directly use user-customized reward function
            return await self.reward_loop.run_single(data)
        else:
            if self.config.reward_model.enable:
                # we assume the rm is disrm
                # genrm must set custom_reward_function
                return await self.compute_score_disrm(data)
            else:
                return await self.reward_loop.run_single(data)

    async def _post_request(self, payload: dict, endpoint: str, max_retries: int = 16):
        url = f"http://{self.reward_router_address}/{endpoint}"
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

    async def _preprocess_reward_inputs(self, data: DataProto) -> str:
        """
        Prepare discriminative reward model inputs: input prompt and output image.
        """
        assert len(data) == 1, "DiffusionRewardLoopWorker only support single data item"
        data_item = data[0]
        assert "raw_prompt" in data_item.non_tensor_batch

        # extract raw prompt
        chat: list = list(data_item.non_tensor_batch["raw_prompt"])

        # extract response
        prompt_str = self.input_tokenizer.decode(data_item.batch["prompts"], skip_special_tokens=True)
        response_image = data_item.batch["responses"]

        # convert to PIL Image
        if isinstance(response_image, torch.Tensor):
            response_image = response_image.float().permute(1, 2, 0).cpu().numpy()
        assert response_image.shape[-1] == 3, "must be in HWC format"
        response_image = (response_image * 255).round().clip(0, 255).astype(np.uint8)
        response_image = Image.fromarray(response_image)

        image_base64 = await self.pil_image_to_base64(response_image)
        query = self.prepare_query(prompt_str, image_base64)

        chat.append({"role": "assistant", "content": query})

        rm_prompt = self.reward_model_tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=False,
            tokenize=False,
        )

        return rm_prompt

    async def compute_score_disrm(self, data: DataProto) -> dict:
        disrm_prompt = await self._preprocess_reward_inputs(data)
        engine_name = self.config.reward_model.rollout.name
        model_name = self.config.reward_model.model.path
        if engine_name == "vllm":
            # TODO (dyy): the "activation" has been changed to "use_activation" in vllm 0.11.2
            payloads = {
                "model": model_name,
                "input": disrm_prompt,
                "activation": False,
                # "add_special_tokens": False,  # vllm >= 0.11.2
            }
            output = await self._post_request(payloads, "classify")
            rm_score = output["data"][-1]["probs"][-1]
        elif engine_name == "sglang":
            payloads = {
                "model": model_name,
                "input": disrm_prompt,
            }
            output = await self._post_request(payloads, "v1/embeddings")
            rm_score = output["data"][-1]["embedding"][-1]
        else:
            raise NotImplementedError(f"DiffusionRewardLoopManager does not support {engine_name}")

        return {"reward_score": rm_score}

    async def pil_image_to_base64(self, image: Image.Image) -> str:
        # To avoid blocking the event loop, run the conversion in a thread pool
        base64_image = await get_event_loop().run_in_executor(self._executor, self._pil_image_to_base64, image)
        return base64_image

    def _pil_image_to_base64(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_image = f"data:image;base64,{encoded_image_text}"
        return base64_image

    def prepare_query(self, prompt, image_base64: str) -> list:
        query = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": image_base64},
            },
        ]
        return query


class DiffusionRewardLoopManager:
    """
    DiffusionRewardLoopManager run in single controller.
    This class will create reward loop workers and manage them.
    """

    def __init__(self, config: DictConfig, rm_resource_pool: RayResourcePool = None):
        self.config = config
        if self.config.reward_model.enable:
            self.reward_model_manager = RewardModelManager(config.reward_model, rm_resource_pool)
            self.reward_router_address = self.reward_model_manager.get_router_address()
        else:
            self.reward_model_manager = None
            self.reward_router_address = None

        self._init_reward_loop_workers()

    def _init_reward_loop_workers(self):
        self.reward_loop_workers = []
        num_workers = self.config.reward_model.num_workers
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]

        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.reward_loop_workers.append(
                DiffusionRewardLoopWorker.options(
                    name=f"reward_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=True,
                    ),
                ).remote(self.config, self.reward_router_address)
            )

    # this func is used to replace the legacy fsdp/megatron RewardModelWorker.compute_rm_score
    def compute_rm_score(self, data: DataProto) -> DataProto:
        if self.reward_model_manager is not None:
            self.reward_model_manager.wake_up()

        chunks = data.chunk(len(self.reward_loop_workers))
        outputs = ray.get(
            [
                worker.compute_score_batch.remote(chunk)
                for worker, chunk in zip(self.reward_loop_workers, chunks, strict=True)
            ]
        )
        outputs_flat = [item for sublist in outputs for item in sublist]

        # compute rm score
        scores = [item["reward_score"] for item in outputs_flat]
        rm_scores = torch.tensor(scores, dtype=torch.float32)
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
        async def run_all():
            return await asyncio.gather(*tasks)

        return asyncio.run(run_all())
