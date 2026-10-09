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
from omegaconf import DictConfig, open_dict
from tensordict import TensorDict

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.runtime import (
    ClassWithInitArgs,
    RemoteCall,
    RemoteWorkerGroup,
    ResourcePool,
    Worker,
    current_runtime,
)
from verl.trainer.ppo.reward import load_reward_manager, resolve_reward_manager_cls
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.ray_utils import get_event_loop

from .reward_model import RewardModelManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def migrate_legacy_reward_impl(config):
    """
    Migrate the legacy reward model implementation to the new one.
    """
    # 1. reward workers migration
    # config.reward_model.num_workers -> config.reward.num_workers
    if config.reward_model.num_workers is not None:
        config.reward.num_workers = config.reward_model.num_workers

    # 2. reward manager migration
    # config.reward_model.reward_manager -> config.reward.reward_manager
    if config.reward_model.reward_manager is not None:
        config.reward.reward_manager.name = config.reward_model.reward_manager
    if config.reward_model.reward_loop_source is not None:
        config.reward.reward_manager.source = config.reward_model.reward_loop_source
        config.reward.reward_manager.module.path = config.reward_model.reward_loop_module_path
        config.reward.reward_manager.module.name = config.reward_model.reward_loop_class_name

    # 3. custom reward function migration
    # config.custom_reward_function -> config.reward.custom_reward_function
    if not all(v is None for v in config.custom_reward_function.values()):
        config.reward.custom_reward_function = config.custom_reward_function

    # 4. reward model migration
    # config.reward_model -> config.reward.reward_model
    for key in ["enable", "enable_resource_pool", "n_gpus_per_node", "nnodes"]:
        if config.reward_model.get(key) is not None:
            config.reward.reward_model[key] = config.reward_model[key]
    if config.reward_model.model.path is not None:
        config.reward.reward_model.model_path = config.reward_model.model.path
    # config.reward_model.reward_kwargs -> config.reward.reward_kwargs (for dapo algo)
    if config.reward_model.get("reward_kwargs") is not None:
        with open_dict(config.reward):
            config.reward["reward_kwargs"] = config.reward_model["reward_kwargs"]
    # config.reward_model.rollout -> config.reward.reward_model.rollout
    legacy_rollout = config.reward_model.rollout
    for key in legacy_rollout.keys():
        if legacy_rollout[key] is not None:
            config.reward.reward_model.rollout[key] = legacy_rollout[key]

    # 5. sandbox_fusion migration
    # config.sandbox_fusion -> reward.sandbox_fusion
    if not all(v is None for v in config.sandbox_fusion.values()):
        config.reward.sandbox_fusion = config.sandbox_fusion

    # 6. delete legacy config from configs
    with open_dict(config):
        del config.reward_model
        del config.custom_reward_function
        del config.sandbox_fusion

    return config


class RewardLoopWorker:
    """
    RewardLoopWork can tackle reward computation:
    (1) rule-based reward computation
    (2) reward model-based reward computation (both disrm and genrm)
    (3) high-flexible user-customized reward function (can access rm by posting requests to reward_model_router)

    Reward Computation Logic:
    - if user-customized reward function is provided:
        -> directly use user-customized reward function
    - if user-customized reward function is not provided:
        -> rm is not enabled: use default rule-based reward function
        -> rm is disrm: compute reward score using disrm
        -> rm is genrm: raise error (user-costomized reward func must be provided)
    """

    def __init__(
        self,
        config: DictConfig,
        reward_router_address: str = None,
        *,
        input_model_config: DictConfig | None = None,
        reward_model_config: DictConfig | None = None,
    ):
        """
        Args:
            config: DictConfig, the config for reward loop worker.
            reward_router_address: str, the address of reward router.
            input_model_config: model config selected for tokenizing policy samples.
            reward_model_config: reward-model config selected by topology ``config_key``.
        """
        self.config = config
        self.input_model_config = (
            input_model_config if input_model_config is not None else config.actor_rollout_ref.model
        )
        self.reward_model_config = (
            reward_model_config if reward_model_config is not None else config.reward.reward_model
        )
        self.reward_router_address = reward_router_address
        self._init_reward_fn()
        self.loop = get_event_loop()

    def _init_reward_fn(self):
        input_tokenizer_path = self.input_model_config.tokenizer_path
        if input_tokenizer_path is None:
            input_tokenizer_path = self.input_model_config.path
        input_tokenizer_local_path = copy_to_local(input_tokenizer_path)
        self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=True)
        self.reward_model_tokenizer = None
        if self.reward_model_config.enable:
            reward_model_tokenizer_local_path = copy_to_local(self.reward_model_config.model_path)
            self.reward_model_tokenizer = hf_tokenizer(reward_model_tokenizer_local_path, trust_remote_code=True)

        self.reward_manager = load_reward_manager(
            self.config,
            self.input_tokenizer,
            reward_router_address=self.reward_router_address,
            reward_model_tokenizer=self.reward_model_tokenizer,
        )

    async def compute_score_batch(self, data: DataProto) -> list[dict]:
        tasks = []
        for i in range(len(data)):
            tasks.append(asyncio.create_task(self.compute_score(data[i : i + 1])))
        outputs = await asyncio.gather(*tasks)
        return outputs

    async def compute_score(self, data: DataProto) -> dict:
        if self.config.reward.custom_reward_function.path is not None:
            # directly use user-customized reward function
            return await self.reward_manager.run_single(data)
        else:
            if self.reward_model_config.enable:
                # we assume the rm is disrm
                # genrm must set custom_reward_function
                return await self.compute_score_disrm(data[-1:])
            else:
                return await self.reward_manager.run_single(data)

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
        assert len(data) == 1, "RewardLoopWorker only support single data item"
        data_item = data[0]
        assert "raw_prompt" in data_item.non_tensor_batch

        # extract raw prompt
        chat: list = list(data_item.non_tensor_batch["raw_prompt"])

        # extract response
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        rollout_response = self.input_tokenizer.decode(valid_response_ids)
        rollout_response = rollout_response.replace(self.input_tokenizer.eos_token, "")

        chat.append({"role": "assistant", "content": rollout_response})

        rm_prompt = self.reward_model_tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=False,
            tokenize=False,
        )

        # llama tokenizer will add bos token by default
        # will be removed in vllm >= 0.11.2, where we can add "add_special_tokens" = False
        if self.reward_model_tokenizer.bos_token is not None and rm_prompt.startswith(
            self.reward_model_tokenizer.bos_token
        ):
            rm_prompt = rm_prompt[len(self.reward_model_tokenizer.bos_token) :]

        return rm_prompt

    async def compute_score_disrm(self, data: DataProto) -> dict:
        disrm_prompt = await self._preprocess_reward_inputs(data)
        engine_name = self.reward_model_config.rollout.name
        model_name = self.reward_model_config.model_path
        if engine_name == "vllm":
            payloads = {
                "model": model_name,
                "input": disrm_prompt,
                "use_activation": False,
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
        elif engine_name == "trtllm":
            # TODO: remove this once TRT-LLM switches to TorchSampler
            raise ValueError("TensorRT-LLM backend does not support reward models currently.")

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

        return {"reward_score": rm_score}


class RuntimeRewardLoopWorker(RewardLoopWorker, Worker):
    """RewardLoopWorker hosted by a backend-neutral Runtime WorkerGroup."""

    def __init__(self, *args, **kwargs) -> None:
        Worker.__init__(self)
        RewardLoopWorker.__init__(self, *args, **kwargs)

    def close(self) -> None:
        close = getattr(self.reward_manager, "close", None)
        if callable(close):
            close()


class RewardLoopManager:
    """
    RewardLoopManager run in single controller.
    This class will create reward loop workers and manage them.
    """

    def __init__(
        self,
        config: DictConfig,
        rm_resource_pool: ResourcePool | None = None,
        *,
        worker_resource_pool: ResourcePool,
        input_model_config: DictConfig | None = None,
        reward_model_config: DictConfig | None = None,
    ):
        """Create reward workers using explicitly selected model configs.

        Args:
            config: Whole trainer config for algorithm and reward-function settings.
            rm_resource_pool: GPU pool for a colocated or dedicated reward model.
            worker_resource_pool: Host pool used to place Runtime reward-loop workers.
            input_model_config: Policy model config used to tokenize reward inputs.
            reward_model_config: Reward-model config selected by topology ``config_key``.
        """
        self.config = config
        self.input_model_config = (
            input_model_config if input_model_config is not None else config.actor_rollout_ref.model
        )
        self.reward_model_config = (
            reward_model_config if reward_model_config is not None else config.reward.reward_model
        )
        if self.reward_model_config.enable:
            rm_rollout = self.reward_model_config.rollout
            # The discriminative /classify (pooling) path is not covered by
            # VLLM_BATCH_INVARIANT (vLLM batch invariance is verified on generation
            # models, not pooling RM architectures). Serialize /classify with
            # max_num_seqs=1 to keep it bitwise reproducible. The generative
            # /v1/chat/completions path (custom reward fn) is user-managed and not
            # forced here — rely on VLLM_BATCH_INVARIANT + per-request seed.
            if (
                rm_rollout.full_determinism
                and self.config.reward.custom_reward_function.path is None
                and rm_rollout.max_num_seqs != 1
            ):
                logger.warning(
                    "[reward_model] full_determinism=True: forcing rollout.max_num_seqs "
                    "from %s to 1 for the /classify pooling path (batch invariance not "
                    "verified for pooling RM). See the determinism doc.",
                    rm_rollout.max_num_seqs,
                )
                with open_dict(self.config):
                    rm_rollout.max_num_seqs = 1
            self.reward_model_manager = RewardModelManager(self.reward_model_config, rm_resource_pool)
            self.reward_router_address = self.reward_model_manager.get_router_address()
        else:
            self.reward_model_manager = None
            self.reward_router_address = None

        self.reward_manager_cls = resolve_reward_manager_cls(config)
        self.worker_group = None
        self._init_runtime_reward_loop_workers(worker_resource_pool)

    @property
    def remote_worker_group(self) -> RemoteWorkerGroup | None:
        """Return the serializable Runtime group used by streaming reward calls."""
        if self.worker_group is not None and (
            not self.reward_model_config.enable or self.reward_model_config.enable_resource_pool
        ):
            return self.worker_group.remote()
        return None

    def _init_runtime_reward_loop_workers(self, worker_resource_pool: ResourcePool) -> None:
        num_workers = self.config.reward.num_workers
        nnodes = worker_resource_pool.nnodes
        if num_workers % nnodes:
            raise ValueError(
                f"reward.num_workers={num_workers} must be divisible by worker ResourcePool nnodes={nnodes}"
            )
        runtime = current_runtime()
        cpu_pool = runtime.create_resource_pool(
            nnodes=nnodes,
            processes_per_node=num_workers // nnodes,
            device_type="cpu",
            on=worker_resource_pool,
        )
        self.worker_group = runtime.create_worker_group(
            ClassWithInitArgs(
                RuntimeRewardLoopWorker,
                config=self.config,
                reward_router_address=self.reward_router_address,
                input_model_config=self.input_model_config,
                reward_model_config=self.reward_model_config,
            ),
            on=cpu_pool,
        )

    def compute_rm_score(self, data: DataProto) -> DataProto:
        if self.reward_model_manager is not None:
            self.reward_model_manager.wake_up()

        num_workers = self.worker_group.world_size
        padded_data, pad_size = pad_dataproto_to_divisor(data, num_workers)
        chunks = padded_data.chunk(num_workers)
        remote_group = self.worker_group.remote()
        outputs = RemoteCall.gather(
            [remote_group.rank(rank).compute_score_batch(chunk) for rank, chunk in enumerate(chunks)]
        ).result()
        outputs_flat = [item for sublist in outputs for item in sublist]
        if pad_size > 0:
            outputs_flat = outputs_flat[: len(data)]

        # compute rm score
        scores = [item["reward_score"] for item in outputs_flat]
        rm_scores = self.reward_manager_cls.assemble_rm_scores(data, scores)
        batch = TensorDict({"rm_scores": rm_scores}, batch_size=len(data))

        reward_extra_infos = [output.get("reward_extra_info", {}) for output in outputs_flat]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        non_tensor_batch = {}
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        if self.reward_model_manager is not None:
            self.reward_model_manager.sleep()

        return data.new_like(
            batch=batch, non_tensor_batch=non_tensor_batch, meta_info={"reward_extra_keys": reward_extra_keys}
        )

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            return await asyncio.gather(*tasks)

        return asyncio.run(run_all())
