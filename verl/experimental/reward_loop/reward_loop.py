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
from omegaconf import DictConfig, open_dict
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local

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

    # 2.reward manager migration
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
        placement_group=None,
        start_bundle_index: int = None,
        num_gpus: int = None,
    ):
        """
        Args:
            config: DictConfig, the config for reward loop worker.
            reward_router_address: str, the address of reward router.
            placement_group: Ray PlacementGroup for sharing global_pool (e.g. forward_rdkit VLLM).
            start_bundle_index: First bundle index for this worker in the placement group.
            num_gpus: Number of GPUs (bundles) this worker needs for its model.
        """
        self.config = config
        self.reward_router_address = reward_router_address
        self.placement_group = placement_group
        self.start_bundle_index = start_bundle_index
        self.num_gpus = num_gpus
        self._init_reward_fn()

    def wait_beamsearch_ready(self) -> bool:
        """Block until this worker's reward manager (e.g. VLLMBeamSearchManager) has finished loading. Returns True. Used to serialize init and avoid GPU memory spike."""
        if hasattr(self.reward_manager, "vllm_beamsearch_manager") and hasattr(
            self.reward_manager.vllm_beamsearch_manager, "wait_ready"
        ):
            self.reward_manager.vllm_beamsearch_manager.wait_ready()
        return True

    def _init_reward_fn(self):
        input_tokenizer_local_path = copy_to_local(self.config.actor_rollout_ref.model.path)
        self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=True)
        self.reward_model_tokenizer = None
        if self.config.reward.reward_model.enable:
            reward_model_tokenizer_local_path = copy_to_local(self.config.reward.reward_model.model_path)
            self.reward_model_tokenizer = hf_tokenizer(reward_model_tokenizer_local_path, trust_remote_code=True)

        reward_manager_kwargs = {
            "reward_router_address": self.reward_router_address,
            "reward_model_tokenizer": self.reward_model_tokenizer,
        }
        if self.placement_group is not None and self.start_bundle_index is not None and self.num_gpus is not None:
            reward_manager_kwargs["placement_group"] = self.placement_group
            reward_manager_kwargs["start_bundle_index"] = self.start_bundle_index
            reward_manager_kwargs["num_gpus"] = self.num_gpus

        self.reward_manager = load_reward_manager(
            self.config,
            self.input_tokenizer,
            **reward_manager_kwargs,
        )

    async def compute_score_batch(self, data: DataProto) -> list[dict]:
        # If reward_manager has run_batch_forward, execute it first
        # This allows each worker to independently process its chunk using local GPUs
        print(f"[RewardLoopWorker] compute_score_batch called with data length: {len(data)}")
        print(f"[RewardLoopWorker] reward_manager type: {type(self.reward_manager)}")
        print(f"[RewardLoopWorker] reward_manager has run_batch_forward: {hasattr(self.reward_manager, 'run_batch_forward')}")
        
        if hasattr(self.reward_manager, 'run_batch_forward'):
            print(f"[RewardLoopWorker] Executing run_batch_forward...")
            logger.info("Running batch rxn forward")
            # Check if run_batch_forward is async (coroutine function)
            import inspect
            if inspect.iscoroutinefunction(self.reward_manager.run_batch_forward):
                data = await self.reward_manager.run_batch_forward(data)
            else:
                data = self.reward_manager.run_batch_forward(data)
            logger.info(f"Batch forward complete, data.batch keys: {list(data.batch.keys())}")
            print(f"[RewardLoopWorker] run_batch_forward complete, is_valid in keys: {'is_valid' in data.batch.keys()}")
        else:
            print(f"[RewardLoopWorker] run_batch_forward NOT FOUND - skipping batch forward")
            print(f"[RewardLoopWorker] Available methods: {[m for m in dir(self.reward_manager) if not m.startswith('_')]}")
        
        tasks = []
        # asyncio application
        for i in range(len(data)):
            # Use direct indexing instead of slicing to preserve all keys
            single_data = data[i : i + 1]
            logger.debug(f"Item {i} batch keys before compute_score: {list(single_data.batch.keys())}")
            tasks.append(asyncio.create_task(self.compute_score(single_data)))
        outputs = await asyncio.gather(*tasks)
        
        return outputs

    async def compute_score(self, data: DataProto) -> dict:
        assert len(data) == 1, "RewardLoopWorker only support single data item"
        if self.config.reward.custom_reward_function.path is not None:
            # directly use user-customized reward function
            return await self.reward_manager.run_single(data)
        else:
            if self.config.reward.reward_model.enable:
                # we assume the rm is disrm
                # genrm must set custom_reward_function
                return await self.compute_score_disrm(data)
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

        # decode
        rollout_response = self.input_tokenizer.decode(valid_response_ids)
        # remove bos and eos
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
        engine_name = self.config.reward.reward_model.rollout.name
        model_name = self.config.reward.reward_model.model_path
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


class RewardLoopManager:
    """
    RewardLoopManager run in single controller.
    This class will create reward loop workers and manage them.
    """

    def __init__(self, config: DictConfig, rm_resource_pool: RayResourcePool = None):
        self.config = config
        self.rm_resource_pool = rm_resource_pool
        if self.config.reward.reward_model.enable:
            self.reward_model_manager = RewardModelManager(config.reward.reward_model, rm_resource_pool)
            self.reward_router_address = self.reward_model_manager.get_router_address()
        else:
            self.reward_model_manager = None
            self.reward_router_address = None

        self.reward_loop_workers_class = ray.remote(RewardLoopWorker)
        self._init_reward_loop_workers()

    def _init_reward_loop_workers(self):
        self.reward_loop_workers = []
        num_workers = self.config.reward.num_workers
        forward_config = self.config.reward.get("forward_model", {})
        num_gpus_per_worker = forward_config.get("num_gpus", 0) if forward_config.get("model_path") else 0

        # When sharing global_pool: assign placement group bundles to each worker
        # With 4 workers × 2 GPUs = 8 VLLMBeamSearchInfer actors across 8 bundles
        # Multi-node: pool may have multiple PGs (e.g. [8, 8]), use all PGs so reward can use 16 bundles.
        placement_group = None
        start_bundle_indices = []
        placement_groups_per_worker = []  # per worker: which pg (when using multiple PGs)
        if self.rm_resource_pool is not None and num_gpus_per_worker > 0:
            pgs = self.rm_resource_pool.get_placement_groups()
            if pgs:
                total_bundles_needed = num_workers * num_gpus_per_worker
                total_bundles_available = sum(pg.bundle_count for pg in pgs)
                if total_bundles_needed <= total_bundles_available:
                    # Build (pg, start_index) for each worker: walk each PG's bundles in steps of num_gpus_per_worker
                    for pg in pgs:
                        for start in range(0, pg.bundle_count - num_gpus_per_worker + 1, num_gpus_per_worker):
                            placement_groups_per_worker.append(pg)
                            start_bundle_indices.append(start)
                            if len(placement_groups_per_worker) >= num_workers:
                                break
                        if len(placement_groups_per_worker) >= num_workers:
                            break
                    if len(placement_groups_per_worker) >= num_workers:
                        placement_groups_per_worker = placement_groups_per_worker[:num_workers]
                        start_bundle_indices = start_bundle_indices[:num_workers]
                        placement_group = placement_groups_per_worker[0] if num_workers == 1 else None
                        logger.info(
                            f"Reward workers will use placement_groups (total {total_bundles_available} bundles) "
                            f"with {num_workers} workers, each occupying {num_gpus_per_worker} bundles"
                        )
                    else:
                        placement_groups_per_worker = []
                        start_bundle_indices = []
                if not placement_groups_per_worker:
                    pg = pgs[0]
                    if total_bundles_needed <= pg.bundle_count:
                        placement_group = pg
                        start_bundle_indices = [i * num_gpus_per_worker for i in range(num_workers)]
                        logger.info(
                            f"Reward workers will use placement_group with {num_workers} workers, "
                            f"each occupying {num_gpus_per_worker} bundles starting at indices: {start_bundle_indices}"
                        )
                    else:
                        logger.warning(
                            f"reward needs {total_bundles_needed} bundles but pool has {total_bundles_available} "
                            f"(first pg has {pg.bundle_count}), reward workers will not use placement group"
                        )
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]

        for i in range(num_workers):
            pg = placement_groups_per_worker[i] if i < len(placement_groups_per_worker) else placement_group
            start_idx = start_bundle_indices[i] if i < len(start_bundle_indices) else None
            num_gpus = num_gpus_per_worker if (pg is not None) else None

            opts = {"name": f"reward_loop_worker_{i}"}
            if pg is None:
                # No pool: round-robin nodes (original behavior)
                node_id = node_ids[i % len(node_ids)]
                opts["scheduling_strategy"] = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=True,
                )

            worker_cls = self.reward_loop_workers_class.options(**opts)
            self.reward_loop_workers.append(
                worker_cls.remote(
                    self.config,
                    self.reward_router_address,
                    placement_group=pg,
                    start_bundle_index=start_idx,
                    num_gpus=num_gpus,
                )
            )
            
    def _deal_with_1000_score(
        self, data: DataProto, rollout_num: int, reward_positions: torch.Tensor = None
    ) -> DataProto:
        """Replace placeholder score 1000 with mean of *other* (valid) scores in the same rollout group.
        When reward_positions is provided, also updates batch['rm_scores'] so training uses the replaced value."""
        for i in range(len(data) // rollout_num):
            start_idx = i * rollout_num
            end_idx = start_idx + rollout_num
            # Mean of valid scores only (exclude 1000 placeholders)
            valid_sum = 0.0
            valid_count = 0
            for j in range(start_idx, end_idx):
                raw = data[j].non_tensor_batch["score"]
                try:
                    if float(raw) < 999.9:
                        valid_sum += float(raw)
                        valid_count += 1
                except (TypeError, ValueError):
                    pass
            mean_score = (valid_sum / valid_count) if valid_count > 0 else 0.0
            for j in range(start_idx, end_idx):
                raw_score = data[j].non_tensor_batch["score"]
                try:
                    is_1000 = float(raw_score) >= 999.9
                except (TypeError, ValueError):
                    is_1000 = False
                if is_1000:
                    data.non_tensor_batch["score"][j] = mean_score
                    data.non_tensor_batch["acc"][j] = 0.0
                    if reward_positions is not None:
                        data.batch["rm_scores"][j, reward_positions[j].item()] = mean_score
        return data
    def wait_reward_workers_ready(self) -> None:
        """Block until all reward loop workers (and their GPU models, e.g. VLLMBeamSearchInfer) have finished initializing. Call before AgentLoopManager.create() to avoid GPU memory spike from concurrent rollout + beamsearch init."""
        if not self.reward_loop_workers:
            return
        ray.get([w.wait_beamsearch_ready.remote() for w in self.reward_loop_workers])

    def compute_rm_score(self, data: DataProto) -> DataProto:
        if self.reward_model_manager is not None:
            self.reward_model_manager.wake_up()

        # Split data into chunks for parallel processing across workers
        chunks = data.chunk(len(self.reward_loop_workers))
        
        # Parallel dispatch to all workers
        # Each worker will independently execute run_batch_forward if available
        print(f"[RewardLoopManager] compute_rm_score called with data length: {len(data)}")
        outputs = ray.get(
            [
                worker.compute_score_batch.remote(chunk)
                for worker, chunk in zip(self.reward_loop_workers, chunks, strict=True)
            ]
        )
        outputs_flat = [item for sublist in outputs for item in sublist]

        # compute rm score
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

        reward_positions = valid_response_length - 1
        return self._deal_with_1000_score(
            DataProto(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
                meta_info={"reward_extra_keys": reward_extra_keys},
            ),
            self.config.actor_rollout_ref.rollout.n,
            reward_positions=reward_positions,
        )

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            return await asyncio.gather(*tasks)

        return asyncio.run(run_all())
