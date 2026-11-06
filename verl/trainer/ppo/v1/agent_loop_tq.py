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

# TODO: move this file to verl.experimental.agent_loop after V1 is stable
"""TransferQueue adapter for AgentLoopManager and AgentLoopWorker"""

import asyncio
import logging
import os
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import ray
import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.experimental.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.experimental.agent_loop.preprocess_cache import PromptPreprocessCache
from verl.trainer.ppo.v1.prompt_sharing import (
    PUBLIC_PARTITION,
    multi_modal_inputs_equal,
    prompt_sharing_enabled,
    prompt_sharing_requested,
    public_multi_modal_key,
    shared_prompt_fields,
)
from verl.utils.ray_utils import auto_await
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    # Greedy REMAX baselines are used for their rewards, not score-centering training heads.
    params.pop("topk_log_probs", None)
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


@dataclass
class _PublicMultiModalInputs:
    lock: asyncio.Lock = dataclass_field(default_factory=asyncio.Lock)
    value: dict | None = None


def _shareable_multi_modal_keys(processor) -> frozenset[str]:
    """Keys of ``multi_modal_inputs`` that depend only on the media, not on the prompt and response text."""
    keys = {"images_seqlens"}
    for name in ("image_processor", "video_processor", "feature_extractor"):
        keys.update(getattr(getattr(processor, name, None), "model_input_names", None) or ())
    # _compute_position_ids rebuilds mm_token_type_ids from input_ids and drops it, but only for M-RoPE processors.
    if hasattr(processor, "get_rope_index"):
        keys.add("mm_token_type_ids")
    return frozenset(keys)


async def _settle_session_tasks(tasks: list[asyncio.Task[Any]]) -> list[BaseException]:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if isinstance(result, BaseException)]


@ray.remote
class AgentLoopWorkerTQ(AgentLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks = set()
        self.share_prompt_fields = prompt_sharing_enabled(self.config.transfer_queue)
        self.share_preprocessing = prompt_sharing_requested(self.config.transfer_queue)
        # uid => fields its trajectories link from the public prompt the trainer put under uid
        self.public_prompt_fields: dict[str, list[str]] = {}
        # uid => multi_modal_inputs the first finished session of uid put under public_multi_modal_key(uid)
        self.public_multi_modal_inputs: dict[str, _PublicMultiModalInputs] = {}
        # uid => media preprocessing the sessions of uid share
        self.preprocess_caches: dict[str, PromptPreprocessCache] = {}
        self.shareable_multi_modal_keys = _shareable_multi_modal_keys(self.processor)

    def _get_preprocess_cache(self, sample_kwargs: dict[str, Any]) -> PromptPreprocessCache | None:
        return self.preprocess_caches.get(sample_kwargs.get("uid"))

    async def generate_sequences(self, batch: TensorDict) -> None:
        """Spawn agent loop for each sample in the batch without waiting for the results."""
        validate = batch["validate"] if "validate" in batch else False
        batch.pop("validate", None)
        public_fields = shared_prompt_fields(batch) if self.share_prompt_fields else []
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )
        # the trainer never consumes the sampler head on validation rollouts
        if config.topk_log_probs and not validate:
            sampling_params["topk_log_probs"] = config.topk_log_probs

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch:
            default_agent_loop = config.agent.default_agent_loop
            batch["agent_name"] = NonTensorData(default_agent_loop)

        trajectory_info = await get_trajectory_info(batch["global_steps"], batch["index"], validate)

        # create background tasks for each sample in the batch
        for i in range(len(batch)):
            # TODO(wuxibin): add trace support
            trace_this_sample = False
            prompt = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[i]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[i].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    logger.exception(f"Unsupported type {type(v)} for key {k}")

            # “fire-and-forget” background tasks
            task = asyncio.create_task(
                self._run_prompt(
                    prompt,
                    sampling_params,
                    trajectory=trajectory_info[i],
                    trace=trace_this_sample,
                    public_fields=public_fields,
                )
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def _run_prompt(
        self,
        prompt: dict,
        sampling_params: dict,
        trajectory: dict,
        trace: bool = False,
        public_fields: list[str] | None = None,
    ) -> None:
        """Spawn multiple agent loops in parallel according to rollout.n or rollout.val_kwargs.n."""
        uid, partition_id = prompt["uid"], "train" if not trajectory["validate"] else "val"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        if self.share_preprocessing:
            self.preprocess_caches[uid] = PromptPreprocessCache()
        if public_fields:
            self.public_prompt_fields[uid] = public_fields
            self.public_multi_modal_inputs[uid] = _PublicMultiModalInputs()
        tasks = []
        try:
            # NOTE: user can dynamically adjust n for each sample here, e.g according to task difficulty.
            config = self.config.actor_rollout_ref.rollout
            n = prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n)
            do_sample = prompt.pop("__do_sample__", True)

            run_sampling_params = dict(sampling_params)
            if not trajectory["validate"] and not do_sample:
                apply_greedy_sampling_params(run_sampling_params)

            tasks = []
            for i in range(n):
                task = asyncio.create_task(
                    self._run_agent_loop(
                        run_sampling_params, trajectory=trajectory, trace=trace, session_id=i, **prompt
                    )
                )
                tasks.append(task)

            # Publish a terminal status only after every session settles, so no sibling can write after
            # ReplayBuffer clears a failed group.
            session_errors = await _settle_session_tasks(tasks)
            if session_errors:
                for error in session_errors:
                    logger.error(
                        f"Error in _run_prompt for uid={uid}",
                        exc_info=(type(error), error, error.__traceback__),
                    )
                status = "failure"
            else:
                status = "finished"
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": status})
        except Exception as e:
            logger.exception(f"Error in _run_prompt: {e}")
            if tasks:
                await _settle_session_tasks(tasks)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})
        finally:
            self.preprocess_caches.pop(uid, None)
            self.public_prompt_fields.pop(uid, None)
            self.public_multi_modal_inputs.pop(uid, None)

    async def _get_multi_modal_inputs(self, uid: str, output: AgentLoopOutput, input_ids: torch.Tensor) -> dict:
        """``multi_modal_inputs`` of an output, computed once for the sessions of uid that saw the same media."""
        cache = self.preprocess_caches.get(uid)
        if cache is None or not output.multi_modal_data:
            return self._compute_multi_modal_inputs(output, input_ids)

        computed = None

        async def compute() -> dict | None:
            nonlocal computed
            computed = self._compute_multi_modal_inputs(output, input_ids)
            # None tells the other sessions to compute their own, since some features depend on the text.
            return dict(computed) if computed.keys() <= self.shareable_multi_modal_keys else None

        shared = await cache.get_or_compute(
            "multi_modal_inputs", (output.multi_modal_data, output.mm_processor_kwargs), compute
        )
        if computed is not None:
            return computed
        if shared is None:
            return self._compute_multi_modal_inputs(output, input_ids)
        # _compute_position_ids pops keys from the returned dict.
        return dict(shared)

    async def _share_multi_modal_inputs(self, uid: str, multi_modal_inputs: list[dict]) -> bool:
        """Whether the outputs of a session can link ``multi_modal_inputs`` from the public key of uid.

        The first session to finish puts its inputs; the others link them only if equal, since multi-turn
        sessions may see different images.
        """
        shared = self.public_multi_modal_inputs.get(uid)
        if shared is None or not multi_modal_inputs[0]:
            return False
        async with shared.lock:
            if shared.value is None:
                await tq.async_kv_batch_put(
                    keys=[public_multi_modal_key(uid)],
                    partition_id=PUBLIC_PARTITION,
                    fields=list_of_dict_to_tensordict([{"multi_modal_inputs": multi_modal_inputs[0]}]),
                )
                shared.value = multi_modal_inputs[0]
        return all(multi_modal_inputs_equal(inputs, shared.value) for inputs in multi_modal_inputs)

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate, **kwargs
    ) -> None:
        """Put agent loop outputs into TransferQueue."""
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            logger.warning(f"Empty output for prompt {uid}_{session_id}")
            return

        await self._compute_score(outputs, kwargs=kwargs)

        final_output = outputs[-1]
        # TODO: Support output:list[AgentLoopOutput]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )

        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                output.reward_score = final_output.reward_score
                output.extra_fields["reward_extra_info"] = final_output.extra_fields["reward_extra_info"]

        # NOTE: agent loop may has multiple outputs, put each output into TransferQueue.
        # key format: {uid}_{session_id}_{index}
        # - uid: raw prompt uid from dataset
        # - session_id: session id for rollout.n sampling
        # - index: index of agent loop output
        # Dataset fields are linked from the public prompt rather than stored n times.
        public_fields = [key for key in self.public_prompt_fields.get(uid, ()) if key in kwargs]
        keys, fields, tags, all_multi_modal_inputs = [], [], [], []
        for i, output in enumerate(outputs):
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = await self._get_multi_modal_inputs(uid, output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)

            keys.append(f"{uid}_{session_id}_{i}")
            field = output.as_dict()
            field.update(kwargs)
            # do not store raw image/video
            field.pop("multi_modal_data", None)
            for key in public_fields:
                field.pop(key)
            # TODO: uniform response_mask and loss_mask
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            all_multi_modal_inputs.append(multi_modal_inputs)
            fields.append(field)
            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            tags.append(
                {
                    "status": "success",
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                    # These tags are used for off-policy staleness control, if a trajectory
                    # spans too many global steps, we need to filter it out.
                    # global_steps: which global steps this sample is from dataloader
                    "global_steps": kwargs["global_steps"],
                    # min_global_steps: start generation model weights version of this trajectory
                    "min_global_steps": field["extra_fields"].get("min_global_steps"),
                    # max_global_steps: end generation model weights version of this trajectory
                    "max_global_steps": field["extra_fields"].get("max_global_steps"),
                }
            )

        partition_id = "train" if not validate else "val"
        if public_fields:
            await tq.async_kv_link(keys=keys, partition_id=partition_id, public_key=uid, fields=public_fields)
        if await self._share_multi_modal_inputs(uid, all_multi_modal_inputs):
            await tq.async_kv_link(
                keys=keys,
                partition_id=partition_id,
                public_key=public_multi_modal_key(uid),
                fields="multi_modal_inputs",
            )
        else:
            for field, multi_modal_inputs in zip(fields, all_multi_modal_inputs, strict=True):
                field["multi_modal_inputs"] = multi_modal_inputs
        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id=partition_id,
        )


class AgentLoopManagerTQ(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = AgentLoopWorkerTQ
        super().__init__(*args, **kwargs)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    def generate_sequences(self, prompts: TensorDict) -> None:
        """
        Dispatch input batch to agent loop workers without blocking. Workers should put agent loop outputs
        into TransferQueue once an agent loop finished.

        Args:
            prompts (TensorDict): Input batch from train or validation dataset.
        """
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=False)
            ]
        )
