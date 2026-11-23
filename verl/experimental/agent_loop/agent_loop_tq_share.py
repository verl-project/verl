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
"""Rollout worker whose sessions share the preprocessing and the TransferQueue storage of their prompt.

Everything prompt sharing adds to the rollout lives in ``AgentLoopWorkerTQShare``: with
``rollout.agent.share_prompts`` off or a storage backend other than SimpleStorage,
``AgentLoopManagerTQ`` runs the plain ``AgentLoopWorkerTQ`` and this module is never imported.
"""

import asyncio
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import hydra
import torch
import transfer_queue as tq
from tensordict import TensorDict

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    DictConfigWrap,
    ToolListWrap,
    _agent_loop_registry,
    _InternalAgentLoopOutput,
)
from verl.experimental.agent_loop.preprocess_cache import PromptPreprocessCache
from verl.experimental.agent_loop.prompt_sharing import (
    PUBLIC_PARTITION,
    async_release_public_prompts,
    multi_modal_inputs_equal,
    public_multi_modal_key,
    publish_public_prompts,
)
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ
from verl.utils.rollout_trace import rollout_trace_attr
from verl.utils.tensordict_utils import list_of_dict_to_tensordict


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


class AgentLoopWorkerTQShare(AgentLoopWorkerTQ):
    """``AgentLoopWorkerTQ`` whose ``rollout.n`` sessions share the work and the storage of their prompt."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # uid => fields its trajectories link from the public prompt put under uid
        self.public_prompt_fields: dict[str, list[str]] = {}
        # uid => multi_modal_inputs the first finished session of uid put under public_multi_modal_key(uid)
        self.public_multi_modal_inputs: dict[str, _PublicMultiModalInputs] = {}
        # uid => media preprocessing the sessions of uid share
        self.preprocess_caches: dict[str, PromptPreprocessCache] = {}
        self.shareable_multi_modal_keys = _shareable_multi_modal_keys(self.processor)

    async def generate_sequences(self, batch: TensorDict) -> None:
        # Freeing runs here because this is the point reached after a step cleared the trajectories
        # that were linking the public prompts of the previous batch.
        await async_release_public_prompts()
        public_fields = await publish_public_prompts(batch)
        # Registered per uid rather than carried into _run_prompt, which the next batch could outrun.
        for uid in batch["uid"]:
            self.public_prompt_fields[uid] = public_fields
            self.public_multi_modal_inputs[uid] = _PublicMultiModalInputs()
        await super().generate_sequences(batch)

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        uid = prompt["uid"]
        self.preprocess_caches[uid] = PromptPreprocessCache()
        try:
            await super()._run_prompt(prompt, sampling_params, trajectory, trace)
        finally:
            self.preprocess_caches.pop(uid, None)
            self.public_prompt_fields.pop(uid, None)
            self.public_multi_modal_inputs.pop(uid, None)

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        """``AgentLoopWorker._run_agent_loop``, with this prompt's cache attached to the agent loop.

        The cache has to be attached between building the agent loop and running it. Overriding the
        whole method keeps that seam out of the shared rollout code, at the cost of repeating the
        base method: keep the two in step.
        """
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop = hydra.utils.instantiate(
                config=_agent_loop_registry[agent_name],
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.llm_client,
                tokenizer=self.tokenizer,
                processor=self.processor,
                hf_model_type=self.hf_model_type,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
                tools=ToolListWrap(self.tools),
            )
            cache = self.preprocess_caches.get(kwargs.get("uid"))
            if cache is not None:
                cache.attach(agent_loop)
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
            return await self._agent_loop_postprocess(output, trajectory["validate"], **kwargs)

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

    async def _put_trajectories(
        self, keys: list[str], fields: list[dict], tags: list[dict], partition_id: str, sample_kwargs: dict[str, Any]
    ) -> None:
        """Link the dataset fields and the multi_modal_inputs of the prompt rather than storing them n times."""
        uid = sample_kwargs["uid"]
        public_fields = [key for key in self.public_prompt_fields.get(uid, ()) if key in sample_kwargs]
        for field in fields:
            for key in public_fields:
                field.pop(key)
        if public_fields:
            await tq.async_kv_link(keys=keys, partition_id=partition_id, public_key=uid, fields=public_fields)

        if await self._share_multi_modal_inputs(uid, [field["multi_modal_inputs"] for field in fields]):
            for field in fields:
                field.pop("multi_modal_inputs")
            await tq.async_kv_link(
                keys=keys,
                partition_id=partition_id,
                public_key=public_multi_modal_key(uid),
                fields="multi_modal_inputs",
            )
        await super()._put_trajectories(keys, fields, tags, partition_id, sample_kwargs)
