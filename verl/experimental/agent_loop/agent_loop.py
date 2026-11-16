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
"""
Agent framework for multi-turn rollout and agentic reinforcement learning.
- AgentLoopBase: coroutine based abstract base class for agent loop.
  - SingleTurnAgentLoop: single turn agent loop.
  - ToolAgentLoop: ReAct agent loop with tool calling, with user defined tools.
- AgentLoopWorker: worker class for running agent loop coroutines in parallel.
- AgentLoopManager: manager class for running agent loop workers in parallel.

AgentLoopManager is one specific agent-framework implementation in verl,
and is designed to be fully replaceable by other agent frameworks such as:
- NVIDIA Nemo-Gym
- AWS Bedrock AgentCore
- SWE-agent
- ...
"""

import asyncio
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional
from uuid import uuid4

import hydra
import numpy as np
import ray
import torch
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel
from tensordict import NonTensorData, NonTensorStack, TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.protocol import DataProto
from verl.tools.tool_registry import load_all_tools
from verl.trainer.distillation import is_distillation_enabled
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import RLHFDataset, get_dataset_class
from verl.utils.import_utils import resolve_config_path
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import simple_timer
from verl.utils.ray_utils import auto_await, get_event_loop
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
)
from verl.utils.tensordict_utils import list_of_dict_to_tensordict
from verl.utils.tokenizer import (
    build_multimodal_processor_inputs,
    get_processor_token_id,
)
from verl.utils.tokenizer.continuous_token_check import verify_continuous_token_builder
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder
from verl.workers.config import (
    HFModelConfig,
    RolloutConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0
    compute_score: float = 0.0
    num_preempted: int = -1  # -1 means not available


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    routed_experts: Optional[Any] = None
    """Routed experts for the total tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""
    mm_processor_kwargs: Optional[dict[str, Any]] = None
    """Processor/backend kwargs that must stay aligned across rollout and training paths."""
    mm_processor_output: Optional[list[dict[str, Any]]] = None
    """SGLang video payload (see AgentLoopBase.build_sglang_video_payload); reused by the teacher
    (OPD) path so an SGLang teacher receives the video instead of raw frames. None otherwise."""

    def as_dict(self) -> dict[str, Any]:
        """Convert agent loop output to a dictionary."""
        output = self.model_dump(exclude_unset=True)

        output["prompts"] = torch.tensor(output.pop("prompt_ids"), dtype=torch.int64)
        output["responses"] = torch.tensor(output.pop("response_ids"), dtype=torch.int64)
        output["response_mask"] = torch.tensor(output.pop("response_mask"), dtype=torch.int64)

        response_logprobs = output.pop("response_logprobs", None)
        if response_logprobs is not None:
            output["rollout_log_probs"] = torch.tensor(response_logprobs, dtype=torch.float32)

        routed_experts = output.pop("routed_experts", None)
        if routed_experts is not None:
            routed_experts = torch.tensor(routed_experts, dtype=torch.int16)
            # Router replay indexes this field by absolute token position, so it must
            # span the whole sequence. The rollout engine records fewer rows than that:
            # it only sees tokens fed through the model, and multi-turn loops stop
            # recording at the last generation. Trailing rows stay zero; replay masks
            # them out instead of consuming them.
            total_length = output["prompts"].size(0) + output["responses"].size(0)
            aligned = routed_experts.new_zeros((total_length, *routed_experts.shape[1:]))
            num_rows = min(routed_experts.size(0), total_length)
            aligned[:num_rows] = routed_experts[:num_rows]
            output["routed_experts"] = aligned

        # rm_scores: reward score for each token
        reward_score = output.pop("reward_score", None)
        if reward_score is not None:
            rm_scores = torch.zeros_like(output["response_mask"], dtype=torch.float32)
            rm_scores[-1] = reward_score
            output["rm_scores"] = rm_scores

        # Mutating a default dict does not add the field to Pydantic's fields-set,
        # so model_dump(exclude_unset=True) can omit populated extra fields.
        extra_fields = output.setdefault("extra_fields", self.extra_fields.copy())
        teacher_ids, teacher_logprobs = (
            extra_fields.pop("teacher_ids", None),
            extra_fields.pop("teacher_logprobs", None),
        )
        if teacher_ids is not None:
            output["teacher_ids"] = teacher_ids
        if teacher_logprobs is not None:
            output["teacher_logprobs"] = teacher_logprobs
        response_topk_ids, response_topk_log_probs = (
            extra_fields.pop("response_topk_ids", None),
            extra_fields.pop("response_topk_log_probs", None),
        )
        if response_topk_ids is not None and response_topk_log_probs is not None:
            from verl.trainer.ppo.score_centering import pad_rollout_topk

            if self.num_turns > 2:
                raise ValueError("rollout.topk_log_probs supports the single-turn agent loop only.")
            if len(response_topk_ids) != output["responses"].size(0):
                raise ValueError(
                    f"sampler top-k heads cover {len(response_topk_ids)} tokens, "
                    f"but the response has {output['responses'].size(0)}."
                )
            output["rollout_topk_ids"], output["rollout_topk_log_probs"] = pad_rollout_topk(
                response_topk_ids, response_topk_log_probs, prompt_length=output["prompts"].size(0)
            )
            # Same prediction-row layout as the heads. Dummy prompt/padding heads are complete
            # by construction; a legacy response without provider evidence remains unknown.
            response_complete = extra_fields.pop("response_support_complete", None)
            if response_complete is None:
                response_complete = torch.zeros(len(response_topk_ids), dtype=torch.bool)
            else:
                response_complete = torch.as_tensor(response_complete)
                if response_complete.dtype != torch.bool or response_complete.shape != (len(response_topk_ids),):
                    raise ValueError("sampler support coverage must have one bool per response token")
            complete = torch.ones(output["rollout_topk_ids"].size(0), dtype=torch.bool)
            start = output["prompts"].size(0) - 1
            complete[start : start + len(response_topk_ids)] = response_complete
            output["rollout_support_complete"] = complete
        return output


class DictConfigWrap:
    """Wrapper for DictConfig to avoid hydra.utils.instantiate recursive resolve."""

    def __init__(self, config: DictConfig):
        self.config = config


class ToolListWrap:
    """Wraps a tool list so ``hydra.utils.instantiate`` doesn't recursively
    resolve its elements (which would demote them to ``DictConfig``)."""

    def __init__(self, tools: list):
        self.tools = tools


class AgentLoopBase(ABC):
    """An agent loop takes an input message, chat with OpenAI compatible LLM server and interact with various
    environments.

    Args:
        trainer_config (DictConfig): whole config for main entrypoint.
        server_manager (LLMServerClient): OpenAI compatible LLM server manager.
        tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        processor (AutoProcessor): Processor for process messages.
        dataset_cls (type[Dataset]): Dataset class for creating dataset, Defaults to RLHFDataset.
        data_config (DictConfigWrap): Dataset config.
        hf_model_type: Root Hugging Face ``model_type`` used for Continuous Token builder selection.
    """

    # Set to True in loops that append tool/user turns to the generated tokens (``ct_merge_context_msg``).
    # AgentLoopWorker checks the Continuous Token builder against the chat template before the first
    # rollout only when a loop the run can dispatch to sets it.
    merges_context_messages: ClassVar[bool] = False

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: LLMServerClient,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        data_config: DictConfigWrap,
        hf_model_type: str | None = None,
        **kwargs,
    ):
        self.config = trainer_config.config
        self.rollout_config = self.config.actor_rollout_ref.rollout
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.data_config = data_config.config
        self.apply_chat_template_kwargs = self.data_config.get("apply_chat_template_kwargs", {})
        self.mm_processor_kwargs = self.data_config.get("mm_processor_kwargs", {})
        # Continuous Token is the only rollout tokenization path for agent loops.
        # The model family (boundary handling) is inferred by exact lookup of the
        # root Hugging Face config's model_type. Unrecognized models use the
        # default builder (or default VL builder with a multimodal processor).
        self.continuous_token_builder = create_continuous_token_builder(
            self.tokenizer,
            hf_model_type=hf_model_type,
            chat_template_kwargs=self.apply_chat_template_kwargs,
            mm_processor_kwargs=self.mm_processor_kwargs,
            processor=self.processor,
        )
        self.loop = get_event_loop()

    def _assert_mm_supported(self, has_multi_modal: bool) -> None:
        """Fail loudly when multimodal inputs are present but unsupported.

        Multimodal rollout requires both a Continuous Token builder that supports
        multimodal boundaries and a non-None processor to render placeholder spans.
        Silent fallback is not allowed; callers must invoke this before mutating any
        rollout state so failures never leave a half-built prompt behind.
        """
        if not has_multi_modal:
            return
        if not (self.continuous_token_builder.supports_multimodal() and self.processor is not None):
            raise ValueError(
                "Multimodal inputs require a Continuous Token builder that supports multimodal "
                "AND a non-None processor, but got "
                f"supports_multimodal={self.continuous_token_builder.supports_multimodal()}, "
                f"processor={'set' if self.processor is not None else 'None'}. "
                "Use a VL base model (with its processor) or remove multimodal inputs."
            )

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]] = None) -> dict[str, Any]:
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
        return mm_processor_kwargs

    async def process_vision_info(self, messages: list[dict]) -> dict:
        """Backward-compatible wrapper for multi-modal extraction."""
        return await self.process_multi_modal_info(messages)

    async def process_multi_modal_info(self, messages: list[dict]) -> dict:
        """Extract images, videos and audios from messages.

        Args:
            messages (list[dict]): Input messages.

        Returns:
            dict: Multi-modal data with keys like "images", "videos" and "audios".
        """
        multi_modal_data = {}
        if self.processor is not None:
            image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
            if hasattr(self.dataset_cls, "process_multi_modal_info"):
                images, videos, audios = await self.dataset_cls.process_multi_modal_info(
                    messages, image_patch_size=image_patch_size, config=self.data_config
                )
            else:
                images, videos = await self.dataset_cls.process_vision_info(
                    messages, image_patch_size=image_patch_size, config=self.data_config
                )
                audios = None
            if images is not None:
                multi_modal_data["images"] = images
            if videos is not None:
                multi_modal_data["videos"] = videos
            if audios is not None:
                multi_modal_data["audios"] = audios

        return multi_modal_data

    async def ct_build_initial_tokens(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple[torch.Tensor, dict]] = None,
        audios: list[Any] = None,
        mm_inputs_out: dict[str, Any] = None,
    ) -> list[int]:
        """Build the initial prompt token ids with Continuous Token.

        Multimodal inputs are forwarded to the builder so that VL builders can
        render placeholder spans through the processor. Text-only builders accept
        and ignore these arguments. ``mm_processor_kwargs`` is not threaded here:
        it is a builder-lifetime constant captured at construction and applied by
        the builder itself during rendering.
        """
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.continuous_token_builder.build_initial_tokens(
                messages,
                tools=tools,
                images=images,
                videos=videos,
                audios=audios,
                mm_inputs_out=mm_inputs_out,
            ),
        )

        # Mirror the response-side ``response_ids[:response_length]`` cap on the prompt side:
        # every prompt produced by the agent loop must fit in ``rollout.prompt_length`` to respect
        # the configured context budget.
        # Multimodal prompts cannot be sliced here because placeholder tokens must remain
        # aligned 1:1 with ``multi_modal_inputs`` features, so we fail loudly instead.
        prompt_length = self.rollout_config.prompt_length
        if (images or videos or audios) and len(prompt_ids) > prompt_length:
            raise ValueError(
                f"Multimodal prompt produced {len(prompt_ids)} tokens, exceeding "
                f"rollout.prompt_length={prompt_length}. Truncating multimodal token "
                f"sequences corrupts vision/audio feature alignment, so this is treated "
                f"as a configuration error. Reduce the multimodal input size "
                f"(e.g. ``total_pixels`` / ``max_pixels`` / fps / number of frames) or "
                f"increase ``rollout.prompt_length``."
            )
        return self._cap_text_prompt_length(prompt_ids)

    @staticmethod
    def build_sglang_video_payload(videos, mm_inputs: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
        """Wrap already-computed video features for SGLang's ``video_data``.

        SGLang cannot ingest raw frames — its ``video_data`` accepts a path / URL / base64 string or
        a dict. ``mm_inputs`` is the processor output captured during tokenization (via the
        ``mm_inputs_out`` hook). Tagging it ``format="processor_output"`` makes
        ``BaseMultimodalProcessor`` pass the dict through untouched instead of re-decoding the video,
        which would re-sample frames and desync the rollout from the actor. Returns ``None`` when
        there is no video, so callers can hand the result straight to ``generate(...)``.

        Shared by SingleTurnAgentLoop and ToolAgentLoop; vLLM keeps the raw frames and ignores this
        (the per-backend split lives in ``LLMServerClient.generate``).
        """
        if not videos or "pixel_values_videos" not in mm_inputs:
            return None
        payload = {
            "format": "processor_output",
            "pixel_values_videos": mm_inputs["pixel_values_videos"],
            "video_grid_thw": mm_inputs["video_grid_thw"],
        }
        if "second_per_grid_ts" in mm_inputs:
            payload["second_per_grid_ts"] = mm_inputs["second_per_grid_ts"]
        return [payload]

    async def ct_merge_context_msg(
        self,
        previous_messages: list[dict],
        updated_messages: list[dict],
        runtime_token_ids: list[int],
        response_mask: list[int],
        response_logprobs: Optional[list[float]] = None,
        tools: list[dict] = None,
    ):
        """Merge appended context messages into runtime tokens and metadata."""
        merge_result = await self.loop.run_in_executor(
            None,
            lambda: self.continuous_token_builder.merge_context_tokens(
                previous_messages,
                updated_messages,
                runtime_token_ids,
                tools=tools,
            ),
        )
        aligned_response_mask, aligned_response_logprobs = self.continuous_token_builder.align_response_metadata(
            merge_result, response_mask, response_logprobs
        )
        return merge_result, aligned_response_mask, aligned_response_logprobs

    async def ct_merge_assistant_token(
        self,
        runtime_token_ids: list[int],
        assistant_token_ids: list[int],
        response_mask: list[int],
        response_logprobs: Optional[list[float]] = None,
        assistant_logprobs: Optional[list[float]] = None,
    ):
        """Merge assistant-generated tokens and align response metadata."""
        merge_result = await self.loop.run_in_executor(
            None,
            lambda: self.continuous_token_builder.merge_assistant_tokens(
                runtime_token_ids,
                assistant_token_ids,
            ),
        )
        aligned_response_mask, aligned_response_logprobs = self.continuous_token_builder.align_response_metadata(
            merge_result,
            response_mask,
            response_logprobs,
            assistant_logprobs=assistant_logprobs,
        )
        return merge_result, aligned_response_mask, aligned_response_logprobs

    def _cap_text_prompt_length(self, prompt_ids: list[int]) -> list[int]:
        prompt_length = self.rollout_config.prompt_length
        if len(prompt_ids) > prompt_length:
            logger.warning(
                "Prompt of %d tokens exceeds rollout.prompt_length=%d; left-truncating.",
                len(prompt_ids),
                prompt_length,
            )
            return prompt_ids[-prompt_length:]
        return prompt_ids

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


def _any_context_merging_agent_loop(agent_names: set[str]) -> bool:
    """Return whether a registered loop among ``agent_names`` declares ``merges_context_messages``."""
    for agent_name in agent_names:
        agent_loop_config = _agent_loop_registry.get(agent_name)
        if agent_loop_config is None:
            continue
        try:
            agent_loop_cls = hydra.utils.get_class(agent_loop_config["_target_"])
        except Exception as exc:
            logger.warning("Could not resolve agent loop %r, assuming it merges context: %s", agent_name, exc)
            return True
        # Registered loops need not subclass AgentLoopBase, so the flag may be missing.
        if getattr(agent_loop_cls, "merges_context_messages", False):
            return True
    return False


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    """Set greedy sampling parameters in place and disable top-k log-probability collection."""
    params.pop("topk_log_probs", None)
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


async def _settle_session_tasks(tasks: list[asyncio.Task[Any]]) -> list[BaseException]:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if isinstance(result, BaseException)]


class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        rollout_config, model_config = config.actor_rollout_ref.rollout, config.actor_rollout_ref.model
        self.rollout_config: RolloutConfig = omega_conf_to_dataclass(rollout_config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)

        self.dataset_cls = get_dataset_class(config.data)
        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor
        hf_model_type = getattr(self.model_config.hf_config, "model_type", None)
        self.hf_model_type: str | None = hf_model_type if isinstance(hf_model_type, str) else None
        self.mm_processor_kwargs = config.data.get("mm_processor_kwargs", {})

        # Online policy distillation
        self.distillation_enabled = is_distillation_enabled(config.distillation)
        if self.distillation_enabled:
            from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager

            self.teacher_key: str = config.distillation.teacher_key
            self.teacher_server_manager = AsyncTeacherLLMServerManager(
                config=config,
                teacher_client=teacher_client,
            )

        # Load tools once per worker; each trajectory just reuses self.tools.
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        function_tool_path = self.rollout_config.multi_turn.function_tool_path
        self.tools = load_all_tools(
            tool_config_path=resolve_config_path(tool_config_path) if tool_config_path else None,
            function_tool_path=resolve_config_path(function_tool_path) if function_tool_path else None,
        )

        # Load custom agent loop implementations from config path
        agent_loop_config_path = self.rollout_config.agent.agent_loop_config_path
        custom_agent_names = []
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
                custom_agent_names.append(agent_loop_config.name)
        if self.model_config.get("custom_chat_template", None) is not None:
            if self.model_config.processor is not None:
                self.model_config.processor.chat_template = self.model_config.custom_chat_template
            self.model_config.tokenizer.chat_template = self.model_config.custom_chat_template

        self._check_continuous_token_builder(custom_agent_names)

        trace_config = self.rollout_config.trace
        if trace_config.get("token2text", False):
            # rollout_trace_op runs on the LLM client, so provide the tokenizer
            # needed to decode each generate call's prompt and response tokens.
            self.llm_client.tokenizer = self.tokenizer
        RolloutTraceConfig.init(
            self.rollout_config.trace.project_name,
            self.rollout_config.trace.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )
        tq.init()
        self.background_tasks = set()

    def _check_continuous_token_builder(self, custom_agent_names: list[str]) -> None:
        """Check the Continuous Token builder against the chat template before the first rollout.

        Only loops that append tool/user turns merge context, so the check runs when a loop the run
        can dispatch to declares ``merges_context_messages``. Samples dispatch to their dataset
        ``agent_name`` or the default loop, and datasets route tool rollouts to the built-in tool
        agent, so it is a candidate whenever multi-turn or tools are configured.
        """
        multi_turn = self.rollout_config.multi_turn
        if multi_turn.tokenization_sanity_check_mode == "disable":
            return
        agent_names = {self.rollout_config.agent.default_agent_loop, *custom_agent_names}
        if multi_turn.enable or self.tools:
            agent_names.add("tool_agent")
        if not _any_context_merging_agent_loop(agent_names):
            return
        builder = create_continuous_token_builder(
            self.tokenizer,
            hf_model_type=self.hf_model_type,
            chat_template_kwargs=self.config.data.get("apply_chat_template_kwargs", {}),
            mm_processor_kwargs=self.mm_processor_kwargs,
            processor=self.processor,
        )
        verify_continuous_token_builder(
            builder,
            mode=multi_turn.tokenization_sanity_check_mode,
            hf_model_type=self.hf_model_type,
            generation_config=self.model_config.generation_config,
        )

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]] = None) -> dict[str, Any]:
        """Return multimodal processor kwargs with audio sampling-rate defaults."""
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
        return mm_processor_kwargs

    async def generate_sequences(self, batch: TensorDict) -> None:
        """Spawn agent loop for each sample in the batch without waiting for the results.

        Completed trajectories are published to TransferQueue without padding.
        For multi-turn conversations, responses include both LLM-generated tokens
        and tool observations. The response_mask is 1 for LLM-generated tokens and
        0 for observation tokens; it is also stored as loss_mask in TransferQueue.

        responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|
        response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 |
        """
        validate = batch["validate"] if "validate" in batch else False
        batch.pop("validate", None)
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
                self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=trace_this_sample)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        """Spawn multiple agent loops in parallel according to rollout.n or rollout.val_kwargs.n."""
        uid, partition_id = prompt["uid"], "train" if not trajectory["validate"] else "val"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
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

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> None:
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

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.llm_client,
                tokenizer=self.tokenizer,
                processor=self.processor,
                hf_model_type=self.hf_model_type,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
                tools=ToolListWrap(self.tools),
            )
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
            return await self._agent_loop_postprocess(output, trajectory["validate"], **kwargs)

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
        keys, fields, tags = [], [], []
        for i, output in enumerate(outputs):
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)

            keys.append(f"{uid}_{session_id}_{i}")
            field = output.as_dict()
            field.update(kwargs)
            # do not store raw image/video
            field.pop("multi_modal_data", None)
            # TODO: uniform response_mask and loss_mask
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs
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

        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        )

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image, video and audio."""
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs

        multi_modal_data = output.multi_modal_data or {}
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        # Collapse expanded vision placeholder runs back to a single placeholder
        # per media item before decoding, so the processor re-expands cleanly.
        # ``input_ids`` already contains fully expanded image/video pad tokens; if
        # those placeholders are NOT registered as special tokens (e.g. Kimi-VL
        # ``<|media_pad|>``, GLM-4.6V ``<|image|>``), ``skip_special_tokens=True``
        # will not strip them, and re-feeding the already-expanded text to the
        # processor triggers a double-expansion error. Collapsing at the token-id
        # level (using the processor-resolved placeholder ids) is model-agnostic
        # and leaves the well-behaved (special-token) placeholders untouched.
        image_token_id = get_processor_token_id(self.processor, "image")
        video_token_id = get_processor_token_id(self.processor, "video")
        collapse_ids = {t for t in (image_token_id, video_token_id) if t is not None}
        collapsed_ids, prev_id = [], None
        for token_id in input_ids.squeeze(0).tolist():
            if token_id in collapse_ids and token_id == prev_id:
                continue
            collapsed_ids.append(token_id)
            prev_id = token_id
        current_text = self.tokenizer.decode(collapsed_ids, skip_special_tokens=True)

        multi_modal_inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[current_text],
            images=images,
            videos=videos,
            audio=audios,
            mm_processor_kwargs=output.mm_processor_kwargs
            if output.mm_processor_kwargs is not None
            else self._get_mm_processor_kwargs(audios),
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
        # because np.array() only keeps the keys for BatchFeature.
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs["images_seqlens"] = images_seqlens
        return multi_modal_inputs

    def _compute_position_ids(
        self,
        input_ids,
        attention_mask,
        multi_modal_inputs,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        # text-only OR non-M-RoPE multimodal (e.g. Gemma4) -> standard 1D positions
        if self.processor is None or not hasattr(self.processor, "get_rope_index"):
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        multi_modal_kwargs = {
            "image_grid_thw": multi_modal_inputs.get("image_grid_thw"),
            "video_grid_thw": multi_modal_inputs.get("video_grid_thw"),
        }
        # For transformers>=5.3.0, mm_token_type_ids is only used to calculate position ids.
        if multi_modal_inputs.pop("mm_token_type_ids", None) is not None:
            mm_token_type_ids = torch.zeros_like(input_ids)
            image_token_id = get_processor_token_id(self.processor, "image")
            video_token_id = get_processor_token_id(self.processor, "video")
            if image_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == image_token_id] = 1
            if video_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == video_token_id] = 2
            multi_modal_kwargs["mm_token_type_ids"] = mm_token_type_ids

        # Allow model-specific processors to contribute additional RoPE inputs.
        get_rope_index_kwargs = getattr(self.processor, "get_rope_index_kwargs", None)
        if get_rope_index_kwargs is not None:
            multi_modal_kwargs.update(get_rope_index_kwargs(multi_modal_inputs))

        # Model's get_rope_index has been dynamically bind to the processor.
        vision_position_ids, _ = self.processor.get_rope_index(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **multi_modal_kwargs,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq_len) => (1, 3, seq_len)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        return position_ids

    async def _compute_score(self, outputs: list[AgentLoopOutput], kwargs: dict) -> None:
        """Compute reward score for all outputs in a trajectory; assigns result to outputs[-1]."""
        enable_async_reward = self.reward_loop_worker_handles is not None

        final_output = outputs[-1]
        if final_output.reward_score is None and enable_async_reward:
            timing = {}
            with simple_timer("compute_score", timing):
                all_prompts, all_responses, all_input_ids, all_attention_mask, all_position_ids = [], [], [], [], []
                for output in outputs:
                    prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
                    responses = torch.tensor(output.response_ids, dtype=torch.int64)
                    input_ids = torch.cat([prompts, responses], dim=0)
                    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
                    multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
                    position_ids = self._compute_position_ids(
                        input_ids.unsqueeze(0),
                        attention_mask.unsqueeze(0),
                        multi_modal_inputs,
                        output.mm_processor_kwargs
                        if output.mm_processor_kwargs is not None
                        else self._get_mm_processor_kwargs(
                            output.multi_modal_data.get("audios") if output.multi_modal_data else None
                        ),
                    ).squeeze(0)
                    all_prompts.append(prompts)
                    all_responses.append(responses)
                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_position_ids.append(position_ids)

                n = len(outputs)
                batch = TensorDict(
                    {
                        "prompts": torch.nn.utils.rnn.pad_sequence(all_prompts, batch_first=True, padding_value=0),
                        "responses": torch.nn.utils.rnn.pad_sequence(all_responses, batch_first=True, padding_value=0),
                        "attention_mask": torch.nn.utils.rnn.pad_sequence(
                            all_attention_mask, batch_first=True, padding_value=0
                        ),
                        "input_ids": torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=0),
                        "position_ids": torch.nn.utils.rnn.pad_sequence(
                            all_position_ids, batch_first=True, padding_value=0
                        ),
                    },
                    batch_size=n,
                )
                non_tensor_batch = {
                    **{k: np.array([v] * n) for k, v in kwargs.items()},
                    "__num_turns__": np.array([o.num_turns for o in outputs]),
                    "tool_extra_fields": np.array([o.extra_fields for o in outputs], dtype=object),
                    "prompt_len": np.array([len(o.prompt_ids) for o in outputs]),
                    "response_len": np.array([len(o.response_ids) for o in outputs]),
                }

                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                )
                selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
                result = await selected_reward_loop_worker_handle.compute_score.remote(data)
                final_output.reward_score = result["reward_score"]
                final_output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
            final_output.metrics.compute_score = timing["compute_score"]

    async def _compute_teacher_logprobs(
        self,
        output: AgentLoopOutput,
        prompt_ids: list[int],
        response_ids: list[int],
        validate: bool,
        sample_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Compute teacher logprobs for single sample."""
        if self.distillation_enabled and not validate:
            routing_key = None
            if sample_kwargs is not None:
                routing_value = sample_kwargs.get(self.teacher_key)
                if routing_value is not None:
                    # Non-tensor batch values arrive as 0-d numpy objects / arrays; normalize to Python.
                    routing_key = routing_value.item() if hasattr(routing_value, "item") else routing_value
            teacher_ids, teacher_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
                sequence_ids=prompt_ids + response_ids,
                multi_modal_data=output.multi_modal_data,
                mm_processor_kwargs=output.mm_processor_kwargs,
                mm_processor_output=getattr(output, "mm_processor_output", None),
                routing_key=routing_key,
            )
            output.extra_fields["teacher_ids"] = teacher_ids
            output.extra_fields["teacher_logprobs"] = teacher_logprobs


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = ray.remote(AgentLoopWorker)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    async def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.rollout_config.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}" + f"_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    self.llm_client,
                    self.teacher_client,
                    self.reward_loop_worker_handles,
                )
            )

    def generate_sequences(self, prompts: TensorDict) -> None:
        """
        Dispatch prompts and wait only for workers to schedule their background tasks.
        Workers publish completed trajectories to TransferQueue; this method returns no batch.

        Args:
            prompts (TensorDict): Input batch from train or validation dataset.
        """
        chunks = prompts.chunk(len(self.agent_loop_workers))
        ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunks, strict=False)
            ]
        )
