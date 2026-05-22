# Copyright 2026 Alibaba Group
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
#
# `_build_step_wise` and `_merge_stepwise` are adapted from SkyRL (Apache 2.0):
#   https://github.com/NovaSky-AI/SkyRL/blob/main/examples/train_integrations/harbor/harbor_generator.py
"""
Harbor agent loop for VeRL.

Bridges VeRL's :class:`AgentLoopBase` to Laude Institute's Harbor framework.
For each sample we spin up a Harbor :class:`Trial`, point it at a VeRL rollout
HTTP endpoint, and convert the multi-turn rollout details Harbor returns into
the linear (prompt_ids + response_ids + response_mask) format VeRL expects.

The conversion uses a two-stage layout: (1) ``_build_step_wise`` flattens
rollout_details into one entry per LLM turn, (2) ``_merge_stepwise`` greedily
merges turns into prefix-coherent groups (re-emitting obs deltas with
response_mask=0 in between), flushing a new group whenever Harbor's
re-rendered prompt diverges from ``prompt[t-1] + completion[t-1]``. A divergent
re-render is treated as a benign group boundary instead of a fatal error so the
loop tolerates chat templates that don't perfectly preserve prior-turn tokens.
"""

import logging
import os
import random
from copy import deepcopy
from typing import Any, Optional
from uuid import uuid4

from omegaconf import DictConfig, OmegaConf

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Suppress noisy LiteLLM logging (Harbor uses LiteLLM internally).
try:
    import logging as _logging

    import litellm

    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    _logging.getLogger("LiteLLM").setLevel(_logging.WARNING)
except ImportError:  # litellm only required at runtime
    pass

# How many times to retry a single trial on transient errors before giving up.
MAX_NUM_RETRIES_PER_TRIAL = 2


def _build_step_wise(rollout_details: list) -> list[dict]:
    """Flatten Harbor rollout_details into one entry per LLM turn.

    Returns ``[{"prompt_ids", "comp_ids", "logprobs"}, ...]`` with one entry per turn.
    """
    assert len(rollout_details) == 1, f"Expected exactly one rollout segment, got {len(rollout_details)}."
    rd = rollout_details[0]
    prompts = rd["prompt_token_ids"]
    completions = rd["completion_token_ids"]
    logprobs = rd["logprobs"]
    n = len(completions)
    assert len(prompts) == n and len(logprobs) == n, (
        f"Malformed rollout_details (prompts={len(prompts)}, completions={n}, logprobs={len(logprobs)})"
    )
    turns = []
    for t in range(n):
        assert len(logprobs[t]) == len(completions[t]), "logprobs and completion lengths must match"
        turns.append({"prompt_ids": list(prompts[t]), "comp_ids": list(completions[t]), "logprobs": list(logprobs[t])})
    return turns


def _is_prefix(maybe_prefix: list[int], candidate: list[int]) -> bool:
    return len(maybe_prefix) <= len(candidate) and maybe_prefix == candidate[: len(maybe_prefix)]


def _merge_stepwise(turns: list[dict]) -> list[dict]:
    """Greedy prefix-aware merge of step-wise turns into trajectory groups.

    Each output group is a self-contained
    ``{prompt_ids, response_ids, response_mask, response_logprobs}``: inside a
    group the response stream is ``comp[t]`` interleaved with obs deltas (the
    prefix-extension between consecutive prompts), with obs tokens masked to 0.
    Whenever ``prompt[t-1] + comp[t-1]`` is *not* a prefix of ``prompt[t]``
    (Harbor re-rendered the history non-trivially), we flush the current group
    and start a fresh one rooted at turn t. Returns one group when prefix holds
    throughout.
    """
    assert turns, "expected at least one turn"

    def _new_group(turn):
        return {
            "prompt_ids": list(turn["prompt_ids"]),
            "response_ids": list(turn["comp_ids"]),
            "response_mask": [1] * len(turn["comp_ids"]),
            "response_logprobs": list(turn["logprobs"]),
        }

    groups: list[dict] = []
    g = _new_group(turns[0])
    for t in turns[1:]:
        cursor_len = len(g["prompt_ids"]) + len(g["response_ids"])
        next_prompt = t["prompt_ids"]
        # cursor == g["prompt_ids"] + g["response_ids"]; build it lazily for the prefix check.
        if len(next_prompt) >= cursor_len and _is_prefix(g["prompt_ids"] + g["response_ids"], next_prompt):
            obs = next_prompt[cursor_len:]
            g["response_ids"].extend(obs)
            g["response_mask"].extend([0] * len(obs))
            g["response_logprobs"].extend([0.0] * len(obs))
            g["response_ids"].extend(t["comp_ids"])
            g["response_mask"].extend([1] * len(t["comp_ids"]))
            g["response_logprobs"].extend(t["logprobs"])
        else:
            groups.append(g)
            g = _new_group(t)
    groups.append(g)
    return groups


@register("harbor_agent")
class HarborAgentLoop(AgentLoopBase):
    """Run one Harbor trial per sample.

    Per-task config is taken from ``harbor_trial_config`` in the agent_loop YAML.
    The model name and api_base are injected at runtime from VeRL's rollout
    server addresses; ``session_id`` is generated per trial.
    """

    def __init__(
        self,
        trainer_config,
        server_manager,
        tokenizer,
        processor,
        dataset_cls,
        data_config,
        harbor_trial_config: Optional[Any] = None,
        served_model_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            trainer_config=trainer_config,
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=processor,
            dataset_cls=dataset_cls,
            data_config=data_config,
            **kwargs,
        )

        if harbor_trial_config is None:
            raise ValueError("HarborAgentLoop requires `harbor_trial_config`; declare it in your agent_loop YAML.")
        if isinstance(harbor_trial_config, DictConfig):
            harbor_trial_config = OmegaConf.to_container(harbor_trial_config, resolve=True)
        self._harbor_template: dict = deepcopy(harbor_trial_config)

        # ``served_model_name`` must match what vLLM advertises. vLLM's async server
        # (vllm_async_server.py) takes ``rollout.prometheus.served_model_name`` and
        # strips it down to the basename when the value contains a path separator,
        # so we apply the same basename rule here to stay in sync. Default for
        # ``prometheus.served_model_name`` is the full model path, which would
        # otherwise yield a malformed ``hosted_vllm//root/...`` LiteLLM target.
        if served_model_name is None:
            prom_name = getattr(self.rollout_config.prometheus, "served_model_name", None)
            served_model_name = prom_name or str(self.config.actor_rollout_ref.model.path)
        served_model_name = os.path.basename(str(served_model_name).rstrip("/"))
        if not served_model_name:
            raise ValueError(
                "Could not resolve served_model_name; pass it explicitly via "
                "agent_loop_kwargs.served_model_name=<name>."
            )
        if not getattr(self.rollout_config.prometheus, "enable", False):
            logger.warning(
                "rollout.prometheus.enable=False; vLLM may advertise the full model path "
                "instead of '%s'. Set prometheus.enable=True (and prometheus.served_model_name) "
                "to keep both sides in sync.",
                served_model_name,
            )
        self._served_model_name = served_model_name

        self._harbor_template.setdefault("agent", {})
        self._harbor_template["agent"]["model_name"] = f"hosted_vllm/{self._served_model_name}"
        agent_kwargs = self._harbor_template["agent"].setdefault("kwargs", {})
        agent_kwargs.setdefault("collect_rollout_details", True)
        if not agent_kwargs.get("collect_rollout_details", False):
            raise ValueError("HarborAgentLoop requires agent.kwargs.collect_rollout_details=true.")
        if agent_kwargs.get("enable_summarize", False):
            raise ValueError("HarborAgentLoop does not support agent.kwargs.enable_summarize=true.")

        self._prompt_length = self.rollout_config.prompt_length
        self._response_length = self.rollout_config.response_length

    async def _resolve_api_base(self) -> str:
        """Pick a VeRL rollout server address and turn it into an OpenAI base URL."""
        servers = await self.server_manager._load_balancer.get_all_servers.remote()
        if not servers:
            raise RuntimeError("No rollout servers registered with the load balancer.")
        return f"http://{random.choice(servers)}/v1"

    async def _run_trial(self, task_path: str, request_id: str):
        """Run a Harbor trial with retries."""
        from harbor.models.trial.config import TrialConfig
        from harbor.trial.trial import Trial

        api_base = await self._resolve_api_base()
        last_results = None
        is_context_length_error = False
        is_agent_timeout_error = False

        for attempt in range(MAX_NUM_RETRIES_PER_TRIAL):
            prefix = f"Trajectory {request_id} attempt {attempt + 1}/{MAX_NUM_RETRIES_PER_TRIAL}"
            try:
                cfg = deepcopy(self._harbor_template)
                cfg["task"] = {"path": task_path}
                cfg["agent"]["kwargs"]["api_base"] = api_base
                cfg["agent"]["kwargs"]["session_id"] = uuid4().hex

                trial = await Trial.create(TrialConfig.model_validate(cfg))
                last_results = await trial.run()

                exc_type = last_results.exception_info.exception_type if last_results.exception_info else None
                is_context_length_error = exc_type == "ContextLengthExceededError"
                is_agent_timeout_error = exc_type == "AgentTimeoutError"

                if is_agent_timeout_error:
                    logger.debug("%s hit AgentTimeoutError (no retry)", prefix)
                    return None, "agent_timeout", 0.0, 0
                if is_context_length_error:
                    reward = 0.0
                elif not last_results.verifier_result:
                    logger.warning("%s missing verifier_result, retrying. info=%s", prefix, last_results.exception_info)
                    continue
                else:
                    reward = float(last_results.verifier_result.rewards["reward"])

                rollout_details = last_results.agent_result.rollout_details
                num_turns = last_results.agent_result.metadata["n_episodes"]
                if (
                    rollout_details
                    and len(rollout_details) >= 1
                    and len(rollout_details[0].get("completion_token_ids", [])) > 0
                ):
                    return (
                        rollout_details,
                        "context_length" if is_context_length_error else "complete",
                        reward,
                        num_turns,
                    )
                logger.warning("%s empty rollout_details, retrying", prefix)
            except Exception as e:
                logger.warning("%s failed: %s", prefix, e)
                continue

        return None, "error", 0.0, 0

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # ``task_path`` is provided by HarborTaskDataset; fall back to extra_info for convenience.
        task_path = kwargs.get("task_path")
        if not task_path:
            extra_info = kwargs.get("extra_info") or {}
            task_path = extra_info.get("task_path")
        if not task_path:
            raise ValueError("HarborAgentLoop requires a `task_path` field on each sample.")

        request_id = uuid4().hex
        metrics: dict[str, float] = {}
        with simple_timer("generate_sequences", metrics):
            rollout_details, stop_reason, reward, num_turns = await self._run_trial(task_path, request_id)

        # fully_async expects per-sample ``min/max_global_steps`` to drive its
        # staleness / partial_rollout stats; standard agent loops get this via
        # ``super().generate()`` returning ``extra_fields["global_steps"]`` from the
        # vLLM server. Harbor goes through LiteLLM/HTTP and bypasses that path,
        # and the server class doesn't expose a public getter we can call from here.
        # We stamp 0 instead: with ``STALENESS_THRESHOLD=0`` + ``partial_rollout=False``
        # (the only mode this loop is exercised in) all samples come from one weight
        # version anyway, so ``abs(end - start) == 0`` is the correct answer; the
        # only loss is the ``param_version_diversity`` log metric (always 1).
        version_fields = {"min_global_steps": 0, "max_global_steps": 0}

        if rollout_details is None:
            # Failed trajectory: emit a single masked token so downstream padding works.
            return AgentLoopOutput(
                prompt_ids=[0],
                response_ids=[0],
                response_mask=[0],
                response_logprobs=[0.0],
                reward_score=0.0,
                num_turns=0,
                metrics=AgentLoopMetrics(**metrics),
                extra_fields={"stop_reason": stop_reason, "harbor_failed": True, **version_fields},
            )

        # Step-wise + prefix-aware merge. With multiple groups, pick one uniformly
        # at random for the policy gradient update; the trajectory reward is attached
        # to whichever group is picked. See README for selection-strategy notes.
        turns = _build_step_wise(rollout_details)
        groups = _merge_stepwise(turns)
        if len(groups) > 1:
            logger.warning(
                "Harbor trajectory %s split into %d merge groups (prefix divergence); "
                "randomly sampling one for policy gradient.",
                request_id,
                len(groups),
            )
        group = random.choice(groups)

        prompt_ids = group["prompt_ids"][: self._prompt_length]
        response_ids = group["response_ids"][: self._response_length]
        response_mask = group["response_mask"][: self._response_length]
        response_logprobs = group["response_logprobs"][: self._response_length]

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            reward_score=reward,
            num_turns=num_turns,
            metrics=AgentLoopMetrics(**metrics),
            extra_fields={
                "stop_reason": stop_reason,
                "harbor_failed": False,
                "harbor_num_merge_groups": len(groups),
                **version_fields,
            },
        )
