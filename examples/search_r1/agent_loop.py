"""Multi-message Search-R1 rollout with loss only on sampled assistant tokens."""

import copy
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer

from .prompt_cache import tokenizer_runtime
from .protocol import parse_action, score_final_turn


class SearchR1AgentLoop(AgentLoopBase):
    def __init__(
        self,
        *args,
        tools=None,
        max_actions=4,
        max_turn_tokens=500,
        max_observation_tokens=500,
        observation_tag="information",
        max_generation_context_tokens=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        available = {tool.name: tool for tool in (tools.tools if tools else [])}
        if "search" not in available:
            raise ValueError("SearchR1AgentLoop requires a configured search BaseTool")
        self.search = available["search"]
        self.max_actions = int(max_actions)
        self.max_turn_tokens = int(max_turn_tokens)
        self.max_observation_tokens = int(max_observation_tokens)
        if observation_tag not in {"information", "documents"}:
            raise ValueError("Unsupported observation tag")
        self.observation_tag = observation_tag
        self.max_generation_context_tokens = max_generation_context_tokens
        if min(self.max_actions, self.max_turn_tokens, self.max_observation_tokens) < 1:
            raise ValueError("Search-R1 budgets must be positive")
        # This recipe uses Qwen ChatML. XML string stops do not produce im_end;
        # we insert that boundary as environment context, with mask/logprob zero.
        self.tokenizer_runtime = tokenizer_runtime(self.tokenizer)
        self.turn_end = list(self.tokenizer_runtime.turn_end)
        self.turn_end_id = self.turn_end[0]

    async def run(self, sampling_params, priority=0, **kwargs):
        reward_metric = (kwargs.get("extra_info") or {}).get("reward_metric", "em")
        if reward_metric != "em":
            raise ValueError("Reward alignment to F1 is deferred; Search-R1 training remains EM")
        messages = copy.deepcopy(list(kwargs["raw_prompt"]))
        initial = await self.tokenizer_runtime.initial_tokens(
            self.tokenizer, self.continuous_token_builder, messages
        )
        # Fail rather than silently drop the beginning of a conversation.
        if len(initial) > self.rollout_config.prompt_length:
            raise ValueError("Initial Search-R1 prompt exceeds prompt_length")
        runtime = list(initial)
        mask, logprobs = [], [] if sampling_params.get("logprobs") else None
        budget = self.rollout_config.response_length
        max_context = self.rollout_config.max_model_len or (self.rollout_config.prompt_length + budget)
        if budget < 1 or len(initial) >= max_context:
            raise ValueError("Search-R1 needs positive generation headroom")
        metrics = {}
        generation_metadata = {}
        tool_metrics = []
        reason, final_text, searches, actions = "action_budget", "", 0, 0
        instance, _ = await self.search.create()
        request_id = uuid4().hex
        try:
            for action in range(self.max_actions):
                if self.max_generation_context_tokens and len(runtime) > self.max_generation_context_tokens:
                    reason = "context_budget"
                    break
                remaining = min(budget - len(mask), max_context - len(runtime))
                if remaining <= 0:
                    reason = "context_budget"
                    break
                params = dict(sampling_params)
                params.pop("max_new_tokens", None)
                max_tokens = min(self.max_turn_tokens, remaining)
                params.update(
                    max_tokens=max_tokens,
                    stop=["</search>", "</answer>"],
                    include_stop_str_in_output=True,
                    stop_token_ids=list(set((params.get("stop_token_ids") or []) + [self.turn_end_id])),
                )
                with simple_timer("generate_sequences", metrics):
                    result = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=runtime,
                        sampling_params=params,
                        priority=int(priority),
                    )
                if not result.token_ids:
                    raise RuntimeError("Empty or aborted Search-R1 generation")
                for key, value in getattr(result, "extra_fields", {}).items():
                    if key in {"min_global_steps", "max_global_steps"} and value is not None:
                        previous = generation_metadata.get(key)
                        combine = min if key == "min_global_steps" else max
                        generation_metadata[key] = value if previous is None else combine(previous, value)
                    else:
                        generation_metadata[key] = value
                if len(result.token_ids) > max_tokens:
                    raise RuntimeError("Generation exceeded its token budget")
                generated = result.token_ids
                final_text = self.tokenizer.decode(generated, skip_special_tokens=True)
                messages.append({"role": "assistant", "content": final_text})
                actions += 1
                if logprobs is not None and (result.log_probs is None or len(result.log_probs) != len(generated)):
                    raise RuntimeError("Missing or misaligned rollout logprobs")
                merged, mask, logprobs = await self.ct_merge_assistant_token(
                    runtime,
                    generated,
                    mask,
                    logprobs,
                    assistant_logprobs=result.log_probs if logprobs is not None else None,
                )
                runtime = merged.token_ids
                kind, value = parse_action(final_text)
                if kind == "answer":
                    reason = "answer"
                    break
                if kind == "invalid" and generated[-1] in {self.tokenizer.eos_token_id, self.turn_end_id}:
                    reason = "eos"
                    break
                if action + 1 >= self.max_actions:
                    reason = "action_budget"
                    break
                if kind == "search":
                    with simple_timer("tool_calls", metrics):
                        response, _, info = await self.search.execute(instance, {"query": value})
                    searches += 1
                    tool_metrics.append(info)
                    observation = response.text or "No passages found."
                else:
                    observation = (
                        "Invalid action. Emit one nonempty <search>query</search> "
                        "or <answer>final answer</answer> outside <think> tags."
                    )
                # sd-search counts the documents wrapper inside its observation
                # budget. A partial closing XML tag is still inside a distinct
                # tool message, whose chat-template boundary is always complete.
                if self.observation_tag == "documents":
                    observation = f"\n\n<documents>{observation.strip()}</documents>\n\n"
                observation_ids = self.tokenizer.encode(observation, add_special_tokens=False)
                if len(observation_ids) > self.max_observation_tokens:
                    observation = self.tokenizer.decode(
                        observation_ids[: self.max_observation_tokens],
                        skip_special_tokens=True,
                    )
                previous = list(messages)
                updated = previous + [
                    {
                        "role": "tool",
                        "name": "search",
                        "content": observation
                        if self.observation_tag == "documents"
                        else (f"<information>{observation}</information>"),
                    }
                ]
                # Stage the inserted assistant end and tool message before committing;
                # do not return a half tool message when the total budget is exhausted.
                candidate = list(runtime)
                next_mask = list(mask)
                next_probs = list(logprobs) if logprobs is not None else None
                if candidate[-len(self.turn_end) :] != self.turn_end:
                    candidate.extend(self.turn_end)
                    next_mask.extend([0] * len(self.turn_end))
                    if next_probs is not None:
                        next_probs.extend([0.0] * len(self.turn_end))
                merged, next_mask, next_probs = await self.ct_merge_context_msg(
                    previous,
                    updated,
                    candidate,
                    next_mask,
                    next_probs,
                )
                if len(next_mask) >= budget or len(merged.token_ids) >= max_context:
                    reason = "context_budget"
                    break
                runtime, mask, logprobs = merged.token_ids, next_mask, next_probs
                messages = updated
        finally:
            await self.search.release(instance)

        ground_truth = kwargs["reward_model"]["ground_truth"]
        outcome = score_final_turn(final_text, ground_truth, reason)
        outcome.update(search_calls=searches, assistant_turns=actions)
        if len(runtime) - len(initial) != len(mask) or (logprobs is not None and len(logprobs) != len(mask)):
            raise RuntimeError("Search-R1 token metadata lost alignment")
        metrics.setdefault("generate_sequences", 0.0)
        metrics.setdefault("tool_calls", 0.0)
        return AgentLoopOutput(
            prompt_ids=initial,
            response_ids=runtime[len(initial) :],
            response_mask=mask,
            response_logprobs=logprobs,
            reward_score=outcome["score"],
            num_turns=len(messages),
            metrics=metrics,
            extra_fields={
                **generation_metadata,
                "search_r1_messages": messages,
                "search_r1_final_assistant": final_text,
                "search_r1_termination": reason,
                "search_r1_reward_metric": reward_metric,
                "search_r1_tool_metrics": tool_metrics,
                "turn_scores": [],
                "tool_rewards": [],
                "reward_extra_info": outcome,
            },
        )
